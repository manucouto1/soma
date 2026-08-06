//! `Worker` — serving a distributed worker from Python.

use crate::prelude::*;

/// Tell the worker's `EnvManager` to use *this* process's Soma.
///
/// A worker builds an isolated venv per pipeline and installs `somatize`
/// into it from PyPI. When the working tree is ahead of the last release —
/// the normal state of a repository — PyPI has no such version and the
/// install falls back to the latest one there. The worker then runs a
/// different Soma than the one that pickled the filters: a
/// `DifferentiableFilter` whose `fit` trained in the older release trained
/// on the worker and diverged, while the same graph run locally did
/// nothing of the sort, and no message anywhere mentioned a version.
///
/// A worker started from Python should run that Python's Soma. Only set
/// when the caller has not said otherwise, and only when the directory
/// really holds the package.
fn point_env_manager_at_this_soma(py: Python<'_>) {
    if std::env::var_os("SOMA_LOCAL_PACKAGE").is_some() {
        return;
    }
    let Ok(module) = py.import("soma") else {
        return;
    };
    let Ok(file) = module
        .getattr("__file__")
        .and_then(|f| f.extract::<String>())
    else {
        return;
    };
    // …/site-packages/soma/__init__.py → …/site-packages
    let Some(parent) = std::path::Path::new(&file)
        .parent()
        .and_then(|p| p.parent())
        .filter(|p| p.join("soma").join("__init__.py").is_file())
    else {
        return;
    };
    // SAFETY: set once, during construction, before any worker thread that
    // reads it exists.
    unsafe { std::env::set_var("SOMA_LOCAL_PACKAGE", parent) };
}

// ── PyWorker ──

/// A Soma worker that can be started from Python.
///
/// Usage:
///   from soma import Worker
///   w = Worker(port=8080, tags=["gpu", "training"], token="sk-xxx")
///   w.serve()  # blocks, serving requests
#[pyclass(name = "Worker")]
pub(crate) struct PyWorker {
    port: u16,
    tags: Vec<String>,
    token: Option<String>,
    cpus: Option<usize>,
    memory: Option<u64>,
    gpus: Option<usize>,
    max_concurrent: usize,
    worker_id: Option<String>,
    coordinator: Option<String>,
    /// Built in `serve`, not here: the Rust worker is constructed on its
    /// own thread, and an `Arc<dyn DataStore>` is easier to make there
    /// than to carry across.
    data_store: Option<StoreConfig>,
}

/// What `set_data_store` was told, kept until `serve` can act on it.
#[derive(Clone)]
struct StoreConfig {
    store_type: String,
    path: Option<String>,
    bucket: Option<String>,
    prefix: Option<String>,
    endpoint: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    cache_dir: Option<String>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyWorker {
    #[new]
    #[pyo3(signature = (port=8080, tags=None, token=None, cpus=None, memory=None, gpus=None, max_concurrent=4, worker_id=None, coordinator=None))]
    fn new(
        py: Python<'_>,
        port: u16,
        tags: Option<Vec<String>>,
        token: Option<String>,
        cpus: Option<usize>,
        memory: Option<u64>,
        gpus: Option<usize>,
        max_concurrent: usize,
        worker_id: Option<String>,
        coordinator: Option<String>,
    ) -> Self {
        point_env_manager_at_this_soma(py);
        Self {
            port,
            tags: tags.unwrap_or_default(),
            token,
            cpus,
            memory,
            gpus,
            max_concurrent,
            worker_id,
            coordinator,
            data_store: None,
        }
    }

    /// Start the worker server (blocking). Releases the GIL so other threads can run.
    /// Give this worker a DataStore, so it can resolve the references a
    /// client sends instead of the data itself.
    ///
    /// Same arguments as `Graph.set_data_store`, and that is the whole
    /// point: the two ends of a transfer have to be told about the same
    /// store. Configuring only the client uploads the payload to a bucket
    /// the worker cannot open, and the plan then fails on the worker
    /// naming the reference it could not resolve.
    ///
    ///   w = soma.Worker(port=8080)
    ///   w.set_data_store("s3", bucket="my-lab", prefix="exp/",
    ///                    endpoint="...", access_key="...", secret_key="...")
    ///   w.serve()
    #[pyo3(signature = (store_type, path=None, bucket=None, prefix=None, endpoint=None, access_key=None, secret_key=None, cache_dir=None))]
    fn set_data_store(
        &mut self,
        store_type: String,
        path: Option<String>,
        bucket: Option<String>,
        prefix: Option<String>,
        endpoint: Option<String>,
        access_key: Option<String>,
        secret_key: Option<String>,
        cache_dir: Option<String>,
    ) -> PyResult<()> {
        // Built once here as well, so a bad configuration is refused at the
        // call rather than inside a thread nobody is watching.
        crate::data::store::build_data_store(
            &store_type,
            path.clone(),
            bucket.clone(),
            prefix.clone(),
            endpoint.clone(),
            access_key.clone(),
            secret_key.clone(),
            cache_dir.clone(),
        )?;
        self.data_store = Some(StoreConfig {
            store_type,
            path,
            bucket,
            prefix,
            endpoint,
            access_key,
            secret_key,
            cache_dir,
        });
        Ok(())
    }

    fn serve(&self, py: Python<'_>) -> PyResult<()> {
        // This interpreter, not whatever `python3` resolves to. A filter
        // arrives cloudpickled by the process that built the graph; only
        // an interpreter of the same version reliably reconstructs it,
        // and a mismatch surfaces inside a subprocess as
        // `'dict' object is not callable` with nothing naming the cause.
        let python: String = py
            .import("sys")?
            .getattr("executable")?
            .extract()
            .unwrap_or_else(|_| "python3".to_string());
        let port = self.port;
        let tags = self.tags.clone();
        let token = self.token.clone();
        let cpus = self.cpus;
        let memory = self.memory;
        let gpus = self.gpus;
        let max_concurrent = self.max_concurrent;
        let worker_id = self.worker_id.clone();
        let coordinator = self.coordinator.clone();
        let store = match &self.data_store {
            Some(c) => Some(crate::data::store::build_data_store(
                &c.store_type,
                c.path.clone(),
                c.bucket.clone(),
                c.prefix.clone(),
                c.endpoint.clone(),
                c.access_key.clone(),
                c.secret_key.clone(),
                c.cache_dir.clone(),
            )?),
            None => None,
        };

        // Build the runtime in a new thread; release GIL so other Python threads can run.
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                // Auto-detect capabilities
                let mut caps = somatize_worker::protocol::Capabilities::detect();
                let limits = somatize_worker::detect::ResourceLimits {
                    max_cpus: cpus,
                    max_memory_bytes: memory,
                    max_gpus: gpus,
                    max_concurrent,
                };
                caps = caps.with_limits(&limits);
                for tag in &tags {
                    if !caps.tags.contains(tag) {
                        caps.tags.push(tag.clone());
                    }
                }

                let id = worker_id.unwrap_or_else(|| {
                    hostname::get()
                        .map(|h| h.to_string_lossy().to_string())
                        .unwrap_or_else(|_| format!("worker_{}", std::process::id()))
                });

                eprintln!("Soma worker '{id}' starting on port {port}");
                eprintln!("Capabilities: {}", caps.summary());

                let mut worker =
                    somatize_worker::Worker::new(&id, caps.clone()).with_python(&python);
                if let Some(store) = store {
                    eprintln!("DataStore configured: references will be resolved");
                    worker = worker.with_data_store(store);
                }
                let addr = format!("0.0.0.0:{port}");

                // Register with coordinator if configured
                if let Some(coord_url) = &coordinator {
                    let url = format!("{coord_url}/register");
                    let body = serde_json::json!({
                        "worker_id": id,
                        "address": format!("ws://0.0.0.0:{port}"),
                        "capabilities": caps,
                    });
                    let mut req = reqwest::Client::new().post(&url).json(&body);
                    if let Some(t) = &token {
                        req = req.query(&[("token", t.as_str())]);
                    }
                    match req.send().await {
                        Ok(resp) if resp.status().is_success() => {
                            eprintln!("Registered with coordinator at {coord_url}");
                        }
                        Ok(resp) => {
                            eprintln!("Coordinator registration failed: {}", resp.status());
                        }
                        Err(e) => {
                            eprintln!("Could not reach coordinator: {e}");
                        }
                    }
                }

                if let Some(t) = token {
                    eprintln!("Authentication enabled");
                    somatize_worker::serve_worker_authenticated(worker, &addr, &t)
                        .await
                        .unwrap();
                } else {
                    somatize_worker::serve_worker(worker, &addr).await.unwrap();
                }
            });
        });

        // Release GIL while waiting for server thread (allows other Python threads to proceed)
        py.allow_threads(|| {
            handle
                .join()
                .map_err(|_| PyRuntimeError::new_err("worker thread panicked"))
        })?;

        Ok(())
    }

    /// A one-line summary of what this worker can run: cpus, memory, gpus.
    fn info(&self) -> PyResult<String> {
        let caps = somatize_worker::protocol::Capabilities::detect();
        Ok(caps.summary())
    }

    fn __repr__(&self) -> String {
        format!(
            "Worker(port={}, tags={:?}, auth={})",
            self.port,
            self.tags,
            self.token.is_some()
        )
    }
}
