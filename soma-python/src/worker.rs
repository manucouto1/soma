//! `Worker` — serving a distributed worker from Python.

use crate::prelude::*;

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
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyWorker {
    #[new]
    #[pyo3(signature = (port=8080, tags=None, token=None, cpus=None, memory=None, gpus=None, max_concurrent=4, worker_id=None, coordinator=None))]
    fn new(
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
        }
    }

    /// Start the worker server (blocking). Releases the GIL so other threads can run.
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

                let worker = somatize_worker::Worker::new(&id, caps.clone()).with_python(&python);
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

    /// Get the worker info as a dict.
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
