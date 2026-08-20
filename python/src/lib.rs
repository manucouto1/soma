//! The seam with Python. It translates, it does not decide.
//!
//! The topology and the contract live in `soma_next_core`, which does not know
//! there is Python behind it. What this crate adds is the one thing the core
//! cannot have: the id → Python object map, and the two calling conventions. If
//! a domain rule ends up written here, it is in the wrong place.

mod codec;
mod node;
mod remote;
mod value;

use codec::Packing;
use node::{PyAwait, PyCtx, PyDone, PyDriver, PyNode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use remote::PyWorker;
use soma_next_core::{
    Catalog, CompileError, Device, DeviceError, Executor, Graph, GraphError, Host, Memory,
    MemoryError, NodeId, Placement, RunError, cacheable, compile, distribute,
};
use soma_next_store::{Cache, Local};
use std::collections::HashMap;
use std::sync::Arc;

/// Translates the core's error into the exception a Python user expects.
fn to_py_err(e: GraphError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The same for a name that names no place to execute.
fn device_err(e: DeviceError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The same for a failure while deciding the shape of the execution.
fn compile_err(e: CompileError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The same for a cache that was declared where it cannot be honoured. It is
/// raised **before** the first node runs, which is the whole point of asking.
fn memory_err(e: MemoryError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The same for a store that cannot be opened.
fn store_err(e: soma_next_store::StoreError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The same for a run failure.
fn run_err(e: RunError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `soma_next.Graph` — the core's topology plus the implementations.
#[pyclass(name = "Graph", module = "soma_next._soma_next", subclass)]
struct PyGraph {
    graph: Graph,
    /// What the engine executes.
    catalog: Catalog,
    /// Where each node runs, for those it was said about.
    placement: Placement,
    /// What is remembered about each node: what it is, whether it is settled,
    /// whether its output is worth keeping.
    memory: Memory,
    /// The object exactly as the user passed it, unwrapped, so it can be
    /// handed back.
    implementations: HashMap<String, PyObject>,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            catalog: Catalog::new(),
            placement: Placement::new(),
            memory: Memory::new(),
            implementations: HashMap::new(),
        }
    }

    /// Adds a node — any object with `forward(input, ctx)` — and returns its
    /// id. `node(obj)` names it for you, `node("id", obj)` you name it.
    #[pyo3(signature = (*args))]
    fn node(&mut self, args: &Bound<'_, PyTuple>) -> PyResult<String> {
        let (id, implementation) = self.name_and_object(args)?;

        // Before touching the graph: an object that cannot be a node fails here.
        let wrapped = PyNode::new(&implementation)?;

        self.graph.add_node(id.clone()).map_err(to_py_err)?;
        // What implements it, said here because here is where the object is:
        // the class's name is half of what a key is built out of.
        self.memory
            .identify(id.clone(), type_name(&implementation)?);
        self.catalog.insert(id.clone(), Arc::new(wrapped));
        self.implementations
            .insert(id.to_string(), implementation.unbind());
        Ok(id.to_string())
    }

    /// Connects two nodes. Both have to exist already.
    fn edge(&mut self, source: &str, target: &str) -> PyResult<()> {
        self.graph.add_edge(source, target).map_err(to_py_err)?;
        Ok(())
    }

    /// Places a node on a device. The primitive the DSL's `.on()` ends up
    /// calling, and what you use when you only have the id.
    fn place(&mut self, node_id: &str, device: &str) -> PyResult<()> {
        let id = self.known(node_id)?;
        let device: Device = device.parse().map_err(device_err)?;
        self.placement.place(id, device);
        Ok(())
    }

    /// Sends a node to a host, by name: the other half of `place`, independent
    /// of it. What the name resolves to is decided in `forward(workers=…)`.
    fn place_at(&mut self, node_id: &str, host: &str) -> PyResult<()> {
        let id = self.known(node_id)?;
        self.placement.place_at(id, Host::from(host));
        Ok(())
    }

    /// Says this node's state does not change from here on, with the digest of
    /// the state it is settled at if whoever calls knows how to hash weights.
    ///
    /// The primitive `.frozen()` ends at, and it **declares**: making it true is
    /// `soma_next.torch.freeze`, exactly as moving a tensor to a GPU is the
    /// node's job and not the core's.
    #[pyo3(signature = (node_id, state = None))]
    fn freeze(&mut self, node_id: &str, state: Option<String>) -> PyResult<()> {
        let id = self.known(node_id)?;
        self.memory.freeze(id, state);
        Ok(())
    }

    /// Says this node's output is worth keeping, with the salt that tells apart
    /// two runs the key cannot tell apart on its own.
    #[pyo3(signature = (node_id, salt = None))]
    fn cache(&mut self, node_id: &str, salt: Option<String>) -> PyResult<()> {
        let id = self.known(node_id)?;
        self.memory.cache(id, salt);
        Ok(())
    }

    /// Notes which version of the code this graph was written against.
    /// **Metadata**: never in a key, compared on a hit and said on `stderr` if
    /// it differs.
    fn written_as(&mut self, node_id: &str, fingerprint: &str) -> PyResult<()> {
        let id = self.known(node_id)?;
        self.memory.written_as(id, fingerprint);
        Ok(())
    }

    /// Which nodes are settled, and at what state — `None` for one with none.
    fn frozen<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.about(py, |memory, id| {
            memory
                .is_frozen(id)
                .then(|| memory.state_of(id).map(str::to_string))
        })
    }

    /// Which nodes are kept, and under what salt.
    fn cached<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.about(py, |memory, id| {
            memory
                .is_cached(id)
                .then(|| memory.salt_of(id).map(str::to_string))
        })
    }

    /// What implements each node, by name.
    fn identities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.about(py, |memory, id| {
            memory.identity_of(id).map(|what| Some(what.to_string()))
        })
    }

    /// Which version of the code each node was written against, for those
    /// where it was noted.
    fn fingerprints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.about(py, |memory, id| {
            memory.fingerprint_of(id).map(|what| Some(what.to_string()))
        })
    }

    /// Which host each node sent away runs on, in declaration order.
    fn hosts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for id in self.graph.nodes() {
            if let Some(host) = self.placement.host_of(id) {
                out.set_item(id.to_string(), host.to_string())?;
            }
        }
        Ok(out)
    }

    /// Where each placed node runs, in declaration order.
    fn devices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for id in self.graph.nodes() {
            if let Some(device) = self.placement.of(id) {
                out.set_item(id.to_string(), device.to_string())?;
            }
        }
        Ok(out)
    }

    /// The ids, in insertion order.
    fn nodes(&self) -> Vec<String> {
        self.graph.nodes().iter().map(NodeId::to_string).collect()
    }

    /// The edges as `(source, target)` pairs, in insertion order.
    fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edges()
            .iter()
            .map(|e| (e.source.to_string(), e.target.to_string()))
            .collect()
    }

    /// The nodes where execution enters.
    fn roots(&self) -> Vec<String> {
        self.graph
            .roots()
            .into_iter()
            .map(NodeId::to_string)
            .collect()
    }

    /// The nodes where it leaves.
    fn leaves(&self) -> Vec<String> {
        self.graph
            .leaves()
            .into_iter()
            .map(NodeId::to_string)
            .collect()
    }

    /// The nodes feeding into `node_id`.
    fn predecessors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let id = self.known(node_id)?;
        Ok(self
            .graph
            .predecessors(&id)
            .into_iter()
            .map(NodeId::to_string)
            .collect())
    }

    /// The nodes `node_id` feeds into.
    fn successors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let id = self.known(node_id)?;
        Ok(self
            .graph
            .successors(&id)
            .into_iter()
            .map(NodeId::to_string)
            .collect())
    }

    /// The nodes in an order where each comes after its predecessors.
    fn topological_sort(&self) -> Vec<String> {
        self.graph
            .topological_sort()
            .into_iter()
            .map(NodeId::to_string)
            .collect()
    }

    /// The object you registered under `node_id`, or `None`.
    fn implementation(&self, node_id: &str) -> Option<&PyObject> {
        self.implementations.get(node_id)
    }

    /// How this graph will be walked: the decided shape, already distributed.
    /// With no host set, `distribute` changes nothing.
    fn plan(&self) -> PyResult<String> {
        let plan = compile(&self.graph, &self.catalog).map_err(compile_err)?;
        Ok(format!("{:?}", distribute(&plan, &self.placement)))
    }

    /// Executes the whole graph and returns what it produced.
    ///
    /// `driver=` is an object with `perform(requests)`; `workers=` says what
    /// each host resolves to. A node sent to a host that is not there is **not
    /// executed here just in case**.
    ///
    /// `store=` is a directory: with one, whatever was declared `.cached()` is
    /// looked up before being computed and kept after. Whether what was
    /// declared can honestly be kept is asked **here**, before the first node
    /// runs, so a `.cached()` in the wrong place fails at once and not as a net
    /// that quietly stopped training.
    #[pyo3(signature = (input = None, *, driver = None, workers = None, store = None))]
    fn forward(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        driver: Option<&Bound<'_, PyAny>>,
        workers: Option<&Bound<'_, PyDict>>,
        store: Option<&str>,
    ) -> PyResult<PyObject> {
        let start = match input {
            Some(obj) => value::from_py(obj)?,
            None => soma_next_core::Value::Null,
        };
        let plan = compile(&self.graph, &self.catalog).map_err(compile_err)?;
        let plan = distribute(&plan, &self.placement);

        // Before releasing the GIL: a `PyRef` cannot survive `allow_threads`.
        let reachable: Vec<(Host, std::sync::Arc<soma_next_transport::Worker>)> = match workers {
            None => Vec::new(),
            Some(dict) => dict
                .iter()
                .map(|(host, worker)| {
                    let host = Host::from(host.extract::<String>()?);
                    let worker = worker.downcast::<PyWorker>().map_err(|_| {
                        PyValueError::new_err(format!(
                            "`workers` takes a dict from host to Worker; for `{host}` something else arrived"
                        ))
                    })?;
                    Ok((host, worker.get().transport()))
                })
                .collect::<PyResult<Vec<_>>>()?,
        };

        // Before anything else, and **whether or not there is a store**: a
        // `.cached()` over something that can still change is wrong today, not
        // the day somebody adds a directory to the call. It costs a walk of the
        // graph, and only to whoever declared a cache.
        if self.keeps_anything() {
            cacheable(&self.graph, &self.memory).map_err(memory_err)?;
        }
        let kept = store.map(Local::at).transpose().map_err(store_err)?;
        let cache = kept.as_ref().map(|kept| Cache::over(kept));
        // The codecs in front, so what reaches the store is bytes and the store
        // never learns Python exists.
        let packing = cache.as_ref().map(|cache| Packing::over(cache));

        let driver = driver.map(PyDriver::new).transpose()?;
        // What is remembered always goes in, keeper or no keeper: it is the
        // graph's, and it travels with a slice sent to a worker that may well be
        // the only one keeping anything.
        let mut executor = Executor::new(&self.catalog)
            .placed(&self.placement)
            .remembering(&self.memory);
        for (host, worker) in &reachable {
            executor = executor.reaching(host.clone(), worker.as_ref());
        }
        if let Some(packing) = &packing {
            executor = executor.keeping(packing);
        }
        let executor = match &driver {
            Some(d) => executor.with_driver(d),
            None => executor,
        };

        // Mandatory, not an optimization: a wave spawns threads that call
        // Python `forward`s, and they would all hang waiting for the GIL.
        let out = py
            .allow_threads(|| executor.run(&plan, start))
            .map_err(run_err)?;
        value::to_py(py, &out)
    }

    fn __len__(&self) -> usize {
        self.graph.len()
    }

    fn __contains__(&self, node_id: &str) -> bool {
        self.graph.contains(&NodeId::from(node_id))
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph({} nodes, {} edges)",
            self.graph.len(),
            self.graph.edges().len()
        )
    }
}

impl PyGraph {
    /// Reads `(obj)` or `(id, obj)`. Without an id the name comes from the
    /// class: `CleanText` → `clean_text`, suffixed if already taken.
    fn name_and_object<'py>(
        &self,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<(NodeId, Bound<'py, PyAny>)> {
        match args.len() {
            1 => {
                let obj = args.get_item(0)?;
                let wanted = snake_case(&type_name(&obj)?);
                Ok((self.graph.free_id(&wanted), obj))
            }
            2 => Ok((
                NodeId::from(args.get_item(0)?.extract::<String>()?),
                args.get_item(1)?,
            )),
            n => Err(PyValueError::new_err(format!(
                "takes (object) or (id, object), not {n} arguments"
            ))),
        }
    }

    /// What is remembered about each node that has anything said about it, in
    /// declaration order. The shape of `devices()` and `hosts()`, written once
    /// for the four questions it answers.
    fn about<'py>(
        &self,
        py: Python<'py>,
        what: impl Fn(&Memory, &NodeId) -> Option<Option<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for id in self.graph.nodes() {
            if let Some(said) = what(&self.memory, id) {
                out.set_item(id.to_string(), said)?;
            }
        }
        Ok(out)
    }

    /// Whether any node of this graph says its output is worth keeping. What
    /// the checks before a run hang off: a graph that declares nothing pays for
    /// nothing.
    fn keeps_anything(&self) -> bool {
        self.graph
            .nodes()
            .iter()
            .any(|id| self.memory.is_cached(id))
    }

    /// An id we know is in the graph, or the same error the core would give.
    fn known(&self, node_id: &str) -> PyResult<NodeId> {
        let id = NodeId::from(node_id);
        if self.graph.contains(&id) {
            Ok(id)
        } else {
            Err(to_py_err(GraphError::UnknownNode(id)))
        }
    }
}

/// The name of a Python object's class.
fn type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type().name()?.extract()
}

/// `CleanText` → `clean_text`, only for default names.
fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[pymodule]
fn _soma_next(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyCtx>()?;
    m.add_class::<PyDone>()?;
    m.add_class::<PyAwait>()?;
    m.add_class::<value::PyOpaque>()?;
    m.add_class::<PyWorker>()?;
    m.add_function(wrap_pyfunction!(codec::codec, m)?)?;
    m.add_function(wrap_pyfunction!(codec::codecs_registered, m)?)?;
    m.add_function(wrap_pyfunction!(remote::serve, m)?)?;
    m.add_function(wrap_pyfunction!(remote::serve_provisioned, m)?)?;
    m.add_function(wrap_pyfunction!(remote::listen, m)?)?;
    m.add_function(wrap_pyfunction!(remote::listen_provisioned, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
