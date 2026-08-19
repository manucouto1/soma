//! The seam with Python. It translates, it does not decide.
//!
//! The topology and the contract live in `soma_next_core`, which does not know
//! there is Python behind it. What this crate adds is the one thing the core
//! cannot have: the id → Python object map, and the two calling conventions. If
//! a domain rule ends up written here, it is in the wrong place.

mod node;
mod remote;
mod value;

use node::{PyAwait, PyCtx, PyDone, PyDriver, PyNode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use remote::PyWorker;
use soma_next_core::{
    Catalog, CompileError, Device, DeviceError, Executor, Graph, GraphError, Host, NodeId,
    Placement, RunError, compile, distribute,
};
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
    #[pyo3(signature = (input = None, *, driver = None, workers = None))]
    fn forward(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        driver: Option<&Bound<'_, PyAny>>,
        workers: Option<&Bound<'_, PyDict>>,
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

        let driver = driver.map(PyDriver::new).transpose()?;
        let mut executor = Executor::new(&self.catalog).placed(&self.placement);
        for (host, worker) in &reachable {
            executor = executor.reaching(host.clone(), worker.as_ref());
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
    m.add_function(wrap_pyfunction!(remote::serve, m)?)?;
    m.add_function(wrap_pyfunction!(remote::serve_provisioned, m)?)?;
    m.add_function(wrap_pyfunction!(remote::listen, m)?)?;
    m.add_function(wrap_pyfunction!(remote::listen_provisioned, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
