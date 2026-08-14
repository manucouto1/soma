//! La costura con Python. Traduce, no decide.
//!
//! La topología vive en `soma_next_core`, que no sabe qué es un filtro. Lo que
//! este crate añade es lo único que el núcleo no puede tener: el mapa de
//! id → objeto Python. Si una regla del dominio acaba escrita aquí, está en el
//! sitio equivocado.

mod filter;
mod step;
mod value;

use filter::PyFilter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use soma_next_core::{
    Catalog, CompileError, Executor, Graph, GraphError, NodeId, RunError, compile,
};
use std::collections::HashMap;
use std::sync::Arc;
use step::{PyDriver, PyStep};

/// Traduce el error del núcleo a la excepción que un usuario de Python espera.
fn to_py_err(e: GraphError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Lo mismo para un fallo al decidir la forma de la ejecución.
fn compile_err(e: CompileError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Lo mismo para un fallo de ejecución.
fn run_err(e: RunError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `soma_next.Graph` — la topología del núcleo más las implementaciones.
#[pyclass(name = "Graph", module = "soma_next._soma_next", subclass)]
struct PyGraph {
    graph: Graph,
    /// Lo que el motor ejecuta.
    catalog: Catalog,
    /// El objeto tal cual lo pasó el usuario, para poder devolvérselo.
    /// No es una copia del catálogo: guarda otra cosa, el original sin envolver.
    implementations: HashMap<String, PyObject>,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            catalog: Catalog::new(),
            implementations: HashMap::new(),
        }
    }

    /// Añade un nodo: `node(filtro)` le pone nombre, `node("id", filtro)` lo
    /// nombras tú. Devuelve el id, que es lo que necesitas para `edge`.
    #[pyo3(signature = (*args))]
    fn node(&mut self, args: &Bound<'_, PyTuple>) -> PyResult<String> {
        let (id, implementation) = self.name_and_object(args)?;

        // Antes de tocar el grafo: un objeto que no puede ser nodo falla aquí,
        // no a mitad de un run.
        let wrapped = PyFilter::new(&implementation)?;

        self.graph.add_node(id.clone()).map_err(to_py_err)?;
        self.catalog.insert_filter(id.clone(), Arc::new(wrapped));
        self.implementations
            .insert(id.to_string(), implementation.unbind());
        Ok(id.to_string())
    }

    /// Conecta dos nodos. Los dos tienen que existir ya.
    fn edge(&mut self, source: &str, target: &str) -> PyResult<()> {
        self.graph.add_edge(source, target).map_err(to_py_err)?;
        Ok(())
    }

    /// Los ids, en orden de inserción.
    fn nodes(&self) -> Vec<String> {
        self.graph.nodes().iter().map(NodeId::to_string).collect()
    }

    /// Las aristas como pares `(origen, destino)`, en orden de inserción.
    fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edges()
            .iter()
            .map(|e| (e.source.to_string(), e.target.to_string()))
            .collect()
    }

    /// Los nodos por donde entra la ejecución.
    fn roots(&self) -> Vec<String> {
        self.graph
            .roots()
            .into_iter()
            .map(NodeId::to_string)
            .collect()
    }

    /// Los nodos por donde sale.
    fn leaves(&self) -> Vec<String> {
        self.graph
            .leaves()
            .into_iter()
            .map(NodeId::to_string)
            .collect()
    }

    /// Los nodos que entran en `node_id`.
    fn predecessors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let id = self.known(node_id)?;
        Ok(self
            .graph
            .predecessors(&id)
            .into_iter()
            .map(NodeId::to_string)
            .collect())
    }

    /// Los nodos a los que sale `node_id`.
    fn successors(&self, node_id: &str) -> PyResult<Vec<String>> {
        let id = self.known(node_id)?;
        Ok(self
            .graph
            .successors(&id)
            .into_iter()
            .map(NodeId::to_string)
            .collect())
    }

    /// Los nodos en un orden en que cada uno va después de sus predecesores.
    fn topological_sort(&self) -> Vec<String> {
        self.graph
            .topological_sort()
            .into_iter()
            .map(NodeId::to_string)
            .collect()
    }

    /// El objeto que registraste bajo `node_id`, o `None`.
    fn implementation(&self, node_id: &str) -> Option<&PyObject> {
        self.implementations.get(node_id)
    }

    /// Añade un step: un objeto con `poll(ctx)`, que puede pedir cosas antes
    /// de terminar. `step(obj)` le pone nombre, `step("id", obj)` lo nombras tú.
    #[pyo3(signature = (*args))]
    fn step(&mut self, args: &Bound<'_, PyTuple>) -> PyResult<String> {
        let (id, implementation) = self.name_and_object(args)?;
        let wrapped = PyStep::new(&implementation)?;

        self.graph.add_node(id.clone()).map_err(to_py_err)?;
        self.catalog.insert_step(id.clone(), Arc::new(wrapped));
        self.implementations
            .insert(id.to_string(), implementation.unbind());
        Ok(id.to_string())
    }

    /// Cómo se va a recorrer este grafo, tal cual lo decide el compilador.
    fn plan(&self) -> PyResult<String> {
        Ok(format!(
            "{:?}",
            compile(&self.graph, &self.catalog).map_err(compile_err)?
        ))
    }

    /// Ejecuta el grafo entero y devuelve lo que produjo.
    ///
    /// Con `driver=` se le da a los steps quien atienda lo que pidan: un objeto
    /// con `perform(peticiones)` que devuelva un resultado por cada una.
    #[pyo3(signature = (input = None, *, driver = None))]
    fn forward(
        &self,
        py: Python<'_>,
        input: Option<&Bound<'_, PyAny>>,
        driver: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let start = match input {
            Some(obj) => value::from_py(obj)?,
            None => soma_next_core::Value::Null,
        };
        let plan = compile(&self.graph, &self.catalog).map_err(compile_err)?;

        let driver = driver.map(PyDriver::new).transpose()?;
        let executor = Executor::new(&self.catalog);
        let executor = match &driver {
            Some(d) => executor.with_driver(d),
            None => executor,
        };

        let out = executor.run(&plan, start).map_err(run_err)?;
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
            "Graph({} nodos, {} aristas)",
            self.graph.len(),
            self.graph.edges().len()
        )
    }
}

impl PyGraph {
    /// Lee `(obj)` o `(id, obj)`, que es como se añade cualquier nodo.
    ///
    /// Sin id, el nombre sale de la clase del objeto: `LimpiarTexto` →
    /// `limpiar_texto`, sufijando si ya estaba cogido.
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
                "toma (objeto) o (id, objeto), no {n} argumentos"
            ))),
        }
    }

    /// Un id que sabemos que está en el grafo, o el mismo error que daría el núcleo.
    fn known(&self, node_id: &str) -> PyResult<NodeId> {
        let id = NodeId::from(node_id);
        if self.graph.contains(&id) {
            Ok(id)
        } else {
            Err(to_py_err(GraphError::UnknownNode(id)))
        }
    }
}

/// El nombre de la clase de un objeto Python.
fn type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type().name()?.extract()
}

/// `LimpiarTexto` → `limpiar_texto`. Es solo para poner nombres por defecto.
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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
