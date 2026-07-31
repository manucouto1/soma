//! The agentic surface, as Python sees it.
//!
//! Deliberately small. A user should be able to write
//!
//! ```python
//! g.step("researcher", soma.Agent(model="ollama/llama3.2", tools=[search]))
//! ```
//!
//! without meeting `Effect`, `Transition` or `StepCtx` — those are the
//! substrate, and the substrate is not the interface. The escape hatch for
//! anyone who does need them is `PyStepBridge`, which lets a Python class
//! implement `Step` directly.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use somatize_core::error::{Result as SomaResult, SomaError};
use somatize_core::step::Step;
use somatize_core::tool::ToolSpec;
use somatize_core::value::Value;
use somatize_llm::tools::ToolOutcome;
use somatize_llm::{JudgeStep, ReactStep};
use std::sync::Arc;

use crate::py_to_value;

/// A tool backed by a Python callable.
///
/// The function is called with keyword arguments taken from the model's
/// JSON, so an ordinary Python signature works unchanged:
/// `def search(query: str) -> str`.
#[pyclass(name = "Tool", module = "soma")]
pub struct PyTool {
    pub(crate) spec: ToolSpec,
    pub(crate) func: Py<PyAny>,
}

// `Py<PyAny>` is only `Clone` behind a feature flag; `clone_ref` is the
// supported way to take another handle to the same object.
impl Clone for PyTool {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            spec: self.spec.clone(),
            func: self.func.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyTool {
    /// Wrap a callable. `schema` is JSON Schema for the arguments; when
    /// omitted, one is derived from the signature.
    #[new]
    #[pyo3(signature = (func, description=None, name=None, schema=None))]
    fn new(
        py: Python<'_>,
        func: Py<PyAny>,
        description: Option<String>,
        name: Option<String>,
        schema: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let bound = func.bind(py);

        let name = match name {
            Some(n) => n,
            None => bound
                .getattr("__name__")
                .and_then(|n| n.extract::<String>())
                .map_err(|_| {
                    PyValueError::new_err(
                        "a tool needs a name: pass `name=` or use a named function",
                    )
                })?,
        };

        // The description is what the model reads to decide whether to call
        // this. A docstring is the natural place for it, so use one when the
        // caller does not pass something better.
        let description = description
            .or_else(|| {
                bound
                    .getattr("__doc__")
                    .ok()
                    .and_then(|d| d.extract::<String>().ok())
                    .map(|d| d.trim().to_string())
                    .filter(|d| !d.is_empty())
            })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "tool `{name}` has no description. Give it a docstring, or pass \
                     `description=`. A model chooses tools by their descriptions, so an \
                     undescribed tool is one it will not use"
                ))
            })?;

        let input_schema = match schema {
            Some(s) => py_to_json(py, s.bind(py))?,
            None => schema_from_signature(py, bound)?,
        };

        Ok(Self {
            spec: ToolSpec::new(name, description, input_schema),
            func,
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }

    #[getter]
    fn description(&self) -> &str {
        &self.spec.description
    }

    /// The JSON Schema advertised to the model.
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        json_to_py(py, &self.spec.input_schema)
    }

    fn __repr__(&self) -> String {
        format!("Tool({})", self.spec.name)
    }
}

impl PyTool {
    pub(crate) fn tool_name(&self) -> &str {
        &self.spec.name
    }

    pub(crate) fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    /// Call the Python function with the model's arguments as kwargs.
    pub(crate) fn call(&self, args: &Value) -> SomaResult<ToolOutcome> {
        Python::with_gil(|py| {
            let kwargs = PyDict::new(py);
            if let serde_json::Value::Object(map) = args.to_plain_json() {
                for (key, value) in map {
                    let converted = json_to_py(py, &value)
                        .map_err(|e| SomaError::Other(format!("tool argument `{key}`: {e}")))?;
                    kwargs
                        .set_item(key, converted)
                        .map_err(|e| SomaError::Other(e.to_string()))?;
                }
            }

            match self.func.bind(py).call((), Some(&kwargs)) {
                Ok(result) => {
                    // A tool returning a string is the common case and needs
                    // no ceremony; anything else goes through the ordinary
                    // Value conversion.
                    let value = match result.extract::<String>() {
                        Ok(text) => Value::text(text),
                        Err(_) => {
                            py_to_value(py, &result).map_err(|e| SomaError::Other(e.to_string()))?
                        }
                    };
                    Ok(ToolOutcome::ok(value))
                }
                // A Python exception is something the *model* should see and
                // work around, not something that ends the run.
                Err(e) => Ok(ToolOutcome::error(format!(
                    "{} raised: {e}",
                    self.spec.name
                ))),
            }
        })
    }
}

/// Presents a Python callable as a [`SomaTool`].
///
/// The driver runs tools on scoped threads, so this acquires the GIL per
/// call — the same discipline Python filters already follow in parallel
/// branches, and the reason the executor releases it before running a plan.
pub(crate) struct PyToolAdapter {
    pub(crate) tool: PyTool,
}

impl somatize_llm::tools::Tool for PyToolAdapter {
    fn spec(&self) -> ToolSpec {
        self.tool.spec().clone()
    }

    fn call(&self, args: &Value) -> SomaResult<ToolOutcome> {
        self.tool.call(args)
    }
}

// ── Searchable constructor arguments ──

/// Read a constructor argument that may be a `search(...)` descriptor.
///
/// A prompt, a model name and a turn budget are hyperparameters like any
/// other, and the interesting question about an agentic graph is usually
/// which of them to use. Writing the space where the value goes keeps the
/// two next to each other:
///
/// ```python
/// soma.Agent(model=soma.search(choices=["ollama/qwen2.5", "kimi/kimi-k2"]),
///            max_turns=soma.search(4, 16))
/// ```
///
/// The argument still resolves to a concrete value — a graph has to be
/// runnable before any study samples it — taken from the descriptor's
/// `default`, else its first choice, else its lower bound.
fn searchable<'py>(
    py: Python<'py>,
    field: &str,
    obj: &Bound<'py, PyAny>,
    space: &mut Vec<serde_json::Value>,
) -> PyResult<Bound<'py, PyAny>> {
    // A SearchDescriptor is the only thing carrying both of these.
    if !(obj.hasattr("to_dict")? && obj.hasattr("field_name")?) {
        return Ok(obj.clone());
    }

    let dim = obj.call_method0("to_dict")?;
    let dim = dim.downcast::<pyo3::types::PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "`{field}`: search descriptor produced no dimension"
        ))
    })?;
    // Declared at the call site, so it has no field name of its own yet.
    dim.set_item("name", field)?;
    space.push(py_to_json(py, dim.as_any())?);

    let default = obj.getattr("default")?;
    if !default.is_none() {
        return Ok(default);
    }
    let choices = obj.getattr("choices")?;
    if !choices.is_none() {
        return choices.get_item(0);
    }
    let low = obj.getattr("low")?;
    if !low.is_none() {
        return Ok(low);
    }
    Err(PyValueError::new_err(format!(
        "`{field}`: this search space has no value to start from. Give the \
         search() a `default=`, choices, or bounds"
    )))
}

/// Extract an optional argument that may itself be a search space.
fn searchable_opt<'py, T>(
    py: Python<'py>,
    field: &str,
    obj: Option<&Bound<'py, PyAny>>,
    space: &mut Vec<serde_json::Value>,
) -> PyResult<Option<T>>
where
    T: FromPyObject<'py>,
{
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => Ok(Some(searchable(py, field, o, space)?.extract::<T>()?)),
    }
}

/// A ReAct agent: asks a model, runs the tools it asks for, repeats.
#[pyclass(name = "Agent", module = "soma")]
pub struct PyAgent {
    model: String,
    system: Option<String>,
    tools: Vec<PyTool>,
    max_turns: usize,
    max_tokens: Option<u32>,
    effort: Option<String>,
    text_only: bool,
    /// Dimensions declared with `search(...)` at construction, named after
    /// the argument they were passed as.
    space: Vec<serde_json::Value>,
}

#[pymethods]
impl PyAgent {
    /// `model` is `provider/name` — `ollama/llama3.2`, `kimi/kimi-k2` — or a
    /// bare name when the graph has a default provider.
    ///
    /// Any of `model`, `system`, `max_turns`, `max_tokens` and `effort` may
    /// be a `search(...)` space instead of a value; see [`searchable`].
    #[new]
    #[pyo3(signature = (
        model,
        system=None,
        tools=None,
        max_turns=None,
        max_tokens=None,
        effort=None,
        text_only=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        system: Option<&Bound<'_, PyAny>>,
        tools: Option<Vec<PyTool>>,
        max_turns: Option<&Bound<'_, PyAny>>,
        max_tokens: Option<&Bound<'_, PyAny>>,
        effort: Option<&Bound<'_, PyAny>>,
        text_only: bool,
    ) -> PyResult<Self> {
        let mut space = Vec::new();
        Ok(Self {
            model: searchable(py, "model", model, &mut space)?.extract()?,
            system: searchable_opt(py, "system", system, &mut space)?,
            tools: tools.unwrap_or_default(),
            max_turns: searchable_opt(py, "max_turns", max_turns, &mut space)?.unwrap_or(12),
            max_tokens: searchable_opt(py, "max_tokens", max_tokens, &mut space)?,
            effort: searchable_opt(py, "effort", effort, &mut space)?,
            text_only,
            space,
        })
    }

    #[getter]
    fn model(&self) -> &str {
        &self.model
    }

    #[setter]
    fn set_model(&mut self, model: String) {
        self.model = model;
    }

    #[getter]
    fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    #[setter]
    fn set_system(&mut self, system: Option<String>) {
        self.system = system;
    }

    #[getter]
    fn max_turns(&self) -> usize {
        self.max_turns
    }

    #[setter]
    fn set_max_turns(&mut self, max_turns: usize) {
        self.max_turns = max_turns;
    }

    #[getter]
    fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    #[setter]
    fn set_max_tokens(&mut self, max_tokens: Option<u32>) {
        self.max_tokens = max_tokens;
    }

    #[getter]
    fn effort(&self) -> Option<&str> {
        self.effort.as_deref()
    }

    #[setter]
    fn set_effort(&mut self, effort: Option<String>) {
        self.effort = effort;
    }

    #[getter]
    fn tools(&self) -> Vec<PyTool> {
        self.tools.clone()
    }

    /// The dimensions this agent contributes to a study's search space.
    fn search_space(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.space.iter().map(|d| json_to_py(py, d)).collect()
    }

    fn __repr__(&self) -> String {
        format!("Agent(model={:?}, tools={})", self.model, self.tools.len())
    }
}

/// Grade something with a model against a rubric.
#[pyclass(name = "Judge", module = "soma")]
pub struct PyJudge {
    model: String,
    rubric: String,
    threshold: f64,
    space: Vec<serde_json::Value>,
}

#[pymethods]
impl PyJudge {
    /// A rubric should be explicitly gradeable — "the CSV has a numeric
    /// `price` column per SKU", not "the data looks good". The grader scores
    /// each criterion on its own, so vague criteria make a noisy metric.
    ///
    /// `model`, `rubric` and `threshold` may each be a `search(...)` space —
    /// how strictly to grade is a real thing to tune, and so is the wording
    /// of the rubric.
    #[new]
    #[pyo3(signature = (model, rubric, threshold=None))]
    fn new(
        py: Python<'_>,
        model: &Bound<'_, PyAny>,
        rubric: &Bound<'_, PyAny>,
        threshold: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut space = Vec::new();
        Ok(Self {
            model: searchable(py, "model", model, &mut space)?.extract()?,
            rubric: searchable(py, "rubric", rubric, &mut space)?.extract()?,
            threshold: searchable_opt(py, "threshold", threshold, &mut space)?.unwrap_or(0.8),
            space,
        })
    }

    #[getter]
    fn model(&self) -> &str {
        &self.model
    }

    #[setter]
    fn set_model(&mut self, model: String) {
        self.model = model;
    }

    #[getter]
    fn rubric(&self) -> &str {
        &self.rubric
    }

    #[setter]
    fn set_rubric(&mut self, rubric: String) {
        self.rubric = rubric;
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.threshold
    }

    #[setter]
    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// The dimensions this judge contributes to a study's search space.
    fn search_space(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.space.iter().map(|d| json_to_py(py, d)).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Judge(model={:?}, threshold={})",
            self.model, self.threshold
        )
    }
}

/// Anything that can become a node in the graph.
pub(crate) enum StepSpec {
    Agent {
        step: Arc<dyn Step>,
        tools: Vec<PyTool>,
    },
    Judge {
        step: Arc<dyn Step>,
    },
}

impl StepSpec {
    pub(crate) fn step(&self) -> Arc<dyn Step> {
        match self {
            Self::Agent { step, .. } | Self::Judge { step } => step.clone(),
        }
    }

    pub(crate) fn tools(&self) -> &[PyTool] {
        match self {
            Self::Agent { tools, .. } => tools,
            Self::Judge { .. } => &[],
        }
    }

    /// A short name for the node kind, shown in graph renderings.
    pub(crate) fn kind(&self) -> String {
        match self {
            Self::Agent { .. } => "Agent".into(),
            Self::Judge { .. } => "Judge".into(),
        }
    }
}

/// Turn a Python object into something the graph can hold.
pub(crate) fn to_step_spec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<StepSpec> {
    if let Ok(agent) = obj.extract::<PyRef<'_, PyAgent>>() {
        let mut react = ReactStep::new(&agent.model)
            .with_max_turns(agent.max_turns)
            .with_tools(agent.tools.iter().map(|t| t.spec.clone()).collect());
        if let Some(system) = &agent.system {
            react = react.with_system(system);
        }
        if let Some(max) = agent.max_tokens {
            react = react.with_max_tokens(max);
        }
        if let Some(effort) = &agent.effort {
            react = react.with_effort(effort);
        }
        if agent.text_only {
            react = react.text_only();
        }
        return Ok(StepSpec::Agent {
            step: Arc::new(react),
            tools: agent.tools.clone(),
        });
    }

    if let Ok(judge) = obj.extract::<PyRef<'_, PyJudge>>() {
        return Ok(StepSpec::Judge {
            step: Arc::new(
                JudgeStep::new(&judge.model, &judge.rubric).with_threshold(judge.threshold),
            ),
        });
    }

    let _ = py;
    Err(PyValueError::new_err(format!(
        "step() expects a soma.Agent or soma.Judge, got {}",
        obj.get_type().name()?
    )))
}

// ── JSON ↔ Python ──

fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let json = py.import("json")?;
    let text: String = json.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    let json = py.import("json")?;
    Ok(json.call_method1("loads", (value.to_string(),))?.unbind())
}

/// Derive a JSON Schema from a Python signature.
///
/// Enough for the common case — named parameters with primitive
/// annotations — and no more. A tool whose arguments do not fit passes
/// `schema=` explicitly rather than fighting an inference engine.
fn schema_from_signature(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let inspect = py.import("inspect")?;
    let signature = inspect.call_method1("signature", (func,)).map_err(|e| {
        PyValueError::new_err(format!(
            "cannot read the signature of this tool ({e}); pass `schema=` instead"
        ))
    })?;
    let parameters = signature.getattr("parameters")?;
    let empty = inspect.getattr("Parameter")?.getattr("empty")?;

    let mut properties = serde_json::Map::new();
    let mut required: Vec<serde_json::Value> = Vec::new();

    let names: Vec<String> = parameters
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k?.extract::<String>())
        .collect::<PyResult<_>>()?;

    for name in names {
        let param = parameters.get_item(&name)?;
        let annotation = param.getattr("annotation")?;

        let json_type = if annotation.is(&empty) {
            // Unannotated: accept anything rather than guess.
            None
        } else {
            let type_name = annotation
                .getattr("__name__")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_default();
            match type_name.as_str() {
                "str" => Some("string"),
                "int" => Some("integer"),
                "float" => Some("number"),
                "bool" => Some("boolean"),
                "list" => Some("array"),
                "dict" => Some("object"),
                _ => None,
            }
        };

        let mut property = serde_json::Map::new();
        if let Some(t) = json_type {
            property.insert("type".into(), serde_json::json!(t));
        }
        properties.insert(name.clone(), serde_json::Value::Object(property));

        if param.getattr("default")?.is(&empty) {
            required.push(serde_json::json!(name));
        }
    }

    Ok(serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    }))
}

/// `soma.tool` — decorator form of [`PyTool`].
///
/// ```python
/// @soma.tool
/// def search(query: str) -> str:
///     """Search the web. Call this when the answer needs current information."""
///     ...
/// ```
#[pyfunction]
#[pyo3(signature = (func=None, *, description=None, name=None, schema=None))]
pub fn tool(
    py: Python<'_>,
    func: Option<Py<PyAny>>,
    description: Option<String>,
    name: Option<String>,
    schema: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    match func {
        // Bare `@soma.tool`
        Some(f) => Ok(PyTool::new(py, f, description, name, schema)?
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        // `@soma.tool(description=...)` — return the decorator itself.
        None => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("description", description)?;
            kwargs.set_item("name", name)?;
            kwargs.set_item("schema", schema)?;
            let partial = py.import("functools")?.getattr("partial")?;
            let tool_fn = py.import("soma")?.getattr("tool")?;
            Ok(partial.call((tool_fn,), Some(&kwargs))?.unbind())
        }
    }
}

/// List the providers Soma knows about and whether each is usable now.
///
/// A provider is *configured* when its key is present (or it needs none),
/// which is the difference between "Soma knows this endpoint exists" and
/// "you can call it".
#[pyfunction]
pub fn providers(py: Python<'_>) -> PyResult<PyObject> {
    let catalog = somatize_llm::Catalog::load()
        .map_err(|e| PyRuntimeError::new_err(format!("reading provider catalog: {e}")))?;
    let configured: Vec<&str> = catalog.configured();

    let rows = PyList::empty(py);
    for id in catalog.ids() {
        let config = catalog.get(id).expect("id came from the catalog");
        let row = PyDict::new(py);
        row.set_item("id", id)?;
        row.set_item("base_url", &config.base_url)?;
        row.set_item("configured", configured.contains(&id))?;
        row.set_item("env", config.auth.env_var())?;
        row.set_item("note", config.note.clone())?;
        rows.append(row)?;
    }
    Ok(rows.into_any().unbind())
}

/// Models a provider currently offers. Reaches the network.
#[pyfunction]
pub fn models(py: Python<'_>, provider: &str) -> PyResult<PyObject> {
    let catalog = somatize_llm::Catalog::load()
        .map_err(|e| PyRuntimeError::new_err(format!("reading provider catalog: {e}")))?;
    let router = somatize_llm::Router::from_catalog(catalog)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let found = router.get(provider).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown provider `{provider}`. Known: {}",
            router.ids().join(", ")
        ))
    })?;

    // Releasing the GIL: this is a blocking network call, and holding it
    // would stall every other Python thread for its duration.
    let listing = py
        .allow_threads(|| found.models())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let names = PyList::empty(py);
    for model in listing {
        names.append(model.qualified())?;
    }
    Ok(names.into_any().unbind())
}
