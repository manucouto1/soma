//! The agentic surface, as Python sees it.
//!
//! Deliberately small. A user should be able to write
//!
//! ```python
//! g.node("researcher", soma.Agent(model="ollama/llama3.2", tools=[search]))
//! ```
//!
//! without meeting `Effect`, `Transition` or `StepCtx` — those are the
//! substrate, and the substrate is not the interface. `node` dispatches on
//! what it is handed, so an agent goes in exactly where a filter would;
//! there is no separate `step` method, and there could not be, because
//! `Graph.step` already means the optimizer's step.
//!
//! The escape hatch, for anyone who does need them, is `PyStepBridge`: any
//! Python object with a `poll(ctx)` method is a step, duck-typed the way a
//! filter's `forward` is. It returns a `{"transition": ...}` mapping — the
//! helpers in `soma.agentic` build them — which is what makes
//! `Transition::Spawn`, and with it dynamic fan-out, reachable from Python
//! at all. `poll` takes the GIL on whichever thread the driver is using, so
//! Python steps fan out for I/O concurrency rather than CPU parallelism.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use somatize_core::cache::CacheKey;
use somatize_core::effect::{
    Effect, EffectResult, JoinPolicy, LlmRequest, NodeSpec, SuspendReason,
};
use somatize_core::error::{Result as SomaResult, SomaError};
use somatize_core::message::Message;
use somatize_core::step::{Step, StepMeta, Transition};
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
    /// JSON Schema the answer must match, if the caller wants one.
    schema: Option<serde_json::Value>,
    max_repairs: usize,
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
        schema=None,
        max_repairs=1,
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
        schema: Option<&Bound<'_, PyAny>>,
        max_repairs: usize,
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
            schema: match schema {
                Some(s) if !s.is_none() => Some(py_to_json(py, s)?),
                _ => None,
            },
            max_repairs,
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

    /// The shape the answer must take, if one was declared.
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.schema.as_ref().map(|s| json_to_py(py, s)).transpose()
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

// ── A Python class implementing `Step` ──

/// What a step's `poll` is handed: this turn, and everything before it.
///
/// Read-only on purpose. A step that accumulates rebuilds what it needs
/// from `history` rather than keeping it on `self`, because replay feeds
/// back the identical results and a field would drift from them.
#[pyclass(name = "StepCtx", module = "soma")]
pub struct PyStepCtx {
    #[pyo3(get)]
    node_id: String,
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    input: PyObject,
    #[pyo3(get)]
    turn: usize,
    /// Results of the effects requested last turn, in request order.
    #[pyo3(get)]
    results: Py<PyList>,
    /// Every turn's results, oldest first.
    #[pyo3(get)]
    history: Py<PyList>,
}

#[pymethods]
impl PyStepCtx {
    /// The single result of a one-effect turn, or None on turn 0.
    fn result(&self, py: Python<'_>) -> Option<PyObject> {
        self.results.bind(py).iter().next().map(|o| o.unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "StepCtx(node_id={:?}, turn={}, results={})",
            self.node_id,
            self.turn,
            self.results.bind(py).len()
        )
    }
}

fn effect_result_to_py(py: Python<'_>, result: &EffectResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    match result {
        EffectResult::Llm(response) => {
            dict.set_item("kind", "llm")?;
            dict.set_item("text", response.message.text())?;
        }
        EffectResult::Tool { output, is_error } => {
            dict.set_item("kind", "tool")?;
            dict.set_item("output", crate::value_to_py(py, output)?)?;
            dict.set_item("is_error", *is_error)?;
        }
        EffectResult::Graph(value) => {
            dict.set_item("kind", "graph")?;
            dict.set_item("output", crate::value_to_py(py, value)?)?;
        }
        EffectResult::Node(value) => {
            dict.set_item("kind", "node")?;
            dict.set_item("output", crate::value_to_py(py, value)?)?;
        }
        EffectResult::Slept => {
            dict.set_item("kind", "slept")?;
        }
        EffectResult::Custom(value) => {
            dict.set_item("kind", "custom")?;
            dict.set_item("output", crate::value_to_py(py, value)?)?;
        }
        EffectResult::Failed { message } => {
            dict.set_item("kind", "failed")?;
            dict.set_item("message", message)?;
        }
        // `EffectResult` is `#[non_exhaustive]`: a variant added upstream
        // must still reach Python as *something* rather than panicking.
        other => {
            dict.set_item("kind", "unknown")?;
            dict.set_item("message", format!("{other:?}"))?;
        }
    }
    dict.set_item("is_error", result.is_error())?;
    Ok(dict.into_any().unbind())
}

/// Read one `{"transition": ...}` mapping into a `Transition`.
///
/// The protocol is a dict rather than a class hierarchy so that the Rust
/// side has one thing to parse and Python keeps ordinary data. The helpers
/// in `soma.agentic` build these dicts.
fn parse_transition(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Transition> {
    let dict = obj.downcast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "poll() must return one of soma.Done / Await / Spawn / Goto / Suspend \
             (a dict with a `transition` key), got {}",
            obj.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })?;

    let kind: String = dict
        .get_item("transition")?
        .ok_or_else(|| {
            PyValueError::new_err(
                "poll() returned a dict with no `transition` key. Use soma.Done(value), \
                 soma.Spawn(...), soma.Await(...), soma.Goto(...) or soma.Suspend(...)",
            )
        })?
        .extract()?;

    let get = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> { dict.get_item(name) };

    match kind.as_str() {
        "done" => {
            let value = match get("value")? {
                Some(v) => py_to_value(py, &v)?,
                None => Value::Empty,
            };
            Ok(Transition::Done(value))
        }

        "spawn" => {
            let specs_obj = get("specs")?.ok_or_else(|| {
                PyValueError::new_err("soma.Spawn needs `specs`, a list of soma.Run(...)")
            })?;
            let mut specs = Vec::new();
            for item in specs_obj.try_iter()? {
                let item = item?;
                let spec = item.downcast::<PyDict>().map_err(|_| {
                    PyValueError::new_err("each spawn spec must be a soma.Run(...) mapping")
                })?;
                let runs: String = spec
                    .get_item("runs")?
                    .ok_or_else(|| {
                        PyValueError::new_err(
                            "a spawn spec needs `runs`: the id of a step registered in \
                             this graph",
                        )
                    })?
                    .extract()?;
                let input = match spec.get_item("input")? {
                    Some(v) => py_to_value(py, &v)?,
                    None => Value::Empty,
                };
                let mut node_spec = NodeSpec::new(runs, input);
                if let Some(label) = spec.get_item("label")?
                    && !label.is_none()
                {
                    node_spec = node_spec.with_label(label.extract::<String>()?);
                }
                specs.push(node_spec);
            }

            let join = match get("join")? {
                Some(j) if !j.is_none() => match j.extract::<String>()?.as_str() {
                    "all" => JoinPolicy::All,
                    "first" => JoinPolicy::First,
                    "all_settled" | "settled" => JoinPolicy::AllSettled,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown join policy {other:?}; expected \"all\", \"first\" or \
                             \"all_settled\""
                        )));
                    }
                },
                _ => JoinPolicy::default(),
            };
            Ok(Transition::Spawn { specs, join })
        }

        "await" => {
            let effects_obj = get("effects")?
                .ok_or_else(|| PyValueError::new_err("soma.Await needs `effects`"))?;
            let mut effects = Vec::new();
            for item in effects_obj.try_iter()? {
                effects.push(parse_effect(py, &item?)?);
            }
            if effects.is_empty() {
                return Err(PyValueError::new_err(
                    "soma.Await with no effects would poll again unchanged, forever. \
                     Return soma.Done(...) instead",
                ));
            }
            Ok(Transition::Await(effects))
        }

        "goto" => {
            let target: String = get("target")?
                .ok_or_else(|| PyValueError::new_err("soma.Goto needs `target`"))?
                .extract()?;
            let carry = match get("carry")? {
                Some(v) => py_to_value(py, &v)?,
                None => Value::Empty,
            };
            Ok(Transition::Goto { target, carry })
        }

        "suspend" => {
            // A suspension from Python is a question for a person: that is
            // what `soma.Suspend("...")` reads as, and `resume_with` answers.
            let reason = match get("reason")? {
                Some(r) if !r.is_none() => SuspendReason::Human {
                    prompt: r.extract::<String>()?,
                    schema: None,
                },
                _ => SuspendReason::Human {
                    prompt: "waiting".into(),
                    schema: None,
                },
            };
            Ok(Transition::Suspend { reason })
        }

        other => Err(PyValueError::new_err(format!(
            "unknown transition {other:?}; expected done, await, spawn, goto or suspend"
        ))),
    }
}

/// Read one `{"effect": ...}` mapping into an `Effect`.
fn parse_effect(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Effect> {
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("an effect must be a mapping, e.g. soma.Llm(...)"))?;
    let kind: String = dict
        .get_item("effect")?
        .ok_or_else(|| PyValueError::new_err("an effect needs an `effect` key"))?
        .extract()?;

    match kind.as_str() {
        "llm" => {
            let model: String = dict
                .get_item("model")?
                .ok_or_else(|| PyValueError::new_err("soma.Llm needs `model`"))?
                .extract()?;
            let prompt: String = dict
                .get_item("prompt")?
                .ok_or_else(|| PyValueError::new_err("soma.Llm needs `prompt`"))?
                .extract()?;
            let mut request = LlmRequest::new(model, vec![Message::user(prompt)].into());
            if let Some(system) = dict.get_item("system")?
                && !system.is_none()
            {
                request = request.with_system(system.extract::<String>()?);
            }
            Ok(Effect::Llm(request))
        }
        "tool" => {
            let name: String = dict
                .get_item("name")?
                .ok_or_else(|| PyValueError::new_err("soma.ToolCall needs `name`"))?
                .extract()?;
            let args = match dict.get_item("args")? {
                Some(a) => py_to_value(py, &a)?,
                None => Value::Empty,
            };
            Ok(Effect::Tool { name, args })
        }
        "sleep" => {
            let secs: f64 = dict
                .get_item("seconds")?
                .ok_or_else(|| PyValueError::new_err("soma.Sleep needs `seconds`"))?
                .extract()?;
            if !secs.is_finite() || secs < 0.0 {
                return Err(PyValueError::new_err(
                    "soma.Sleep needs a finite, non-negative number of seconds",
                ));
            }
            Ok(Effect::Sleep(std::time::Duration::from_secs_f64(secs)))
        }
        "custom" => {
            let kind: String = dict
                .get_item("kind")?
                .ok_or_else(|| PyValueError::new_err("soma.Custom needs `kind`"))?
                .extract()?;
            let payload = match dict.get_item("payload")? {
                Some(p) => py_to_value(py, &p)?,
                None => Value::Empty,
            };
            Ok(Effect::Custom { kind, payload })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown effect {other:?}; expected llm, tool, sleep or custom"
        ))),
    }
}

/// A Python class acting as a `Step`.
///
/// The mirror of `PyFilterBridge`, and the reason `Transition::Spawn` is
/// reachable from Python at all: dynamic fan-out needs a step that decides
/// the width at runtime, and until now every step was Rust.
///
/// `poll` is called on whichever thread the driver is using — spawned
/// children run concurrently — so each call takes the GIL. Python steps
/// therefore fan out for I/O concurrency, not for CPU parallelism.
struct PyStepBridge {
    py_obj: PyObject,
    name: String,
    config_hash_val: CacheKey,
    max_turns: usize,
}

impl Step for PyStepBridge {
    fn config_hash(&self) -> CacheKey {
        self.config_hash_val.clone()
    }

    fn meta(&self) -> StepMeta {
        StepMeta::new(&self.name).with_max_turns(self.max_turns)
    }

    fn poll(&self, ctx: &somatize_core::step::StepCtx<'_>) -> SomaResult<Transition> {
        Python::with_gil(|py| {
            // Build the context, then poll. Everything Python-side is fallible
            // in PyErr terms, so it is assembled in one closure and converted
            // at the boundary rather than at every `?`.
            let build = || -> PyResult<Transition> {
                let results = PyList::empty(py);
                for result in ctx.results {
                    results.append(effect_result_to_py(py, result)?)?;
                }
                let history = PyList::empty(py);
                for turn in ctx.history {
                    let entry = PyList::empty(py);
                    for result in turn {
                        entry.append(effect_result_to_py(py, result)?)?;
                    }
                    history.append(entry)?;
                }

                let py_ctx = PyStepCtx {
                    node_id: ctx.node_id.to_string(),
                    run_id: ctx.run_id.to_string(),
                    input: crate::value_to_py(py, ctx.input)?,
                    turn: ctx.turn,
                    results: results.unbind(),
                    history: history.unbind(),
                };

                let returned = self.py_obj.call_method1(py, "poll", (py_ctx,))?;
                parse_transition(py, returned.bind(py))
            };
            build().map_err(|e| SomaError::Other(format!("Python step `{}`: {e}", self.name)))
        })
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
    /// A Python class with a `poll` method.
    Custom {
        step: Arc<dyn Step>,
        name: String,
    },
}

impl StepSpec {
    pub(crate) fn step(&self) -> Arc<dyn Step> {
        match self {
            Self::Agent { step, .. } | Self::Judge { step } | Self::Custom { step, .. } => {
                step.clone()
            }
        }
    }

    pub(crate) fn tools(&self) -> &[PyTool] {
        match self {
            Self::Agent { tools, .. } => tools,
            Self::Judge { .. } | Self::Custom { .. } => &[],
        }
    }

    /// A short name for the node kind, shown in graph renderings.
    pub(crate) fn kind(&self) -> String {
        match self {
            Self::Agent { .. } => "Agent".into(),
            Self::Judge { .. } => "Judge".into(),
            Self::Custom { name, .. } => name.clone(),
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
        if let Some(schema) = &agent.schema {
            react = react
                .with_schema(schema.clone())
                .with_repairs(agent.max_repairs);
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

    // Anything with a `poll` method is a step. Duck typing rather than a
    // required base class, for the same reason filters accept duck-typed
    // `forward`: a user's class should not have to inherit from us.
    if obj.hasattr("poll")? {
        // Same identity ladder as a filter — qualname, canonical config, then
        // the source-hash fallback — so a step's journal key is as stable as a
        // filter's cache key, and unstable for the same reasons.
        let identity_mod = py.import("soma._identity")?;
        let identity = identity_mod.call_method1("filter_identity", (obj,))?;
        let qualname: String = identity.get_item("qualname")?.extract()?;
        let config_json: String = identity.get_item("config_json")?.extract()?;
        let code_fp: String = identity.get_item("code_fp")?.extract()?;

        let name: String = obj
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "Step".to_string());
        let max_turns: usize = match obj.getattr("max_turns") {
            Ok(v) if !v.is_none() => v.extract()?,
            _ => StepMeta::new(&name).max_turns,
        };

        return Ok(StepSpec::Custom {
            step: Arc::new(PyStepBridge {
                py_obj: obj.clone().unbind(),
                name: name.clone(),
                config_hash_val: CacheKey::from_parts(&[
                    b"soma-step-v1",
                    qualname.as_bytes(),
                    config_json.as_bytes(),
                    code_fp.as_bytes(),
                ]),
                max_turns,
            }),
            name,
        });
    }

    Err(PyValueError::new_err(format!(
        "expected a soma.Agent, a soma.Judge, or an object with a poll() method, got {}",
        obj.get_type().name()?
    )))
}

// ── JSON ↔ Python ──

/// The conversion pair lives in the parent module. There used to be a
/// copy here — and the two disagreed: the other one stringified arrays
/// and objects.
use crate::{json_to_py, py_any_to_json};

fn py_to_json(_py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    py_any_to_json(obj)
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
