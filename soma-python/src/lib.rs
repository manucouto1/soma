//! PyO3 bindings for Soma — exposes Graph, Study, and Filter to Python.
//!
//! Bridges Python Filter classes to the Rust Filter trait, converts
//! between Python lists/dicts and Soma Values, and wraps the StudyRunner
//! for hyperparameter optimization from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

use somatize_compiler::{CompileMode, compile};
use somatize_core::cache::CacheKey;
use somatize_core::error::{Result as SomaResult, SomaError};
use somatize_core::event::MetricRecord;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::fingerprint::ArchitectureFingerprint;
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::search::{Scale, SearchDimension, SearchSpace};
use somatize_core::study::{Direction, Objective, PruningStrategy, SearchStrategy, Study};
use somatize_core::tracking::{GraphSummaryInfo, RunKind, RunState, Tracker};
use somatize_core::value::Value;
mod agentic;

use agentic::{PyAgent, PyJudge, PyStepCtx, PyTool, PyToolAdapter, to_step_spec};
use somatize_runtime::EventBus;
use somatize_runtime::cache::{FsActionStore, MemoryCache, TieredCache};
use somatize_runtime::effects::{EffectDriver, EffectJournal};
use somatize_runtime::executor::{self, Context, GraphInfo};
use somatize_runtime::executors::study::{
    FnTrialExecutor, StudyRunner, TrialContext, TrialOutcome,
};
use somatize_runtime::node_catalog::NodeCatalog;
use somatize_runtime::runner::{LocalRunner, Runner};
use somatize_runtime::sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
use somatize_runtime::tracking::{
    LocalTracker, RunReader, advance_head, load_manifest, resolve_parent, summarize,
};

// All four derive from `RuntimeError`, which is what every `SomaError`
// used to become. Adding a more specific type should let a caller catch
// less, not make an existing `except RuntimeError` stop catching.
pyo3::create_exception!(
    _soma,
    SomaSuspended,
    PyRuntimeError,
    "A run stopped, waiting for something outside it.\n\n\
     Not a failure. Carries `run_id`, `node_id`, `turn` and `reason`, \
     which is what `Graph.resume(...)` needs to answer it."
);
pyo3::create_exception!(
    _soma,
    SomaPruned,
    PyRuntimeError,
    "A study trial was stopped early by a pruner."
);
pyo3::create_exception!(
    _soma,
    SomaSchemaMismatch,
    PyRuntimeError,
    "Two connected nodes disagree about what flows between them."
);
pyo3::create_exception!(
    _soma,
    SomaNodeNotFound,
    PyRuntimeError,
    "A plan named a node the catalog does not hold."
);

/// A `SomaError` as the Python exception that matches it.
///
/// Every variant used to become a flat `RuntimeError` carrying only its
/// `Display` text, including the structured ones. `Suspended` is the case
/// that mattered: it is not a failure, it is a pause, and answering it
/// needs the run id, node id and turn — all of which were legible only by
/// reading the message.
fn soma_err_to_py(e: SomaError) -> PyErr {
    let text = e.to_string();
    match e {
        SomaError::Suspended {
            run_id,
            node_id,
            turn,
            reason,
        } => Python::with_gil(|py| {
            let err = SomaSuspended::new_err(text);
            let obj = err.value(py);
            let _ = obj.setattr("run_id", run_id);
            let _ = obj.setattr("node_id", node_id);
            let _ = obj.setattr("turn", turn);
            let _ = obj.setattr("kind", reason.kind());
            if let Ok(json) = serde_json::to_value(&*reason)
                && let Ok(py_reason) = json_to_py(py, &json)
            {
                let _ = obj.setattr("reason", py_reason);
            }
            err
        }),
        SomaError::Pruned { .. } => SomaPruned::new_err(text),
        SomaError::SchemaMismatch { .. } => SomaSchemaMismatch::new_err(text),
        SomaError::NodeNotFound(_) => SomaNodeNotFound::new_err(text),
        _ => PyRuntimeError::new_err(text),
    }
}

fn py_err_to_soma(e: PyErr) -> SomaError {
    SomaError::Other(e.to_string())
}

/// The shared persistent cache root: `$SOMA_CACHE_DIR`, else `~/.soma/cache`.
/// Shared across processes and projects so a re-run (or another
/// investigation over the same data) reuses previous compute.
fn default_cache_dir() -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("SOMA_CACHE_DIR")
        && !dir.is_empty()
    {
        return Some(std::path::PathBuf::from(dir));
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| std::path::PathBuf::from(home).join(".soma").join("cache"))
}

/// A Python object as the JSON value it describes, via the `json` module.
pub(crate) fn py_any_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let py = obj.py();
    let json_mod = py.import("json")?;
    let text: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("not JSON-serializable: {e}")))
}

/// A `serde_json::Value` as the Python object it describes.
///
/// Through `json.loads`, so a list arrives as a list and an object as a
/// dict. The hand-written match this replaces ended in
/// `other => other.to_string()`: every array and every object reached
/// Python as the *string* of its JSON. A study whose search space held a
/// list gave `"[1, 2, 3]"` back from `trial["params"]`.
pub(crate) fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    let json = py.import("json")?;
    Ok(json.call_method1("loads", (v.to_string(),))?.unbind())
}

// ── Value conversion ──

/// A dict's JSON form, if JSON can hold it without changing it.
///
/// `json.dumps` is lenient — it turns tuples into lists and integer-keyed
/// dicts into string-keyed ones — so dumping is not enough. The value has to
/// survive the round trip unchanged to count as JSON; anything else keeps
/// the pickle it would have had before.
fn as_json_dict(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<serde_json::Value>> {
    let json_mod = py.import("json")?;
    let Ok(dumped) = json_mod.call_method1("dumps", (obj,)) else {
        return Ok(None);
    };
    let text: String = dumped.extract()?;

    let restored = json_mod.call_method1("loads", (&text,))?;
    if !restored.eq(obj)? {
        return Ok(None);
    }

    Ok(serde_json::from_str(&text).ok())
}

fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(lists) = obj.extract::<Vec<Vec<f64>>>() {
        let rows = lists.len();
        let cols = if rows > 0 { lists[0].len() } else { 0 };
        let flat: Vec<f64> = lists.into_iter().flatten().collect();
        return Ok(Value::tensor(flat, vec![rows, cols]));
    }

    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        let len = arr.len();
        return Ok(Value::tensor(arr, vec![len]));
    }

    // A non-numeric list — `["summarise", "critique"]`, a plan, a list of
    // records. Numeric lists were caught above and stay tensors, so no
    // existing cache key moves; this only rescues what used to be a flat
    // "cannot convert" on the most ordinary thing to hand a fan-out.
    if obj.is_instance_of::<PyList>()
        && let Some(json) = as_json_dict(py, obj)?
    {
        return Ok(Value::json(json));
    }

    if obj.is_instance_of::<PyDict>() {
        // A dict that JSON can hold *becomes* JSON. An opaque pickle is
        // unreadable to everything outside this process: a loop cannot read
        // a stop signal out of it, a branch cannot read an arm label, a
        // report cannot show it and a remote worker in another language
        // cannot receive it. Round-tripping is what decides — a dict with
        // tuples or ndarrays inside comes back changed (or not at all), and
        // those keep the pickle.
        if let Some(json) = as_json_dict(py, obj)? {
            return Ok(Value::json(json));
        }
        let pickle = py.import("pickle")?;
        let data: Vec<u8> = pickle.call_method1("dumps", (obj, 5i32))?.extract()?;
        return Ok(Value::object(data));
    }

    // A control value is usually one of these. They used to be outright
    // errors, so nothing can be relying on the old behaviour.
    if obj.is_none() {
        return Ok(Value::Empty);
    }
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Value::json(serde_json::Value::Bool(obj.extract::<bool>()?)));
    }

    // A bare number, like a bare bool, used to be an outright error — which
    // made `Spawn([Run("worker", 3)])` fail on the most obvious thing anyone
    // would write. JSON rather than a 1-element tensor so it comes back as a
    // number, and so a loop or a branch can still read it.
    //
    // After the bool check, never before it: in Python `bool` *is* an `int`,
    // and `True` would extract as `1`.
    if obj.is_instance_of::<pyo3::types::PyInt>()
        && let Ok(i) = obj.extract::<i64>()
    {
        return Ok(Value::json(serde_json::Value::from(i)));
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>()
        && let Ok(f) = obj.extract::<f64>()
    {
        // NaN and the infinities have no JSON spelling; they stay tensors
        // rather than becoming null and losing the value silently.
        return Ok(match serde_json::Number::from_f64(f) {
            Some(n) => Value::json(serde_json::Value::Number(n)),
            None => Value::tensor(vec![f], vec![1]),
        });
    }

    if let Ok(s) = obj.extract::<String>() {
        // A string that parses as JSON stays JSON — changing that would move
        // the cache key of every pipeline already passing JSON strings.
        // Anything else is plain text (a prompt, a label, a completion),
        // which used to be an outright error.
        return Ok(match serde_json::from_str(&s) {
            Ok(val) => Value::json(val),
            Err(_) => Value::text(s),
        });
    }

    Err(PyRuntimeError::new_err(
        "Cannot convert Python object to Value. Expected list, 2D list, dict, str, or JSON string.",
    ))
}

fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Tensor { values, shape } => {
            if shape.len() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                let result = PyList::empty(py);
                for r in 0..rows {
                    let row: Vec<f64> = values[r * cols..(r + 1) * cols].to_vec();
                    result.append(row)?;
                }
                Ok(result.into_any().unbind())
            } else {
                Ok(values.as_slice().into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::Text(s) => Ok(s.as_ref().into_pyobject(py)?.into_any().unbind()),
        Value::Json(v) => {
            let json_str = v.to_string();
            let json_mod = py.import("json")?;
            let obj = json_mod.call_method1("loads", (json_str,))?;
            Ok(obj.unbind())
        }
        Value::Object(data) => {
            let pickle = py.import("pickle")?;
            let py_bytes = PyBytes::new(py, data.as_slice());
            let obj = pickle.call_method1("loads", (py_bytes,))?;
            Ok(obj.unbind())
        }
        Value::Bytes(b) => Ok(b.as_slice().into_pyobject(py)?.into_any().unbind()),
        Value::Empty => Ok(py.None()),
        _ => Ok(py.None()),
    }
}

// ── Python Filter wrapper ──

struct PyFilterBridge {
    py_obj: PyObject,
    name: String,
    config_hash_val: CacheKey,
    /// cloudpickle.dumps() bytes — serializes the full object (bytecode + closures + deps).
    pickled_bytes: Vec<u8>,
    /// Full module source code (imports + classes + helpers) for introspection by Nous agents.
    source: String,
    /// Pip requirements detected from the filter's imports.
    requirements: Vec<String>,
    /// Whether this filter is trainable (has meaningful fit()).
    trainable: bool,
}

impl PyFilterBridge {
    fn new(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = obj.get_type().name()?.to_string();

        // Deterministic identity: qualified class name + canonical
        // config (search defaults merged, sorted keys, typed
        // CacheConfigError on unhashable attrs) + code fingerprint
        // (_cache_version → source hash → cloudpickle ladder).
        // See python/soma/_identity.py.
        let identity_mod = py.import("soma._identity")?;
        let identity = identity_mod.call_method1("filter_identity", (obj,))?;
        let qualname: String = identity.get_item("qualname")?.extract()?;
        let config_json: String = identity.get_item("config_json")?.extract()?;
        let code_fp: String = identity.get_item("code_fp")?.extract()?;

        // Serialize with cloudpickle (like Spark/Dask/Ray).
        // Register the filter's module AND all its transitive local dependencies
        // for by-value pickling. Without this, the worker would need the exact
        // same source tree installed (e.g. src.filters.classifiers, src.utils).
        let cloudpickle = py.import("cloudpickle")?;
        let inspect = py.import("inspect")?;
        let module = inspect.call_method1("getmodule", (obj.get_type(),))?;
        // Python helper: walk module globals, find all non-stdlib imported modules,
        // register them for by-value serialization (transitive).
        py.run(
            c"
import sys, types, sysconfig, site, os, cloudpickle as _cp

# Python 3.10+ exposes the canonical stdlib module name set. Fall back to
# empty frozenset on older versions (combined with other heuristics below).
_STDLIB = getattr(sys, 'stdlib_module_names', frozenset())
_BUILTINS = frozenset(sys.builtin_module_names)
# Never pickle-by-value modules the worker already has installed.
_NEVER = {'soma', 'cloudpickle', 'numpy', 'pandas', 'torch', 'sklearn', 'scipy'}
# Site-packages directories (platform/installer independent: works on
# Debian, RHEL/Fedora with lib64, Windows, conda, virtualenvs, --user installs).
_site_dirs = {sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')}
_site_dirs.add(getattr(site, 'getusersitepackages', lambda: None)())
_SITE_PREFIXES = tuple(
    os.path.realpath(p) + os.sep
    for p in _site_dirs
    if p
)

def _is_stdlib_or_installed(mod):
    name = getattr(mod, '__name__', '') or ''
    top = name.split('.')[0]
    if top in _STDLIB or top in _BUILTINS or top in _NEVER:
        return True
    f = getattr(mod, '__file__', None)
    if f is None:
        # C extension or frozen without __file__: treat as stdlib/builtin.
        return True
    # Python 3.11+ frozen stdlib modules: __file__ == '<frozen io>' etc.
    if f.startswith('<'):
        return True
    # Authoritative site-packages check via sysconfig (handles lib64,
    # virtualenvs, conda). Fall back to substring match for pathological
    # cases where realpath resolution differs.
    rf = os.path.realpath(f)
    if _SITE_PREFIXES and rf.startswith(_SITE_PREFIXES):
        return True
    if 'site-packages' in f or 'dist-packages' in f:
        return True
    return False

def _register_transitive(mod, visited=None):
    if visited is None:
        visited = set()
    name = getattr(mod, '__name__', '')
    if not name or name in visited or name == '__main__':
        return
    visited.add(name)
    if _is_stdlib_or_installed(mod):
        return
    _cp.register_pickle_by_value(mod)
    # Walk globals for imported modules
    for v in vars(mod).values():
        if isinstance(v, types.ModuleType):
            _register_transitive(v, visited)
        elif isinstance(v, type):
            m = sys.modules.get(v.__module__)
            if m and m.__name__ not in visited:
                _register_transitive(m, visited)

if _soma_module is not None:
    _register_transitive(_soma_module)
",
            Some(&[("_soma_module", &module)].into_py_dict(py)?),
            None,
        )?;
        let pickled = cloudpickle.call_method1("dumps", (obj,))?;
        let pickled_bytes: Vec<u8> = pickled.extract()?;
        let source = if !module.is_none() {
            inspect
                .call_method1("getsource", (&module,))
                .and_then(|s| s.extract::<String>())
                .unwrap_or_default()
        } else {
            inspect
                .call_method1("getsource", (obj.get_type(),))
                .and_then(|s| s.extract::<String>())
                .unwrap_or_default()
        };

        // Detect pip requirements from the filter module's imports.
        // Collects top-level package names of all site-packages imports.
        let reqs_result = py.run(
            c"
import types, sys
_reqs = set()
if _mod is not None:
    for v in vars(_mod).values():
        if isinstance(v, types.ModuleType):
            f = getattr(v, '__file__', '') or ''
            if 'site-packages' in f:
                _reqs.add(v.__name__.split('.')[0])
        elif isinstance(v, type):
            m = sys.modules.get(v.__module__)
            if m:
                f = getattr(m, '__file__', '') or ''
                if 'site-packages' in f:
                    _reqs.add(m.__name__.split('.')[0])
_reqs = sorted(_reqs)
",
            Some(&[("_mod", &module)].into_py_dict(py)?),
            None,
        );
        let requirements: Vec<String> = if reqs_result.is_ok() {
            py.eval(c"_reqs", None, None)
                .and_then(|r| r.extract())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Detect if the filter is trainable (has _kind = "trainable")
        let trainable = obj
            .get_type()
            .getattr("_kind")
            .and_then(|v| v.extract::<String>())
            .map(|s| s != "stateless")
            .unwrap_or(true); // default to trainable if not specified

        // The environment is part of the identity: the same code under
        // different dependency sets can produce different results.
        let env = requirements.join(",");
        let config_hash = CacheKey::from_parts(&[
            b"soma-id-v2",
            qualname.as_bytes(),
            config_json.as_bytes(),
            code_fp.as_bytes(),
            env.as_bytes(),
        ]);

        Ok(Self {
            py_obj: obj.clone().unbind(),
            name,
            config_hash_val: config_hash,
            pickled_bytes,
            source,
            requirements,
            trainable,
        })
    }
}

impl Filter for PyFilterBridge {
    fn config_hash(&self) -> CacheKey {
        self.config_hash_val.clone()
    }

    fn fit(&self, x: &Value, y: Option<&Value>) -> SomaResult<Value> {
        Python::with_gil(|py| {
            let py_x = value_to_py(py, x).map_err(py_err_to_soma)?;
            let py_y = match y {
                Some(v) => value_to_py(py, v).map_err(py_err_to_soma)?,
                None => py.None(),
            };
            let result = self
                .py_obj
                .call_method1(py, "fit", (py_x, py_y))
                .map_err(|e| SomaError::Other(format!("Python fit() error: {e}")))?;
            let bound = result.bind(py);
            py_to_value(py, bound).map_err(py_err_to_soma)
        })
    }

    fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
        Python::with_gil(|py| {
            let py_x = value_to_py(py, x).map_err(py_err_to_soma)?;
            let py_state = value_to_py(py, state).map_err(py_err_to_soma)?;
            let result = self
                .py_obj
                .call_method1(py, "forward", (py_x, py_state))
                .map_err(|e| SomaError::Other(format!("Python forward() error: {e}")))?;
            let bound = result.bind(py);
            // Filter forward may return either ``out`` directly (legacy)
            // or ``(out, aux_dict)`` (new DifferentiableFilter contract).
            // Auxiliary signals are runtime-only — drop them at the
            // serialization boundary so cached/remote callers see the
            // same shape as before.
            if let Ok(tuple) = bound.downcast::<pyo3::types::PyTuple>()
                && tuple.len() == 2
                && tuple
                    .get_item(1)
                    .is_ok_and(|v| v.is_instance_of::<PyDict>())
            {
                let out = tuple
                    .get_item(0)
                    .map_err(|e| SomaError::Other(format!("forward tuple unpack: {e}")))?;
                return py_to_value(py, &out).map_err(py_err_to_soma);
            }
            py_to_value(py, bound).map_err(py_err_to_soma)
        })
    }

    fn meta(&self) -> FilterMeta {
        // Read optional meta attributes from the Python class.
        // Users can set: _cacheable = False, _differentiable = False,
        //                _deterministic = False, _kind = "stateless",
        //                _stream_mode = "evolving"
        let (kind, cacheable, differentiable, deterministic, stream_mode) =
            Python::with_gil(|py| {
                let obj = self.py_obj.bind(py);

                let cacheable = obj
                    .getattr("_cacheable")
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(true);

                let differentiable = obj
                    .getattr("_differentiable")
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(false);

                let deterministic = obj
                    .getattr("_deterministic")
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(true);

                let kind = obj
                    .getattr("_kind")
                    .and_then(|v| v.extract::<String>())
                    .map(|s| match s.as_str() {
                        "stateless" => FilterKind::Stateless,
                        _ => FilterKind::Trainable,
                    })
                    .unwrap_or(FilterKind::Trainable);

                let stream_mode = obj
                    .getattr("_stream_mode")
                    .and_then(|v| v.extract::<String>())
                    .map(|s| match s.as_str() {
                        "evolving" => StreamMode::Evolving {
                            checkpoint_every: 100,
                        },
                        "barrier" => StreamMode::Barrier,
                        _ => StreamMode::FixedState,
                    })
                    .unwrap_or(StreamMode::FixedState);

                (kind, cacheable, differentiable, deterministic, stream_mode)
            });

        FilterMeta {
            name: self.name.clone(),
            kind,
            cacheable,
            differentiable,
            deterministic,
            stream_mode,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn composite_fit(
        &self,
        peers: &[(String, std::sync::Arc<dyn Filter>)],
        x: &Value,
        y: Option<&Value>,
    ) -> Option<SomaResult<(Value, std::collections::HashMap<String, Value>)>> {
        Python::with_gil(|py| {
            let obj = self.py_obj.bind(py);

            // Only trigger when the user has overridden composite_fit.
            // The base Filter class does not declare the method, so
            // ``hasattr`` is the simplest "is it overridden?" probe.
            match obj.hasattr("composite_fit") {
                Ok(true) => {}
                _ => return None,
            }

            // Build a Python dict {node_id: filter_py_obj} for the whole block.
            // If any peer isn't a PyFilterBridge (non-Python filter) we bail
            // out so the runner falls back to sequential execution.
            let peers_dict = PyDict::new(py);
            for (node_id, filter) in peers {
                let bridge = filter.as_any().downcast_ref::<PyFilterBridge>()?;
                if let Err(e) = peers_dict.set_item(node_id, bridge.py_obj.clone_ref(py)) {
                    return Some(Err(SomaError::Other(format!(
                        "composite_fit: building peers dict: {e}"
                    ))));
                }
            }

            // Convert x and y for the Python call.
            let py_x = match value_to_py(py, x) {
                Ok(v) => v,
                Err(e) => return Some(Err(py_err_to_soma(e))),
            };
            let py_y = match y {
                Some(v) => match value_to_py(py, v) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(py_err_to_soma(e))),
                },
                None => py.None(),
            };

            let result = match obj.call_method1("composite_fit", (peers_dict, py_x, py_y)) {
                Ok(r) => r,
                Err(e) => {
                    return Some(Err(SomaError::Other(format!(
                        "Python composite_fit() error: {e}"
                    ))));
                }
            };

            // Expected return: (output, {node_id: state}).
            let tuple = match result.downcast::<pyo3::types::PyTuple>() {
                Ok(t) => t.clone(),
                Err(_) => {
                    return Some(Err(SomaError::Other(
                        "composite_fit must return (output, states_dict)".into(),
                    )));
                }
            };
            if tuple.len() != 2 {
                return Some(Err(SomaError::Other(format!(
                    "composite_fit must return a 2-tuple, got length {}",
                    tuple.len()
                ))));
            }
            let output_py = tuple.get_item(0).ok()?;
            let states_py = tuple.get_item(1).ok()?;

            let output = match py_to_value(py, &output_py) {
                Ok(v) => v,
                Err(e) => return Some(Err(py_err_to_soma(e))),
            };

            let states_dict = match states_py.downcast::<PyDict>() {
                Ok(d) => d.clone(),
                Err(_) => {
                    return Some(Err(SomaError::Other(
                        "composite_fit states must be a dict[node_id, state]".into(),
                    )));
                }
            };
            let mut states_map: std::collections::HashMap<String, Value> =
                std::collections::HashMap::new();
            for (k, v) in states_dict.iter() {
                let key: String = match k.extract() {
                    Ok(s) => s,
                    Err(e) => {
                        return Some(Err(SomaError::Other(format!(
                            "composite_fit state key: {e}"
                        ))));
                    }
                };
                let val = match py_to_value(py, &v) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(py_err_to_soma(e))),
                };
                states_map.insert(key, val);
            }

            Some(Ok((output, states_map)))
        })
    }
}

// ── Search dimension parsing ──

fn parse_py_search_dim(_py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<SearchDimension> {
    let dict = item.downcast::<PyDict>()?;
    let dtype: String = dict
        .get_item("type")?
        .ok_or_else(|| PyRuntimeError::new_err("missing 'type'"))?
        .extract()?;
    let name: String = dict
        .get_item("name")?
        .ok_or_else(|| PyRuntimeError::new_err("missing 'name'"))?
        .extract()?;

    match dtype.as_str() {
        "float" => {
            let low: f64 = dict.get_item("low")?.unwrap().extract()?;
            let high: f64 = dict.get_item("high")?.unwrap().extract()?;
            let scale_str: String = dict
                .get_item("scale")?
                .map(|s| s.extract().unwrap_or_default())
                .unwrap_or("linear".into());
            let scale = match scale_str.as_str() {
                "log" => Scale::Log,
                "reverse_log" => Scale::ReverseLog,
                _ => Scale::Linear,
            };
            Ok(SearchDimension::Float {
                name,
                low,
                high,
                scale,
                default: None,
            })
        }
        "int" => {
            let low: i64 = dict.get_item("low")?.unwrap().extract()?;
            let high: i64 = dict.get_item("high")?.unwrap().extract()?;
            Ok(SearchDimension::Int {
                name,
                low,
                high,
                scale: Scale::Linear,
            })
        }
        "categorical" => {
            let choices_py = dict.get_item("choices")?.unwrap();
            let choices_list = choices_py.downcast::<PyList>()?;
            let choices: Vec<serde_json::Value> = choices_list
                .iter()
                .map(|c| {
                    if let Ok(s) = c.extract::<String>() {
                        serde_json::json!(s)
                    } else if let Ok(b) = c.extract::<bool>() {
                        serde_json::json!(b)
                    } else if let Ok(i) = c.extract::<i64>() {
                        serde_json::json!(i)
                    } else if let Ok(f) = c.extract::<f64>() {
                        serde_json::json!(f)
                    } else {
                        serde_json::json!(c.to_string())
                    }
                })
                .collect();
            Ok(SearchDimension::Categorical { name, choices })
        }
        _ => Err(PyRuntimeError::new_err(format!(
            "unknown search dim type: {dtype}"
        ))),
    }
}

// ── PyTrial ──

/// Handle passed to a study's objective function.
///
/// Behaves like a read-only params mapping (`trial["Encoder.lr"]`,
/// `trial.get("x", 0.5)`) so legacy `fn(params) -> dict` executors keep
/// working, and adds `report(name, value, step)` / `should_prune()`
/// for pruning-aware training loops.
#[pyclass(name = "Trial")]
struct PyTrial {
    ctx: TrialContext,
    params: HashMap<String, serde_json::Value>,
}

#[pymethods]
impl PyTrial {
    /// Sampled parameters as a dict.
    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.params {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.unbind())
    }

    /// Trial id (`trial_0003`).
    #[getter]
    fn id(&self) -> String {
        self.ctx.trial_id().to_string()
    }

    /// Report an intermediate metric. Returns True when the trial
    /// should stop (the pruner decided against it).
    fn report(&self, name: &str, value: f64, step: usize) -> bool {
        self.ctx.report(name, value, step)
    }

    /// Whether the pruner has decided to stop this trial.
    fn should_prune(&self) -> bool {
        self.ctx.should_prune()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.params.get(key) {
            Some(v) => json_to_py(py, v),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyResult<PyObject> {
        match self.params.get(key) {
            Some(v) => json_to_py(py, v),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn keys(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }
}

// ── PyStudy ──

#[pyclass(name = "Study")]
struct PyStudy {
    study: Study,
    /// Python callable metrics-dict -> float; recorded as metric "score".
    objective_cb: Option<PyObject>,
    tracking: bool,
    root: std::path::PathBuf,
    run_dir: Option<std::path::PathBuf>,
}

fn parse_pruning(pruning: &Bound<'_, PyAny>) -> PyResult<PruningStrategy> {
    if let Ok(s) = pruning.extract::<String>() {
        return match s.as_str() {
            "median" => Ok(PruningStrategy::Median { n_warmup_steps: 0 }),
            other => Err(PyRuntimeError::new_err(format!(
                "unknown pruning '{other}'; use 'median', ('median', warmup) or ('percentile', pct, warmup)"
            ))),
        };
    }
    let tuple = pruning.downcast::<pyo3::types::PyTuple>().map_err(|_| {
        PyRuntimeError::new_err(
            "pruning must be 'median', ('median', warmup) or ('percentile', pct, warmup)",
        )
    })?;
    let kind: String = tuple.get_item(0)?.extract()?;
    match kind.as_str() {
        "median" => Ok(PruningStrategy::Median {
            n_warmup_steps: tuple.get_item(1)?.extract()?,
        }),
        "percentile" => Ok(PruningStrategy::Percentile {
            percentile: tuple.get_item(1)?.extract()?,
            n_warmup_steps: tuple.get_item(2)?.extract()?,
        }),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown pruning '{other}'"
        ))),
    }
}

/// Append a completed study's summary as an ExperimentRecord.
///
/// Goes through the same run-directory path as every other run, so a
/// study lands in the pool with a conclusion, a lineage and a trial
/// breakdown instead of a hand-rolled record. The best trial's
/// configuration and metrics are the extras only the study knows.
fn record_study_experiment(run_dir: &std::path::Path, study: &Study) {
    let (params, metrics) = match study.best_trial() {
        Some(best) => (
            best.params.clone().into_iter().collect(),
            best.metrics
                .iter()
                .map(|m| (m.name.clone(), m.value))
                .collect(),
        ),
        None => (HashMap::new(), HashMap::new()),
    };
    append_run_record(run_dir, params, metrics);
}

fn trial_to_py(py: Python<'_>, trial: &somatize_core::study::Trial) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &trial.id)?;
    let params_dict = PyDict::new(py);
    for (k, v) in &trial.params {
        params_dict.set_item(k, json_to_py(py, v)?)?;
    }
    dict.set_item("params", params_dict)?;
    dict.set_item(
        "state",
        match &trial.state {
            somatize_core::study::TrialState::Pending => "pending".to_string(),
            somatize_core::study::TrialState::Running => "running".to_string(),
            somatize_core::study::TrialState::Completed => "completed".to_string(),
            somatize_core::study::TrialState::Pruned { .. } => "pruned".to_string(),
            somatize_core::study::TrialState::Failed { .. } => "failed".to_string(),
        },
    )?;
    let metrics_dict = PyDict::new(py);
    for m in &trial.metrics {
        metrics_dict.set_item(&m.name, m.value)?;
    }
    dict.set_item("metrics", metrics_dict)?;
    // Full metric series (one record per report), for learning curves —
    // "metrics" above keeps only the last value per name.
    let series = pyo3::types::PyList::empty(py);
    for m in &trial.metrics {
        let rec = PyDict::new(py);
        rec.set_item("name", &m.name)?;
        rec.set_item("value", m.value)?;
        rec.set_item("step", m.step)?;
        rec.set_item("timestamp", m.timestamp.to_rfc3339())?;
        series.append(rec)?;
    }
    dict.set_item("series", series)?;
    dict.set_item("started_at", trial.started_at.map(|t| t.to_rfc3339()))?;
    dict.set_item("finished_at", trial.finished_at.map(|t| t.to_rfc3339()))?;
    dict.set_item("duration_ms", trial.duration_ms)?;
    Ok(dict.unbind())
}

#[pymethods]
impl PyStudy {
    #[new]
    #[pyo3(signature = (name, search_space=None, strategy="grid".to_string(), n_trials=10,
                        objectives=None, seed=None, objective=None, direction="maximize".to_string(),
                        pruning=None, tracking=true, root=".soma".to_string(), tags=None,
                        seeds=None, frozen=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python<'_>,
        name: String,
        search_space: Option<&Bound<'_, PyList>>,
        strategy: String,
        n_trials: usize,
        objectives: Option<Vec<(String, String)>>,
        seed: Option<u64>,
        objective: Option<PyObject>,
        direction: String,
        pruning: Option<&Bound<'_, PyAny>>,
        tracking: bool,
        root: String,
        tags: Option<Vec<String>>,
        seeds: Option<Vec<i64>>,
        frozen: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut space = SearchSpace::new();
        if let Some(dims) = search_space {
            for item in dims.iter() {
                let dim = parse_py_search_dim(item.py(), &item)?;
                space.add(dim);
            }
        }

        let strat = match strategy.as_str() {
            "grid" => SearchStrategy::Grid {
                points_per_dim: n_trials,
            },
            "random" => SearchStrategy::Random { n_trials, seed },
            "bayesian" => SearchStrategy::Bayesian {
                n_trials,
                n_startup: (n_trials / 5).max(5),
                seed,
            },
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown strategy: {strategy}. Use 'grid', 'random', or 'bayesian'."
                )));
            }
        };

        // Strict: a typo like "minimise" must not silently maximize.
        let parse_dir = |d: &str| -> PyResult<Direction> {
            match d {
                "minimize" => Ok(Direction::Minimize),
                "maximize" => Ok(Direction::Maximize),
                other => Err(PyRuntimeError::new_err(format!(
                    "unknown direction '{other}'; use 'maximize' or 'minimize'"
                ))),
            }
        };

        // With a Python objective callable, the composite score is
        // recorded as metric "score" — that becomes the objective.
        let objs: Vec<Objective> = match (&objective, objectives) {
            (Some(_), _) => vec![Objective {
                metric: "score".into(),
                direction: parse_dir(&direction)?,
            }],
            (None, Some(list)) => list
                .into_iter()
                .map(|(metric, dir)| {
                    Ok(Objective {
                        metric,
                        direction: parse_dir(&dir)?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
            (None, None) => vec![],
        };

        let mut study = Study::new(name, space, strat, objs);
        if let Some(p) = pruning {
            study.pruning = parse_pruning(p)?;
        }
        study.tags = tags.unwrap_or_default();
        // Experiment seeds: every sampled config runs once per seed;
        // trial params carry "seed" (wire torch.manual_seed(trial["seed"])
        // in the trial callable — the cache keys per seed automatically).
        study.seeds = seeds.unwrap_or_default();
        if let Some(f) = frozen {
            for (k, v) in f.iter() {
                let key: String = k.extract()?;
                let value: serde_json::Value = py_any_to_json(&v)?;
                study.frozen.insert(key, value);
            }
        }

        Ok(Self {
            study,
            objective_cb: objective,
            tracking,
            root: std::path::PathBuf::from(root),
            run_dir: None,
        })
    }

    /// Load a study from a run directory (its `study.json`).
    ///
    /// Continue it from anywhere::
    ///
    ///     study = soma.Study.load(".soma/runs/study_.../")
    ///     study.run(train, resume=True)
    ///
    /// A composite `objective=` callable cannot be persisted — re-pass
    /// it here when resuming a study that was created with one, or the
    /// new trials won't produce the "score" metric.
    #[staticmethod]
    #[pyo3(signature = (run_dir, objective=None))]
    fn load(run_dir: String, objective: Option<PyObject>) -> PyResult<Self> {
        let dir = std::path::PathBuf::from(run_dir);
        let study = Study::load(dir.join("study.json")).map_err(soma_err_to_py)?;
        // <root>/runs/<run_id> → root
        let root = dir
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(".soma"));
        Ok(Self {
            study,
            objective_cb: objective,
            tracking: true,
            root,
            run_dir: Some(dir),
        })
    }

    /// Save the study as JSON (defaults to `<run_dir>/study.json`).
    #[pyo3(signature = (path=None))]
    fn save(&self, path: Option<String>) -> PyResult<()> {
        let path = match (path, &self.run_dir) {
            (Some(p), _) => std::path::PathBuf::from(p),
            (None, Some(dir)) => dir.join("study.json"),
            (None, None) => {
                return Err(PyRuntimeError::new_err(
                    "no path given and the study has no run directory yet",
                ));
            }
        };
        self.study.save(path).map_err(soma_err_to_py)
    }

    /// Run the study.
    ///
    /// `fn` receives a `Trial` handle: read params via `trial.params`,
    /// `trial["name"]` or `trial.get(...)`; report intermediate metrics
    /// with `trial.report(name, value, step)` (returns True → stop);
    /// return a dict of final metrics (or None / a bare float).
    #[pyo3(signature = (executor, on_event=None, resume=false))]
    fn run(
        &mut self,
        py: Python<'_>,
        executor: &Bound<'_, PyAny>,
        on_event: Option<PyObject>,
        resume: bool,
    ) -> PyResult<()> {
        // A study created with objective= scores through the "score"
        // metric; resuming it without re-passing the callable would
        // silently stop producing that metric. Only prior trials that
        // actually carry "score" are evidence a callable existed.
        if self.objective_cb.is_none()
            && self
                .study
                .objectives
                .first()
                .is_some_and(|o| o.metric == "score")
            && self
                .study
                .trials
                .iter()
                .any(|t| t.metrics.iter().any(|m| m.name == "score"))
        {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "this study scores on the 'score' metric but no objective= callable \
                     is set — pass it to Study.load(run_dir, objective=...) when resuming, \
                     or return a {'score': ...} dict from the executor",
                ),
            )?;
        }

        let tracker: Option<Arc<LocalTracker>> = if self.tracking {
            let t = match (&self.run_dir, resume) {
                (Some(dir), true) => LocalTracker::open(dir).map_err(soma_err_to_py)?,
                _ => {
                    let t = LocalTracker::create(&self.root, RunKind::Study, &self.study.name)
                        .map_err(soma_err_to_py)?;
                    // Enrich the manifest exactly like begin_run does.
                    if let Ok(mut manifest) = load_manifest(t.run_dir()) {
                        manifest.tags = self.study.tags.clone();
                        manifest.python_version =
                            Some(py.version().split_whitespace().next().unwrap_or("").into());
                        manifest.parent_run_id = resolve_parent(&self.root, None);
                        // Record experiment seeds (and the sampler seed)
                        // so a reader can see what this run covers.
                        for (i, s) in self.study.seeds.iter().enumerate() {
                            manifest.seeds.insert(format!("seed_{i}"), *s);
                        }
                        match &self.study.strategy {
                            somatize_core::study::SearchStrategy::Random {
                                seed: Some(s), ..
                            }
                            | somatize_core::study::SearchStrategy::Bayesian {
                                seed: Some(s),
                                ..
                            } => {
                                manifest.seeds.insert("sampler".into(), *s as i64);
                            }
                            _ => {}
                        }
                        let _ = t.save_manifest(&manifest);
                    }
                    t
                }
            };
            self.run_dir = Some(t.run_dir().to_path_buf());
            Some(Arc::new(t))
        } else {
            None
        };

        let bus = Arc::new(EventBus::new(1024));
        if let Some(t) = &tracker {
            bus.add_sink(t.sink());
        }
        if let Some(callback) = on_event {
            let mut rx = bus.subscribe();
            std::thread::spawn(move || {
                while let Ok(event) = rx.blocking_recv() {
                    if let Ok(json_str) = serde_json::to_string(&event) {
                        Python::with_gil(|py| {
                            let json_mod = py.import("json").unwrap();
                            if let Ok(dict) = json_mod.call_method1("loads", (json_str,)) {
                                let _ = callback.call1(py, (dict,));
                            }
                        });
                    }
                }
            });
        }

        let mut runner = StudyRunner::new(bus.clone());
        if let Some(t) = &tracker {
            runner = runner.with_tracker(t.clone() as Arc<dyn Tracker>);
        }

        let executor_obj = executor.clone().unbind();
        let objective_cb = self
            .objective_cb
            .as_ref()
            .map(|cb| Python::with_gil(|py| cb.clone_ref(py)));

        let trial_executor = FnTrialExecutor(
            move |params: &HashMap<String, serde_json::Value>,
                  ctx: &TrialContext|
                  -> SomaResult<TrialOutcome> {
                Python::with_gil(|py| {
                    let trial = PyTrial {
                        ctx: ctx.clone(),
                        params: params.clone(),
                    };
                    let result = executor_obj
                        .call1(py, (trial,))
                        .map_err(|e| SomaError::Other(format!("Python executor error: {e}")))?;
                    let bound = result.bind(py);

                    // Accepted returns: None, a bare number, or a dict
                    // of final metrics.
                    let mut metrics: Vec<MetricRecord> = Vec::new();
                    if bound.is_none() {
                        // metrics were reported via trial.report()
                    } else if let Ok(v) = bound.extract::<f64>() {
                        metrics.push(MetricRecord {
                            name: "score".into(),
                            value: v,
                            step: 0,
                            timestamp: chrono::Utc::now(),
                        });
                    } else if let Ok(dict) = bound.downcast::<PyDict>() {
                        for (k, v) in dict.iter() {
                            let name: String = k.extract().map_err(py_err_to_soma)?;
                            let value: f64 = v.extract().map_err(py_err_to_soma)?;
                            metrics.push(MetricRecord {
                                name,
                                value,
                                step: 0,
                                timestamp: chrono::Utc::now(),
                            });
                        }
                    } else {
                        return Err(SomaError::Other(
                            "executor must return None, a number, or a dict of metrics".into(),
                        ));
                    }

                    // Composite objective callable → recorded as "score".
                    if let Some(cb) = &objective_cb
                        && !metrics.iter().any(|m| m.name == "score")
                    {
                        let merged = PyDict::new(py);
                        for m in ctx.metrics() {
                            merged.set_item(&m.name, m.value).map_err(py_err_to_soma)?;
                        }
                        for m in &metrics {
                            merged.set_item(&m.name, m.value).map_err(py_err_to_soma)?;
                        }
                        let score: f64 = cb
                            .call1(py, (merged,))
                            .map_err(|e| SomaError::Other(format!("objective error: {e}")))?
                            .extract(py)
                            .map_err(|_| {
                                SomaError::Other("objective must return a number".into())
                            })?;
                        metrics.push(MetricRecord {
                            name: "score".into(),
                            value: score,
                            step: 0,
                            timestamp: chrono::Utc::now(),
                        });
                    }
                    Ok(TrialOutcome::Completed(metrics))
                })
            },
        );

        let mut sampler: Box<dyn Sampler> = match &self.study.strategy {
            SearchStrategy::Grid { points_per_dim } => Box::new(GridSampler::new(*points_per_dim)),
            SearchStrategy::Random { n_trials, seed } => {
                Box::new(RandomSampler::new(*n_trials, *seed))
            }
            SearchStrategy::Bayesian {
                n_trials,
                n_startup,
                seed,
                ..
            } => Box::new(BayesianSampler::new(*n_trials, *n_startup, *seed)),
            _ => return Err(PyRuntimeError::new_err("Unsupported strategy")),
        };

        // Release the GIL for the whole study: the executor re-acquires
        // it per trial, and on_event callbacks can run between trials.
        let study = &mut self.study;
        let result = py.allow_threads(move || runner.run(study, sampler.as_mut(), &trial_executor));

        if let Some(t) = &tracker {
            let state = if result.is_ok() {
                RunState::Completed
            } else {
                RunState::Failed
            };
            let _ = t.finalize(state);
            if result.is_ok() {
                record_study_experiment(t.run_dir(), &self.study);
                advance_head(&self.root, t.run_id());
            }
        }
        result.map_err(soma_err_to_py)
    }

    #[getter]
    fn best_trial(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        match self.study.best_trial() {
            Some(trial) => Ok(Some(trial_to_py(py, trial)?)),
            None => Ok(None),
        }
    }

    /// All recorded trials as dicts.
    #[getter]
    fn trials(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.study
            .trials
            .iter()
            .map(|t| trial_to_py(py, t))
            .collect()
    }

    #[getter]
    fn n_trials(&self) -> usize {
        self.study.trials.len()
    }

    #[getter]
    fn progress(&self) -> f64 {
        self.study.progress()
    }

    /// Declared objectives as `(metric, direction)` pairs, e.g.
    /// `[("f1", "maximize")]`. A composite objective reports as
    /// `("score", direction)`.
    #[getter]
    fn objectives(&self) -> Vec<(String, String)> {
        let dir_str = |d: &somatize_core::study::Direction| match d {
            somatize_core::study::Direction::Maximize => "maximize".to_string(),
            somatize_core::study::Direction::Minimize => "minimize".to_string(),
        };
        if self.study.objectives.is_empty()
            && let Some(composite) = &self.study.composite
        {
            return vec![("score".to_string(), dir_str(&composite.direction))];
        }
        self.study
            .objectives
            .iter()
            .map(|o| (o.metric.clone(), dir_str(&o.direction)))
            .collect()
    }

    /// The study's display name.
    #[getter]
    fn name(&self) -> String {
        self.study.name.clone()
    }

    /// Run directory holding study.json/events.jsonl (None if tracking
    /// is disabled and the study never ran).
    #[getter]
    fn run_dir(&self) -> Option<String> {
        self.run_dir.as_ref().map(|p| p.display().to_string())
    }
}

// ── PyGraph ──

/// What a registered node *does*, once `register_behaviour` has filed it
/// away — enough to build the graph node, and nothing else.
enum Behaviour {
    /// An effectful step, carrying its `step_name`.
    Step(String),
    /// An ordinary filter, carrying its `filter_name`.
    Filter(String),
}

impl Behaviour {
    fn node(&self, id: &str) -> Node {
        match self {
            Behaviour::Step(kind) => Node::step(id, kind),
            Behaviour::Filter(name) => Node::filter_with_id(id, name),
        }
    }
}

#[pyclass(name = "Graph")]
struct PyGraph {
    graph: Graph,
    library: NodeCatalog,
    cache: Arc<dyn somatize_core::cache::CacheStore>,
    event_bus: Arc<EventBus>,
    fitted: bool,
    /// Registered remote workers: (address, token, tags).
    workers: Vec<(String, Option<String>, Vec<String>)>,
    /// Coordinator URL + token.
    coordinator: Option<(String, Option<String>)>,
    /// Pickled filter bytes + requirements for remote serialization.
    /// node_id → (cloudpickle bytes, pip requirements)
    pickled_filters: std::collections::HashMap<String, (Vec<u8>, Vec<String>)>,
    /// Module source code per filter for Nous agent introspection/editing.
    /// node_id → full module source (imports + classes + helpers)
    filter_sources: std::collections::HashMap<String, String>,
    /// Optional DataStore for persistent data transport (opt-in, costs storage).
    data_store: Option<Arc<dyn somatize_core::store::DataStore>>,
    /// Whether each filter is trainable (node_id → bool).
    filter_trainable: std::collections::HashMap<String, bool>,
    /// Live Python filter instances retained by node id. Used by the
    /// in-process training path (graph.train/forward/freeze) so that a
    /// filter's persistent state (e.g. an nn.Module attached to self)
    /// survives across forward calls instead of being deserialised each
    /// time. Distinct from `pickled_filters`, which exists only for
    /// remote-worker dispatch.
    live_filters: std::collections::HashMap<String, Py<PyAny>>,
    /// Effectful step nodes, by node id. Empty for a purely computational
    /// graph, in which case none of the agentic machinery is built.
    /// The live Python `Agent`/`Judge` behind each step node. A `Step` is
    /// immutable once built, so a study that samples a new prompt or model
    /// writes to these and the library is rebuilt from them — the same
    /// arrangement `live_filters` has for the computational path.
    live_steps: std::collections::HashMap<String, Py<PyAny>>,
    /// Data edges a study may cut, in declaration order.
    optional_edges: Vec<(String, String)>,
    /// Optional edges currently cut, held whole together with the position
    /// they came from, so restoring one restores its id, kind, label *and*
    /// place — a trial that cuts an edge has to leave the graph the next
    /// trial starts from byte-identical.
    cut_edges: std::collections::HashMap<(String, String), (usize, Edge)>,
    /// Tools every agent in this graph may call, by name. Collected from the
    /// agents as they are added, so a tool declared once is callable by any
    /// node that lists it.
    tools: std::collections::HashMap<String, PyTool>,
    /// Which provider serves a bare (unqualified) model name.
    default_provider: Option<String>,
    /// Tool sets from MCP servers. Held so the servers stay alive for the
    /// graph's lifetime — dropping a client kills its subprocess.
    mcp_toolboxes: Vec<somatize_llm::Toolbox>,
    /// Generic Python-side scratch dict for orchestration state that
    /// doesn't belong on the Rust struct (e.g. the registered optimiser).
    /// Lazily initialised on first access. PyGraph deliberately doesn't
    /// expose `__dict__`, so this dict is the supported way to attach
    /// per-graph Python state.
    py_state: Option<Py<PyDict>>,
}

impl PyGraph {
    /// Does anything in this graph need fitting before it can run?
    fn has_trainable_filters(&self) -> bool {
        self.filter_trainable.values().any(|t| *t)
    }

    /// A node id not yet taken, suffixing `_2`, `_3`, … as needed.
    fn free_id(&self, wanted: &str) -> String {
        if self.graph.node(wanted).is_none() {
            return wanted.to_string();
        }
        let mut i = 2;
        loop {
            let candidate = format!("{wanted}_{i}");
            if self.graph.node(&candidate).is_none() {
                return candidate;
            }
            i += 1;
        }
    }

    /// Register what a node *does*, without saying what shape it has in the
    /// graph. A branch node runs a classifier and routes; a plain node runs
    /// the same classifier and stops. The behaviour registration is
    /// identical, so it lives here and the two callers differ only in the
    /// [`Node`] they add.
    ///
    /// Returns what the caller needs to build the graph node itself.
    fn register_behaviour(
        &mut self,
        py: Python<'_>,
        node_id: &str,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<Behaviour> {
        if let Ok(spec) = to_step_spec(py, obj) {
            // Tools travel with the graph, not with the node: one agent may
            // declare a tool and another list the same one, and both should
            // reach the same implementation.
            for tool in spec.tools() {
                self.tools
                    .insert(tool.tool_name().to_string(), tool.clone());
            }
            let kind = spec.kind().to_string();
            self.library.register_step_arc(node_id, spec.step());
            // Keep the live Agent/Judge: a study samples by writing to it,
            // and the step is rebuilt from it before the next run.
            self.live_steps
                .insert(node_id.to_string(), obj.clone().unbind());
            return Ok(Behaviour::Step(kind));
        }

        let bridge = PyFilterBridge::new(py, obj)?;
        let name = bridge.name.clone();
        self.pickled_filters.insert(
            node_id.to_string(),
            (bridge.pickled_bytes.clone(), bridge.requirements.clone()),
        );
        self.filter_sources
            .insert(node_id.to_string(), bridge.source.clone());
        self.filter_trainable
            .insert(node_id.to_string(), bridge.trainable);
        self.live_filters
            .insert(node_id.to_string(), obj.clone().unbind());
        self.library.register(node_id.to_string(), Box::new(bridge));
        Ok(Behaviour::Filter(name))
    }

    /// Resolve one arm of a branch or one entry of a loop body: either the
    /// id of a node already in the graph, or a filter/agent to add as one.
    fn resolve_member(
        &mut self,
        py: Python<'_>,
        fallback_id: &str,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        if let Ok(existing) = obj.extract::<String>() {
            if self.graph.node(&existing).is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "`{existing}` names no node in this graph. Pass a filter or an \
                     agent to create one, or an id already added with node()"
                )));
            }
            return Ok(existing);
        }

        let id = self.free_id(fallback_id);
        let node = self.register_behaviour(py, &id, obj)?.node(&id);
        self.graph.add_node(node);
        Ok(id)
    }

    /// Add a labelled control edge — the wire the compiler reads to decide
    /// which nodes a loop or branch owns.
    fn control_edge(&mut self, source: &str, target: &str, label: Option<&str>) {
        let id = format!("e_{}", self.graph.edges.len());
        let mut edge = Edge::control(id, source, target);
        if let Some(label) = label {
            edge = edge.with_label(label);
        }
        self.graph.add_edge(edge);
    }

    /// Build the step library and effect driver an agentic plan needs.
    ///
    /// Returns `None` for a graph with no steps, so a purely computational
    /// pipeline never constructs a provider router, reads a catalog, or
    /// touches an environment variable.
    /// The catalog as it stands *now* — filters and steps together.
    ///
    /// A [`Step`] is immutable once built, so a study that samples a new
    /// prompt or model has no way to change one in place — it writes to the
    /// live `Agent` instead, and the steps are rebuilt from those here,
    /// before every compile and every run. Cheap: rebuilding a step is
    /// reading a handful of fields off a Python object.
    ///
    /// Every entry point passes this one value, which is what stops
    /// `compile()` from type-checking a different graph than `run()` does.
    fn rebuild_catalog(&self, py: Python<'_>) -> PyResult<NodeCatalog> {
        if self.live_steps.is_empty() {
            return Ok(self.library.clone());
        }
        let mut catalog = self.library.clone();
        for (node_id, obj) in &self.live_steps {
            catalog.register_step_arc(node_id, to_step_spec(py, obj.bind(py))?.step());
        }
        Ok(catalog)
    }

    fn step_runtime(
        &self,
        py: Python<'_>,
        catalog: &NodeCatalog,
    ) -> PyResult<Option<EffectDriver>> {
        if !catalog.has_steps() {
            return Ok(None);
        }

        // Python tools and MCP tools land in one toolbox: to a model they
        // are the same thing, and a step names them the same way. Tools
        // declared on a live agent are collected here too, so an agent that
        // gained one since the graph was built can still call it.
        let mut toolbox = somatize_llm::Toolbox::new();
        for tool in self.tools.values() {
            toolbox.add(Arc::new(PyToolAdapter { tool: tool.clone() }));
        }
        for obj in self.live_steps.values() {
            for tool in to_step_spec(py, obj.bind(py))?.tools() {
                toolbox.add(Arc::new(PyToolAdapter { tool: tool.clone() }));
            }
        }
        for mcp in &self.mcp_toolboxes {
            toolbox.merge_from(mcp);
        }

        let catalog = somatize_llm::Catalog::load().map_err(soma_err_to_py)?;
        let mut router = somatize_llm::Router::from_catalog(catalog).map_err(soma_err_to_py)?;
        if let Some(default) = &self.default_provider {
            router = router.with_default(default);
        }

        // The journal shares the graph's cache directory, so an agentic run
        // is resumable by the same mechanism a computational one is.
        let cache_dir = default_cache_dir().ok_or_else(|| {
            PyRuntimeError::new_err(
                "an agentic graph needs somewhere to journal its effects; \
                 set SOMA_CACHE_DIR or HOME",
            )
        })?;
        let store = Arc::new(FsActionStore::new(cache_dir).map_err(soma_err_to_py)?);
        let journal = EffectJournal::new(store.clone(), store);

        let driver = EffectDriver::new(journal)
            .with_event_bus(self.event_bus.clone())
            .with_handler(Arc::new(somatize_llm::LlmHandler::new(router)))
            .with_handler(Arc::new(toolbox))
            .with_handler(Arc::new(somatize_runtime::effects::SleepHandler));

        Ok(Some(driver))
    }

    /// Write the run's topology snapshot: `graph.json` (the machine
    /// contract), `graph.mmd` (the human one) and `fingerprint.json`
    /// (structural identity, with each node's filter config hash).
    ///
    /// Called from `begin_run` — the single writer. The fingerprint is
    /// best-effort: a graph whose canonical form will not serialize
    /// must not stop a run from starting.
    fn snapshot_topology(&self, tracker: &LocalTracker) -> PyResult<()> {
        let graph_json = serde_json::to_string_pretty(&self.graph)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        tracker
            .save_artifact("graph.json", graph_json.as_bytes())
            .map_err(soma_err_to_py)?;
        tracker
            .save_artifact("graph.mmd", self.graph.to_mermaid().as_bytes())
            .map_err(soma_err_to_py)?;

        if let Ok(fingerprint) = ArchitectureFingerprint::of(&self.graph) {
            let node_config: std::collections::BTreeMap<String, String> = self
                .graph
                .nodes
                .iter()
                .filter_map(|node| {
                    let hash = self.library.get(&node.id)?.config_hash();
                    Some((node.id.clone(), hash.to_hex()))
                })
                .collect();
            let fingerprint = fingerprint.with_node_config(node_config);
            if let Ok(json) = serde_json::to_string_pretty(&fingerprint) {
                tracker
                    .save_artifact("fingerprint.json", json.as_bytes())
                    .map_err(soma_err_to_py)?;
            }
        }
        Ok(())
    }

    /// Split a Value into batches along the first axis.
    /// For Tensor: splits rows. For Json dict with lists: splits list values.
    #[allow(dead_code)]
    fn split_value_into_batches(value: &Value, batch_size: usize) -> Vec<Value> {
        match value {
            Value::Tensor { values, shape } if !shape.is_empty() && shape[0] > batch_size => {
                let rows = shape[0];
                let row_size: usize = shape[1..].iter().product::<usize>().max(1);
                let mut batches = Vec::new();
                for start in (0..rows).step_by(batch_size) {
                    let end = (start + batch_size).min(rows);
                    let flat_start = start * row_size;
                    let flat_end = end * row_size;
                    let batch_vals = values[flat_start..flat_end].to_vec();
                    let mut batch_shape = shape.clone();
                    batch_shape[0] = end - start;
                    batches.push(Value::tensor(batch_vals, batch_shape));
                }
                batches
            }
            Value::Json(json_val) if json_val.is_object() => {
                let map = json_val.as_object().unwrap();
                let total = map
                    .values()
                    .find_map(|v| v.as_array().map(|a| a.len()))
                    .unwrap_or(0);
                if total <= batch_size {
                    return vec![value.clone()];
                }
                let mut batches = Vec::new();
                for start in (0..total).step_by(batch_size) {
                    let end = (start + batch_size).min(total);
                    let mut batch_map = serde_json::Map::new();
                    for (k, v) in map {
                        if let Some(arr) = v.as_array() {
                            let slice = arr[start..end.min(arr.len())].to_vec();
                            batch_map.insert(k.clone(), serde_json::Value::Array(slice));
                        } else {
                            batch_map.insert(k.clone(), v.clone());
                        }
                    }
                    batches.push(Value::json(serde_json::Value::Object(batch_map)));
                }
                batches
            }
            _ => vec![value.clone()],
        }
    }

    /// Build a transport from the first registered worker (if any).
    fn make_transport(&self) -> Option<Arc<dyn somatize_runtime::runner::Transport>> {
        if self.workers.is_empty() {
            return None;
        }
        let (addr, token, _tags) = &self.workers[0];
        let transport = somatize_worker::WsTransport::new(addr, token.clone());
        Some(Arc::new(transport))
    }

    /// Send a Shutdown message to a worker via WebSocket.
    fn send_shutdown(address: &str, token: Option<&str>, reason: &str) -> PyResult<()> {
        somatize_worker::WsTransport::new(address, token.map(str::to_string))
            .notify(&somatize_worker::protocol::CoordinatorToWorker::Shutdown {
                reason: reason.to_string(),
            })
            .map_err(soma_err_to_py)
    }

    /// Decide how to transport input data to the worker.
    ///
    /// - DataStore configured → upload to store, return Reference
    /// - Large payload (≥ 10MB) → HTTP bulk upload to worker, return Reference
    /// - Small payload → Inline (current WS behavior)
    fn resolve_transport(
        &self,
        x: &Value,
        transport: &somatize_worker::WsTransport,
    ) -> Result<somatize_worker::protocol::InputSource, PyErr> {
        use somatize_core::cache::CacheKey;
        use somatize_worker::protocol::InputSource;

        // DataStore configured → always use it (user opted in)
        if let Some(store) = &self.data_store {
            let data_bytes = serde_json::to_vec(x).unwrap_or_default();
            let key = CacheKey::hash_data(&data_bytes);
            let data_ref = store.put(&key, x).map_err(soma_err_to_py)?;
            return Ok(InputSource::Reference { data_ref });
        }

        // Estimate payload size
        let size_bytes = serde_json::to_vec(x).map(|v| v.len()).unwrap_or(0);
        if size_bytes >= somatize_core::store::INLINE_THRESHOLD_BYTES {
            let data_ref = transport.upload(x).map_err(soma_err_to_py)?;
            return Ok(InputSource::Reference { data_ref });
        }

        Ok(InputSource::Inline { value: x.clone() })
    }

    /// Send a plan to a remote worker via WebSocket.
    /// Returns (output, trained_states) — states are non-empty after Fit mode.
    fn dispatch_to_worker(
        &self,
        x: &Value,
        mode: somatize_worker::protocol::ExecutionMode,
    ) -> Result<(Value, std::collections::HashMap<String, Value>), PyErr> {
        use somatize_worker::protocol::*;

        let compile_mode = match &mode {
            ExecutionMode::Fit { .. } => CompileMode::NoCache,
            ExecutionMode::Forward => CompileMode::Inference,
            _ => CompileMode::Inference,
        };

        let compile_result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            compile_mode,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        // Find the best worker by target tag or first available
        let first_target = self
            .graph
            .nodes
            .iter()
            .find_map(|n| n.target.as_deref())
            .unwrap_or("default");

        let (addr, token) = self
            .workers
            .iter()
            .find(|(_, _, tags)| {
                first_target == "default" || tags.contains(&first_target.to_string())
            })
            .or_else(|| self.workers.first())
            .map(|(a, t, _)| (a.clone(), t.clone()))
            .ok_or_else(|| PyRuntimeError::new_err("no workers available"))?;

        // Serialize filters with cloudpickle bytes so the worker can reconstruct them
        let filters: Vec<SerializedFilter> = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| {
                let (pickled, reqs) = self.pickled_filters.get(&node.id)?;
                let state = self.library.get_state(&node.id).map(|arc| (*arc).clone());
                let trainable = self.filter_trainable.get(&node.id).copied().unwrap_or(true);
                let config_hash = self.library.get(&node.id).map(|f| f.config_hash());
                Some(SerializedFilter {
                    node_id: node.id.clone(),
                    pickled_filter: pickled.clone(),
                    state,
                    requirements: reqs.clone(),
                    trainable,
                    config_hash,
                })
            })
            .collect();

        let transport = somatize_worker::WsTransport::new(&addr, token.clone());
        let input_source = self.resolve_transport(x, &transport)?;
        let plan = SerializedPlan {
            plan_id: somatize_core::util::timestamp_id("remote_plan"),
            plan: compile_result.plan,
            input: Some(input_source),
            filters,
            mode,
            metadata: serde_json::json!({}),
        };

        // The socket, the framing and the size limits belong to the
        // transport. This function decides *which* worker gets *which*
        // filters — policy — and hands the result over.
        let reply = transport
            .send_msg(&CoordinatorToWorker::AssignPlan { plan })
            .map_err(soma_err_to_py)?;

        match reply {
            WorkerToCoordinator::PlanResult { result, .. } => match result {
                PlanResult::Success { output, states, .. } => {
                    let value = transport.resolve_output(&output).map_err(soma_err_to_py)?;
                    Ok((value, states))
                }
                PlanResult::Failed { error, .. } => {
                    Err(PyRuntimeError::new_err(format!("remote: {error}")))
                }
            },
            other => Err(PyRuntimeError::new_err(format!(
                "worker answered with {other:?} instead of a plan result"
            ))),
        }
    }

    /// Stream data to a worker in chunks via WebSocket Binary frames.
    /// Chunks are processed by StreamExecutor on the worker — no full materialization.
    fn dispatch_streamed(&self, x: &Value, chunk_size: usize) -> Result<Value, PyErr> {
        use somatize_worker::protocol::*;

        let compile_result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            CompileMode::Inference,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let first_target = self
            .graph
            .nodes
            .iter()
            .find_map(|n| n.target.as_deref())
            .unwrap_or("default");

        let (addr, token) = self
            .workers
            .iter()
            .find(|(_, _, tags)| {
                first_target == "default" || tags.contains(&first_target.to_string())
            })
            .or_else(|| self.workers.first())
            .map(|(a, t, _)| (a.clone(), t.clone()))
            .ok_or_else(|| PyRuntimeError::new_err("no workers available"))?;

        let filters: Vec<SerializedFilter> = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| {
                let (pickled, reqs) = self.pickled_filters.get(&node.id)?;
                let state = self.library.get_state(&node.id).map(|arc| (*arc).clone());
                let trainable = self.filter_trainable.get(&node.id).copied().unwrap_or(true);
                let config_hash = self.library.get(&node.id).map(|f| f.config_hash());
                Some(SerializedFilter {
                    node_id: node.id.clone(),
                    pickled_filter: pickled.clone(),
                    state,
                    requirements: reqs.clone(),
                    trainable,
                    config_hash,
                })
            })
            .collect();

        let chunks = Self::chunk_value(x, chunk_size);
        let stream_id = somatize_core::util::timestamp_id("stream");

        let plan = SerializedPlan {
            plan_id: stream_id,
            plan: compile_result.plan,
            input: None, // input comes via chunks
            filters,
            mode: ExecutionMode::Forward,
            metadata: serde_json::json!({}),
        };

        somatize_worker::WsTransport::new(&addr, token)
            .stream_plan(plan, chunks)
            .map_err(soma_err_to_py)
    }

    /// Split a Value into chunks for streaming.
    fn chunk_value(x: &Value, chunk_size: usize) -> Vec<Value> {
        match x {
            Value::Tensor { values, shape } if !values.is_empty() => {
                // Split along first dimension
                let row_size = if shape.len() > 1 {
                    shape[1..].iter().product()
                } else {
                    1
                };
                let n_rows = shape[0];
                let mut chunks = Vec::new();
                for start in (0..n_rows).step_by(chunk_size) {
                    let end = (start + chunk_size).min(n_rows);
                    let flat_start = start * row_size;
                    let flat_end = end * row_size;
                    let chunk_vals = values[flat_start..flat_end].to_vec();
                    let mut chunk_shape = shape.clone();
                    chunk_shape[0] = end - start;
                    chunks.push(Value::tensor(chunk_vals, chunk_shape));
                }
                chunks
            }
            // For non-tensor or small data, single chunk
            _ => vec![x.clone()],
        }
    }
}

#[pymethods]
impl PyGraph {
    /// Create a new Graph.
    ///
    /// Optional keyword arguments:
    ///
    /// * `cache` — cache backend: `"memory"`, `"local"`, or `"tiered"`.
    ///   Default: a persistent tiered cache (memory LRU over a shared
    ///   on-disk store at `$SOMA_CACHE_DIR` or `~/.soma/cache`), so fit
    ///   states and forward outputs survive crashes and are shared
    ///   across processes and projects. Pass `cache="memory"` to opt out.
    /// * `cache_path` — directory for `"local"` / `"tiered"` cache (required for those).
    /// * `cache_max_bytes` — max bytes for the in-memory LRU (default 1 GB).
    #[new]
    #[pyo3(signature = (*, cache=None, cache_path=None, cache_max_bytes=None))]
    fn new(
        cache: Option<&str>,
        cache_path: Option<String>,
        cache_max_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let max_bytes = cache_max_bytes.unwrap_or(1024 * 1024 * 1024);
        let cache_store: Arc<dyn somatize_core::cache::CacheStore> = match cache {
            None => match default_cache_dir().and_then(|dir| FsActionStore::new(dir).ok()) {
                Some(local) => Arc::new(TieredCache::memory_and_local(
                    Box::new(MemoryCache::new(max_bytes)),
                    Box::new(local),
                )),
                // No writable cache dir (sandbox, read-only home):
                // degrade to memory-only rather than failing.
                None => Arc::new(MemoryCache::new(max_bytes)),
            },
            Some("memory") => Arc::new(MemoryCache::new(max_bytes)),
            Some("local") => {
                let path = cache_path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cache_path is required for cache=\"local\"",
                    )
                })?;
                Arc::new(
                    FsActionStore::new(path)
                        .map_err(|e| PyRuntimeError::new_err(format!("cache init: {e}")))?,
                )
            }
            Some("tiered") => {
                let path = cache_path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cache_path is required for cache=\"tiered\"",
                    )
                })?;
                let local = FsActionStore::new(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("cache init: {e}")))?;
                Arc::new(TieredCache::memory_and_local(
                    Box::new(MemoryCache::new(max_bytes)),
                    Box::new(local),
                ))
            }
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown cache type: {other:?} (expected \"memory\", \"local\", or \"tiered\")"
                )));
            }
        };

        Ok(Self {
            graph: Graph::new(),
            library: NodeCatalog::new(),
            cache: cache_store,
            event_bus: Arc::new(EventBus::new(256)),
            fitted: false,
            workers: Vec::new(),
            coordinator: None,
            pickled_filters: std::collections::HashMap::new(),
            filter_sources: std::collections::HashMap::new(),
            data_store: None,
            filter_trainable: std::collections::HashMap::new(),
            live_filters: std::collections::HashMap::new(),
            live_steps: std::collections::HashMap::new(),
            optional_edges: Vec::new(),
            cut_edges: std::collections::HashMap::new(),
            tools: std::collections::HashMap::new(),
            default_provider: None,
            mcp_toolboxes: Vec::new(),
            py_state: None,
        })
    }

    /// Add a filter node. Returns the node id.
    ///
    /// Usage:
    ///   g.node(MyFilter())                        # auto-named
    ///   g.node("scaler", MyFilter())              # explicit id
    ///   g.node(MyFilter(), target="gpu")           # route to gpu worker
    ///   g.node("model", MyFilter(), target="local") # force local execution
    #[pyo3(signature = (*args, target=None))]
    fn node(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        target: Option<String>,
    ) -> PyResult<String> {
        let (node_id, filter_obj) = match args.len() {
            1 => {
                let filter_obj = args.get_item(0)?;
                let class_name = filter_obj
                    .getattr("__class__")?
                    .getattr("__name__")?
                    .extract::<String>()?;
                let snake = to_snake_case(&class_name);
                (snake, filter_obj.to_owned())
            }
            2 => {
                let id = args.get_item(0)?.extract::<String>()?;
                let filter_obj = args.get_item(1)?;
                (id, filter_obj.to_owned())
            }
            n => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node() takes 1 or 2 positional arguments, got {n}"
                )));
            }
        };

        // An Agent or a Judge is a node too — it just runs a turn loop
        // instead of a function. `register_behaviour` dispatches, so there
        // is one way to add a node rather than a second method whose name
        // would collide with the optimiser's `step()`.
        let actual_id = self.free_id(&node_id);
        let mut node = self
            .register_behaviour(py, &actual_id, &filter_obj)?
            .node(&actual_id);
        if let Some(t) = target {
            node = node.with_target(t);
        }
        self.graph.add_node(node);

        Ok(actual_id)
    }

    /// Register a step that can be *spawned* but is not a node in the graph.
    ///
    /// `Spawn` names the work it wants by id, and that id is looked up in the
    /// step library — which `node()` also fills. But a node with no edges is
    /// a root, so registering a spawn target with `node()` makes it run once
    /// on the graph's own input as well, which is wasted work and a confusing
    /// reading of the diagram.
    ///
    /// ```python
    /// g.node("fanout", Planner())        # decides the width at runtime
    /// g.register_step("worker", Worker())  # spawnable, never a root
    /// ```
    ///
    /// The returned id is the one `Spawn` should name.
    fn register_step(
        &mut self,
        py: Python<'_>,
        step_id: &str,
        obj: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        let spec = to_step_spec(py, obj)?;
        for tool in spec.tools() {
            self.tools
                .insert(tool.tool_name().to_string(), tool.clone());
        }
        self.library.register_step_arc(step_id, spec.step());
        self.live_steps
            .insert(step_id.to_string(), obj.clone().unbind());
        Ok(step_id.to_string())
    }

    /// Add a node that routes: it runs `condition`, reads the arm label out
    /// of the result, and executes only that arm.
    ///
    /// ```python
    /// g.branch("router", Classifier(), {
    ///     "billing": soma.Agent(model="ollama/llama3.2", system="Billing."),
    ///     "tech":    "tech_team",     # a node already in the graph
    ///     "default": Escalate(),
    /// })
    /// ```
    ///
    /// The arms are declared, so the compiler rejects one that no edge
    /// reaches and one that no arm declares — the silent-drop failure that
    /// the multi-agent literature files under inter-agent misalignment.
    /// An arm labelled `default` (or `else`) catches anything unmatched;
    /// without one, an unrecognised label is an error rather than a guess.
    #[pyo3(signature = (node_id, condition, arms, target=None))]
    fn branch(
        &mut self,
        py: Python<'_>,
        node_id: String,
        condition: &Bound<'_, PyAny>,
        arms: &Bound<'_, PyDict>,
        target: Option<String>,
    ) -> PyResult<String> {
        if arms.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "branch() needs at least one arm; a router with nowhere to \
                 route is just a node",
            ));
        }

        let actual_id = self.free_id(&node_id);
        // The branch node *is* the condition: the executor runs it and reads
        // the arm label from its output.
        self.register_behaviour(py, &actual_id, condition)?;

        let labels: Vec<String> = arms
            .keys()
            .iter()
            .map(|k| k.extract::<String>())
            .collect::<PyResult<_>>()?;

        let mut node = Node::branch_over(&actual_id, labels);
        if let Some(t) = target {
            node = node.with_target(t);
        }
        self.graph.add_node(node);

        for (key, value) in arms.iter() {
            let label = key.extract::<String>()?;
            let arm_id = self.resolve_member(py, &label, &value)?;
            self.control_edge(&actual_id, &arm_id, Some(&label));
        }

        Ok(actual_id)
    }

    /// Add a node that repeats a body until it signals completion.
    ///
    /// ```python
    /// g.node("draft", Draft())
    /// g.node("critic", soma.Judge(model="ollama/llama3.2", rubric="..."))
    /// g.connect("draft", "critic")
    /// g.loop("refine", body="draft", until="critic", max_iterations=3)
    /// ```
    ///
    /// `body` names the entry node(s); the loop owns those and everything
    /// only reachable through them.
    ///
    /// `until` says when to stop:
    ///
    /// - a node id — that node's output carries the signal: a bool,
    ///   `"done"`/`"stop"`, or a mapping with a `done` key, which is exactly
    ///   what [`Judge`] emits;
    /// - unset (the default) — the body's single terminal node is used, and
    ///   a body with several terminals is a compile error rather than a race;
    /// - `False` — never stop early; run the full `max_iterations`.
    ///
    /// The loop's value is its *carry*: seeded from the loop's input, then
    /// replaced after each pass by the condition node's output. That is what
    /// the body reads on the next round, so a refine loop refines instead of
    /// redrafting the same thing.
    #[pyo3(name = "loop", signature = (node_id, body, until=None, max_iterations=None))]
    fn loop_(
        &mut self,
        py: Python<'_>,
        node_id: String,
        body: &Bound<'_, PyAny>,
        until: Option<&Bound<'_, PyAny>>,
        max_iterations: Option<usize>,
    ) -> PyResult<String> {
        // One entry or several: a list is the general case, a bare value the
        // one people write.
        let entries: Vec<Bound<'_, PyAny>> = match body.try_iter() {
            Ok(iter) if !body.is_instance_of::<pyo3::types::PyString>() => {
                iter.collect::<PyResult<_>>()?
            }
            _ => vec![body.clone()],
        };
        if entries.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "loop() needs a body",
            ));
        }

        let actual_id = self.free_id(&node_id);
        use somatize_core::control::LoopCondition;
        let until = match until {
            None => LoopCondition::BodyTerminal,
            // `False` is the only bool that means anything here: "run the
            // whole count". `True` would have to mean "stop immediately",
            // which nobody writes on purpose.
            Some(u) if u.is_instance_of::<pyo3::types::PyBool>() => {
                if u.extract::<bool>()? {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "until=True says the loop stops before it runs. Pass a node \
                         id to read the signal from, or False to run the full count",
                    ));
                }
                LoopCondition::Exhaust
            }
            Some(u) => {
                let cond = u.extract::<String>()?;
                if self.graph.node(&cond).is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "`{cond}` names no node in this graph, so it cannot be the \
                         loop's stop condition"
                    )));
                }
                LoopCondition::WhenSignaled(cond)
            }
        };

        self.graph
            .add_node(Node::loop_until(&actual_id, max_iterations, until));

        for (i, entry) in entries.iter().enumerate() {
            let fallback = format!("{actual_id}_body_{i}");
            let entry_id = self.resolve_member(py, &fallback, entry)?;
            self.control_edge(&actual_id, &entry_id, None);
        }

        Ok(actual_id)
    }

    /// Set the provider that serves model names given without a prefix.
    ///
    /// ```python
    /// g.use_provider("ollama")
    /// g.step("a", soma.Agent(model="llama3.2"))   # → ollama/llama3.2
    /// ```
    fn use_provider(&mut self, provider: String) {
        self.default_provider = Some(provider);
    }

    /// Make a data edge part of the search space: a study may keep it or
    /// cut it.
    ///
    /// This is topology as a hyperparameter — whether the critic should see
    /// the retriever's output at all is exactly the kind of question a
    /// search answers better than an argument does. Control edges are not
    /// eligible: they are what makes a loop a loop, not a design choice.
    ///
    /// ```python
    /// g.optional("retriever", "critic")
    /// study = g.study("shape", n_trials=20)   # gains `edge:retriever->critic`
    /// ```
    fn optional(&mut self, source: String, target: String) -> PyResult<()> {
        let found = self
            .graph
            .edges
            .iter()
            .find(|e| e.source == source && e.target == target);

        match found {
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "there is no edge `{source}` → `{target}` to make optional"
            ))),
            Some(e) if e.kind != somatize_core::graph::EdgeKind::Data => {
                Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "`{source}` → `{target}` is a control edge; cutting it would \
                     change what the loop or branch owns, not just what flows"
                )))
            }
            Some(_) => {
                let pair = (source, target);
                if !self.optional_edges.contains(&pair) {
                    self.optional_edges.push(pair);
                }
                Ok(())
            }
        }
    }

    /// The edges a study may cut, as `(source, target)`.
    fn optional_edges(&self) -> Vec<(String, String)> {
        self.optional_edges.clone()
    }

    /// Keep or cut one of the optional edges.
    ///
    /// A cut edge is set aside whole, so restoring it restores its id, kind
    /// and label — a trial that cuts an edge must leave the graph identical
    /// to the one the next trial starts from.
    fn set_edge(&mut self, source: String, target: String, enabled: bool) -> PyResult<()> {
        let pair = (source, target);
        if !self.optional_edges.contains(&pair) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "`{}` → `{}` was never declared optional; call optional() first",
                pair.0, pair.1
            )));
        }

        if enabled {
            if let Some((at, edge)) = self.cut_edges.remove(&pair) {
                // Back where it was, not on the end. Appending would leave a
                // graph that is semantically the same and renders, hashes and
                // fingerprints differently — so two trials of the same
                // topology would not compare equal.
                self.graph
                    .edges
                    .insert(at.min(self.graph.edges.len()), edge);
            }
        } else if !self.cut_edges.contains_key(&pair)
            && let Some(i) = self
                .graph
                .edges
                .iter()
                .position(|e| e.source == pair.0 && e.target == pair.1)
        {
            let edge = self.graph.edges.remove(i);
            self.cut_edges.insert(pair, (i, edge));
        }
        Ok(())
    }

    /// The live `Agent`/`Judge` behind each step node, as `(node_id, obj)`.
    ///
    /// The counterpart of `filters()`. A study reads their search spaces and
    /// writes sampled values straight onto them.
    fn steps(&self, py: Python<'_>) -> Vec<(String, PyObject)> {
        let mut items: Vec<(String, PyObject)> = self
            .live_steps
            .iter()
            .map(|(id, obj)| (id.clone(), obj.clone_ref(py)))
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        items
    }

    /// Register a tool without attaching it to a particular agent.
    fn add_tool(&mut self, tool: PyTool) {
        self.tools.insert(tool.tool_name().to_string(), tool);
    }

    /// Start an MCP server and make everything it publishes callable.
    ///
    /// Returns the tool names discovered. Discovery happens now, so a
    /// misconfigured server fails here rather than mid-run.
    #[pyo3(signature = (command, args=None))]
    fn add_mcp_server(
        &mut self,
        py: Python<'_>,
        command: String,
        args: Option<Vec<String>>,
    ) -> PyResult<Vec<String>> {
        let args = args.unwrap_or_default();
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();

        // Spawning and handshaking is I/O; do not hold the GIL for it.
        let mut toolbox = somatize_llm::Toolbox::new();
        py.allow_threads(|| toolbox.add_mcp_server(&command, &refs))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let names: Vec<String> = toolbox.names().into_iter().map(String::from).collect();
        self.mcp_toolboxes.push(toolbox);
        Ok(names)
    }

    /// Connect two nodes with a data edge.
    fn edge(&mut self, source: String, target: String) {
        let id = format!("e_{}", self.graph.edges.len());
        self.graph.add_edge(Edge::data(id, source, target));
    }

    /// Alias for edge().
    fn connect(&mut self, source: String, target: String) {
        self.edge(source, target);
    }

    /// Declare that `source` may hand control to `target`.
    ///
    /// This is what `soma.Goto(target)` needs: a handoff transfers control
    /// rather than passing data, so it is a control edge and not a
    /// `connect`. Declaring it is deliberate — a step that hands control
    /// somewhere the graph never said it could is an error rather than a
    /// silent jump, which is the inter-agent misalignment the multi-agent
    /// literature keeps finding.
    ///
    /// ```python
    /// g.node("triage", Triage())
    /// g.node("billing", soma.Agent(model="ollama/qwen2.5"))
    /// g.handoff("triage", "billing")   # now Goto("billing") is allowed
    /// ```
    fn handoff(&mut self, source: &str, target: &str) {
        self.control_edge(source, target, None);
    }

    /// Fit all trainable filters in topological order.
    ///
    /// If `batch_size` is set, the input is split into batches and each batch
    /// is processed through the entire pipeline (encoder → classifier) before
    /// moving to the next. This keeps memory bounded.
    ///
    /// If workers are registered and no node forces local, training is
    /// dispatched to a remote worker.
    #[pyo3(signature = (x, y=None, batch_size=None, mode="inference", seed=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        y: Option<&Bound<'_, pyo3::types::PyAny>>,
        batch_size: Option<usize>,
        mode: &str,
        seed: Option<i64>,
    ) -> PyResult<()> {
        let x_val = py_to_value(py, x)?;
        let y_val = match y {
            Some(v) => Some(py_to_value(py, v)?),
            None => None,
        };

        // Differentiable mode: compile with CompileMode::Differentiable (which
        // collapses consecutive differentiable filters into a Composite block)
        // and execute via LocalRunner, which delegates the block to the first
        // filter's ``composite_fit``. Gradients flow end-to-end inside the
        // user-provided composite_fit implementation.
        if mode == "differentiable" {
            if !self.workers.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "mode='differentiable' is only supported for local execution \
                     (no workers). Remote differentiable training is not yet implemented.",
                ));
            }
            self.graph.validate().map_err(soma_err_to_py)?;
            let catalog = self.rebuild_catalog(py)?;
            let compile_result = compile(
                &self.graph,
                &catalog,
                CompileMode::Differentiable,
                Some(self.cache.as_ref()),
            )
            .map_err(soma_err_to_py)?;
            let runner = LocalRunner;
            let run_id = somatize_core::util::timestamp_id("fit");
            self.event_bus
                .emit(somatize_core::event::Event::RunStarted {
                    run_id: run_id.clone(),
                    plan_summary: compile_result.plan.summary(),
                });
            let run_start = std::time::Instant::now();
            let run_ctx = somatize_runtime::runner::RunContext::new(
                &catalog,
                self.cache.as_ref(),
                &self.event_bus,
                &run_id,
                GraphInfo::from_graph(&self.graph),
            );
            let result = runner.fit(&compile_result.plan, &run_ctx, &x_val, y_val.as_ref());
            let (_output, states) = match result {
                Ok(out) => {
                    self.event_bus
                        .emit(somatize_core::event::Event::RunCompleted {
                            run_id,
                            duration: run_start.elapsed(),
                        });
                    out
                }
                Err(e) => {
                    self.event_bus.emit(somatize_core::event::Event::RunFailed {
                        run_id,
                        error: e.to_string(),
                    });
                    return Err(soma_err_to_py(e));
                }
            };
            // LocalRunner tags composite-produced states with "__state_{id}".
            // Regular sequential states appear under the bare node_id.
            for (key, state) in states {
                let node_id = key.strip_prefix("__state_").unwrap_or(&key).to_string();
                if let Err(e) = self.library.try_set_state(node_id, state) {
                    return Err(soma_err_to_py(e));
                }
            }
            self.fitted = true;
            return Ok(());
        }
        if mode != "inference" {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown mode={mode:?}. Use 'inference' or 'differentiable'."
            )));
        }

        // Dispatch fit to worker if possible.
        // Release GIL during WS dispatch so worker thread can acquire it for Python execution.
        if !self.workers.is_empty() && self.graph.nodes.iter().all(|n| !n.is_local()) {
            if let Some(bs) = batch_size {
                // Batched fit: dispatch once with batch_size so the worker handles batching
                let mode = somatize_worker::protocol::ExecutionMode::Fit {
                    y: y_val.clone(),
                    batch_size: Some(bs),
                };
                let result = py.allow_threads(|| self.dispatch_to_worker(&x_val, mode));
                let (_output, states) = result?;
                for (node_id, state) in states {
                    if let Err(e) = self.library.try_set_state(&node_id, state) {
                        return Err(soma_err_to_py(e));
                    }
                }
                self.fitted = true;
                return Ok(());
            }

            let mode = somatize_worker::protocol::ExecutionMode::Fit {
                y: y_val.clone(),
                batch_size: None,
            };
            let result = py.allow_threads(|| self.dispatch_to_worker(&x_val, mode));
            let (_output, states) = result?;
            for (node_id, state) in states {
                if let Err(e) = self.library.try_set_state(&node_id, state) {
                    return Err(soma_err_to_py(e));
                }
            }
            self.fitted = true;
            return Ok(());
        }

        // Local fit.
        //
        // Through the compiler and the runner, like every other entry
        // point. This used to be a topological loop written here, walking
        // `graph.topological_sort()` and calling fit/forward node by node
        // — so it ignored parallelism, loops and branches, and it was the
        // only fit anywhere that salted its state keys with the seed. Now
        // the runner salts, and the loop is gone.
        self.graph.validate().map_err(soma_err_to_py)?;
        let catalog = self.rebuild_catalog(py)?;
        let compile_result = compile(
            &self.graph,
            &catalog,
            CompileMode::NoCache,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let run_id = somatize_core::util::timestamp_id("graph_fit");
        self.event_bus
            .emit(somatize_core::event::Event::RunStarted {
                run_id: run_id.clone(),
                plan_summary: compile_result.plan.summary(),
            });
        let run_start = std::time::Instant::now();

        let run_ctx = somatize_runtime::runner::RunContext::new(
            &catalog,
            self.cache.as_ref(),
            &self.event_bus,
            &run_id,
            GraphInfo::from_graph(&self.graph),
        )
        .with_seed(seed);

        // Release the GIL: a Parallel plan runs branches on scoped threads
        // whose Python filters must acquire it.
        let result = py.allow_threads(|| {
            LocalRunner.fit(&compile_result.plan, &run_ctx, &x_val, y_val.as_ref())
        });

        let (_output, states) = match result {
            Ok(out) => {
                self.event_bus
                    .emit(somatize_core::event::Event::RunCompleted {
                        run_id,
                        duration: run_start.elapsed(),
                    });
                out
            }
            Err(e) => {
                self.event_bus.emit(somatize_core::event::Event::RunFailed {
                    run_id,
                    error: e.to_string(),
                });
                return Err(soma_err_to_py(e));
            }
        };

        // Only the `__state_` keys. The map also holds each node's
        // *output* under its bare id, so stripping a prefix that is not
        // there stores an output as a state — and which one wins depends
        // on `HashMap` order, so a scaler ends up with no learned mean
        // roughly half the time.
        for (key, state) in states {
            if let Some(node_id) = key.strip_prefix("__state_") {
                self.library
                    .try_set_state(node_id, state)
                    .map_err(soma_err_to_py)?;
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Forward data through the compiled graph (inference mode).
    ///
    /// Routing:
    /// - stream=True → chunks sent via WS Binary to StreamExecutor on worker
    /// - No workers → local execution
    /// - Workers + all nodes non-local → entire plan dispatched to worker
    /// - Workers + mixed (some local) → local execution with remote fallback
    #[pyo3(signature = (x, stream=false, chunk_size=1024, seed=None, run_id=None))]
    fn forward(
        &self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        stream: bool,
        chunk_size: usize,
        seed: Option<i64>,
        run_id: Option<String>,
    ) -> PyResult<PyObject> {
        // A step has no fit phase — its behaviour comes from a model and a
        // prompt, not from learned state. A graph with nothing trainable in
        // it therefore has nothing to fit, and demanding a fit first would
        // be asking for a no-op.
        if !self.fitted && self.has_trainable_filters() {
            return Err(PyRuntimeError::new_err(
                "graph must be fitted before forward",
            ));
        }
        let x_val = py_to_value(py, x)?;

        // Streaming mode: remote via WS Binary, local via StreamExecutor
        if stream {
            if !self.workers.is_empty() {
                // Release GIL during WS dispatch so worker thread can acquire
                // it for Python execution.
                let output = py.allow_threads(|| self.dispatch_streamed(&x_val, chunk_size))?;
                return value_to_py(py, &output);
            }
            // Local streaming: compile as Stream plan, execute via StreamExecutor
            let catalog = self.rebuild_catalog(py)?;
            let compile_result =
                somatize_compiler::compile_stream(&self.graph, &catalog, chunk_size)
                    .map_err(soma_err_to_py)?;

            let graph_info = GraphInfo::from_graph(&self.graph);
            let mut ctx = Context::new(
                self.event_bus.clone(),
                somatize_core::util::timestamp_id("stream_forward"),
            )
            .with_graph_info(graph_info)
            .with_seed(seed);

            let roots = self.graph.roots();
            if roots.len() == 1 {
                ctx.set(format!("__input_{}", roots[0]), x_val.clone());
            }
            ctx.set("__input__", x_val);

            py.allow_threads(|| {
                executor::execute(
                    &compile_result.plan,
                    &mut ctx,
                    &catalog,
                    self.cache.as_ref(),
                )
            })
            .map_err(soma_err_to_py)?;

            let leaves = self.graph.leaves();
            let output = if let Some(leaf_id) = leaves.first() {
                ctx.store
                    .remove(*leaf_id)
                    .and_then(|vv| vv.as_value().cloned())
                    .unwrap_or(Value::Empty)
            } else {
                ctx.execution_order
                    .last()
                    .and_then(|id| ctx.store.remove(id))
                    .and_then(|vv| vv.as_value().cloned())
                    .unwrap_or(Value::Empty)
            };

            return value_to_py(py, &output);
        }

        // Dispatch entire plan remotely if workers registered and no node forces local
        if !self.workers.is_empty() && self.graph.nodes.iter().all(|n| !n.is_local()) {
            let (output, _states) = py.allow_threads(|| {
                self.dispatch_to_worker(&x_val, somatize_worker::protocol::ExecutionMode::Forward)
            })?;
            return value_to_py(py, &output);
        }

        // Local execution (with optional remote executor for mixed graphs)
        let catalog = self.rebuild_catalog(py)?;
        let compile_result = somatize_compiler::compile(
            &self.graph,
            &catalog,
            CompileMode::Inference,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let graph_info = GraphInfo::from_graph(&self.graph);
        // A caller resuming a suspended run passes its id back. The
        // journal keys an impure effect by `(run, node, turn, index)`, so
        // a fresh id would replay nothing and the answer already recorded
        // would never be found — which is why resuming did not work.
        let run_id = run_id.unwrap_or_else(|| somatize_core::util::timestamp_id("graph_forward"));
        let mut ctx = Context::new(self.event_bus.clone(), run_id)
            .with_graph_info(graph_info)
            .with_seed(seed);

        if let Some(driver) = self.step_runtime(py, &catalog)? {
            ctx = ctx.with_driver(driver, Arc::new(catalog.clone()));
        }
        if let Some(transport) = self.make_transport() {
            ctx = ctx.with_transport(transport);
        }

        let roots = self.graph.roots();
        if roots.len() == 1 {
            ctx.set(format!("__input_{}", roots[0]), x_val.clone());
        }
        ctx.set("__input__", x_val);

        // Release the GIL: Parallel plans run branches on scoped threads
        // whose Python filters must acquire it — holding it here would
        // deadlock the join.
        py.allow_threads(|| {
            executor::execute(
                &compile_result.plan,
                &mut ctx,
                &catalog,
                self.cache.as_ref(),
            )
        })
        .map_err(soma_err_to_py)?;

        // Which leaf is "the output" when there are several? Prefer one that
        // actually ran. A branch makes every arm a leaf, so declaration
        // order alone would return the arm that was *not* taken — an empty
        // value, from a node that never executed.
        //
        // Among leaves that did produce something, declaration order still
        // decides, so a parallel fan-out answers the same as it always has.
        let leaves = self.graph.leaves();
        let produced = leaves
            .iter()
            .find(|id| ctx.store.contains_key(**id))
            .or_else(|| leaves.first());

        let output = if let Some(leaf_id) = produced {
            ctx.store
                .remove(*leaf_id)
                .and_then(|vv| vv.as_value().cloned())
                .unwrap_or(Value::Empty)
        } else {
            ctx.execution_order
                .last()
                .and_then(|id| ctx.store.remove(id))
                .and_then(|vv| vv.as_value().cloned())
                .unwrap_or(Value::Empty)
        };

        value_to_py(py, &output)
    }

    /// Compile and execute, returning all node outputs as a dict.
    fn run(&self, py: Python<'_>) -> PyResult<PyObject> {
        let catalog = self.rebuild_catalog(py)?;
        let compile_result = somatize_compiler::compile(
            &self.graph,
            &catalog,
            CompileMode::NoCache,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let graph_info = GraphInfo::from_graph(&self.graph);
        let run_id = somatize_core::util::timestamp_id("graph_run");
        let mut ctx =
            Context::new(self.event_bus.clone(), run_id.clone()).with_graph_info(graph_info);

        if let Some(driver) = self.step_runtime(py, &catalog)? {
            ctx = ctx.with_driver(driver, Arc::new(catalog.clone()));
        }
        if let Some(transport) = self.make_transport() {
            ctx = ctx.with_transport(transport);
        }

        self.event_bus
            .emit(somatize_core::event::Event::RunStarted {
                run_id: run_id.clone(),
                plan_summary: compile_result.plan.summary(),
            });
        let run_start = std::time::Instant::now();

        // Release the GIL: Parallel plans run branches on scoped threads
        // whose Python filters must acquire it — holding it here would
        // deadlock the join.
        let result = py.allow_threads(|| {
            executor::execute(
                &compile_result.plan,
                &mut ctx,
                &catalog,
                self.cache.as_ref(),
            )
        });
        match &result {
            Ok(()) => self
                .event_bus
                .emit(somatize_core::event::Event::RunCompleted {
                    run_id,
                    duration: run_start.elapsed(),
                }),
            Err(e) => self.event_bus.emit(somatize_core::event::Event::RunFailed {
                run_id,
                error: e.to_string(),
            }),
        };
        result.map_err(soma_err_to_py)?;

        let dict = PyDict::new(py);
        for (k, vv) in &ctx.store {
            if let Some(v) = vv.as_value() {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
        }
        Ok(dict.into_any().unbind())
    }

    /// Answer what a suspended run was waiting for.
    ///
    /// Every argument comes off the `SomaSuspended` exception that stopped
    /// the run, `reason` included — it is part of the journal key, so the
    /// answer has to be filed against the same pause the step described,
    /// not one reconstructed from a guess.
    ///
    /// The answer lands at the exact site the step paused. Running the
    /// graph again replays every prior effect from the record, reaches
    /// that point, and finds it waiting. There is no checkpoint file: the
    /// journal is the checkpoint.
    ///
    /// This existed in Rust and nowhere else, which meant nowhere at all —
    /// the only entry point that runs steps is this one.
    fn resume(
        &mut self,
        py: Python<'_>,
        run_id: &str,
        node_id: &str,
        turn: usize,
        reason: &Bound<'_, PyAny>,
        answer: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let reason: somatize_core::effect::SuspendReason =
            serde_json::from_value(py_any_to_json(reason)?).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "`reason` should be the one from the SomaSuspended exception: {e}"
                ))
            })?;

        let catalog = self.rebuild_catalog(py)?;
        let driver = self.step_runtime(py, &catalog)?.ok_or_else(|| {
            PyRuntimeError::new_err(
                "this graph has no effectful nodes, so nothing in it can suspend",
            )
        })?;

        driver
            .resume_with(run_id, node_id, turn, &reason, py_to_value(py, answer)?)
            .map_err(soma_err_to_py)
    }

    /// Compile the graph and return diagnostic information.
    #[pyo3(signature = (mode="inference"))]
    fn compile(&self, py: Python<'_>, mode: &str) -> PyResult<PyObject> {
        let compile_mode = match mode {
            "inference" => CompileMode::Inference,
            "differentiable" => CompileMode::Differentiable,
            "no_cache" => CompileMode::NoCache,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown mode: {mode}. Use 'inference', 'differentiable', or 'no_cache'."
                )));
            }
        };

        // The rebuilt catalog, not `self.library`: passing the filter half
        // alone is how `.compile()` came to skip every step's schema while
        // `.run()` checked them.
        let catalog = self.rebuild_catalog(py)?;
        let result = somatize_compiler::compile(
            &self.graph,
            &catalog,
            compile_mode,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let dict = PyDict::new(py);
        let summary = result.plan.summary();
        dict.set_item("total_nodes", summary.total_nodes)?;
        dict.set_item("cached_nodes", summary.cached_nodes)?;
        dict.set_item("parallel_branches", summary.parallel_branches)?;

        // Structured diagnostics: {node, level, message} dicts, not
        // Debug strings — readable and machine-consumable.
        let diags = PyList::empty(py);
        for d in &result.diagnostics {
            let entry = PyDict::new(py);
            entry.set_item("node", &d.node_id)?;
            entry.set_item(
                "level",
                match d.level {
                    somatize_compiler::DiagnosticLevel::Warning => "warning",
                    somatize_compiler::DiagnosticLevel::Info => "info",
                },
            )?;
            entry.set_item("message", &d.message)?;
            diags.append(entry)?;
        }
        dict.set_item("diagnostics", diags)?;
        dict.set_item("plan_text", format!("{}", result.plan))?;
        dict.set_item("plan_mermaid", result.plan.to_mermaid())?;
        dict.set_item("plan_svg", result.plan.to_graph().to_svg())?;

        Ok(dict.into_any().unbind())
    }

    // ── Visualization ──

    /// Render the graph as a Mermaid diagram string.
    ///
    /// `overlay` is an optional dict of per-node execution annotations
    /// (the shape `RunView.overlay()` returns): status coloring plus a
    /// duration/cache/flags label line per node.
    #[pyo3(signature = (overlay=None))]
    fn to_mermaid(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_mermaid()),
            Some(ov) => Ok(self.graph.to_mermaid_with(&py_overlay(py, ov)?)),
        }
    }

    /// Render the graph as a self-contained SVG diagram (same optional
    /// `overlay` as `to_mermaid`). No JavaScript — displays inline in
    /// any notebook viewer.
    #[pyo3(signature = (overlay=None))]
    fn to_svg(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_svg()),
            Some(ov) => Ok(self.graph.to_svg_with(&py_overlay(py, ov)?)),
        }
    }

    /// Notebook display: the architecture as an inline SVG diagram
    /// (falls back to the text tree for very large graphs).
    fn _repr_html_(&self) -> String {
        if self.graph.nodes.is_empty() {
            return "<i>empty graph — add nodes with g.node(...)</i>".to_string();
        }
        if self.graph.nodes.len() > 80 {
            return format!(
                "<pre style='font-family:ui-monospace,monospace'>{}</pre>",
                self.graph
                    .to_text()
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
            );
        }
        self.graph.to_svg()
    }

    /// Render the graph as a Graphviz DOT string (same optional
    /// `overlay` as `to_mermaid`).
    #[pyo3(signature = (overlay=None))]
    fn to_graphviz(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_graphviz()),
            Some(ov) => Ok(self.graph.to_graphviz_with(&py_overlay(py, ov)?)),
        }
    }

    /// Render the graph as an ASCII text tree.
    fn to_text(&self) -> String {
        self.graph.to_text()
    }

    // ── Events ──

    /// Register a Python callback to receive events during execution.
    ///
    /// The callback is called with a dict for each event. Events are
    /// delivered in a background thread; the callback must be thread-safe.
    ///
    /// Usage:
    /// ```python
    /// def on_event(event):
    ///     print(event["event_type"], event.get("node_id", ""))
    /// g.on_event(on_event)
    /// g.fit(data)
    /// ```
    fn on_event(&self, callback: PyObject) -> PyResult<()> {
        let mut rx = self.event_bus.subscribe();
        std::thread::spawn(move || {
            while let Ok(event) = rx.blocking_recv() {
                if let Ok(json_str) = serde_json::to_string(&event) {
                    Python::with_gil(|py| {
                        // Parse JSON string into Python dict via json.loads
                        let json_mod = py.import("json").unwrap();
                        if let Ok(dict) = json_mod.call_method1("loads", (json_str,)) {
                            let _ = callback.call1(py, (dict,));
                        }
                    });
                }
            }
        });
        Ok(())
    }

    /// Emit an event onto the graph's bus from Python.
    ///
    /// The dict must carry an `event_type` matching a Soma event
    /// variant (e.g. `StepCompleted`, `MetricReported`, `HealthFlag`)
    /// plus that variant's fields. Used by the native training loop and
    /// the gradient audit to make Python-side progress visible to
    /// trackers and subscribers.
    fn emit_event(&self, py: Python<'_>, event: &Bound<'_, PyDict>) -> PyResult<()> {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (event,))?.extract()?;
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid event JSON: {e}")))?;
        let event: somatize_core::event::Event = serde_json::from_value(value)
            .map_err(|e| PyRuntimeError::new_err(format!("unknown or malformed event: {e}")))?;
        self.event_bus.emit(event);
        Ok(())
    }

    /// Serialized graph topology (nodes/edges) as JSON — written into
    /// run directories so a front-end can draw the architecture.
    fn graph_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.graph)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Start a tracked run: creates `.soma/runs/<run_id>/`, snapshots
    /// the graph topology into it (`graph.json`, `graph.mmd`,
    /// `fingerprint.json`) and attaches its lossless sink to this
    /// graph's event bus. Prefer the `graph.track_run(...)` context
    /// manager from Python.
    ///
    /// This is the only writer of those three files: it is the one place
    /// where the `Graph` and the `NodeCatalog` are both in scope, so
    /// it is the only place that can stamp per-node config hashes into
    /// the fingerprint.
    ///
    /// `params` are the hyperparameters that live outside the graph;
    /// they are what makes a `ParamChanged` derivation possible when a
    /// later run varies one. `parent` names the run this one descends
    /// from — omit it and soma resolves one from `$SOMA_PARENT_RUN` or
    /// `.soma/HEAD` (see `soma.checkout`).
    #[pyo3(signature = (name, root=".soma".to_string(), kind="train".to_string(), tags=None, params=None, parent=None, hypothesis=None))]
    #[allow(clippy::too_many_arguments)]
    fn begin_run(
        &self,
        py: Python<'_>,
        name: String,
        root: String,
        kind: String,
        tags: Option<Vec<String>>,
        params: Option<&Bound<'_, PyDict>>,
        parent: Option<String>,
        hypothesis: Option<String>,
    ) -> PyResult<PyRun> {
        let kind = match kind.as_str() {
            "fit" => RunKind::Fit,
            "train" => RunKind::Train,
            "study" => RunKind::Study,
            "trial" => RunKind::Trial,
            _ => RunKind::Other,
        };
        let tracker = LocalTracker::create(&root, kind, &name).map_err(soma_err_to_py)?;
        self.snapshot_topology(&tracker)?;

        // Enrich the manifest with Python-side context.
        let mut manifest = load_manifest(tracker.run_dir()).map_err(soma_err_to_py)?;
        manifest.tags = tags.unwrap_or_default();
        manifest.python_version = Some(py.version().split_whitespace().next().unwrap_or("").into());
        manifest.params = match params {
            Some(dict) => match py_any_to_json(dict.as_any())? {
                serde_json::Value::Object(map) => map.into_iter().collect(),
                _ => HashMap::new(),
            },
            None => HashMap::new(),
        };
        manifest.parent_run_id = resolve_parent(&root, parent.as_deref());
        manifest.hypothesis = hypothesis;
        manifest.graph = Some(GraphSummaryInfo {
            n_nodes: self.graph.nodes.len(),
            node_ids: self.graph.nodes.iter().map(|n| n.id.clone()).collect(),
            graph_path: Some("graph.json".into()),
            mermaid_path: Some("graph.mmd".into()),
        });
        tracker.save_manifest(&manifest).map_err(soma_err_to_py)?;

        let sink = tracker.sink();
        self.event_bus.add_sink(sink.clone());
        Ok(PyRun {
            tracker: Arc::new(tracker),
            bus: self.event_bus.clone(),
            sink,
            finished: std::sync::atomic::AtomicBool::new(false),
            summary: std::sync::Mutex::new(HashMap::new()),
        })
    }

    // ── Workers ──

    /// Register a remote worker for direct connection (mode B).
    ///
    /// Usage:
    ///   g.add_worker("ws://gpu-0:8080", token="sk-xxx", tags=["gpu"])
    #[pyo3(signature = (address, token=None, tags=None))]
    fn add_worker(&mut self, address: String, token: Option<String>, tags: Option<Vec<String>>) {
        self.workers
            .push((address, token, tags.unwrap_or_default()));
    }

    /// Configure a DataStore for persistent data transport (opt-in).
    ///
    /// When set, large payloads are uploaded to the store and workers read
    /// via DataRef instead of receiving data inline or via HTTP upload.
    ///
    /// Usage:
    ///   g.set_data_store("local", path="/data/soma")
    ///   g.set_data_store("s3", bucket="my-lab", prefix="exp/",
    ///                    endpoint="s3.amazonaws.com",
    ///                    access_key="AK...", secret_key="SK...")
    #[pyo3(signature = (store_type, path=None, bucket=None, prefix=None, endpoint=None, access_key=None, secret_key=None, cache_dir=None))]
    #[allow(clippy::too_many_arguments)]
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
        match store_type.as_str() {
            "local" => {
                let p = path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("local store requires 'path'")
                })?;
                let store = somatize_core::store::LocalDataStore::new(p);
                self.data_store = Some(Arc::new(store));
                Ok(())
            }
            "s3" => {
                let bucket = bucket.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("s3 store requires 'bucket'")
                })?;
                let prefix = prefix.unwrap_or_default();
                let endpoint = endpoint.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("s3 store requires 'endpoint'")
                })?;
                let ak = access_key
                    .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "s3 store requires 'access_key' or AWS_ACCESS_KEY_ID env var",
                        )
                    })?;
                let sk = secret_key
                    .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "s3 store requires 'secret_key' or AWS_SECRET_ACCESS_KEY env var",
                        )
                    })?;
                let cache = cache_dir.unwrap_or_else(|| {
                    std::env::temp_dir()
                        .join(format!("soma-s3-cache-{bucket}"))
                        .to_string_lossy()
                        .to_string()
                });
                let store =
                    somatize_core::store::S3DataStore::new(bucket, prefix, endpoint, ak, sk, cache)
                        .map_err(soma_err_to_py)?;
                self.data_store = Some(Arc::new(store));
                Ok(())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown store type: '{other}'. Available: local, s3"
            ))),
        }
    }

    /// Shutdown a specific worker by address.
    ///
    /// Usage:
    ///   g.shutdown_worker("ws://worker:8080")
    ///   g.shutdown_worker("ws://worker:8080", reason="maintenance")
    #[pyo3(signature = (address, reason=None))]
    fn shutdown_worker(&self, address: String, reason: Option<String>) -> PyResult<()> {
        let token = self
            .workers
            .iter()
            .find(|(a, _, _)| *a == address)
            .and_then(|(_, t, _)| t.clone());
        Self::send_shutdown(&address, token.as_deref(), &reason.unwrap_or_default())
    }

    /// Shutdown all registered workers.
    ///
    /// Usage:
    ///   g.shutdown_workers()
    ///   g.shutdown_workers(reason="end of experiment")
    #[pyo3(signature = (reason=None))]
    fn shutdown_workers(&self, reason: Option<String>) -> PyResult<()> {
        let reason = reason.unwrap_or_default();
        for (addr, token, _) in &self.workers {
            if let Err(e) = Self::send_shutdown(addr, token.as_deref(), &reason) {
                eprintln!("Warning: failed to shutdown {addr}: {e}");
            }
        }
        Ok(())
    }

    /// Set a coordinator for auto-discovery (mode C).
    ///
    /// Usage:
    ///   g.set_coordinator("http://coord:9090", token="sk-xxx")
    #[pyo3(signature = (url, token=None))]
    fn set_coordinator(&mut self, url: String, token: Option<String>) {
        self.coordinator = Some((url, token));
    }

    /// List known workers (from add_worker or coordinator).
    ///
    /// Returns a list of dicts with worker info.
    fn workers(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);

        // Direct workers
        for (addr, _token, tags) in &self.workers {
            let dict = PyDict::new(py);
            dict.set_item("address", addr)?;
            dict.set_item("tags", tags)?;
            dict.set_item("source", "direct")?;
            list.append(dict)?;
        }

        // If coordinator is set, query it for registered workers
        if let Some((url, token)) = &self.coordinator {
            let workers_url = format!("{url}/workers");
            // Synchronous HTTP request (blocking in Python context is fine)
            let client = reqwest::blocking::Client::new();
            let mut request = client.get(&workers_url);
            if let Some(t) = token {
                request = request.query(&[("token", t.as_str())]);
            }
            if let Ok(resp) = request.send()
                && let Ok(text) = resp.text()
            {
                let json_mod = py.import("json")?;
                if let Ok(parsed) = json_mod.call_method1("loads", (text,))
                    && let Ok(items) = parsed.downcast::<PyList>()
                {
                    for item in items.iter() {
                        list.append(item)?;
                    }
                }
            }
        }

        Ok(list.into_any().unbind())
    }

    /// Get the full module source code for a filter node (for Nous agent introspection).
    /// Returns None if the node has no captured source.
    fn filter_source(&self, node_id: String) -> Option<String> {
        self.filter_sources.get(&node_id).cloned()
    }

    /// Get all filter sources as a dict: {node_id: source_code}.
    fn filter_sources_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (node_id, source) in &self.filter_sources {
            dict.set_item(node_id, source)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Retrieve the live Python filter instance registered under `node_id`.
    ///
    /// Returns ``None`` if the node doesn't exist or wasn't added through
    /// the Python `node()` API (e.g. nodes materialised from a serialised
    /// graph have only pickled bytes, not a live instance).
    ///
    /// Used by the in-process training path so callers can manipulate the
    /// filter directly — e.g. toggle `self.training`, read `_module`, or
    /// extract `state_dict()` — without round-tripping through a pickle.
    fn filter(&self, py: Python<'_>, node_id: String) -> Option<PyObject> {
        self.live_filters.get(&node_id).map(|o| o.clone_ref(py))
    }

    /// List node ids with live Python filter instances, in topological order.
    ///
    /// Falls back to insertion order if topological sort fails (e.g. the
    /// graph hasn't been validated yet — possible during construction).
    /// Callers that drive training need the topo order so output of one
    /// filter feeds the next.
    fn filter_ids(&self) -> Vec<String> {
        match self.graph.topological_sort() {
            Ok(sorted) => sorted
                .into_iter()
                .filter(|id| self.live_filters.contains_key(*id))
                .map(|id| id.to_string())
                .collect(),
            Err(_) => self.live_filters.keys().cloned().collect(),
        }
    }

    /// Return live Python filter instances as an ordered list of
    /// ``(node_id, filter)`` tuples in topological order.
    ///
    /// Returning a list (vs. a dict) preserves the order — callers
    /// iterating to chain forwards get inputs threaded correctly.
    fn filters(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for node_id in self.filter_ids() {
            if let Some(obj) = self.live_filters.get(&node_id) {
                let tuple = (node_id, obj.clone_ref(py));
                list.append(tuple)?;
            }
        }
        Ok(list.into_any().unbind())
    }

    /// Store a Python state value for a filter node.
    ///
    /// Used by ``Graph.freeze()`` (Python side) to push each live
    /// ``DifferentiableFilter`` module's serialised ``state_dict`` into
    /// the runtime's filter-state library, so subsequent eval calls go
    /// through the Rust forward path with state pre-populated.
    fn set_node_state(
        &mut self,
        py: Python<'_>,
        node_id: String,
        state: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value = py_to_value(py, &state)?;
        self.library
            .try_set_state(node_id, value)
            .map_err(soma_err_to_py)?;
        Ok(())
    }

    /// List data edges as ``[(source, target), ...]`` in insertion order.
    ///
    /// Used by :meth:`Graph.save` to record topology in the manifest so
    /// :meth:`Graph.load` can reconstruct non-linear graphs (forks,
    /// joins) instead of falling back to a linear chain.
    fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect()
    }

    /// Retrieve the stored state value for a filter node, or ``None``.
    ///
    /// Mirror of :meth:`set_node_state`. Used by ``Graph.state()`` to
    /// snapshot every node's state for checkpointing.
    fn get_node_state(&self, py: Python<'_>, node_id: String) -> PyResult<Option<PyObject>> {
        match self.library.get_state(&node_id) {
            Some(arc) => Ok(Some(value_to_py(py, arc.as_ref())?)),
            None => Ok(None),
        }
    }

    /// Mark the graph as fitted without running ``fit()``.
    ///
    /// The Rust ``forward`` path refuses to run on an un-fitted graph.
    /// When the user trains via the Python autograd loop (``train()`` /
    /// ``forward(x)`` / ``backward`` / ``step``) and then calls
    /// ``freeze()``, no Rust ``fit()`` ran — but state has been pushed
    /// via ``set_node_state``. ``freeze()`` calls this so the
    /// subsequent eval ``forward`` is allowed.
    fn mark_fitted(&mut self) {
        self.fitted = true;
    }

    /// Per-graph scratch dict for Python-side orchestration state.
    ///
    /// PyGraph doesn't expose ``__dict__``, so callers (e.g. the
    /// _orchestrator module) use this dict to attach things like the
    /// registered optimiser without monkey-patching the class.
    /// Lazily created on first access.
    #[getter]
    fn py_state(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if self.py_state.is_none() {
            self.py_state = Some(PyDict::new(py).unbind());
        }
        Ok(self.py_state.as_ref().unwrap().clone_ref(py))
    }

    /// Number of nodes in the graph.
    fn __len__(&self) -> usize {
        self.graph.nodes.len()
    }

    fn __repr__(&self) -> String {
        let n = self.graph.nodes.len();
        let e = self.graph.edges.len();
        format!(
            "Graph({n} nodes, {e} edges, fitted={fitted})",
            fitted = self.fitted
        )
    }

    fn __str__(&self) -> String {
        self.graph.to_text()
    }
}

fn to_snake_case(name: &str) -> String {
    name.chars()
        .enumerate()
        .fold(String::new(), |mut s, (i, c)| {
            if c.is_uppercase() && i > 0 {
                s.push('_');
            }
            s.push(c.to_ascii_lowercase());
            s
        })
}

// ── PyWorker ──

/// A Soma worker that can be started from Python.
///
/// Usage:
///   from soma import Worker
///   w = Worker(port=8080, tags=["gpu", "training"], token="sk-xxx")
///   w.serve()  # blocks, serving requests
#[pyclass(name = "Worker")]
struct PyWorker {
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

// ── PyRun ──

/// A tracked run bound to a graph's event bus.
///
/// Created by `Graph.begin_run` (or the `graph.track_run(...)` context
/// manager). Metrics logged here become `MetricReported` events, which
/// the run's sink persists to `metrics.jsonl`/`events.jsonl`.
#[pyclass(name = "Run")]
struct PyRun {
    tracker: Arc<LocalTracker>,
    bus: Arc<EventBus>,
    sink: Arc<dyn somatize_core::tracking::EventSink>,
    finished: std::sync::atomic::AtomicBool,
    /// W&B-style summary: last logged value per metric name, written
    /// into the experiments journal on finish.
    summary: std::sync::Mutex<HashMap<String, f64>>,
}

#[pymethods]
impl PyRun {
    #[getter]
    fn id(&self) -> String {
        self.tracker.run_id().to_string()
    }

    /// Absolute path of the run directory.
    #[getter]
    fn dir(&self) -> String {
        self.tracker.run_dir().display().to_string()
    }

    /// Log a scalar metric (optionally scoped to a node).
    #[pyo3(signature = (name, value, step=None, node=None))]
    fn log(&self, name: String, value: f64, step: Option<usize>, node: Option<String>) {
        if let Ok(mut summary) = self.summary.lock() {
            summary.insert(name.clone(), value);
        }
        self.bus.emit(somatize_core::event::Event::MetricReported {
            run_id: self.tracker.run_id().to_string(),
            metric: MetricRecord {
                name,
                value,
                step: step.unwrap_or(0),
                timestamp: chrono::Utc::now(),
            },
            node_id: node,
            trial_id: None,
        });
    }

    /// Mark the start of an epoch.
    #[pyo3(signature = (epoch, total=None))]
    fn log_epoch(&self, epoch: usize, total: Option<usize>) {
        self.bus.emit(somatize_core::event::Event::EpochStarted {
            run_id: self.tracker.run_id().to_string(),
            epoch,
            total_epochs: total,
        });
        let _ = self.tracker.heartbeat();
    }

    /// Mark the end of an epoch with its summary metrics.
    #[pyo3(signature = (epoch, metrics=None))]
    fn log_epoch_completed(
        &self,
        epoch: usize,
        metrics: Option<std::collections::HashMap<String, f64>>,
    ) {
        let now = chrono::Utc::now();
        let records: Vec<MetricRecord> = metrics
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| {
                if let Ok(mut summary) = self.summary.lock() {
                    summary.insert(name.clone(), value);
                }
                MetricRecord {
                    name,
                    value,
                    step: epoch,
                    timestamp: now,
                }
            })
            .collect();
        self.bus.emit(somatize_core::event::Event::EpochCompleted {
            run_id: self.tracker.run_id().to_string(),
            epoch,
            metrics: records,
        });
        let _ = self.tracker.heartbeat();
    }

    /// Mark one optimizer step (used by the native training loop).
    #[pyo3(signature = (step, epoch=None))]
    fn step_completed(&self, step: usize, epoch: Option<usize>) {
        self.bus.emit(somatize_core::event::Event::StepCompleted {
            run_id: self.tracker.run_id().to_string(),
            step,
            epoch,
        });
    }

    /// Refresh the run's heartbeat (liveness for external readers).
    fn heartbeat(&self) -> PyResult<()> {
        self.tracker.heartbeat().map_err(soma_err_to_py)
    }

    /// Finalize the run: flush logs, set terminal status, detach the
    /// sink from the graph's bus, and append the run's summary to the
    /// experiments journal. Idempotent.
    #[pyo3(signature = (status="completed".to_string()))]
    fn finish(&self, status: String) -> PyResult<()> {
        if self
            .finished
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }
        self.bus.remove_sink(&self.sink);
        let state = match status.as_str() {
            "failed" => RunState::Failed,
            _ => RunState::Completed,
        };
        self.tracker.finalize(state).map_err(soma_err_to_py)?;
        if matches!(state, RunState::Completed) {
            let metrics = self.summary.lock().map(|s| s.clone()).unwrap_or_default();
            append_run_record(self.tracker.run_dir(), HashMap::new(), metrics);
            // HEAD advances only on success: a run that died must never
            // become the parent of everything that follows it.
            if let Some(root) = tracking_root(self.tracker.run_dir()) {
                advance_head(root, self.tracker.run_id());
            }
        }
        Ok(())
    }
}

/// The tracking root (`.soma`) a run directory lives under.
fn tracking_root(run_dir: &std::path::Path) -> Option<&std::path::Path> {
    run_dir.parent().and_then(|p| p.parent())
}

/// Append a finished run's summary to `<root>/experiments.jsonl`,
/// linked to its parent by the derivation move between them.
///
/// This is the single path from "a run happened" to "the pool knows
/// about it": `RunReader` → `summarize` → `ExperimentRecord::from_run`.
/// `extra_params` and `extra_metrics` carry what the run directory does
/// not know (a study's best-trial configuration, a training loop's
/// W&B-style summary metrics).
///
/// Best-effort throughout: recording an experiment must never fail a
/// training run that already produced its results.
fn append_run_record(
    run_dir: &std::path::Path,
    extra_params: HashMap<String, serde_json::Value>,
    extra_metrics: HashMap<String, f64>,
) {
    use somatize_memory::{ExperimentRecord, FileKnowledgeBase, KnowledgeBase};

    let Some(root) = tracking_root(run_dir) else {
        return;
    };
    let reader = match RunReader::open(run_dir) {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!(
                "soma: cannot read run {} to record it: {e}",
                run_dir.display()
            );
            return;
        }
    };
    let summary = match summarize(&reader) {
        Ok(summary) => summary,
        Err(e) => {
            eprintln!("soma: cannot summarize run {}: {e}", run_dir.display());
            return;
        }
    };

    let mut record = ExperimentRecord::from_run(&summary)
        .with_extra_params(extra_params)
        .with_extra_metrics(extra_metrics);

    let mut kb = match FileKnowledgeBase::open(root.join("experiments.jsonl")) {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("soma: failed to open experiments.jsonl: {e}");
            return;
        }
    };
    if let Some(parent_id) = summary.parent_run_id.clone()
        && let Ok(Some(parent)) = kb.get(&parent_id)
    {
        record = record.descended_from(&parent);
    }
    if let Err(e) = kb.record(record) {
        eprintln!("soma: failed to record experiment: {e}");
    }
}

// ── Cache management (backs the `soma cache` CLI) ──

fn resolve_cache_root(path: Option<String>) -> PyResult<std::path::PathBuf> {
    match path {
        Some(p) => Ok(p.into()),
        None => default_cache_dir().ok_or_else(|| {
            PyRuntimeError::new_err("no cache directory: set SOMA_CACHE_DIR or HOME")
        }),
    }
}

fn open_store(path: Option<String>) -> PyResult<FsActionStore> {
    let root = resolve_cache_root(path)?;
    FsActionStore::new(root).map_err(|e| PyRuntimeError::new_err(format!("cache open: {e}")))
}

/// Stats about the shared persistent cache.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn cache_stats(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    let store = open_store(path)?;
    let actions = store
        .actions()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let cas_bytes = store
        .cas_bytes()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pinned = store
        .pinned()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let total_compute_ms: u64 = actions.iter().map(|a| a.compute_ms).sum();
    let mut blob_hashes = std::collections::HashSet::new();
    let mut available = 0usize;
    for record in &actions {
        for hash in record.outputs.values() {
            if blob_hashes.insert(*hash)
                && somatize_core::action::BlobStore::contains(&store, hash).unwrap_or(false)
            {
                available += 1;
            }
        }
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("root", store.root().display().to_string())?;
    dict.set_item("actions", actions.len())?;
    dict.set_item("unique_outputs", blob_hashes.len())?;
    dict.set_item("blobs_available", available)?;
    dict.set_item("blobs_evicted", blob_hashes.len() - available)?;
    dict.set_item("cas_bytes", cas_bytes)?;
    dict.set_item("pinned", pinned.len())?;
    dict.set_item("saved_compute_ms", total_compute_ms)?;
    Ok(dict.into_any().unbind())
}

/// Cost-aware GC: evict blobs (records retained) down to `max_bytes`.
#[pyfunction]
#[pyo3(signature = (max_bytes, min_age_secs=3600, path=None))]
fn cache_gc(
    py: Python<'_>,
    max_bytes: u64,
    min_age_secs: u64,
    path: Option<String>,
) -> PyResult<PyObject> {
    let store = open_store(path)?;
    let policy = somatize_runtime::cache::gc::GcPolicy {
        max_bytes,
        min_age: std::time::Duration::from_secs(min_age_secs),
    };
    let report = somatize_runtime::cache::gc::collect(&store, &policy)
        .map_err(|e| PyRuntimeError::new_err(format!("gc: {e}")))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("bytes_before", report.bytes_before)?;
    dict.set_item("bytes_after", report.bytes_after)?;
    dict.set_item("blobs_evicted", report.blobs_evicted)?;
    dict.set_item("blobs_kept", report.blobs_kept)?;
    dict.set_item("pinned_blobs", report.pinned_blobs)?;
    Ok(dict.into_any().unbind())
}

/// Pin an action as a GC root under a human-readable name.
#[pyfunction]
#[pyo3(signature = (name, key_hex, path=None))]
fn cache_pin(name: &str, key_hex: &str, path: Option<String>) -> PyResult<()> {
    if key_hex.len() != 64 || !key_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "key must be a 64-char hex action key",
        ));
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&key_hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    }
    let store = open_store(path)?;
    store
        .pin(name, &somatize_core::cache::CacheKey(digest))
        .map_err(|e| PyRuntimeError::new_err(format!("pin: {e}")))
}

/// Verify every referenced blob against its content hash.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn cache_verify(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    use somatize_core::action::BlobStore;
    let store = open_store(path)?;
    let actions = store
        .actions()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (mut ok, mut missing, mut corrupt) = (0usize, 0usize, 0usize);
    let mut seen = std::collections::HashSet::new();
    for record in &actions {
        for hash in record.outputs.values() {
            if !seen.insert(*hash) {
                continue;
            }
            // get_bytes verifies content and maps corrupt blobs to None,
            // so distinguish via raw existence.
            match (
                store.contains(hash).unwrap_or(false),
                store.get_bytes(hash).ok().flatten(),
            ) {
                (true, Some(_)) => ok += 1,
                (true, None) => corrupt += 1,
                (false, _) => missing += 1,
            }
        }
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("ok", ok)?;
    dict.set_item("missing", missing)?;
    dict.set_item("corrupt", corrupt)?;
    Ok(dict.into_any().unbind())
}

/// Remove legacy Phase-1 (v1) cache entries: `<hh>/<hh>/<hex>.json`
/// shard trees at the store root. The v2 key namespace makes them
/// unreachable — they are pure dead weight.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn cache_purge_v1(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    let root = resolve_cache_root(path)?;
    let mut removed_files = 0u64;
    let mut removed_bytes = 0u64;
    if root.exists() {
        for entry in std::fs::read_dir(&root).map_err(|e| PyRuntimeError::new_err(e.to_string()))? {
            let entry = entry.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().to_string();
            // v1 shard dirs are exactly two lowercase hex chars.
            let is_v1_shard = name.len() == 2
                && name.chars().all(|c| c.is_ascii_hexdigit())
                && entry.path().is_dir();
            if is_v1_shard {
                fn walk(dir: &std::path::Path, files: &mut u64, bytes: &mut u64) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for e in entries.filter_map(|e| e.ok()) {
                            let p = e.path();
                            if p.is_dir() {
                                walk(&p, files, bytes);
                            } else {
                                *files += 1;
                                *bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
                walk(&entry.path(), &mut removed_files, &mut removed_bytes);
                std::fs::remove_dir_all(entry.path())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("removed_files", removed_files)?;
    dict.set_item("removed_bytes", removed_bytes)?;
    Ok(dict.into_any().unbind())
}

// ── Module ──

// ── Run-directory readers (back `soma.runs()` / `soma.RunView`) ──
//
// Each function returns a JSON string (the serde shape of the
// soma-runtime reader structs); the Python wrapper in `soma/_runs.py`
// parses it. Same pattern as `Graph.graph_json`.

/// Convert an overlay dict (the `RunView.overlay()` shape) into the
/// core overlay struct, via JSON like `Graph.emit_event`.
fn py_overlay(
    py: Python<'_>,
    overlay: &Bound<'_, PyDict>,
) -> PyResult<somatize_core::viz::GraphOverlay> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (overlay,))?.extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid overlay: {e}")))
}

fn open_run_reader(dir: &str) -> PyResult<somatize_runtime::tracking::RunReader> {
    somatize_runtime::tracking::RunReader::open(dir)
        .map_err(|e| PyRuntimeError::new_err(format!("open run dir {dir}: {e}")))
}

fn to_json_py<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// One run folded into a `RunSummary`: identity, cost, metrics and a
/// templated conclusion. Works on runs recorded before the experiment
/// pool existed — everything it cannot read becomes a warning.
#[pyfunction]
fn run_summary_json(dir: String) -> PyResult<String> {
    let summary =
        summarize(&open_run_reader(&dir)?).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&summary)
}

/// Point `.soma/HEAD` at an existing run so the next run branches from
/// it — the handle for going back and trying something else.
///
/// Errors when the run is unknown to this root: attaching the next
/// experiment to a parent that does not exist is worse than not
/// branching at all.
#[pyfunction]
#[pyo3(signature = (run_id, root=".soma".to_string()))]
fn checkout_run(run_id: String, root: String) -> PyResult<()> {
    somatize_runtime::tracking::checkout(&root, &run_id).map_err(soma_err_to_py)
}

/// The run id in `.soma/HEAD`, or None when the next run starts a new
/// line.
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
fn read_head_run(root: String) -> Option<String> {
    somatize_runtime::tracking::read_head(&root)
}

/// Detach HEAD: the next run starts its own research line.
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
fn clear_head_run(root: String) -> PyResult<()> {
    somatize_runtime::tracking::clear_head(&root).map_err(soma_err_to_py)
}

/// Open the project's experiment journal, refreshed.
fn open_kb(root: &str) -> PyResult<somatize_memory::FileKnowledgeBase> {
    somatize_memory::FileKnowledgeBase::open(std::path::Path::new(root).join("experiments.jsonl"))
        .map_err(|e| PyRuntimeError::new_err(format!("cannot open the experiment pool: {e}")))
}

/// Rank past experiments against a query — the Python half of
/// `kb_find_similar`. Returns a JSON array of scored records.
///
/// Text, architecture, recency and importance, added rather than
/// multiplied; see `somatize_memory::retrieval` for the formula.
#[pyfunction]
#[pyo3(signature = (query="".to_string(), like_run=None, limit=5, research_line=None, tags=None, half_life_days=None, root=".soma".to_string()))]
#[allow(clippy::too_many_arguments)]
fn kb_find_similar_json(
    query: String,
    like_run: Option<String>,
    limit: usize,
    research_line: Option<String>,
    tags: Option<Vec<String>>,
    half_life_days: Option<f64>,
    root: String,
) -> PyResult<String> {
    use somatize_memory::{KnowledgeBase, RetrievalQuery};

    if query.trim().is_empty() && like_run.is_none() {
        return Err(PyRuntimeError::new_err(
            "find_similar needs a query, a like_run, or both",
        ));
    }
    let kb = open_kb(&root)?;
    let mut retrieval = RetrievalQuery::new(&query, chrono::Utc::now());
    retrieval.limit = limit.clamp(1, 100);
    retrieval.research_line = research_line;
    retrieval.tags = tags.unwrap_or_default();
    if let Some(days) = half_life_days.filter(|d| *d > 0.0) {
        retrieval.half_life_days = days;
    }
    if let Some(run_id) = &like_run {
        match kb.get(run_id).ok().flatten().and_then(|r| r.architecture) {
            Some(architecture) => retrieval.architecture = Some(architecture),
            None => {
                return Err(PyRuntimeError::new_err(format!(
                    "no architecture recorded for '{run_id}' — it may predate fingerprinting. \
                     Pass a query instead."
                )));
            }
        }
    }

    let hits = kb
        .retrieve(&retrieval)
        .map_err(|e| PyRuntimeError::new_err(format!("retrieval failed: {e}")))?;
    let payload: Vec<serde_json::Value> = hits
        .iter()
        .map(|hit| {
            serde_json::json!({
                "score": hit.score,
                "why": hit.why(),
                "components": hit.components,
                "record": hit.record,
            })
        })
        .collect();
    to_json_py(&payload)
}

/// Retain a conclusion about a run, as an append-only amendment.
///
/// The original record is never rewritten: a note added today cannot
/// corrupt what was recorded when the run happened. Returns the
/// amendment's id.
#[pyfunction]
#[pyo3(signature = (run_id, notes, hypothesis=None, tags=None, root=".soma".to_string()))]
fn kb_record_conclusion(
    run_id: String,
    notes: String,
    hypothesis: Option<String>,
    tags: Option<Vec<String>>,
    root: String,
) -> PyResult<String> {
    use somatize_memory::{ExperimentRecord, KnowledgeBase};

    let mut kb = open_kb(&root)?;
    let Some(target) = kb.get(&run_id).ok().flatten() else {
        return Err(PyRuntimeError::new_err(format!(
            "no experiment '{run_id}' in {root}/experiments.jsonl"
        )));
    };
    let id = somatize_core::util::timestamp_id("amend");
    let mut amendment = ExperimentRecord::amendment(&id, &run_id, notes);
    amendment.research_line = target.research_line.clone();
    if let Some(hypothesis) = hypothesis {
        amendment = amendment.with_hypothesis(hypothesis);
    }
    if let Some(tags) = tags {
        amendment = amendment.with_tags(tags);
    }
    kb.record(amendment)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to record the conclusion: {e}")))?;
    Ok(id)
}

/// One experiment with its ancestors and descendants, as JSON.
#[pyfunction]
#[pyo3(signature = (run_id, root=".soma".to_string()))]
fn kb_lineage_json(run_id: String, root: String) -> PyResult<Option<String>> {
    use somatize_memory::KnowledgeBase;

    let kb = open_kb(&root)?;
    match kb
        .lineage(&run_id)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    {
        Some(lineage) => to_json_py(&lineage).map(Some),
        None => Ok(None),
    }
}

/// The move between any two experiments, related or not.
///
/// `derivation` only exists on a parent→child edge; this computes the
/// same diff for two records that never met — which is exactly the
/// comparison you want between sibling branches.
#[pyfunction]
#[pyo3(signature = (a, b, root=".soma".to_string()))]
fn kb_diff_json(a: String, b: String, root: String) -> PyResult<String> {
    use somatize_memory::{KnowledgeBase, derive};

    let kb = open_kb(&root)?;
    let fetch = |id: &str| {
        kb.get(id)
            .ok()
            .flatten()
            .ok_or_else(|| PyRuntimeError::new_err(format!("no experiment '{id}'")))
    };
    to_json_py(&derive(&fetch(&a)?, &fetch(&b)?))
}

/// Rebuild `<root>/experiments.jsonl` from `<root>/runs/*`.
///
/// One operation covering three needs: migrating runs recorded before
/// the pool existed, backfilling runs whose journal line was lost, and
/// disaster recovery when the journal itself is gone. The run
/// directories are the source of truth; the journal is an index.
///
/// Writes to a temp file and renames, so an interrupted reindex leaves
/// the previous journal intact. Returns the number of records written.
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
fn kb_reindex(root: String) -> PyResult<usize> {
    use somatize_memory::{ExperimentRecord, RecordKind};

    let root = std::path::PathBuf::from(&root);
    let journal = root.join("experiments.jsonl");

    // Amendments have no run directory to be rebuilt from, so they are
    // carried across verbatim rather than dropped.
    let amendments: Vec<ExperimentRecord> = std::fs::read_to_string(&journal)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<ExperimentRecord>(line).ok())
        .filter(|r| r.kind == RecordKind::Amendment)
        .collect();

    let infos = somatize_runtime::tracking::list_runs(&root)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Oldest first, so a parent is always in place before its children.
    let mut records: Vec<ExperimentRecord> = Vec::with_capacity(infos.len());
    for info in infos.into_iter().rev() {
        let Ok(reader) = RunReader::open(&info.dir) else {
            continue;
        };
        let Ok(summary) = summarize(&reader) else {
            continue;
        };
        let mut record = ExperimentRecord::from_run(&summary);
        if let Some(parent_id) = summary.parent_run_id.clone()
            && let Some(parent) = records.iter().find(|r| r.id == parent_id)
        {
            record = record.descended_from(parent);
        }
        records.push(record);
    }

    let mut body = String::new();
    for record in records.iter().chain(&amendments) {
        let line =
            serde_json::to_string(record).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        body.push_str(&line);
        body.push('\n');
    }
    let tmp = journal.with_extension("jsonl.tmp");
    std::fs::write(&tmp, body).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    std::fs::rename(&tmp, &journal).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(records.len() + amendments.len())
}

/// All runs under `<root>/runs/`, newest first (JSON array of RunInfo).
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
fn list_runs_json(root: String) -> PyResult<String> {
    let infos = somatize_runtime::tracking::list_runs(&root)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&infos)
}

/// Listing entry (manifest identity + derived state) for one run dir.
#[pyfunction]
fn run_info_json(dir: String) -> PyResult<String> {
    let info = open_run_reader(&dir)?
        .info()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&info)
}

/// The run's manifest.json contents.
#[pyfunction]
fn run_manifest_json(dir: String) -> PyResult<String> {
    let manifest = open_run_reader(&dir)?
        .manifest()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&manifest)
}

/// All parseable event envelopes from events.jsonl, in log order.
#[pyfunction]
fn run_events_json(dir: String) -> PyResult<String> {
    let events = open_run_reader(&dir)?
        .events()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&events)
}

/// Metric time series (optionally filtered by name).
#[pyfunction]
#[pyo3(signature = (dir, name=None))]
fn run_metric_series_json(dir: String, name: Option<String>) -> PyResult<String> {
    let points = open_run_reader(&dir)?
        .metric_series(name.as_deref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&points)
}

/// Per-node execution spans (gantt/overlay substrate).
#[pyfunction]
fn run_node_timings_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .node_timings()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Cache hit/miss counts, total and per node.
#[pyfunction]
fn run_cache_activity_json(dir: String) -> PyResult<String> {
    let activity = open_run_reader(&dir)?
        .cache_activity()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&activity)
}

/// HealthFlag events with wall time.
#[pyfunction]
fn run_health_flags_json(dir: String) -> PyResult<String> {
    let flags = open_run_reader(&dir)?
        .health_flags()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&flags)
}

/// Trial lifetimes from study.json (empty for non-study runs).
#[pyfunction]
fn run_trial_timeline_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .trial_timeline()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Rendering overlay aggregated from this run's events (JSON
/// GraphOverlay: per-node status, duration, cache tier, flags).
#[pyfunction]
fn run_overlay_json(dir: String) -> PyResult<String> {
    let overlay = open_run_reader(&dir)?
        .overlay()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&overlay)
}

/// Mermaid diagram of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
fn run_to_mermaid(dir: String, overlay: bool) -> PyResult<String> {
    let reader = open_run_reader(&dir)?;
    if overlay {
        reader
            .to_mermaid()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        let graph = reader
            .graph()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err(format!("run dir {dir} has no graph.json")))?;
        Ok(graph.to_mermaid())
    }
}

/// Render a serialized soma-core Graph (the `graph.json` schema) to
/// mermaid, with an optional overlay dict (the `RunView.overlay()`
/// shape). Backs inner-architecture rendering (`diagnostics/modules/`).
#[pyfunction]
#[pyo3(signature = (graph_json, overlay=None))]
fn graph_json_to_mermaid(
    py: Python<'_>,
    graph_json: &str,
    overlay: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let graph: somatize_core::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid graph JSON: {e}")))?;
    match overlay {
        None => Ok(graph.to_mermaid()),
        Some(ov) => Ok(graph.to_mermaid_with(&py_overlay(py, ov)?)),
    }
}

/// Render a serialized soma-core Graph to a self-contained SVG diagram
/// (no JavaScript), with an optional overlay dict.
#[pyfunction]
#[pyo3(signature = (graph_json, overlay=None))]
fn graph_json_to_svg(
    py: Python<'_>,
    graph_json: &str,
    overlay: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let graph: somatize_core::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid graph JSON: {e}")))?;
    match overlay {
        None => Ok(graph.to_svg()),
        Some(ov) => Ok(graph.to_svg_with(&py_overlay(py, ov)?)),
    }
}

/// Self-contained SVG of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
fn run_to_svg(dir: String, overlay: bool) -> PyResult<String> {
    let reader = open_run_reader(&dir)?;
    if overlay {
        reader
            .to_svg()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        let graph = reader
            .graph()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err(format!("run dir {dir} has no graph.json")))?;
        Ok(graph.to_svg())
    }
}

/// Graphviz DOT of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
fn run_to_graphviz(dir: String, overlay: bool) -> PyResult<String> {
    let reader = open_run_reader(&dir)?;
    if overlay {
        reader
            .to_graphviz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        let graph = reader
            .graph()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err(format!("run dir {dir} has no graph.json")))?;
        Ok(graph.to_graphviz())
    }
}

#[pymodule]
fn _soma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyAgent>()?;
    m.add_class::<PyJudge>()?;
    m.add_class::<PyTool>()?;
    m.add_class::<PyStepCtx>()?;
    m.add_function(wrap_pyfunction!(agentic::tool, m)?)?;
    m.add_function(wrap_pyfunction!(agentic::providers, m)?)?;
    m.add_function(wrap_pyfunction!(agentic::models, m)?)?;
    m.add_class::<PyStudy>()?;
    m.add_class::<PyTrial>()?;
    m.add_class::<PyRun>()?;
    m.add_class::<PyWorker>()?;
    m.add_function(wrap_pyfunction!(cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(cache_gc, m)?)?;
    m.add_function(wrap_pyfunction!(cache_pin, m)?)?;
    m.add_function(wrap_pyfunction!(cache_verify, m)?)?;
    m.add_function(wrap_pyfunction!(cache_purge_v1, m)?)?;
    m.add_function(wrap_pyfunction!(list_runs_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_summary_json, m)?)?;
    m.add_function(wrap_pyfunction!(checkout_run, m)?)?;
    m.add_function(wrap_pyfunction!(read_head_run, m)?)?;
    m.add_function(wrap_pyfunction!(clear_head_run, m)?)?;
    m.add_function(wrap_pyfunction!(kb_reindex, m)?)?;
    m.add_function(wrap_pyfunction!(kb_find_similar_json, m)?)?;
    m.add_function(wrap_pyfunction!(kb_record_conclusion, m)?)?;
    m.add_function(wrap_pyfunction!(kb_lineage_json, m)?)?;
    m.add_function(wrap_pyfunction!(kb_diff_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_info_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_manifest_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_events_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_metric_series_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_node_timings_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_cache_activity_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_health_flags_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_trial_timeline_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_overlay_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_to_mermaid, m)?)?;
    m.add_function(wrap_pyfunction!(run_to_graphviz, m)?)?;
    m.add_function(wrap_pyfunction!(graph_json_to_mermaid, m)?)?;
    m.add_function(wrap_pyfunction!(graph_json_to_svg, m)?)?;
    m.add_function(wrap_pyfunction!(run_to_svg, m)?)?;
    m.add("SomaSuspended", m.py().get_type::<SomaSuspended>())?;
    m.add("SomaPruned", m.py().get_type::<SomaPruned>())?;
    m.add(
        "SomaSchemaMismatch",
        m.py().get_type::<SomaSchemaMismatch>(),
    )?;
    m.add("SomaNodeNotFound", m.py().get_type::<SomaNodeNotFound>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
