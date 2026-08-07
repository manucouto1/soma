//! A Python `Filter` seen as a Rust [`Filter`].

use crate::prelude::*;

// ── Python Filter wrapper ──

pub(crate) struct PyFilterBridge {
    pub(crate) py_obj: PyObject,
    pub(crate) name: String,
    config_hash_val: CacheKey,
    /// cloudpickle.dumps() bytes — serializes the full object (bytecode + closures + deps).
    pub(crate) pickled_bytes: Vec<u8>,
    /// Full module source code (imports + classes + helpers) for introspection by Nous agents.
    pub(crate) source: String,
    /// Pip requirements detected from the filter's imports.
    pub(crate) requirements: Vec<String>,
    /// Whether this filter is trainable (has meaningful fit()).
    pub(crate) trainable: bool,
    /// Declared via `_input_schema` / `_output_schema`; parsed once at
    /// registration so a typo fails at build time, and `meta()` — which
    /// cannot fail — just clones.
    input_schema: Option<somatize_core::data::schema::Schema>,
    output_schema: Option<somatize_core::data::schema::Schema>,
}

impl PyFilterBridge {
    pub(crate) fn new(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
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
        // register them for by-value serialization (transitive), and collect
        // the third-party distributions the worker will have to install.
        //
        // The globals dict is named rather than built inline because the
        // script's results are read back out of it below.
        let helper_globals = [("_soma_module", &module)].into_py_dict(py)?;
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

# ── pip requirements ─────────────────────────────────────────
# What the worker must install to unpickle and run this filter:
# third-party distributions it imports. Deliberately the complement
# of the check above rather than a second heuristic — this used to be
# a separate script with its own `'site-packages' in f` substring
# test, which disagreed with this one on conda and lib64 layouts.
# Soma itself is excluded: a worker that is running this code has it.
_SELF = {'soma', '_soma', 'somatize'}

def _distribution_of(mod):
    name = getattr(mod, '__name__', '') or ''
    top = name.split('.')[0]
    if not top or top in _STDLIB or top in _BUILTINS or top in _SELF:
        return None
    f = getattr(mod, '__file__', None)
    if not f or f.startswith('<'):
        return None
    rf = os.path.realpath(f)
    if _SITE_PREFIXES and rf.startswith(_SITE_PREFIXES):
        return top
    if 'site-packages' in rf or 'dist-packages' in rf:
        return top
    return None

_reqs = set()
if _soma_module is not None:
    for _v in vars(_soma_module).values():
        if isinstance(_v, types.ModuleType):
            _d = _distribution_of(_v)
        elif isinstance(_v, type):
            _d = _distribution_of(sys.modules.get(_v.__module__))
        else:
            continue
        if _d:
            _reqs.add(_d)
_reqs = sorted(_reqs)
",
            Some(&helper_globals),
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

        // Read back what the helper collected.
        //
        // This used to run a second script and then read `_reqs` with
        // `py.eval(c"_reqs", None, None)`. Passing `None` for globals means
        // `__main__`, where the script never bound anything — so it raised
        // `NameError` on every call, `unwrap_or_default()` swallowed it, and
        // `requirements` was empty for every filter ever built. The
        // environment silently left the cache key below, and every remote
        // plan told the worker it needed nothing installed.
        let requirements: Vec<String> = match helper_globals.get_item("_reqs")? {
            Some(value) => value.extract()?,
            None => {
                return Err(PyRuntimeError::new_err(
                    "internal: the filter-identity helper did not produce `_reqs`",
                ));
            }
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
            input_schema: crate::agentic::parse_schema_attr(py, obj, "_input_schema")?,
            output_schema: crate::agentic::parse_schema_attr(py, obj, "_output_schema")?,
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
                        "evolving" => StreamMode::Evolving,
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
            distribution: somatize_core::graph::filter::Distribution::Local,
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
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
