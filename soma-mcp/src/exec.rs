//! Running a graph the model described.
//!
//! `run_pipeline` and `run_study` used to echo their arguments back. The
//! reason given was that the server cannot load user code — true of the
//! *server*, which is a Rust binary, and irrelevant: Soma already runs
//! Python filters in a subprocess everywhere else it needs to (that is
//! what `soma-worker` is). This does the same thing, with the project
//! directory on `sys.path`, so a model can build a graph out of the
//! filters it just read with `read_filter_source` and actually run it.
//!
//! The subprocess writes its answer to a file rather than stdout,
//! because a `print` inside a user's filter is the most natural thing in
//! the world to write and must not corrupt the reply. Stdout and stderr
//! come back as diagnostics either way.
//!
//! **This executes code from the project directory.** So does
//! `write_filter_source`, which the same server has always offered — the
//! model could already write a filter and ask the user to run it. What
//! changes is the loop's length, not who is trusted.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The Python driver, embedded. Reads a spec on stdin, writes a JSON
/// result to `spec["result_path"]`.
const DRIVER: &str = r#"
import base64, importlib, importlib.util, json, os, sys, traceback

spec = json.loads(sys.stdin.read())
sys.path.insert(0, os.getcwd())
result_path = spec["result_path"]

def _fail(message, detail=None):
    with open(result_path, "w") as fh:
        json.dump({"ok": False, "error": message, "detail": detail}, fh)
    sys.exit(0)

def _load_from_file(path, cls_name):
    mod_name = "_soma_mcp_" + os.path.basename(path).replace(".", "_")
    spec_ = importlib.util.spec_from_file_location(mod_name, path)
    if spec_ is None or spec_.loader is None:
        raise ImportError("cannot load %s" % path)
    module = importlib.util.module_from_spec(spec_)
    sys.modules[mod_name] = module
    spec_.loader.exec_module(module)
    return getattr(module, cls_name)

def _resolve(ref, candidates):
    """`module.Class`, `path/to/file.py:Class`, or a bare class name.

    A bare name is only looked for in the files the server already found
    with `list_filters` — importing every .py under the project to go
    looking would run whatever else is down there.
    """
    if ":" in ref:
        path, _, cls_name = ref.partition(":")
        return _load_from_file(path, cls_name)
    if "." in ref:
        mod_name, _, cls_name = ref.rpartition(".")
        try:
            return getattr(importlib.import_module(mod_name), cls_name)
        except (ImportError, AttributeError):
            pass  # may still be a bare dotted name; fall through to the scan
    tried = []
    for path in candidates:
        try:
            obj = _load_from_file(path, ref.rpartition(".")[2] or ref)
        except Exception as e:
            tried.append("%s: %s" % (path, e))
            continue
        return obj
    raise ImportError(
        "cannot resolve filter %r. Give it as `module.Class` or "
        "`path/to/file.py:Class`. Tried: %s" % (ref, "; ".join(tried) or "nothing")
    )

def _config(value):
    """Config values, with `{"__search__": {...}}` becoming a search dimension.

    That is what makes a study expressible without a second vocabulary:
    the same node spec describes a fixed value and a searched one, and
    `graph.search_space()` picks the dimension up on its own.
    """
    import soma
    if isinstance(value, dict):
        if "__search__" in value:
            return soma.search(**value["__search__"])
        return {k: _config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_config(v) for v in value]
    return value

def _searchable(cls, config):
    """Move `{"__search__": …}` values onto a subclass, where they count.

    `search()` is a CLASS-level descriptor — `FilterMeta` collects the
    ones it finds in a class body into `_soma_search_space`, and that is
    what `graph.search_space()` reads. Passing one to a constructor sets
    an instance attribute holding a descriptor, which is not a dimension
    and cannot be hashed into a cache key either.

    So the searched values become a subclass's body, and the constructor
    is called WITHOUT them: its own default supplies a concrete value for
    the identity, and each trial supplies the sampled one.
    """
    from soma.search import SearchDescriptor
    searched = {k: v for k, v in config.items() if isinstance(v, SearchDescriptor)}
    if not searched:
        return cls, config
    subclass = type(cls)(cls.__name__, (cls,), dict(searched))
    return subclass, {k: v for k, v in config.items() if k not in searched}

def _build(spec, overrides=None):
    import soma
    overrides = overrides or {}
    g = soma.Graph(cache=spec.get("cache", "memory"))
    candidates = spec.get("filter_files", [])
    for node in spec["nodes"]:
        cls = _resolve(node["filter"], candidates)
        config = {k: _config(v) for k, v in (node.get("config") or {}).items()}
        cls, config = _searchable(cls, config)
        for key, value in overrides.get(node["id"], {}).items():
            config[key] = value
        try:
            instance = cls(**config) if config else cls()
        except TypeError as e:
            raise TypeError(
                "constructing node %r as %s(%s): %s"
                % (node["id"], getattr(cls, "__name__", node["filter"]),
                   ", ".join("%s=%r" % kv for kv in config.items()), e))
        if node.get("target"):
            g.node(node["id"], instance, target=node["target"])
        else:
            g.node(node["id"], instance)
    for edge in spec.get("edges", []):
        g.connect(edge[0], edge[1])
    return g

def _run_once(spec, g):
    """Fit when asked (or when targets were given), then forward.

    A graph with a trainable node refuses to forward before it is fitted,
    so `fit` defaults to true rather than to "only when y is present".
    """
    x, y = spec.get("input"), spec.get("y")
    if spec.get("fit", True):
        g.fit(x, y)
    return g.forward(x)

def _jsonable(value, limit=2000):
    """Outputs can be large; the model needs the shape more than the tail."""
    if isinstance(value, list):
        if len(value) > limit:
            return {"truncated": True, "length": len(value),
                    "head": _jsonable(value[:limit], limit)}
        return [_jsonable(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v, limit) for k, v in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    for attr in ("tolist", "item"):
        if hasattr(value, attr):
            try:
                return _jsonable(getattr(value, attr)(), limit)
            except Exception:
                pass
    return repr(value)

try:
    import soma
except ImportError as e:
    _fail("soma is not importable by %s: %s" % (sys.executable, e),
          "Install it in the interpreter the server uses, or set SOMA_PYTHON "
          "to one that has it.")

try:
    if spec["kind"] == "pipeline":
        g = _build(spec)
        payload = {"plan": str(g.compile())}
        if spec.get("track", True):
            with g.track_run(spec.get("name", "mcp-run"),
                             tags=list(spec.get("tags", [])),
                             params=spec.get("params") or None) as run:
                out = _run_once(spec, g)
                # Absolute: the contract is that a model with file tools
                # can go and read the directory, and it does not share
                # this subprocess's working directory.
                payload["run_dir"] = os.path.abspath(run.dir)
        else:
            out = _run_once(spec, g)
        payload["ok"] = True
        payload["output"] = _jsonable(out)
        payload["state"] = _jsonable(g.state())
        payload["mermaid"] = g.to_mermaid()
    else:
        g = _build(spec)
        space = g.search_space()
        if not space:
            _fail("this graph has no search space: no node config used "
                  '{"__search__": {...}}, so every trial would be identical',
                  "Mark the dimensions to search in the node configs.")
        metric = spec.get("metric", "score")
        study = soma.Study(
            spec.get("name", "mcp-study"),
            search_space=space,
            strategy=spec.get("strategy", "random"),
            n_trials=int(spec.get("n_trials", 10)),
            objectives=[(metric, spec.get("direction", "minimize"))],
            seed=spec.get("seed"),
        )

        def executor(trial):
            # A trial names its dimensions `node.field` — the same names
            # `search_space()` produced — so the graph is rebuilt with the
            # sampled values in place rather than mutated afterwards.
            overrides = {}
            for key in trial.keys():
                node_id, _, field = key.rpartition(".")
                overrides.setdefault(node_id, {})[field] = trial[key]
            out = _run_once(spec, _build(spec, overrides))
            value = out
            if isinstance(value, dict):
                if metric not in value:
                    raise KeyError(
                        "the graph produced %s, which has no %r to optimize. "
                        "Name the metric your last node emits, or end the graph "
                        "with one that emits it (soma.library.Eval does)."
                        % (sorted(value), metric))
                value = value[metric]
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(
                        "the graph produced %d values; an objective needs one. "
                        "End it with a node that reduces to a single number."
                        % len(value))
                value = value[0]
            return {metric: float(value)}

        study.run(executor)
        payload = {
            "ok": True,
            "n_trials": study.n_trials,
            "best_trial": _jsonable(study.best_trial),
            "trials": _jsonable(study.trials),
            "run_dir": os.path.abspath(study.run_dir) if study.run_dir else None,
            "objectives": [list(o) for o in study.objectives],
        }
    with open(result_path, "w") as fh:
        json.dump(payload, fh)
except BaseException as e:
    _fail("%s: %s" % (type(e).__name__, e), traceback.format_exc())
"#;

/// Runs a graph spec in a Python subprocess rooted at the project.
pub struct GraphRunner {
    project_dir: PathBuf,
    python: String,
}

impl GraphRunner {
    /// A runner for `project_dir`, using `$SOMA_PYTHON` when set.
    ///
    /// The interpreter matters: it must be one that can `import soma` and
    /// import the project's own modules. Defaulting to `python3` is right
    /// far more often than guessing, and when it is wrong the error says
    /// which interpreter could not find soma.
    pub fn new(project_dir: impl Into<PathBuf>) -> Self {
        Self {
            project_dir: project_dir.into(),
            python: std::env::var("SOMA_PYTHON").unwrap_or_else(|_| "python3".into()),
        }
    }

    /// Run `spec`, returning the driver's JSON payload.
    ///
    /// `Err` is for the subprocess itself going wrong — no interpreter, a
    /// crash, no answer written. A graph that failed to build or to run
    /// comes back as `Ok` with `"ok": false`, because that is a result the
    /// model can act on rather than an infrastructure problem.
    pub fn run(&self, spec: &serde_json::Value) -> Result<serde_json::Value, String> {
        let result_file = std::env::temp_dir().join(format!(
            "soma-mcp-{}-{}.json",
            std::process::id(),
            somatize_core::util::timestamp_id("r")
        ));
        let mut spec = spec.clone();
        spec["result_path"] = serde_json::json!(result_file.to_string_lossy());
        spec["filter_files"] = serde_json::json!(self.filter_files());

        let mut child = Command::new(&self.python)
            .arg("-c")
            .arg(DRIVER)
            .current_dir(&self.project_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "cannot start `{}`: {e}. Set SOMA_PYTHON to an interpreter \
                     that can import soma",
                    self.python
                )
            })?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| "the driver has no stdin".to_string())?;
            stdin
                .write_all(spec.to_string().as_bytes())
                .map_err(|e| format!("sending the spec to the driver: {e}"))?;
        }

        let out = child
            .wait_with_output()
            .map_err(|e| format!("waiting for the driver: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();

        let raw = std::fs::read_to_string(&result_file).map_err(|_| {
            format!(
                "the driver wrote no result (exit {}).\nstderr:\n{}\nstdout:\n{}",
                out.status,
                if stderr.is_empty() {
                    "(empty)"
                } else {
                    &stderr
                },
                if stdout.is_empty() {
                    "(empty)"
                } else {
                    &stdout
                },
            )
        })?;
        let _ = std::fs::remove_file(&result_file);

        let mut payload: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("the driver's result is not JSON ({e}): {raw}"))?;
        // A filter's own prints are worth keeping — they are often the
        // only trace of what a run was doing.
        if !stdout.is_empty() {
            payload["stdout"] = serde_json::json!(stdout);
        }
        if !stderr.is_empty() {
            payload["stderr"] = serde_json::json!(stderr);
        }
        Ok(payload)
    }

    /// The project's filter files, so a node may name a bare class.
    fn filter_files(&self) -> Vec<String> {
        crate::context::find_filter_files(&self.project_dir)
            .unwrap_or_default()
            .iter()
            .map(|p| relative_to(p, &self.project_dir))
            .collect()
    }
}

/// Paths the driver can open from its own working directory.
fn relative_to(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}
