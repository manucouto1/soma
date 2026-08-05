//! Python subprocess — persistent daemon for filter execution.
//!
//! Spawns a Python child process that loads filters via cloudpickle
//! and executes fit/forward commands via stdin/stdout JSON Lines.
//! The GIL is completely isolated from the Rust process — no segfaults.

use crate::error::{Result, WorkerError};
use base64::engine::{Engine, general_purpose::STANDARD};
use somatize_core::cache::CacheKey;
use somatize_core::error::SomaError;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

/// The Python daemon script, embedded as a Rust string.
const DAEMON_SCRIPT: &str = r#"
import json, sys, base64, cloudpickle, io, pickle

# stdout is the protocol. A `print` inside a user's filter — the most
# natural thing in the world to write while debugging one — landed in the
# middle of the JSON dialogue and the worker died parsing its own reply.
# The protocol keeps the real handle; everything else is sent to stderr,
# where the worker already forwards it.
_protocol = sys.stdout
sys.stdout = sys.stderr

def _reply(payload):
    print(json.dumps(payload), file=_protocol, flush=True)

filters = {}

def _unwrap(out):
    """A filter's forward may return `(out, aux)`; the chain wants both."""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[0], out[1]
    return out, {}

def _backward_pass(f, data, y):
    """Forward, loss, backward — leaving the gradients on the parameters.

    This is what a *remote* fit of a DifferentiableFilter has to do and did
    not: its `fit` learns no state (the parameters live in `_module`), so a
    worker ran `fit`, got `{}` back, and reported a trained model whose
    parameters had never seen a gradient. `data_parallel` then averaged
    nothing across replicas.

    Deliberately does NOT step. Whoever owns the round owns the step: in a
    data-parallel round the gradients are averaged across replicas first,
    and stepping here would apply each replica's own gradient before the
    average.

    Returns the filter's state, or None when it is not a differentiable
    filter and this does not apply.
    """
    module = getattr(f, "_module", None)
    if module is None and not getattr(f, "_differentiable", False):
        return None
    import torch
    was_training = getattr(f, "training", False)
    f.training = True          # so forward returns a live tensor, not a list
    try:
        out, aux = _unwrap(f.forward(data, {}))
        if not hasattr(out, "backward"):
            return None
        y_t = y if hasattr(y, "shape") else torch.tensor(y, dtype=torch.float32)
        if y_t.shape != out.shape and y_t.numel() == out.numel():
            y_t = y_t.reshape(out.shape)
        if y_t.shape != out.shape:
            raise ValueError(
                "the targets have shape %s and the output %s; they cannot be "
                "paired" % (tuple(y_t.shape), tuple(out.shape)))
        loss = f.compute_loss(out, y_t, aux) if hasattr(f, "compute_loss") \
            else torch.nn.functional.mse_loss(out, y_t)
        loss.backward()
    finally:
        f.training = was_training
    module = getattr(f, "_module", None)
    if module is None:
        return None
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return {"weights_b64": base64.b64encode(buf.getvalue()).decode()}

def _encode(obj):
    """Encode a Python object to JSON-safe format."""
    if obj is None:
        return None
    if isinstance(obj, (list, int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    # Fall back to pickle + base64
    return {"__pickle_b64__": base64.b64encode(pickle.dumps(obj)).decode()}

def _decode(obj):
    """Decode from JSON-safe format back to Python object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "__pickle_b64__" in obj:
            return pickle.loads(base64.b64decode(obj["__pickle_b64__"]))
        if "type" in obj and "data" in obj:
            # Soma Value format
            t, d = obj["type"], obj["data"]
            if t == "Tensor":
                # The shape travels with the values and used to be dropped
                # here, so a (8, 2) tensor reached the filter as 16 loose
                # floats. Anything that reads `x.shape[1:]` — every
                # DifferentiableFilter, when it sizes its module — then saw
                # an empty tuple and died with "tuple index out of range",
                # seven layers from the cause.
                vals, shape = d.get("values", []), d.get("shape") or []
                if len(shape) <= 1:
                    return vals
                def _nest(flat, dims):
                    if len(dims) == 1:
                        return list(flat)
                    step = 1
                    for k in dims[1:]:
                        step *= k
                    return [_nest(flat[i * step:(i + 1) * step], dims[1:])
                            for i in range(dims[0])]
                return _nest(vals, list(shape))
            if t == "Text":
                return d
            if t == "Json":
                return d
            if t == "Empty":
                return {}
            if t == "Bytes":
                return bytes(d)
            if t == "Object":
                return pickle.loads(bytes(d))
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
    except json.JSONDecodeError as e:
        _reply(({"ok": False, "error": f"invalid JSON: {e}"}))
        continue

    try:
        action = cmd.get("cmd", "")

        if action == "LOAD":
            for f in cmd["filters"]:
                obj = cloudpickle.loads(base64.b64decode(f["pickle_b64"]))
                filters[f["id"]] = {"obj": obj, "trainable": f.get("trainable", True)}
            _reply(({"ok": True}))

        elif action == "FIT":
            f = filters[cmd["node_id"]]["obj"]
            data = _decode(cmd.get("data"))
            y = _decode(cmd.get("y"))
            result = f.fit(data, y)
            if y is not None:
                trained = _backward_pass(f, data, y)
                if trained is not None:
                    result = trained
            _reply(({"ok": True, "result": _encode(result)}))

        elif action == "FORWARD":
            f = filters[cmd["node_id"]]["obj"]
            data = _decode(cmd.get("data"))
            state = _decode(cmd.get("state", {}))
            result = f.forward(data, state)
            _reply(({"ok": True, "result": _encode(result)}))

        elif action == "COMPOSITE_FORWARD":
            node_ids = cmd["node_ids"]
            data = _decode(cmd.get("data"))
            try:
                import torch
                if isinstance(data, list):
                    x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
                else:
                    x = data
            except ImportError:
                x = data

            out = x
            for nid in node_ids:
                f = filters[nid]["obj"]
                state = _decode(cmd.get("states", {}).get(nid, {}))
                out, _ = _unwrap(f.forward(out, state))

            result = out.detach().tolist() if hasattr(out, 'detach') else out
            _reply(({"ok": True, "result": _encode(result)}))

        elif action == "COMPOSITE_FIT":
            node_ids = cmd["node_ids"]
            data = _decode(cmd.get("data"))
            y = _decode(cmd.get("y"))

            # Step 1: fit each trainable filter to get states
            fit_states = {}
            fit_input = data
            for nid in node_ids:
                f = filters[nid]["obj"]
                if filters[nid].get("trainable", True):
                    state = f.fit(fit_input, y)
                    fit_states[nid] = state
                else:
                    fit_states[nid] = {}
                # Forward to propagate output to next filter. `forward`
                # may answer `(out, aux)` — chaining the tuple fed the next
                # filter a 2-tuple where it wanted a tensor.
                fit_input, _ = _unwrap(f.forward(fit_input, fit_states[nid]))

            # Step 2: forward with autograd if torch available
            try:
                import torch
                if isinstance(data, list):
                    x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
                else:
                    x = data
            except ImportError:
                x = data

            out = x
            aux = {}
            for nid in node_ids:
                f = filters[nid]["obj"]
                # Training mode, so a DifferentiableFilter hands back a live
                # tensor rather than a detached list — without this the
                # `hasattr(out, 'backward')` below was never true and the
                # whole backward block was dead code.
                _was = getattr(f, "training", False)
                f.training = True
                try:
                    out, aux = _unwrap(f.forward(out, fit_states.get(nid, {})))
                finally:
                    f.training = _was

            # Backward. This used to be wrapped in `except Exception: pass`,
            # so a loss that could not be computed — wrong shape, wrong dtype,
            # a `compute_loss` that raised — produced a fit that reported
            # success and had trained nothing. A backward that cannot run is
            # an error; the parameters are the whole point of being here.
            #
            # The gradients are deliberately LEFT on the parameters. A
            # data-parallel round reads them with GET_GRADIENTS after this
            # returns, averages them across replicas, and steps in
            # APPLY_GRADIENTS. Stepping here as well would apply each
            # replica's own gradient before the average, which is not
            # data-parallel SGD and not anything else either.
            if y is not None and hasattr(out, 'backward'):
                last = filters[node_ids[-1]]["obj"]
                try:
                    import torch
                except ImportError:
                    _reply(({"ok": False, "error":
                        "composite fit of %s produced a differentiable output "
                        "but torch is not importable" % node_ids}))
                    continue
                if isinstance(y, list):
                    y_t = torch.tensor(y, dtype=torch.float32)
                else:
                    y_t = y
                try:
                    # `compute_loss` is the DifferentiableFilter contract;
                    # `loss_fn` is the older duck-typed attribute, still read
                    # so a filter that predates the base class keeps working.
                    if hasattr(last, 'compute_loss'):
                        loss = last.compute_loss(out, y_t, aux)
                    elif hasattr(last, 'loss_fn'):
                        loss = last.loss_fn(out, y_t)
                    else:
                        loss = torch.nn.functional.mse_loss(out, y_t)
                    loss.backward()
                except Exception as e:
                    _reply(({"ok": False, "error":
                        "composite fit of %s: the backward pass failed: %s: %s"
                        % (node_ids, type(e).__name__, e)}))
                    continue
                # A filter carrying its own optimizer keeps its own loop: it
                # is not part of a gradient-averaging round.
                for nid in node_ids:
                    f = filters[nid]["obj"]
                    if hasattr(f, 'optimizer'):
                        f.optimizer.step()
                        f.optimizer.zero_grad()

            states = {}
            for nid in node_ids:
                f = filters[nid]["obj"]
                if hasattr(f, 'state_dict'):
                    buf = io.BytesIO()
                    try:
                        import torch
                        torch.save(f.state_dict(), buf)
                    except ImportError:
                        buf.write(cloudpickle.dumps(f))
                    states[nid] = base64.b64encode(buf.getvalue()).decode()

            result = out.detach().tolist() if hasattr(out, 'detach') else out
            _reply(({"ok": True, "result": _encode(result), "states": states}))

        elif action == "GET_STATE":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
            _mod = getattr(f, "_module", None)
            if _mod is not None and hasattr(_mod, "state_dict"):
                # A DifferentiableFilter is not itself an nn.Module, so the
                # branch below would cloudpickle the whole filter object —
                # a state no local graph could load. Its state is the
                # `{"weights_b64": …}` dict its own `forward` reads back and
                # the local fit path writes, so send exactly that.
                import torch
                buf = io.BytesIO()
                torch.save(_mod.state_dict(), buf)
                _reply(({"ok": True, "state": {
                    "weights_b64": base64.b64encode(buf.getvalue()).decode()}}))
                continue
            buf = io.BytesIO()
            if hasattr(f, 'state_dict'):
                try:
                    import torch
                    torch.save(f.state_dict(), buf)
                except ImportError:
                    buf.write(cloudpickle.dumps(f))
            else:
                buf.write(cloudpickle.dumps(f))
            state_b64 = base64.b64encode(buf.getvalue()).decode()
            _reply(({"ok": True, "state_b64": state_b64}))

        elif action == "SET_STATE":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
            _state = cmd.get("state")
            if isinstance(_state, dict) and "weights_b64" in _state:
                # The symmetric read of the GET_STATE branch above.
                import torch
                _mod = getattr(f, "_module", None)
                if _mod is None:
                    _reply(({"ok": False, "error":
                        "`%s` was sent weights but has no module to load them "
                        "into: it was never materialized on this worker" % nid}))
                    continue
                _mod.load_state_dict(torch.load(
                    io.BytesIO(base64.b64decode(_state["weights_b64"])),
                    weights_only=True))
                _reply(({"ok": True}))
                continue
            state_bytes = base64.b64decode(cmd["state_b64"])
            buf = io.BytesIO(state_bytes)
            if hasattr(f, 'load_state_dict'):
                try:
                    import torch
                    f.load_state_dict(torch.load(buf, weights_only=True))
                except ImportError:
                    filters[nid]["obj"] = cloudpickle.loads(buf.read())
            else:
                filters[nid]["obj"] = cloudpickle.loads(buf.read())
            _reply(({"ok": True}))

        elif action == "GET_GRADIENTS":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
            # A DifferentiableFilter is not itself an nn.Module: it builds
            # one and keeps it in `_module`. Looking only at `f` found no
            # parameters and returned an EMPTY buffer, so AllReduce averaged
            # nothing and the round reported success — the gradients simply
            # never left the worker.
            module = getattr(f, "_module", None) or f
            if not hasattr(module, "named_parameters"):
                _reply(({"ok": False, "error":
                    "`%s` has no parameters: it is neither an nn.Module nor a "
                    "materialized DifferentiableFilter, so there are no "
                    "gradients to read" % nid}))
                continue
            try:
                import torch
            except ImportError:
                _reply(({"ok": False, "error":
                    "`%s` cannot produce gradients: torch is not installed in "
                    "this worker's environment" % nid}))
                continue
            # Nested lists, not a torch pickle. The aggregator that averages
            # these lives in Rust, and a pickle is opaque to it: AllReduce
            # over two `Value::Bytes` blobs could only refuse. Plain JSON is
            # also what makes the average independent of the torch version
            # each worker happens to have.
            grads = {n: p.grad.detach().cpu().tolist()
                     for n, p in module.named_parameters() if p.grad is not None}
            if not grads:
                _reply(({"ok": False, "error":
                    "`%s` has parameters but none carry a gradient. Run a "
                    "backward pass before asking for them" % nid}))
                continue
            _reply(({"ok": True, "gradients": grads}))

        elif action == "APPLY_GRADIENTS":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
            module = getattr(f, "_module", None) or f
            if not hasattr(module, "named_parameters"):
                _reply(({"ok": False, "error":
                    "`%s` has no parameters to apply gradients to" % nid}))
                continue
            try:
                import torch
            except ImportError:
                _reply(({"ok": False, "error":
                    "`%s` cannot take gradients: torch is not installed" % nid}))
                continue
            grads = cmd.get("gradients") or {}
            applied = 0
            mismatch = None
            for name, p in module.named_parameters():
                if name not in grads:
                    continue
                g = torch.tensor(grads[name], dtype=p.dtype, device=p.device)
                if tuple(g.shape) != tuple(p.shape):
                    mismatch = ("`%s`: aggregated gradient for `%s` has shape %s, "
                                "the parameter has %s"
                                % (nid, name, tuple(g.shape), tuple(p.shape)))
                    break
                p.grad = g
                applied += 1
            if mismatch is not None:
                _reply(({"ok": False, "error": mismatch}))
                continue
            if applied == 0 and grads:
                _reply(({"ok": False, "error":
                    "`%s`: none of the %d aggregated gradients matched a "
                    "parameter name. The replicas are not the same model"
                    % (nid, len(grads))}))
                continue
            # Applying an averaged gradient and stopping there would leave
            # every replica exactly where the round started: data-parallel
            # SGD *is* the step. The optimizer is built once and kept on the
            # filter, so its moments survive across rounds — an Adam rebuilt
            # every round is plain SGD wearing its name.
            stepped = False
            if applied and hasattr(f, "make_optimizer"):
                opt = getattr(f, "_soma_dp_optimizer", None)
                if opt is None:
                    opt = f.make_optimizer([module])
                    f._soma_dp_optimizer = opt
                opt.step()
                opt.zero_grad()
                stepped = True
            _reply(({"ok": True, "applied": applied, "stepped": stepped}))

        elif action == "BATCHED_FIT":
            # Process dataset in batches — model loaded ONCE, batches processed in loop
            node_ids = cmd["node_ids"]
            data = _decode(cmd.get("data"))
            y = _decode(cmd.get("y"))
            batch_size = cmd.get("batch_size", 32)

            # Find the list dimension to batch on
            if isinstance(data, dict):
                list_keys = [k for k, v in data.items() if isinstance(v, list)]
                total = len(data[list_keys[0]]) if list_keys else 0
            elif isinstance(data, list):
                total = len(data)
            else:
                total = 0

            all_states = {}
            n_batches = (total + batch_size - 1) // batch_size if total > 0 else 1

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, total)

                # Slice the batch
                if isinstance(data, dict):
                    batch = {}
                    for k, v in data.items():
                        if isinstance(v, list):
                            batch[k] = v[start:end]
                        else:
                            batch[k] = v
                elif isinstance(data, list):
                    batch = data[start:end]
                else:
                    batch = data

                y_batch = None
                if y is not None:
                    if isinstance(y, list):
                        y_batch = y[start:end]
                    elif isinstance(y, dict):
                        y_batch = {k: (v[start:end] if isinstance(v, list) else v) for k, v in y.items()}
                    else:
                        y_batch = y

                # Fit + forward for this batch through all filters
                batch_input = batch
                for nid in node_ids:
                    f = filters[nid]["obj"]
                    if filters[nid].get("trainable", True):
                        state = f.fit(batch_input, y_batch)
                        all_states[nid] = state
                    else:
                        if nid not in all_states:
                            all_states[nid] = {}
                    batch_input = f.forward(batch_input, all_states.get(nid, {}))

                import sys
                print(f"    Batch {b+1}/{n_batches} complete", file=sys.stderr)

            # Encode final states
            encoded_states = {}
            for nid, state in all_states.items():
                encoded_states[nid] = _encode(state)

            result = _encode(batch_input) if batch_input is not None else None
            _reply(({"ok": True, "result": result, "states": encoded_states}))

        elif action == "SHUTDOWN":
            _reply(({"ok": True}))
            break

        else:
            _reply(({"ok": False, "error": f"unknown command: {action}"}))

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        _reply(({"ok": False, "error": str(e), "traceback": tb}))
"#;

/// A persistent Python child process that executes filter commands.
pub struct PythonProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    node_ids: Vec<String>,
}

impl PythonProcess {
    /// Spawn a Python daemon and load filters into it.
    pub fn spawn(
        python_path: &str,
        filters: &[(String, Vec<u8>, bool)], // (node_id, pickled_bytes, trainable)
    ) -> Result<Self> {
        let mut child = Command::new(python_path)
            .args(["-c", DAEMON_SCRIPT])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Python stderr → worker stderr (for logs/tracing)
            .spawn()
            .map_err(|e| WorkerError::Python(format!("failed to spawn python: {e}")))?;

        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| WorkerError::Python("no stdin".into()))?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| WorkerError::Python("no stdout".into()))?,
        );

        let node_ids: Vec<String> = filters.iter().map(|(id, _, _)| id.clone()).collect();

        let mut proc = Self {
            child,
            stdin,
            stdout,
            node_ids,
        };

        // Send LOAD command with all filters
        let filter_specs: Vec<serde_json::Value> = filters
            .iter()
            .map(|(id, pickled, trainable)| {
                serde_json::json!({
                    "id": id,
                    "pickle_b64": STANDARD.encode(pickled),
                    "trainable": trainable,
                })
            })
            .collect();

        let resp = proc.send(serde_json::json!({
            "cmd": "LOAD",
            "filters": filter_specs,
        }))?;

        if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let error = resp
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown error");
            return Err(WorkerError::Python(format!("LOAD failed: {error}")));
        }

        Ok(proc)
    }

    /// Send a JSON command and read the JSON response.
    fn send(&mut self, cmd: serde_json::Value) -> Result<serde_json::Value> {
        let action = cmd
            .get("cmd")
            .and_then(|c| c.as_str())
            .unwrap_or("?")
            .to_string();
        let node_id = cmd
            .get("node_id")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();

        tracing::debug!(action = %action, node_id = %node_id, "→ Python");
        let start = std::time::Instant::now();

        let line = serde_json::to_string(&cmd)
            .map_err(|e| WorkerError::Encoding(format!("serialize cmd: {e}")))?;

        writeln!(self.stdin, "{line}")
            .map_err(|e| WorkerError::Python(format!("write to python stdin: {e}")))?;
        self.stdin
            .flush()
            .map_err(|e| WorkerError::Python(format!("flush stdin: {e}")))?;

        let mut response = String::new();
        self.stdout
            .read_line(&mut response)
            .map_err(|e| WorkerError::Python(format!("read from python stdout: {e}")))?;

        let duration_ms = start.elapsed().as_millis();

        if response.is_empty() {
            tracing::error!(action = %action, "Python process closed stdout (crashed?)");
            return Err(WorkerError::Python(
                "python process closed stdout (crashed?)".into(),
            ));
        }

        let parsed: serde_json::Value = serde_json::from_str(&response).map_err(|e| {
            WorkerError::Python(format!("parse python response: {e}\nraw: {response}"))
        })?;

        let ok = parsed.get("ok") == Some(&serde_json::Value::Bool(true));
        if ok {
            tracing::debug!(action = %action, node_id = %node_id, duration_ms, "← Python OK");
        } else {
            let error = parsed.get("error").and_then(|e| e.as_str()).unwrap_or("?");
            let traceback = parsed
                .get("traceback")
                .and_then(|t| t.as_str())
                .unwrap_or("");
            tracing::error!(action = %action, node_id = %node_id, error, "Python filter error");
            if !traceback.is_empty() {
                tracing::error!("Python traceback:\n{traceback}");
            }
        }

        Ok(parsed)
    }

    /// Convert a response to a Value, handling errors.
    fn response_to_value(resp: &serde_json::Value) -> Result<Value> {
        if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let error = resp
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown error");
            let traceback = resp.get("traceback").and_then(|t| t.as_str()).unwrap_or("");
            return Err(WorkerError::Python(format!(
                "Python error: {error}\n{traceback}"
            )));
        }

        if let Some(result) = resp.get("result") {
            return Self::json_to_value(result);
        }

        Ok(Value::Empty)
    }

    /// Convert a JSON value to a Soma Value.
    fn json_to_value(v: &serde_json::Value) -> Result<Value> {
        if v.is_null() {
            return Ok(Value::Empty);
        }
        if let Some(arr) = v.as_array() {
            let values: Vec<f64> = arr.iter().filter_map(|x| x.as_f64()).collect();
            if values.len() == arr.len() && !values.is_empty() {
                return Ok(Value::tensor(values.clone(), vec![values.len()]));
            }
            // Could be nested array
            if let Some(first) = arr.first()
                && first.is_array()
            {
                let rows = arr.len();
                let cols = first.as_array().map(|a| a.len()).unwrap_or(0);
                let flat: Vec<f64> = arr
                    .iter()
                    .filter_map(|row| row.as_array())
                    .flat_map(|row| row.iter().filter_map(|x| x.as_f64()))
                    .collect();
                if flat.len() == rows * cols {
                    return Ok(Value::tensor(flat, vec![rows, cols]));
                }
            }
        }
        // A bare string from Python becomes text unless it parses as JSON —
        // byte-for-byte the rule `py_to_value` applies in-process. The two
        // paths must agree: the same filter run locally and on a worker has
        // to produce the same `Value`, or its cache key changes with where
        // it ran.
        if let Some(s) = v.as_str() {
            return Ok(match serde_json::from_str(s) {
                Ok(parsed) => Value::json(parsed),
                Err(_) => Value::text(s),
            });
        }
        Ok(Value::json(v.clone()))
    }

    /// Encode a Value to JSON for the Python process.
    fn value_to_json(v: &Value) -> serde_json::Value {
        serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
    }

    // ── Public API ──

    /// Fit the filter loaded under `node_id` on `data` (and optional
    /// labels `y`), returning what its `fit` returned — the trained state.
    pub fn fit(&mut self, node_id: &str, data: &Value, y: Option<&Value>) -> Result<Value> {
        let mut cmd = serde_json::json!({
            "cmd": "FIT",
            "node_id": node_id,
            "data": Self::value_to_json(data),
        });
        if let Some(y_val) = y {
            cmd["y"] = Self::value_to_json(y_val);
        }
        let resp = self.send(cmd)?;
        Self::response_to_value(&resp)
    }

    /// Run the filter's `forward` on `data` with a previously trained
    /// `state`, returning its output.
    pub fn forward(&mut self, node_id: &str, data: &Value, state: &Value) -> Result<Value> {
        let resp = self.send(serde_json::json!({
            "cmd": "FORWARD",
            "node_id": node_id,
            "data": Self::value_to_json(data),
            "state": Self::value_to_json(state),
        }))?;
        Self::response_to_value(&resp)
    }

    /// Fit a chain of filters in one command: each trainable filter fits,
    /// then forwards to feed the next; if torch is importable the daemon
    /// follows with one autograd forward/backward pass over the chain.
    /// Returns the chain's output plus each node's serialized state
    /// (torch `state_dict` bytes when available, cloudpickle otherwise).
    /// One round-trip — intermediate values never cross the process
    /// boundary, and the autograd graph stays whole.
    pub fn composite_fit(
        &mut self,
        node_ids: &[String],
        data: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let mut cmd = serde_json::json!({
            "cmd": "COMPOSITE_FIT",
            "node_ids": node_ids,
            "data": Self::value_to_json(data),
        });
        if let Some(y_val) = y {
            cmd["y"] = Self::value_to_json(y_val);
        }
        let resp = self.send(cmd)?;
        let output = Self::response_to_value(&resp)?;

        let mut states = HashMap::new();
        if let Some(state_map) = resp.get("states").and_then(|s| s.as_object()) {
            for (id, b64) in state_map {
                if let Some(s) = b64.as_str() {
                    let bytes = STANDARD
                        .decode(s)
                        .map_err(|e| WorkerError::Encoding(format!("decode state: {e}")))?;
                    states.insert(id.clone(), Value::bytes(bytes));
                }
            }
        }
        Ok((output, states))
    }

    /// Batched fit: send full dataset + batch_size, daemon splits internally.
    /// Model loaded ONCE, batches processed in a loop.
    pub fn batched_fit(
        &mut self,
        node_ids: &[String],
        data: &Value,
        y: Option<&Value>,
        batch_size: usize,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let mut cmd = serde_json::json!({
            "cmd": "BATCHED_FIT",
            "node_ids": node_ids,
            "data": Self::value_to_json(data),
            "batch_size": batch_size,
        });
        if let Some(y_val) = y {
            cmd["y"] = Self::value_to_json(y_val);
        }
        let resp = self.send(cmd)?;
        let output = Self::response_to_value(&resp)?;

        let mut states = HashMap::new();
        if let Some(state_map) = resp.get("states").and_then(|s| s.as_object()) {
            for (id, val) in state_map {
                if let Ok(v) = Self::json_to_value(val) {
                    states.insert(id.clone(), v);
                }
            }
        }
        Ok((output, states))
    }

    /// Forward `data` through a chain of filters, in order, inside one
    /// command — the composite counterpart of [`PythonProcess::forward`].
    pub fn composite_forward(&mut self, node_ids: &[String], data: &Value) -> Result<Value> {
        let resp = self.send(serde_json::json!({
            "cmd": "COMPOSITE_FORWARD",
            "node_ids": node_ids,
            "data": Self::value_to_json(data),
        }))?;
        Self::response_to_value(&resp)
    }

    /// Extract one filter's state.
    ///
    /// A materialized `DifferentiableFilter` answers with its own state
    /// convention — `Value::Json({"weights_b64": …})`, the dict its
    /// `forward` reads back and the local fit path writes — so a state
    /// read off a worker is loadable by a local graph. Anything else is
    /// opaque bytes: a torch `state_dict` when the filter has one, the
    /// cloudpickled filter otherwise.
    pub fn get_state(&mut self, node_id: &str) -> Result<Value> {
        let resp = self.send(serde_json::json!({"cmd": "GET_STATE", "node_id": node_id}))?;
        if let Some(state) = resp.get("state") {
            return Ok(Value::json(state.clone()));
        }
        if let Some(b64) = resp.get("state_b64").and_then(|s| s.as_str()) {
            let bytes = STANDARD
                .decode(b64)
                .map_err(|e| WorkerError::Encoding(format!("decode state: {e}")))?;
            Ok(Value::bytes(bytes))
        } else {
            Self::response_to_value(&resp)
        }
    }

    /// Load what [`PythonProcess::get_state`] produced back into the
    /// filter — how FedAvg-style aggregated states reach a worker.
    ///
    /// Mirrors `get_state` in both of its forms: a `Value::Json` state goes
    /// as-is (and `{"weights_b64": …}` is loaded into the filter's module),
    /// bytes go base64. Anything else is an encoding error.
    pub fn set_state(&mut self, node_id: &str, state: &Value) -> Result<()> {
        if let Value::Json(j) = state {
            let resp = self.send(serde_json::json!({
                "cmd": "SET_STATE", "node_id": node_id, "state": (**j).clone(),
            }))?;
            if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
                let error = resp.get("error").and_then(|e| e.as_str()).unwrap_or("?");
                return Err(WorkerError::Python(format!("set_state: {error}")));
            }
            return Ok(());
        }
        let b64 = match state {
            Value::Bytes(b) => STANDARD.encode(b.as_slice()),
            _ => {
                return Err(WorkerError::Encoding(
                    "set_state expects Value::Bytes".into(),
                ));
            }
        };
        let resp = self
            .send(serde_json::json!({"cmd": "SET_STATE", "node_id": node_id, "state_b64": b64}))?;
        if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let error = resp.get("error").and_then(|e| e.as_str()).unwrap_or("?");
            return Err(WorkerError::Python(format!("set_state: {error}")));
        }
        Ok(())
    }

    /// Collect the filter's current gradients, one nested list per named
    /// parameter, for AllReduce aggregation.
    ///
    /// Returns `Value::Json({param_name: nested list})` rather than the
    /// torch pickle this used to send. The aggregator is in Rust
    /// ([`somatize_runtime::strategy`]), and a pickle is opaque to it: the
    /// average of two `Value::Bytes` blobs is not a thing that can be
    /// computed, so the round died at the aggregation step having done all
    /// the work. Plain JSON also makes the average independent of the
    /// torch version each worker happens to have installed.
    ///
    /// A filter with no parameters, or no gradient on them, is an **error**
    /// rather than `Value::Empty`. Returning empty meant the average was
    /// taken over nothing and applied as nothing, and the round reported
    /// success — a data-parallel step that trained no one. The daemon says
    /// which of the three it is; this passes that on.
    pub fn get_gradients(&mut self, node_id: &str) -> Result<Value> {
        let resp = self.send(serde_json::json!({"cmd": "GET_GRADIENTS", "node_id": node_id}))?;
        if let Some(grads) = resp.get("gradients") {
            return Ok(Value::json(grads.clone()));
        }
        let error = resp
            .get("error")
            .and_then(|e| e.as_str())
            .unwrap_or("the worker returned no gradients and said nothing about why");
        Err(WorkerError::Python(format!("get_gradients: {error}")))
    }

    /// Hand aggregated gradients (the post-AllReduce mean of what
    /// [`PythonProcess::get_gradients`] returned) to the filter: the daemon
    /// writes them onto the matching parameters and steps the optimizer, so
    /// the replica actually moves.
    pub fn apply_gradients(&mut self, node_id: &str, gradients: &Value) -> Result<()> {
        let json = match gradients {
            Value::Json(j) => (**j).clone(),
            other => {
                return Err(WorkerError::Python(format!(
                    "apply_gradients expects the JSON gradients get_gradients \
                     returns, got {}",
                    other.type_name()
                )));
            }
        };
        let resp = self.send(
            serde_json::json!({"cmd": "APPLY_GRADIENTS", "node_id": node_id, "gradients": json}),
        )?;
        // The reply used to be discarded, so every failure the daemon
        // reported here — a shape mismatch, a model that is not the same
        // model — was a successful round that changed nothing.
        if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let error = resp.get("error").and_then(|e| e.as_str()).unwrap_or("?");
            return Err(WorkerError::Python(format!("apply_gradients: {error}")));
        }
        Ok(())
    }

    /// Ask the daemon to exit its command loop. Best-effort — the reply
    /// is ignored, and [`Drop`] kills the child regardless.
    pub fn shutdown(&mut self) {
        let _ = self.send(serde_json::json!({"cmd": "SHUTDOWN"}));
    }

    /// The node ids of the filters loaded into this process, in load order.
    pub fn node_ids(&self) -> &[String] {
        &self.node_ids
    }
}

impl Drop for PythonProcess {
    fn drop(&mut self) {
        self.shutdown();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ── SubprocessFilter: implements Filter trait via PythonProcess ──

/// A filter that delegates to a shared PythonProcess via stdin/stdout.
/// Multiple SubprocessFilters can share the same process (`Arc<Mutex>`).
pub struct SubprocessFilter {
    pub(crate) process: Arc<Mutex<PythonProcess>>,
    node_id: String,
    trainable: bool,
    /// Real config hash — must reflect the filter's configuration, never
    /// just the node id (two configs under the same node id must not
    /// share cache entries).
    config_hash: CacheKey,
}

impl SubprocessFilter {
    /// A proxy for the filter loaded under `node_id` in `process` —
    /// shared, so every sibling filter of one plan talks to the same
    /// interpreter.
    pub fn new(
        process: Arc<Mutex<PythonProcess>>,
        node_id: String,
        trainable: bool,
        config_hash: CacheKey,
    ) -> Self {
        Self {
            process,
            node_id,
            trainable,
            config_hash,
        }
    }

    /// Fallback identity for payloads that carry no explicit config hash:
    /// hash the pickled filter bytes — any config change changes the
    /// pickle, so stale cache hits are still impossible (the pickle is
    /// merely less stable across environments than a real config hash).
    pub fn fallback_config_hash(node_id: &str, pickled_filter: &[u8]) -> CacheKey {
        CacheKey::from_parts(&[b"subprocess-filter", node_id.as_bytes(), pickled_filter])
    }
}

/// The seam.
///
/// `Filter` is a `soma-core` trait, so these three return `SomaError`
/// while everything behind them is typed as a [`WorkerError`]. A subprocess
/// that died and a payload that would not decode stay distinguishable
/// right up to here, which is as far as the shared type can carry them.
impl Filter for SubprocessFilter {
    fn config_hash(&self) -> CacheKey {
        self.config_hash.clone()
    }

    fn fit(&self, x: &Value, y: Option<&Value>) -> somatize_core::error::Result<Value> {
        Ok(self
            .process
            .lock()
            .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))?
            .fit(&self.node_id, x, y)?)
    }

    fn forward(&self, x: &Value, state: &Value) -> somatize_core::error::Result<Value> {
        Ok(self
            .process
            .lock()
            .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))?
            .forward(&self.node_id, x, state)?)
    }

    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: self.node_id.clone(),
            kind: if self.trainable {
                FilterKind::Trainable
            } else {
                FilterKind::Stateless
            },
            cacheable: true,
            differentiable: self.trainable,
            deterministic: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn composite_fit(
        &self,
        peers: &[(String, std::sync::Arc<dyn somatize_core::filter::Filter>)],
        x: &Value,
        y: Option<&Value>,
    ) -> Option<somatize_core::error::Result<(Value, HashMap<String, Value>)>> {
        // Subprocess transport serialises the node_ids only — other filters
        // aren't shipped; the worker already has them deserialised from the
        // preceding prepare step.
        let node_ids: Vec<String> = peers.iter().map(|(id, _)| id.clone()).collect();
        tracing::info!(nodes = ?node_ids, "Composite fit via subprocess");
        Some(
            self.process
                .lock()
                .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))
                .and_then(|mut proc| proc.composite_fit(&node_ids, x, y))
                .map_err(SomaError::from),
        )
    }
}
