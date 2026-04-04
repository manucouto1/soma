//! Python subprocess — persistent daemon for filter execution.
//!
//! Spawns a Python child process that loads filters via cloudpickle
//! and executes fit/forward commands via stdin/stdout JSON Lines.
//! The GIL is completely isolated from the Rust process — no segfaults.

use base64::engine::{Engine, general_purpose::STANDARD};
use somatize_core::cache::CacheKey;
use somatize_core::error::{Result, SomaError};
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

/// The Python daemon script, embedded as a Rust string.
const DAEMON_SCRIPT: &str = r#"
import json, sys, base64, cloudpickle, io, pickle

filters = {}

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
                return d.get("values", [])
            if t == "Json":
                return d
            if t == "Empty":
                return {}
            if t == "Bytes":
                return bytes(d)
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
        print(json.dumps({"ok": False, "error": f"invalid JSON: {e}"}), flush=True)
        continue

    try:
        action = cmd.get("cmd", "")

        if action == "LOAD":
            for f in cmd["filters"]:
                obj = cloudpickle.loads(base64.b64decode(f["pickle_b64"]))
                filters[f["id"]] = {"obj": obj, "trainable": f.get("trainable", True)}
            print(json.dumps({"ok": True}), flush=True)

        elif action == "FIT":
            f = filters[cmd["node_id"]]["obj"]
            data = _decode(cmd.get("data"))
            y = _decode(cmd.get("y"))
            result = f.fit(data, y)
            print(json.dumps({"ok": True, "result": _encode(result)}), flush=True)

        elif action == "FORWARD":
            f = filters[cmd["node_id"]]["obj"]
            data = _decode(cmd.get("data"))
            state = _decode(cmd.get("state", {}))
            result = f.forward(data, state)
            print(json.dumps({"ok": True, "result": _encode(result)}), flush=True)

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
                out = f.forward(out, state)

            result = out.detach().tolist() if hasattr(out, 'detach') else out
            print(json.dumps({"ok": True, "result": _encode(result)}), flush=True)

        elif action == "COMPOSITE_FIT":
            node_ids = cmd["node_ids"]
            data = _decode(cmd.get("data"))
            y = _decode(cmd.get("y"))

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
                out = f.forward(out, {})

            # Backward
            if y is not None and hasattr(out, 'backward'):
                last = filters[node_ids[-1]]["obj"]
                try:
                    import torch
                    if isinstance(y, list):
                        y_t = torch.tensor(y, dtype=torch.float32)
                    else:
                        y_t = y
                    if hasattr(last, 'loss_fn'):
                        loss = last.loss_fn(out, y_t)
                    else:
                        loss = torch.nn.functional.mse_loss(out, y_t)
                    loss.backward()
                    for nid in node_ids:
                        f = filters[nid]["obj"]
                        if hasattr(f, 'optimizer'):
                            f.optimizer.step()
                            f.optimizer.zero_grad()
                except Exception:
                    pass

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
            print(json.dumps({"ok": True, "result": _encode(result), "states": states}), flush=True)

        elif action == "GET_STATE":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
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
            print(json.dumps({"ok": True, "state_b64": state_b64}), flush=True)

        elif action == "SET_STATE":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
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
            print(json.dumps({"ok": True}), flush=True)

        elif action == "GET_GRADIENTS":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
            buf = io.BytesIO()
            if hasattr(f, 'parameters'):
                try:
                    import torch
                    grads = {name: p.grad.clone() for name, p in f.named_parameters() if p.grad is not None}
                    torch.save(grads, buf)
                except ImportError:
                    pass
            grad_b64 = base64.b64encode(buf.getvalue()).decode()
            print(json.dumps({"ok": True, "gradients_b64": grad_b64}), flush=True)

        elif action == "APPLY_GRADIENTS":
            nid = cmd["node_id"]
            f = filters[nid]["obj"]
            grad_bytes = base64.b64decode(cmd["gradients_b64"])
            if hasattr(f, 'named_parameters'):
                try:
                    import torch
                    buf = io.BytesIO(grad_bytes)
                    grads = torch.load(buf, weights_only=True)
                    for name, p in f.named_parameters():
                        if name in grads:
                            p.grad = grads[name]
                    if hasattr(f, 'optimizer'):
                        f.optimizer.step()
                except ImportError:
                    pass
            print(json.dumps({"ok": True}), flush=True)

        elif action == "SHUTDOWN":
            print(json.dumps({"ok": True}), flush=True)
            break

        else:
            print(json.dumps({"ok": False, "error": f"unknown command: {action}"}), flush=True)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(json.dumps({"ok": False, "error": str(e), "traceback": tb}), flush=True)
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
            .map_err(|e| SomaError::Other(format!("failed to spawn python: {e}")))?;

        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| SomaError::Other("no stdin".into()))?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| SomaError::Other("no stdout".into()))?,
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
            return Err(SomaError::Other(format!("LOAD failed: {error}")));
        }

        Ok(proc)
    }

    /// Send a JSON command and read the JSON response.
    fn send(&mut self, cmd: serde_json::Value) -> Result<serde_json::Value> {
        let line = serde_json::to_string(&cmd)
            .map_err(|e| SomaError::Other(format!("serialize cmd: {e}")))?;

        writeln!(self.stdin, "{line}")
            .map_err(|e| SomaError::Other(format!("write to python stdin: {e}")))?;
        self.stdin
            .flush()
            .map_err(|e| SomaError::Other(format!("flush stdin: {e}")))?;

        let mut response = String::new();
        self.stdout
            .read_line(&mut response)
            .map_err(|e| SomaError::Other(format!("read from python stdout: {e}")))?;

        if response.is_empty() {
            // Process probably died
            return Err(SomaError::Other(
                "python process closed stdout (crashed?)".into(),
            ));
        }

        serde_json::from_str(&response)
            .map_err(|e| SomaError::Other(format!("parse python response: {e}\nraw: {response}")))
    }

    /// Convert a response to a Value, handling errors.
    fn response_to_value(resp: &serde_json::Value) -> Result<Value> {
        if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let error = resp
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown error");
            let traceback = resp.get("traceback").and_then(|t| t.as_str()).unwrap_or("");
            return Err(SomaError::Other(format!(
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
        Ok(Value::Json(v.clone()))
    }

    /// Encode a Value to JSON for the Python process.
    fn value_to_json(v: &Value) -> serde_json::Value {
        serde_json::to_value(v).unwrap_or(serde_json::Value::Null)
    }

    // ── Public API ──

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

    pub fn forward(&mut self, node_id: &str, data: &Value, state: &Value) -> Result<Value> {
        let resp = self.send(serde_json::json!({
            "cmd": "FORWARD",
            "node_id": node_id,
            "data": Self::value_to_json(data),
            "state": Self::value_to_json(state),
        }))?;
        Self::response_to_value(&resp)
    }

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
                        .map_err(|e| SomaError::Other(format!("decode state: {e}")))?;
                    states.insert(id.clone(), Value::Bytes(bytes));
                }
            }
        }
        Ok((output, states))
    }

    pub fn composite_forward(&mut self, node_ids: &[String], data: &Value) -> Result<Value> {
        let resp = self.send(serde_json::json!({
            "cmd": "COMPOSITE_FORWARD",
            "node_ids": node_ids,
            "data": Self::value_to_json(data),
        }))?;
        Self::response_to_value(&resp)
    }

    pub fn get_state(&mut self, node_id: &str) -> Result<Value> {
        let resp = self.send(serde_json::json!({"cmd": "GET_STATE", "node_id": node_id}))?;
        if let Some(b64) = resp.get("state_b64").and_then(|s| s.as_str()) {
            let bytes = STANDARD
                .decode(b64)
                .map_err(|e| SomaError::Other(format!("decode state: {e}")))?;
            Ok(Value::Bytes(bytes))
        } else {
            Self::response_to_value(&resp)
        }
    }

    pub fn set_state(&mut self, node_id: &str, state: &Value) -> Result<()> {
        let b64 = match state {
            Value::Bytes(b) => STANDARD.encode(b),
            _ => return Err(SomaError::Other("set_state expects Value::Bytes".into())),
        };
        let resp = self
            .send(serde_json::json!({"cmd": "SET_STATE", "node_id": node_id, "state_b64": b64}))?;
        if resp.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let error = resp.get("error").and_then(|e| e.as_str()).unwrap_or("?");
            return Err(SomaError::Other(format!("set_state: {error}")));
        }
        Ok(())
    }

    pub fn get_gradients(&mut self, node_id: &str) -> Result<Value> {
        let resp = self.send(serde_json::json!({"cmd": "GET_GRADIENTS", "node_id": node_id}))?;
        if let Some(b64) = resp.get("gradients_b64").and_then(|s| s.as_str()) {
            let bytes = STANDARD
                .decode(b64)
                .map_err(|e| SomaError::Other(format!("decode gradients: {e}")))?;
            Ok(Value::Bytes(bytes))
        } else {
            Ok(Value::Empty)
        }
    }

    pub fn apply_gradients(&mut self, node_id: &str, gradients: &Value) -> Result<()> {
        let b64 = match gradients {
            Value::Bytes(b) => STANDARD.encode(b),
            _ => {
                return Err(SomaError::Other(
                    "apply_gradients expects Value::Bytes".into(),
                ));
            }
        };
        self.send(
            serde_json::json!({"cmd": "APPLY_GRADIENTS", "node_id": node_id, "gradients_b64": b64}),
        )?;
        Ok(())
    }

    pub fn shutdown(&mut self) {
        let _ = self.send(serde_json::json!({"cmd": "SHUTDOWN"}));
    }

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
/// Multiple SubprocessFilters can share the same process (Arc<Mutex>).
pub struct SubprocessFilter {
    process: Arc<Mutex<PythonProcess>>,
    node_id: String,
    trainable: bool,
}

impl SubprocessFilter {
    pub fn new(process: Arc<Mutex<PythonProcess>>, node_id: String, trainable: bool) -> Self {
        Self {
            process,
            node_id,
            trainable,
        }
    }
}

impl Filter for SubprocessFilter {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[self.node_id.as_bytes()])
    }

    fn fit(&self, x: &Value, y: Option<&Value>) -> Result<Value> {
        self.process
            .lock()
            .map_err(|e| SomaError::Other(format!("process mutex poisoned: {e}")))?
            .fit(&self.node_id, x, y)
    }

    fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
        self.process
            .lock()
            .map_err(|e| SomaError::Other(format!("process mutex poisoned: {e}")))?
            .forward(&self.node_id, x, state)
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
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
