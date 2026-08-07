//! What a run records, and who hears about it.

use super::PyGraph;
use crate::prelude::*;
use crate::tracking::run::PyRun;

/// Start a tracked run: create `.soma/runs/<run_id>/`, snapshot the graph
/// topology into it and attach its lossless sink to the graph's event bus.
#[allow(clippy::too_many_arguments)]
pub(super) fn begin_run(
    g: &PyGraph,
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
    snapshot_topology(g, &tracker)?;

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
        n_nodes: g.graph.nodes.len(),
        node_ids: g.graph.nodes.iter().map(|n| n.id.clone()).collect(),
        graph_path: Some("graph.json".into()),
        mermaid_path: Some("graph.mmd".into()),
    });
    tracker.save_manifest(&manifest).map_err(soma_err_to_py)?;

    let sink = tracker.sink();
    g.event_bus.add_sink(sink.clone());
    Ok(PyRun {
        tracker: Arc::new(tracker),
        bus: g.event_bus.clone(),
        sink,
        finished: std::sync::atomic::AtomicBool::new(false),
        summary: std::sync::Mutex::new(HashMap::new()),
    })
}

/// Write the run's topology snapshot: `graph.json` (the machine
/// contract), `graph.mmd` (the human one) and `fingerprint.json`
/// (structural identity, with each node's filter config hash).
///
/// Called from [`begin_run`] — the single writer. The fingerprint is
/// best-effort: a graph whose canonical form will not serialize must not
/// stop a run from starting.
fn snapshot_topology(g: &PyGraph, tracker: &LocalTracker) -> PyResult<()> {
    let graph_json = serde_json::to_string_pretty(&g.graph)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    tracker
        .save_artifact("graph.json", graph_json.as_bytes())
        .map_err(soma_err_to_py)?;
    tracker
        .save_artifact("graph.mmd", g.graph.to_mermaid().as_bytes())
        .map_err(soma_err_to_py)?;

    if let Ok(fingerprint) = ArchitectureFingerprint::of(&g.graph) {
        let node_config: std::collections::BTreeMap<String, String> = g
            .graph
            .nodes
            .iter()
            .filter_map(|node| {
                let hash = g.library.get(&node.id)?.config_hash();
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

/// Register a Python callback to receive events during execution.
///
/// Events are delivered on a background thread, so the callback must be
/// thread-safe.
pub(super) fn on_event(g: &PyGraph, callback: PyObject) -> PyResult<()> {
    let mut rx = g.event_bus.subscribe();
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
    Ok(())
}

/// Emit an event onto the graph's bus from Python.
pub(super) fn emit_event(g: &PyGraph, py: Python<'_>, event: &Bound<'_, PyDict>) -> PyResult<()> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (event,))?.extract()?;
    let value: serde_json::Value = serde_json::from_str(&json_str)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid event JSON: {e}")))?;
    let event: somatize_core::tracking::event::Event = serde_json::from_value(value)
        .map_err(|e| PyRuntimeError::new_err(format!("unknown or malformed event: {e}")))?;
    g.event_bus.emit(event);
    Ok(())
}
