//! Reading run directories and the experiment pool.

use crate::prelude::*;

// ── Module ──

// ── Run-directory readers (back `soma.runs()` / `soma.RunView`) ──
//
// Each function returns a JSON string (the serde shape of the
// soma-runtime reader structs); the Python wrapper in `soma/_runs.py`
// parses it. Same pattern as `Graph.graph_json`.

/// Convert an overlay dict (the `RunView.overlay()` shape) into the
/// core overlay struct, via JSON like `Graph.emit_event`.
pub(crate) fn py_overlay(
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
pub(crate) fn run_summary_json(dir: String) -> PyResult<String> {
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
pub(crate) fn checkout_run(run_id: String, root: String) -> PyResult<()> {
    somatize_runtime::tracking::checkout(&root, &run_id).map_err(soma_err_to_py)
}

/// The run id in `.soma/HEAD`, or None when the next run starts a new
/// line.
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
pub(crate) fn read_head_run(root: String) -> Option<String> {
    somatize_runtime::tracking::read_head(&root)
}

/// Detach HEAD: the next run starts its own research line.
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
pub(crate) fn clear_head_run(root: String) -> PyResult<()> {
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
pub(crate) fn kb_find_similar_json(
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
pub(crate) fn kb_record_conclusion(
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
pub(crate) fn kb_lineage_json(run_id: String, root: String) -> PyResult<Option<String>> {
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
pub(crate) fn kb_diff_json(a: String, b: String, root: String) -> PyResult<String> {
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
pub(crate) fn kb_reindex(root: String) -> PyResult<usize> {
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
pub(crate) fn list_runs_json(root: String) -> PyResult<String> {
    let infos = somatize_runtime::tracking::list_runs(&root)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&infos)
}

/// Listing entry (manifest identity + derived state) for one run dir.
#[pyfunction]
pub(crate) fn run_info_json(dir: String) -> PyResult<String> {
    let info = open_run_reader(&dir)?
        .info()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&info)
}

/// The run's manifest.json contents.
#[pyfunction]
pub(crate) fn run_manifest_json(dir: String) -> PyResult<String> {
    let manifest = open_run_reader(&dir)?
        .manifest()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&manifest)
}

/// All parseable event envelopes from events.jsonl, in log order.
#[pyfunction]
pub(crate) fn run_events_json(dir: String) -> PyResult<String> {
    let events = open_run_reader(&dir)?
        .events()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&events)
}

/// Metric time series (optionally filtered by name).
#[pyfunction]
#[pyo3(signature = (dir, name=None))]
pub(crate) fn run_metric_series_json(dir: String, name: Option<String>) -> PyResult<String> {
    let points = open_run_reader(&dir)?
        .metric_series(name.as_deref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&points)
}

/// Per-node execution spans (gantt/overlay substrate).
#[pyfunction]
pub(crate) fn run_node_timings_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .node_timings()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Cache hit/miss counts, total and per node.
#[pyfunction]
pub(crate) fn run_cache_activity_json(dir: String) -> PyResult<String> {
    let activity = open_run_reader(&dir)?
        .cache_activity()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&activity)
}

/// HealthFlag events with wall time.
#[pyfunction]
pub(crate) fn run_health_flags_json(dir: String) -> PyResult<String> {
    let flags = open_run_reader(&dir)?
        .health_flags()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&flags)
}

/// Trial lifetimes from study.json (empty for non-study runs).
#[pyfunction]
pub(crate) fn run_trial_timeline_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .trial_timeline()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Agent-level activity aggregated per step node (turns, tokens,
/// effects by label, tool calls, suspensions).
#[pyfunction]
pub(crate) fn run_agentic_activity_json(dir: String) -> PyResult<String> {
    let activity = open_run_reader(&dir)?
        .agentic_activity()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&activity)
}

/// Per-effect execution spans (the gantt substrate for agent runs).
#[pyfunction]
pub(crate) fn run_agentic_timeline_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .agentic_timeline()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Rendering overlay aggregated from this run's events (JSON
/// GraphOverlay: per-node status, duration, cache tier, flags).
#[pyfunction]
pub(crate) fn run_overlay_json(dir: String) -> PyResult<String> {
    let overlay = open_run_reader(&dir)?
        .overlay()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&overlay)
}

/// Mermaid diagram of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
pub(crate) fn run_to_mermaid(dir: String, overlay: bool) -> PyResult<String> {
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
pub(crate) fn graph_json_to_mermaid(
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
pub(crate) fn graph_json_to_svg(
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
pub(crate) fn run_to_svg(dir: String, overlay: bool) -> PyResult<String> {
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
