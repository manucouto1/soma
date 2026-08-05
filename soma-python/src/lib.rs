// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! PyO3 bindings for Soma — exposes Graph, Study, and Filter to Python.
//!
//! Bridges Python Filter classes to the Rust Filter trait, converts
//! between Python lists/dicts and Soma Values, and wraps the StudyRunner
//! for hyperparameter optimization from Python.
//!
//! One module per thing exposed, rather than one file. This was 4718
//! lines in a single `mod`, which meant every reader of any one binding
//! scrolled past all the others, and nothing could be `pub(crate)`
//! because there was no boundary to be private to.

mod agentic;
mod bridge;
mod cache;
mod convert;
mod graph;
mod readers;
mod run;
mod store;
mod study;
mod worker;

/// What every binding module needs.
///
/// A glob rather than thirty `use` lines repeated nine times. The types
/// here are the vocabulary of the whole crate — a `Value`, a `Graph`, a
/// `PyResult` — so naming them once is the honest description.
pub(crate) mod prelude {
    pub(crate) use super::agentic::{
        PyAgent, PyJudge, PyStepCtx, PyTool, PyToolAdapter, to_step_spec,
    };
    pub(crate) use super::bridge::PyFilterBridge;
    pub(crate) use super::convert::{json_to_py, py_any_to_json, py_to_value, value_to_py};
    pub(crate) use super::{default_cache_dir, py_err_to_soma, soma_err_to_py};
    pub(crate) use pyo3::exceptions::PyRuntimeError;
    pub(crate) use pyo3::exceptions::PyValueError;
    pub(crate) use pyo3::prelude::*;
    pub(crate) use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};
    pub(crate) use somatize_compiler::{CompileMode, compile};
    pub(crate) use somatize_core::cache::CacheKey;
    pub(crate) use somatize_core::error::{Result as SomaResult, SomaError};
    pub(crate) use somatize_core::event::MetricRecord;
    pub(crate) use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
    pub(crate) use somatize_core::fingerprint::ArchitectureFingerprint;
    pub(crate) use somatize_core::graph::{Edge, Graph, Node};
    pub(crate) use somatize_core::search::{Scale, SearchDimension, SearchSpace};
    pub(crate) use somatize_core::study::{
        Direction, Objective, PruningStrategy, SearchStrategy, Study,
    };
    pub(crate) use somatize_core::tracking::{GraphSummaryInfo, RunKind, RunState, Tracker};
    pub(crate) use somatize_core::value::Value;
    pub(crate) use somatize_runtime::EventBus;
    pub(crate) use somatize_runtime::StudyIo;
    pub(crate) use somatize_runtime::cache::{FsActionStore, MemoryCache, TieredCache};
    pub(crate) use somatize_runtime::effects::{EffectDriver, EffectJournal};
    pub(crate) use somatize_runtime::executor::{self, Context, GraphInfo};
    pub(crate) use somatize_runtime::executors::study::{
        FnTrialExecutor, StudyRunner, TrialContext, TrialOutcome,
    };
    pub(crate) use somatize_runtime::node_catalog::NodeCatalog;
    pub(crate) use somatize_runtime::runner::{LocalRunner, Runner};
    pub(crate) use somatize_runtime::sampler::{
        BayesianSampler, GridSampler, RandomSampler, Sampler,
    };
    pub(crate) use somatize_runtime::tracking::{
        LocalTracker, RunReader, advance_head, load_manifest, resolve_parent, summarize,
    };
    pub(crate) use std::collections::HashMap;
    pub(crate) use std::sync::Arc;
}

use crate::cache::{cache_gc, cache_pin, cache_purge_v1, cache_stats, cache_verify};
use crate::graph::PyGraph;
use crate::prelude::*;
use crate::readers::{
    checkout_run, clear_head_run, graph_json_to_mermaid, graph_json_to_svg, kb_diff_json,
    kb_find_similar_json, kb_lineage_json, kb_record_conclusion, kb_reindex, list_runs_json,
    read_head_run, run_agentic_activity_json, run_agentic_timeline_json, run_cache_activity_json,
    run_events_json, run_health_flags_json, run_info_json, run_manifest_json,
    run_metric_series_json, run_node_timings_json, run_overlay_json, run_summary_json,
    run_to_graphviz, run_to_mermaid, run_to_svg, run_trial_timeline_json,
};
use crate::run::PyRun;
use crate::study::{PyStudy, PyTrial};
use crate::worker::PyWorker;

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
    m.add_function(wrap_pyfunction!(run_agentic_activity_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_agentic_timeline_json, m)?)?;
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
