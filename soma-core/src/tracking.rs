//! Experiment tracking types: run manifests, status, event envelopes,
//! and the [`EventSink`]/[`Tracker`] traits.
//!
//! A *run* is the unit of tracking — one training session, study, or
//! fit — materialized as a directory of append-only logs plus small
//! atomic JSON files. This module holds only the schema and trait
//! contracts; the file-writing implementation lives in `soma-runtime`
//! (`LocalTracker`), and a future remote backend implements the same
//! [`Tracker`] trait.

use crate::error::Result;
use crate::event::Event;
use crate::study::Study;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Version of the on-disk run schema (manifest + logs layout).
pub const RUN_SCHEMA_VERSION: u32 = 1;

/// What kind of work a run tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunKind {
    /// A `GraphSession::fit` over a static pipeline.
    Fit,
    /// A native training loop (materialize/forward/backward/step).
    Train,
    /// A hyperparameter study.
    Study,
    /// A single trial within a study.
    Trial,
    /// Anything else.
    Other,
}

/// Lifecycle state of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunState {
    Running,
    Completed,
    Failed,
}

/// Best-effort git context captured at run start.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GitInfo {
    #[serde(default)]
    pub sha: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub dirty: Option<bool>,
}

/// Compact description of the graph a run executed, with pointers to
/// the full topology files inside the run directory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphSummaryInfo {
    pub n_nodes: usize,
    pub node_ids: Vec<String>,
    /// Relative path to the serialized graph (e.g. `graph.json`).
    #[serde(default)]
    pub graph_path: Option<String>,
    /// Relative path to the mermaid rendering (e.g. `graph.mmd`).
    #[serde(default)]
    pub mermaid_path: Option<String>,
}

/// Run manifest — written once, atomically, at run start.
///
/// Mutable lifecycle state (running/completed/failed, heartbeat) lives
/// in the separate [`RunStatus`] file so the manifest never needs
/// rewriting after creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub kind: RunKind,
    pub name: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub soma_version: Option<String>,
    #[serde(default)]
    pub python_version: Option<String>,
    #[serde(default)]
    pub hostname: Option<String>,
    #[serde(default)]
    pub git: GitInfo,
    #[serde(default)]
    pub entrypoint: Option<String>,
    #[serde(default)]
    pub argv: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    /// Named seeds, e.g. `{"torch": 42}`.
    #[serde(default)]
    pub seeds: HashMap<String, i64>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub parent_run_id: Option<String>,
    #[serde(default)]
    pub graph: Option<GraphSummaryInfo>,
    /// Relative path to `study.json` for study runs.
    #[serde(default)]
    pub study_path: Option<String>,
}

impl RunManifest {
    /// Minimal manifest; callers fill in environment fields.
    pub fn new(run_id: impl Into<String>, kind: RunKind, name: impl Into<String>) -> Self {
        Self {
            schema_version: RUN_SCHEMA_VERSION,
            run_id: run_id.into(),
            kind,
            name: name.into(),
            created_at: Utc::now(),
            soma_version: None,
            python_version: None,
            hostname: None,
            git: GitInfo::default(),
            entrypoint: None,
            argv: Vec::new(),
            cwd: None,
            seeds: HashMap::new(),
            tags: Vec::new(),
            notes: None,
            parent_run_id: None,
            graph: None,
            study_path: None,
        }
    }
}

/// Mutable run status — small, atomically rewritten (`status.json`).
///
/// The heartbeat lets a reader distinguish a live run from a crashed
/// one without any protocol: `state == Running` with a stale
/// `heartbeat_at` means the process died.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStatus {
    pub state: RunState,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub heartbeat_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
}

impl RunStatus {
    pub fn running() -> Self {
        let now = Utc::now();
        Self {
            state: RunState::Running,
            updated_at: now,
            heartbeat_at: Some(now),
            finished_at: None,
        }
    }
}

/// One line of `events.jsonl`: a monotonic sequence number and wall
/// timestamp wrapped around the event, with the event's own
/// `event_type` tag flattened into the same object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    pub seq: u64,
    pub ts: DateTime<Utc>,
    #[serde(flatten)]
    pub event: Event,
}

/// A lossless, ordered consumer of events.
///
/// Unlike broadcast subscribers (which may lag and drop), sinks are
/// invoked synchronously from `EventBus::emit` and must never lose an
/// event. Implementations should buffer writes and must not panic;
/// I/O errors are to be swallowed (optionally logged), never surfaced
/// into the training loop.
pub trait EventSink: Send + Sync {
    /// Record one event. Called synchronously on the emitting thread.
    fn record(&self, event: &Event);

    /// Flush any buffered state to durable storage.
    fn flush(&self) {}
}

/// A tracking backend bound to one run.
///
/// The local implementation writes a run directory; a remote backend
/// can implement the same contract over HTTP.
pub trait Tracker: Send + Sync {
    fn run_id(&self) -> &str;

    /// Root directory of the run (for file-based backends).
    fn run_dir(&self) -> &Path;

    /// The sink that persists events for this run.
    fn sink(&self) -> Arc<dyn EventSink>;

    /// Atomically write the manifest.
    fn save_manifest(&self, manifest: &RunManifest) -> Result<()>;

    /// Write an artifact at a path relative to the run directory,
    /// creating parent directories as needed.
    fn save_artifact(&self, rel_path: &str, bytes: &[u8]) -> Result<()>;

    /// Atomically write `study.json` (tmp + rename — readers never see
    /// a partial study, and a crash mid-write preserves the previous
    /// complete version).
    fn save_study(&self, study: &Study) -> Result<()>;

    /// Refresh `heartbeat_at` in the status file.
    fn heartbeat(&self) -> Result<()>;

    /// Set the terminal state, stamp `finished_at`, and flush the sink.
    fn finalize(&self, state: RunState) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::MetricRecord;

    #[test]
    fn manifest_roundtrip_and_defaults() {
        let mut m = RunManifest::new("run_x", RunKind::Train, "baseline");
        m.tags = vec!["mos".into()];
        m.seeds.insert("torch".into(), 42);
        let json = serde_json::to_string(&m).unwrap();
        let back: RunManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run_id, "run_x");
        assert_eq!(back.schema_version, RUN_SCHEMA_VERSION);
        assert_eq!(back.seeds["torch"], 42);

        // Old manifests without the optional fields still load.
        let minimal = serde_json::json!({
            "schema_version": 1,
            "run_id": "r",
            "kind": "fit",
            "name": "n",
            "created_at": "2026-07-26T10:00:00Z",
        });
        let back: RunManifest = serde_json::from_value(minimal).unwrap();
        assert!(back.git.sha.is_none());
        assert!(back.argv.is_empty());
    }

    #[test]
    fn envelope_flattens_event_type() {
        let env = EventEnvelope {
            seq: 7,
            ts: Utc::now(),
            event: Event::MetricReported {
                run_id: "r1".into(),
                metric: MetricRecord {
                    name: "val_f1".into(),
                    value: 0.9,
                    step: 3,
                    timestamp: Utc::now(),
                },
                node_id: None,
                trial_id: None,
            },
        };
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["seq"], 7);
        assert_eq!(json["event_type"], "MetricReported");
        assert_eq!(json["metric"]["name"], "val_f1");
        let back: EventEnvelope = serde_json::from_value(json).unwrap();
        assert_eq!(back.seq, 7);
        assert!(matches!(back.event, Event::MetricReported { .. }));
    }

    #[test]
    fn run_status_serde() {
        let s = RunStatus::running();
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"running\""));
        let back: RunStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back.state, RunState::Running);
        assert!(back.finished_at.is_none());
    }
}
