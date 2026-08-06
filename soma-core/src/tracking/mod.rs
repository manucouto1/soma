//! Experiment tracking types: run manifests, status, event envelopes,
//! and the [`EventSink`]/[`Tracker`] traits.
//!
//! A *run* is the unit of tracking — one training session, study, or
//! fit — materialized as a directory of append-only logs plus small
//! atomic JSON files. This module holds only the schema and trait
//! contracts; the file-writing implementation lives in `soma-runtime`
//! (`LocalTracker`), and a future remote backend implements the same
//! [`Tracker`] trait.

pub mod event;
pub mod fingerprint;
pub mod summary;

use crate::error::Result;
use crate::optimizer::study::Study;
use crate::tracking::event::Event;
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
    /// Anything else — also the fallback when deserializing a kind
    /// written by a newer soma, so old readers never fail on new kinds.
    #[serde(other)]
    Other,
}

/// Lifecycle state of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RunState {
    /// The run's process claims to be alive; trust it only while
    /// [`RunStatus::heartbeat_at`] is fresh.
    Running,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
}

/// Best-effort git context captured at run start.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GitInfo {
    /// Commit hash of `HEAD`.
    #[serde(default)]
    pub sha: Option<String>,
    /// Checked-out branch name, `None` on a detached head.
    #[serde(default)]
    pub branch: Option<String>,
    /// Whether the working tree had uncommitted changes — a dirty run
    /// is one the recorded `sha` cannot fully reproduce.
    #[serde(default)]
    pub dirty: Option<bool>,
}

/// Compact description of the graph a run executed, with pointers to
/// the full topology files inside the run directory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphSummaryInfo {
    /// Number of nodes in the executed graph.
    pub n_nodes: usize,
    /// Node ids, in the graph's insertion order.
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
    /// On-disk layout version this run was written with
    /// (see [`RUN_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Unique run identifier — also the run directory's name.
    pub run_id: String,
    /// What kind of work this run tracks.
    pub kind: RunKind,
    /// Human-readable run name (not required to be unique).
    pub name: String,
    /// When the run started.
    pub created_at: DateTime<Utc>,
    /// Version of soma that wrote this run.
    #[serde(default)]
    pub soma_version: Option<String>,
    /// Python interpreter version, for runs started from the bindings.
    #[serde(default)]
    pub python_version: Option<String>,
    /// Host the run executed on.
    #[serde(default)]
    pub hostname: Option<String>,
    /// Best-effort git context captured at run start.
    #[serde(default)]
    pub git: GitInfo,
    /// Script or module that started the run.
    #[serde(default)]
    pub entrypoint: Option<String>,
    /// Command-line arguments of the launching process.
    #[serde(default)]
    pub argv: Vec<String>,
    /// Working directory the run was started from.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Named seeds, e.g. `{"torch": 42}`.
    #[serde(default)]
    pub seeds: HashMap<String, i64>,
    /// Hyperparameters the caller declared for this run — the knobs
    /// that live outside the graph (learning rate, batch size, …) and
    /// so cannot be recovered from a filter's config hash. What makes
    /// a `ParamChanged` derivation possible at all.
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
    /// What the person starting this run expected, and why. Recorded at
    /// the start rather than the end on purpose: a hypothesis written
    /// after seeing the result is a conclusion.
    #[serde(default)]
    pub hypothesis: Option<String>,
    /// Free-form labels for filtering run listings.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Free-form notes attached at run start.
    #[serde(default)]
    pub notes: Option<String>,
    /// Run this one derives from — the edge the experiment pool's
    /// lineage is built on. Set explicitly, never inferred.
    #[serde(default)]
    pub parent_run_id: Option<String>,
    /// Compact description of the executed graph, absent for
    /// graph-less runs (e.g. a study run).
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
            params: HashMap::new(),
            hypothesis: None,
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
    /// Current lifecycle state.
    pub state: RunState,
    /// When this status file was last rewritten, for any reason.
    pub updated_at: DateTime<Utc>,
    /// Last liveness ping. Stale while `state` is
    /// [`RunState::Running`] means the process died.
    #[serde(default)]
    pub heartbeat_at: Option<DateTime<Utc>>,
    /// When the run reached a terminal state, `None` while running.
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
}

impl RunStatus {
    /// Fresh status for a run that just started: state
    /// [`RunState::Running`] with the heartbeat stamped now.
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
    /// Monotonic sequence number within the run — the order events were
    /// recorded in, which timestamps alone cannot guarantee.
    pub seq: u64,
    /// Wall-clock time the event was recorded.
    pub ts: DateTime<Utc>,
    /// The event itself, flattened into the envelope's JSON object.
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
    /// Identifier of the run this tracker is bound to.
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
    use crate::tracking::event::MetricRecord;

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

    #[test]
    fn run_status_terminal_states_roundtrip() {
        for state in [RunState::Completed, RunState::Failed] {
            let now = Utc::now();
            let s = RunStatus {
                state,
                updated_at: now,
                heartbeat_at: Some(now),
                finished_at: Some(now),
            };
            let back: RunStatus =
                serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
            assert_eq!(back.state, state);
            assert_eq!(back.finished_at, Some(now));
        }
        // Back-compat: a status without the optional timestamps loads.
        let minimal = serde_json::json!({
            "state": "completed",
            "updated_at": "2026-07-26T10:00:00Z",
        });
        let back: RunStatus = serde_json::from_value(minimal).unwrap();
        assert_eq!(back.state, RunState::Completed);
        assert!(back.heartbeat_at.is_none());
        assert!(back.finished_at.is_none());
    }

    #[test]
    fn unknown_run_kind_falls_back_to_other() {
        // A manifest written by a future soma with a new kind must not
        // break `LocalTracker::open` on this version.
        let manifest = serde_json::json!({
            "schema_version": 2,
            "run_id": "r",
            "kind": "evaluation",
            "name": "n",
            "created_at": "2026-07-26T10:00:00Z",
            "some_future_field": {"nested": true},
        });
        let back: RunManifest = serde_json::from_value(manifest).unwrap();
        assert_eq!(back.kind, RunKind::Other);
        // A reader can detect the newer schema explicitly.
        assert!(back.schema_version > RUN_SCHEMA_VERSION);
    }

    #[test]
    fn envelope_roundtrips_one_event_per_level() {
        let now = Utc::now();
        let metric = MetricRecord {
            name: "f1".into(),
            value: 0.5,
            step: 1,
            timestamp: now,
        };
        let events = vec![
            Event::RunFailed {
                run_id: "r".into(),
                error: "boom".into(),
            },
            Event::TrialMetric {
                study_id: "s".into(),
                trial_id: "t".into(),
                metric: metric.clone(),
            },
            Event::StudyProgress {
                study_id: "s".into(),
                completed: 1,
                total: 4,
                best_value: 0.5,
            },
            Event::MemberExploited {
                study_id: "s".into(),
                generation: 1,
                replaced_id: "a".into(),
                donor_id: "b".into(),
            },
            Event::HealthFlag {
                run_id: "r".into(),
                node_id: "n".into(),
                step: 3,
                flag: "LEAKAGE".into(),
                detail: "cka=0.99".into(),
            },
        ];
        for (i, event) in events.into_iter().enumerate() {
            let env = EventEnvelope {
                seq: i as u64,
                ts: now,
                event,
            };
            let json = serde_json::to_value(&env).unwrap();
            // The envelope's own fields never collide with payloads.
            assert_eq!(json["seq"], i as u64);
            assert!(json["event_type"].is_string());
            let back: EventEnvelope = serde_json::from_value(json).unwrap();
            assert_eq!(back.seq, i as u64);
            assert_eq!(back.ts, now);
        }
    }

    #[test]
    fn git_info_and_graph_summary_serde() {
        let git = GitInfo {
            sha: Some("abc123".into()),
            branch: Some("main".into()),
            dirty: Some(true),
        };
        let back: GitInfo = serde_json::from_str(&serde_json::to_string(&git).unwrap()).unwrap();
        assert_eq!(back, git);
        assert_eq!(GitInfo::default(), GitInfo::default());
        assert!(GitInfo::default().sha.is_none());

        let summary = GraphSummaryInfo {
            n_nodes: 2,
            node_ids: vec!["a".into(), "b".into()],
            graph_path: Some("graph.json".into()),
            mermaid_path: None,
        };
        let back: GraphSummaryInfo =
            serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();
        assert_eq!(back, summary);
        // Back-compat: path fields are optional.
        let minimal: GraphSummaryInfo =
            serde_json::from_value(serde_json::json!({"n_nodes": 1, "node_ids": ["x"]})).unwrap();
        assert_eq!(minimal.n_nodes, 1);
        assert!(minimal.graph_path.is_none());
        assert_eq!(GraphSummaryInfo::default().n_nodes, 0);
    }
}
