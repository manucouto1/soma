//! Read-side of run tracking: list run directories and aggregate their
//! logs into chart-ready data.
//!
//! [`RunReader`] is the reader counterpart of [`LocalTracker`](super::LocalTracker):
//! it consumes the files a tracker writes (`manifest.json`, `status.json`,
//! `events.jsonl`, `metrics.jsonl`, `study.json`) and never writes anything.
//! Every aggregate it produces is a plain serde struct so the same shapes
//! serve Python bindings, CLI output, and any future front-end.
//!
//! Wall-clock times come from the envelope `ts` stamped by the sink at
//! emit time (sinks are synchronous); start events themselves carry no
//! timestamp. Unparseable lines — a torn tail from a crash, or an event
//! kind written by a newer soma — are skipped, never an error.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::graph::Graph;
use somatize_core::study::{Study, TrialState};
use somatize_core::tracking::{EventEnvelope, RunManifest, RunState, RunStatus};
use somatize_core::viz::{GraphOverlay, NodeStatus};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::local_tracker::{load_manifest, load_status};

/// A `Running` status whose heartbeat is older than this is reported as
/// crashed: the process died without finalizing.
pub const STALE_HEARTBEAT_SECS: i64 = 300;

/// Reader over one run directory.
pub struct RunReader {
    dir: PathBuf,
}

/// Listing entry for one run: manifest identity plus derived liveness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub run_id: String,
    /// Manifest `kind` as its snake_case string (`fit`, `train`, `study`, …).
    pub kind: String,
    pub name: String,
    /// `running` | `completed` | `failed` | `crashed`.
    pub state: String,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    /// Wall time from creation to finish, when finished.
    pub duration_ms: Option<u64>,
    pub tags: Vec<String>,
    /// Absolute path of the run directory.
    pub dir: String,
}

/// One execution span of a node, in event order. A node appears once
/// per execution (re-runs and stream chunks produce separate spans).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeSpan {
    pub node_id: String,
    /// Envelope timestamp of the opening event.
    pub started_ts: Option<DateTime<Utc>>,
    /// Envelope timestamp of the closing event.
    pub finished_ts: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    /// `completed` | `failed` | `cache_hit` | `running`.
    pub outcome: String,
    /// Cache tier that served a hit (`memory`, `local`, …).
    pub cache_tier: Option<String>,
    pub error: Option<String>,
}

/// Per-run cache effectiveness, reconstructed from hit/miss events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheActivity {
    pub hits: u64,
    pub misses: u64,
    pub by_node: BTreeMap<String, NodeCacheCounts>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeCacheCounts {
    pub hits: u64,
    pub misses: u64,
    pub last_tier: Option<String>,
}

/// One line of `metrics.jsonl` (also derivable from events).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub ts: DateTime<Utc>,
    pub name: String,
    pub value: f64,
    pub step: u64,
    #[serde(default)]
    pub trial_id: Option<String>,
    #[serde(default)]
    pub node_id: Option<String>,
}

/// One `HealthFlag` event with its wall time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthFlagRecord {
    pub ts: DateTime<Utc>,
    pub node_id: String,
    pub step: usize,
    pub flag: String,
    pub detail: String,
}

/// One trial's lifetime, from `study.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialSpan {
    pub trial_id: String,
    /// `completed` | `pruned` | `failed` | `running` | `pending`.
    pub state: String,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
}

impl RunReader {
    /// Open a run directory. Fails only if the manifest is missing or
    /// unreadable — everything else is tolerated per-file.
    pub fn open(run_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = run_dir.as_ref().to_path_buf();
        load_manifest(&dir)?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn manifest(&self) -> Result<RunManifest> {
        load_manifest(&self.dir)
    }

    pub fn status(&self) -> Result<RunStatus> {
        load_status(&self.dir)
    }

    /// Listing entry for this run (state includes crash detection).
    pub fn info(&self) -> Result<RunInfo> {
        let manifest = self.manifest()?;
        Ok(run_info(
            &self.dir,
            manifest,
            self.status().ok(),
            Utc::now(),
        ))
    }

    /// All parseable event envelopes, in log order. Torn or unknown
    /// lines are skipped; `seq` gaps let a consumer detect the skips.
    pub fn events(&self) -> Result<Vec<EventEnvelope>> {
        let path = self.dir.join("events.jsonl");
        let file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => return Ok(Vec::new()), // no events yet
        };
        let mut envelopes = Vec::new();
        for line in BufReader::new(file).lines() {
            let line = line.map_err(SomaError::Io)?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(env) = serde_json::from_str::<EventEnvelope>(&line) {
                envelopes.push(env);
            }
        }
        Ok(envelopes)
    }

    /// Per-node execution spans in event order — the gantt/overlay
    /// substrate. Cache hits are standalone spans (a hit node never
    /// starts); an unclosed span means the run died mid-node.
    pub fn node_timings(&self) -> Result<Vec<NodeSpan>> {
        let mut spans: Vec<NodeSpan> = Vec::new();
        let mut open: BTreeMap<String, usize> = BTreeMap::new();
        for env in self.events()? {
            match env.event {
                Event::NodeStarted { node_id, .. } => {
                    open.insert(node_id.clone(), spans.len());
                    spans.push(NodeSpan {
                        node_id,
                        started_ts: Some(env.ts),
                        finished_ts: None,
                        duration_ms: None,
                        outcome: "running".into(),
                        cache_tier: None,
                        error: None,
                    });
                }
                Event::NodeCacheHit {
                    node_id,
                    tier,
                    load_time,
                    ..
                } => {
                    spans.push(NodeSpan {
                        node_id,
                        started_ts: Some(env.ts),
                        finished_ts: Some(env.ts),
                        duration_ms: Some(load_time.as_millis() as u64),
                        outcome: "cache_hit".into(),
                        cache_tier: Some(format!("{tier:?}").to_lowercase()),
                        error: None,
                    });
                }
                Event::NodeCompleted {
                    node_id, duration, ..
                } => {
                    let idx = open.remove(&node_id);
                    let span = match idx {
                        Some(i) => &mut spans[i],
                        None => {
                            spans.push(NodeSpan {
                                node_id: node_id.clone(),
                                started_ts: None,
                                finished_ts: None,
                                duration_ms: None,
                                outcome: String::new(),
                                cache_tier: None,
                                error: None,
                            });
                            spans.last_mut().expect("just pushed")
                        }
                    };
                    span.finished_ts = Some(env.ts);
                    span.duration_ms = Some(duration.as_millis() as u64);
                    span.outcome = "completed".into();
                }
                Event::NodeFailed { node_id, error, .. } => {
                    let idx = open.remove(&node_id);
                    let span = match idx {
                        Some(i) => &mut spans[i],
                        None => {
                            spans.push(NodeSpan {
                                node_id: node_id.clone(),
                                started_ts: None,
                                finished_ts: None,
                                duration_ms: None,
                                outcome: String::new(),
                                cache_tier: None,
                                error: None,
                            });
                            spans.last_mut().expect("just pushed")
                        }
                    };
                    span.finished_ts = Some(env.ts);
                    span.outcome = "failed".into();
                    span.error = Some(error);
                }
                _ => {}
            }
        }
        Ok(spans)
    }

    /// Cache hit/miss counts, total and per node.
    pub fn cache_activity(&self) -> Result<CacheActivity> {
        let mut activity = CacheActivity::default();
        for env in self.events()? {
            match env.event {
                Event::NodeCacheHit { node_id, tier, .. } => {
                    activity.hits += 1;
                    let counts = activity.by_node.entry(node_id).or_default();
                    counts.hits += 1;
                    counts.last_tier = Some(format!("{tier:?}").to_lowercase());
                }
                Event::NodeCacheMiss { node_id, .. } => {
                    activity.misses += 1;
                    activity.by_node.entry(node_id).or_default().misses += 1;
                }
                _ => {}
            }
        }
        Ok(activity)
    }

    /// Metric time series, optionally filtered by name. Reads the flat
    /// `metrics.jsonl` tee; falls back to deriving the same points from
    /// `events.jsonl` when the tee is absent.
    pub fn metric_series(&self, name: Option<&str>) -> Result<Vec<MetricPoint>> {
        let path = self.dir.join("metrics.jsonl");
        let mut points: Vec<MetricPoint> = Vec::new();
        if let Ok(file) = fs::File::open(&path) {
            for line in BufReader::new(file).lines() {
                let line = line.map_err(SomaError::Io)?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(p) = serde_json::from_str::<MetricPoint>(&line) {
                    points.push(p);
                }
            }
        } else {
            for env in self.events()? {
                match env.event {
                    Event::TrialMetric {
                        trial_id, metric, ..
                    } => points.push(MetricPoint {
                        ts: metric.timestamp,
                        name: metric.name,
                        value: metric.value,
                        step: metric.step as u64,
                        trial_id: Some(trial_id),
                        node_id: None,
                    }),
                    Event::MetricReported {
                        metric,
                        node_id,
                        trial_id,
                        ..
                    } => points.push(MetricPoint {
                        ts: metric.timestamp,
                        name: metric.name,
                        value: metric.value,
                        step: metric.step as u64,
                        trial_id,
                        node_id,
                    }),
                    _ => {}
                }
            }
        }
        if let Some(name) = name {
            points.retain(|p| p.name == name);
        }
        Ok(points)
    }

    /// All `HealthFlag` events with wall time.
    pub fn health_flags(&self) -> Result<Vec<HealthFlagRecord>> {
        let mut flags = Vec::new();
        for env in self.events()? {
            if let Event::HealthFlag {
                node_id,
                step,
                flag,
                detail,
                ..
            } = env.event
            {
                flags.push(HealthFlagRecord {
                    ts: env.ts,
                    node_id,
                    step,
                    flag,
                    detail,
                });
            }
        }
        Ok(flags)
    }

    /// The graph this run executed (`graph.json`), if snapshotted.
    pub fn graph(&self) -> Result<Option<Graph>> {
        let path = self.dir.join("graph.json");
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| SomaError::Serialization(e.to_string()))
    }

    /// Fold this run's node spans and health flags into a rendering
    /// overlay: status + total duration + cache tier per node, `×N`
    /// when a node ran more than once, deduplicated flags.
    pub fn overlay(&self) -> Result<GraphOverlay> {
        let mut overlay = GraphOverlay::default();
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for span in self.node_timings()? {
            let entry = overlay.nodes.entry(span.node_id.clone()).or_default();
            *counts.entry(span.node_id).or_default() += 1;
            // Last span wins for status/tier; durations accumulate.
            entry.status = Some(match span.outcome.as_str() {
                "completed" => NodeStatus::Completed,
                "cache_hit" => NodeStatus::Cached,
                "failed" => NodeStatus::Failed,
                _ => NodeStatus::Running,
            });
            entry.cache_tier = span.cache_tier;
            if let Some(ms) = span.duration_ms {
                entry.duration_ms = Some(entry.duration_ms.unwrap_or(0) + ms);
            }
        }
        for (node_id, n) in counts {
            if n > 1
                && let Some(entry) = overlay.nodes.get_mut(&node_id)
            {
                entry.sublabel = Some(format!("×{n}"));
            }
        }
        for flag in self.health_flags()? {
            let entry = overlay.nodes.entry(flag.node_id).or_default();
            if !entry.flags.contains(&flag.flag) {
                entry.flags.push(flag.flag);
            }
        }
        Ok(overlay)
    }

    /// Mermaid rendering of the run's graph, annotated with this run's
    /// overlay. Errors if the run has no `graph.json` snapshot.
    pub fn to_mermaid(&self) -> Result<String> {
        let graph = self.graph()?.ok_or_else(|| {
            SomaError::Other(format!("run dir {} has no graph.json", self.dir.display()))
        })?;
        Ok(graph.to_mermaid_with(&self.overlay()?))
    }

    /// Graphviz rendering of the run's graph, annotated with this
    /// run's overlay. Errors if the run has no `graph.json` snapshot.
    pub fn to_graphviz(&self) -> Result<String> {
        let graph = self.graph()?.ok_or_else(|| {
            SomaError::Other(format!("run dir {} has no graph.json", self.dir.display()))
        })?;
        Ok(graph.to_graphviz_with(&self.overlay()?))
    }

    /// The study attached to this run, if any.
    pub fn study(&self) -> Result<Option<Study>> {
        let path = self.dir.join("study.json");
        if !path.exists() {
            return Ok(None);
        }
        Study::load(&path).map(Some)
    }

    /// Trial lifetimes from `study.json` (empty for non-study runs) —
    /// the timeline/gantt substrate for HPO charts.
    pub fn trial_timeline(&self) -> Result<Vec<TrialSpan>> {
        let Some(study) = self.study()? else {
            return Ok(Vec::new());
        };
        Ok(study
            .trials
            .iter()
            .map(|t| TrialSpan {
                trial_id: t.id.clone(),
                state: trial_state_str(&t.state).to_string(),
                started_at: t.started_at,
                finished_at: t.finished_at,
                duration_ms: t.duration_ms,
            })
            .collect())
    }
}

fn trial_state_str(state: &TrialState) -> &'static str {
    match state {
        TrialState::Pending => "pending",
        TrialState::Running => "running",
        TrialState::Completed => "completed",
        TrialState::Pruned { .. } => "pruned",
        TrialState::Failed { .. } => "failed",
    }
}

/// Derive a [`RunInfo`] from manifest + status. `now` is a parameter so
/// crash detection is testable.
fn run_info(
    dir: &Path,
    manifest: RunManifest,
    status: Option<RunStatus>,
    now: DateTime<Utc>,
) -> RunInfo {
    let state = match &status {
        None => "running".to_string(),
        Some(s) => match s.state {
            RunState::Completed => "completed".to_string(),
            RunState::Failed => "failed".to_string(),
            RunState::Running => {
                let last_beat = s.heartbeat_at.unwrap_or(s.updated_at);
                if (now - last_beat).num_seconds() > STALE_HEARTBEAT_SECS {
                    "crashed".to_string()
                } else {
                    "running".to_string()
                }
            }
            // Forward-compat: RunState is non_exhaustive.
            _ => "running".to_string(),
        },
    };
    let finished_at = status.as_ref().and_then(|s| s.finished_at);
    let duration_ms = finished_at
        .map(|end| (end - manifest.created_at).num_milliseconds())
        .filter(|ms| *ms >= 0)
        .map(|ms| ms as u64);
    let kind = serde_json::to_value(manifest.kind)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| "other".to_string());
    RunInfo {
        run_id: manifest.run_id,
        kind,
        name: manifest.name,
        state,
        created_at: manifest.created_at,
        finished_at,
        duration_ms,
        tags: manifest.tags,
        dir: dir.display().to_string(),
    }
}

/// All runs under `<root>/runs/`, newest first. Directories without a
/// readable manifest are skipped.
pub fn list_runs(root: impl AsRef<Path>) -> Result<Vec<RunInfo>> {
    let runs_dir = root.as_ref().join("runs");
    let entries = match fs::read_dir(&runs_dir) {
        Ok(e) => e,
        Err(_) => return Ok(Vec::new()), // no runs yet
    };
    let now = Utc::now();
    let mut infos: Vec<RunInfo> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let dir = e.path();
            let manifest = load_manifest(&dir).ok()?;
            let status = load_status(&dir).ok();
            Some(run_info(&dir, manifest, status, now))
        })
        .collect();
    infos.sort_by_key(|info| std::cmp::Reverse(info.created_at));
    Ok(infos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration as ChronoDuration;
    use somatize_core::tracking::RunKind;

    fn manifest(run_id: &str) -> RunManifest {
        RunManifest::new(run_id, RunKind::Train, "test-run")
    }

    #[test]
    fn run_info_detects_crash_from_stale_heartbeat() {
        let now = Utc::now();
        let stale = RunStatus {
            state: RunState::Running,
            updated_at: now - ChronoDuration::seconds(STALE_HEARTBEAT_SECS + 60),
            heartbeat_at: Some(now - ChronoDuration::seconds(STALE_HEARTBEAT_SECS + 60)),
            finished_at: None,
        };
        let info = run_info(Path::new("/tmp/r"), manifest("r1"), Some(stale), now);
        assert_eq!(info.state, "crashed");

        let fresh = RunStatus::running();
        let info = run_info(Path::new("/tmp/r"), manifest("r1"), Some(fresh), now);
        assert_eq!(info.state, "running");
    }

    #[test]
    fn run_info_duration_and_kind() {
        let now = Utc::now();
        let mut m = manifest("r2");
        m.created_at = now - ChronoDuration::milliseconds(1500);
        let status = RunStatus {
            state: RunState::Completed,
            updated_at: now,
            heartbeat_at: Some(now),
            finished_at: Some(now),
        };
        let info = run_info(Path::new("/tmp/r"), m, Some(status), now);
        assert_eq!(info.state, "completed");
        assert_eq!(info.kind, "train");
        assert_eq!(info.duration_ms, Some(1500));
    }
}
