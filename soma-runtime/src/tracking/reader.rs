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

use crate::optimizer::study_io::StudyIo;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use somatize_core::error::{Result, SomaError};
use somatize_core::graph::Graph;
use somatize_core::optimizer::study::{Study, TrialState};
use somatize_core::tracking::event::Event;
use somatize_core::tracking::{EventEnvelope, RunManifest, RunState, RunStatus};
use somatize_core::viz::{GraphOverlay, NodeStatus};
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::local_tracker::{load_manifest, load_status};

/// A `Running` status whose heartbeat is older than this is reported as
/// crashed: the process died without finalizing.
pub const STALE_HEARTBEAT_SECS: i64 = 300;

/// Reader over one run directory.
///
/// `events.jsonl` is parsed **once** per reader and memoized. Every
/// aggregate below is a fold over the same envelopes, and each used to
/// re-open and re-parse the file for itself — so `summarize`, which calls
/// five of them, was five full reads and five full parses of one file
/// (D-63). The memo is per reader, so a reader is a snapshot: a run still
/// being written needs a fresh one to see new events.
pub struct RunReader {
    dir: PathBuf,
    envelopes: OnceCell<Vec<EventEnvelope>>,
}

/// Listing entry for one run: manifest identity plus derived liveness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    /// Run identifier from the manifest; also tags every event of the run.
    pub run_id: String,
    /// Manifest `kind` as its snake_case string (`fit`, `train`, `study`, …).
    pub kind: String,
    /// Human-readable name from the manifest.
    pub name: String,
    /// `running` | `completed` | `failed` | `crashed`.
    pub state: String,
    /// When the run directory was created.
    pub created_at: DateTime<Utc>,
    /// When the run finalized, if it did.
    pub finished_at: Option<DateTime<Utc>>,
    /// Wall time from creation to finish, when finished.
    pub duration_ms: Option<u64>,
    /// Free-form tags from the manifest.
    pub tags: Vec<String>,
    /// Absolute path of the run directory.
    pub dir: String,
}

/// One execution span of a node, in event order. A node appears once
/// per execution (re-runs and stream chunks produce separate spans).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeSpan {
    /// Graph node this span belongs to.
    pub node_id: String,
    /// Envelope timestamp of the opening event.
    pub started_ts: Option<DateTime<Utc>>,
    /// Envelope timestamp of the closing event.
    pub finished_ts: Option<DateTime<Utc>>,
    /// Wall time between the two envelope timestamps.
    pub duration_ms: Option<u64>,
    /// `completed` | `failed` | `cache_hit` | `running`.
    pub outcome: String,
    /// Cache tier that served a hit (`memory`, `local`, …).
    pub cache_tier: Option<String>,
    /// Failure message when `outcome` is `failed`.
    pub error: Option<String>,
    /// The node was a step (from `NodeStarted`); defaults for spans
    /// reconstructed from logs that predate the field.
    #[serde(default)]
    pub effectful: bool,
}

/// Per-run cache effectiveness, reconstructed from hit/miss events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheActivity {
    /// Cache hits across the whole run.
    pub hits: u64,
    /// Cache misses across the whole run.
    pub misses: u64,
    /// Per-node breakdown, keyed by node id.
    pub by_node: BTreeMap<String, NodeCacheCounts>,
}

/// One node's share of [`CacheActivity`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeCacheCounts {
    /// Hits recorded for this node.
    pub hits: u64,
    /// Misses recorded for this node.
    pub misses: u64,
    /// Tier that served the most recent hit (`memory`, `local`, …).
    pub last_tier: Option<String>,
}

/// One line of `metrics.jsonl` (also derivable from events).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    /// Wall time the point was logged.
    pub ts: DateTime<Utc>,
    /// Metric name (`loss`, `accuracy`, …).
    pub name: String,
    /// The recorded scalar.
    pub value: f64,
    /// Logger-supplied step index within the run.
    pub step: u64,
    /// Owning trial, when logged inside a study.
    #[serde(default)]
    pub trial_id: Option<String>,
    /// Emitting node, when the metric came from inside a node.
    #[serde(default)]
    pub node_id: Option<String>,
}

/// One `HealthFlag` event with its wall time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthFlagRecord {
    /// Wall time the flag was raised.
    pub ts: DateTime<Utc>,
    /// Node the flag fired on — hierarchical (`node/module.path`) when it
    /// came from an intra-node audit hook.
    pub node_id: String,
    /// Training step at which the flag fired.
    pub step: usize,
    /// Flag family name, as emitted by the audit.
    pub flag: String,
    /// Human-readable description of what was detected.
    pub detail: String,
}

/// Agent-level activity for one run, aggregated from the step events
/// (`AgentTurnStarted`, `EffectCompleted`, `ToolCalled`, `Suspended`,
/// `AgentStepCompleted`, …). Empty `by_node` means the run had no agent
/// steps — or predates their telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgenticActivity {
    /// Agent turns across all step nodes.
    pub turns: u64,
    /// Prompt tokens consumed across all step nodes.
    pub input_tokens: u64,
    /// Completion tokens produced across all step nodes.
    pub output_tokens: u64,
    /// Effects performed or replayed across all step nodes.
    pub effects: u64,
    /// Of `effects`, how many were served from the journal (a resumed
    /// or replayed run should be nearly all replays).
    pub replayed: u64,
    /// Tool invocations across all step nodes.
    pub tool_calls: u64,
    /// Step nodes that completed successfully.
    pub steps_completed: u64,
    /// Step nodes that completed failed.
    pub steps_failed: u64,
    /// `Suspended` transitions observed across the run.
    pub suspensions: u64,
    /// Per-step-node breakdown, keyed by node id.
    pub by_node: BTreeMap<String, AgentNodeActivity>,
}

/// One step node's share of the run's agentic work. Spawned instances
/// appear under their own hierarchical ids (`parent/label`).
///
/// Token, turn and duration totals come from the accounting events
/// (`AgentStepCompleted`, whose totals are cumulative, or the cost a
/// `Suspended` carries when nothing completed after it) — never by
/// summing both, which would double-count a resumed run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentNodeActivity {
    /// Agent turns this node ran.
    pub turns: u64,
    /// Prompt tokens this node consumed.
    pub input_tokens: u64,
    /// Completion tokens this node produced.
    pub output_tokens: u64,
    /// Wall time from the accounting events (see the type docs).
    pub duration_ms: u64,
    /// Effects this node performed or replayed.
    pub effects: u64,
    /// Effect counts by label (`llm:<model>`, `tool:<name>`, …).
    pub effects_by_label: BTreeMap<String, u64>,
    /// Effects that completed carrying an error result.
    pub effect_errors: u64,
    /// Of `effects`, how many were served from the journal.
    pub replayed: u64,
    /// Tool invocations this node made.
    pub tool_calls: u64,
    /// Tool invocations that returned an error.
    pub tool_errors: u64,
    /// Control handoffs this node emitted to other nodes.
    pub handoffs_out: u64,
    /// Times this node suspended awaiting external input.
    pub suspensions: u64,
    /// Instances this node fanned out (sum over its `AgentSpawned`s).
    pub spawned: u64,
    /// `AgentStepCompleted` events with `failed: false` / `true`.
    pub completions: u64,
    /// Completions that reported `failed: true`.
    pub failures: u64,
}

/// One effect's execution inside a step — the gantt substrate for
/// agent runs, the per-effect analogue of [`NodeSpan`]. An unclosed
/// span (`outcome: "running"`) means the run died mid-effect.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EffectSpan {
    /// Step node the effect ran inside.
    pub node_id: String,
    /// Turn index within the step's loop.
    pub turn: usize,
    /// `Effect::label()` — e.g. `llm:qwen2.5:14b`, `tool:search`.
    pub effect: String,
    /// Envelope timestamp of the opening event.
    pub started_ts: Option<DateTime<Utc>>,
    /// Envelope timestamp of the closing event.
    pub finished_ts: Option<DateTime<Utc>>,
    /// Wall time between the two envelope timestamps.
    pub duration_ms: Option<u64>,
    /// Served from the journal instead of being performed.
    pub replayed: bool,
    /// The effect completed carrying an error result.
    pub is_error: bool,
    /// `completed` | `running`.
    pub outcome: String,
}

/// One trial's lifetime, from `study.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialSpan {
    /// Trial identifier from `study.json`.
    pub trial_id: String,
    /// `completed` | `pruned` | `failed` | `running` | `pending`.
    pub state: String,
    /// When the trial started, if it did.
    pub started_at: Option<DateTime<Utc>>,
    /// When the trial finished, if it did.
    pub finished_at: Option<DateTime<Utc>>,
    /// Wall time between the two, when both are known.
    pub duration_ms: Option<u64>,
}

impl RunReader {
    /// Open a run directory. Fails only if the manifest is missing or
    /// unreadable — everything else is tolerated per-file.
    pub fn open(run_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = run_dir.as_ref().to_path_buf();
        load_manifest(&dir)?;
        Ok(Self {
            dir,
            envelopes: OnceCell::new(),
        })
    }

    /// The run directory this reader was opened on.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Parse `manifest.json` — the run's immutable identity record.
    pub fn manifest(&self) -> Result<RunManifest> {
        load_manifest(&self.dir)
    }

    /// Parse `status.json` — the latest state + heartbeat snapshot.
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
    ///
    /// Parsed once and memoized — see the type's note. This clones; the
    /// aggregates below fold over the memo without one.
    pub fn events(&self) -> Result<Vec<EventEnvelope>> {
        self.envelopes().map(<[EventEnvelope]>::to_vec)
    }

    /// The memoized envelopes. A read error is not cached: an empty or
    /// unreadable `events.jsonl` reads as no events, which is what an
    /// unfinished run legitimately has.
    fn envelopes(&self) -> Result<&[EventEnvelope]> {
        if let Some(cached) = self.envelopes.get() {
            return Ok(cached);
        }
        let path = self.dir.join("events.jsonl");
        let parsed = match fs::File::open(&path) {
            Err(_) => Vec::new(), // no events yet
            Ok(file) => {
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
                envelopes
            }
        };
        Ok(self.envelopes.get_or_init(|| parsed))
    }

    /// Per-node execution spans in event order — the gantt/overlay
    /// substrate. Cache hits are standalone spans (a hit node never
    /// starts); an unclosed span means the run died mid-node.
    pub fn node_timings(&self) -> Result<Vec<NodeSpan>> {
        let mut spans: Vec<NodeSpan> = Vec::new();
        let mut open: BTreeMap<String, usize> = BTreeMap::new();
        for env in self.envelopes()? {
            match &env.event {
                Event::NodeStarted {
                    node_id, effectful, ..
                } => {
                    open.insert(node_id.clone(), spans.len());
                    spans.push(NodeSpan {
                        node_id: node_id.clone(),
                        started_ts: Some(env.ts),
                        finished_ts: None,
                        duration_ms: None,
                        outcome: "running".into(),
                        cache_tier: None,
                        error: None,
                        effectful: *effectful,
                    });
                }
                Event::NodeCacheHit {
                    node_id,
                    tier,
                    load_time,
                    ..
                } => {
                    spans.push(NodeSpan {
                        node_id: node_id.clone(),
                        started_ts: Some(env.ts),
                        finished_ts: Some(env.ts),
                        duration_ms: Some(load_time.as_millis() as u64),
                        outcome: "cache_hit".into(),
                        cache_tier: Some(format!("{tier:?}").to_lowercase()),
                        error: None,
                        effectful: false,
                    });
                }
                Event::NodeCompleted {
                    node_id, duration, ..
                } => {
                    let idx = open.remove(node_id.as_str());
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
                                effectful: false,
                            });
                            spans.last_mut().expect("just pushed")
                        }
                    };
                    span.finished_ts = Some(env.ts);
                    span.duration_ms = Some(duration.as_millis() as u64);
                    span.outcome = "completed".into();
                }
                Event::NodeFailed { node_id, error, .. } => {
                    let idx = open.remove(node_id.as_str());
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
                                effectful: false,
                            });
                            spans.last_mut().expect("just pushed")
                        }
                    };
                    span.finished_ts = Some(env.ts);
                    span.outcome = "failed".into();
                    span.error = Some(error.clone());
                }
                _ => {}
            }
        }
        Ok(spans)
    }

    /// Cache hit/miss counts, total and per node.
    pub fn cache_activity(&self) -> Result<CacheActivity> {
        let mut activity = CacheActivity::default();
        for env in self.envelopes()? {
            match &env.event {
                Event::NodeCacheHit { node_id, tier, .. } => {
                    activity.hits += 1;
                    let counts = activity.by_node.entry(node_id.clone()).or_default();
                    counts.hits += 1;
                    counts.last_tier = Some(format!("{tier:?}").to_lowercase());
                }
                Event::NodeCacheMiss { node_id, .. } => {
                    activity.misses += 1;
                    activity.by_node.entry(node_id.clone()).or_default().misses += 1;
                }
                // A streamed node probes once per chunk and reports the
                // total, once. Without this arm a streamed run read as
                // zero cache activity while its node spans said otherwise.
                Event::NodeCacheSummary {
                    node_id,
                    hits,
                    misses,
                    ..
                } => {
                    activity.hits += hits;
                    activity.misses += misses;
                    let counts = activity.by_node.entry(node_id.clone()).or_default();
                    counts.hits += hits;
                    counts.misses += misses;
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
            for env in self.envelopes()? {
                match &env.event {
                    Event::TrialMetric {
                        trial_id, metric, ..
                    } => points.push(MetricPoint {
                        ts: metric.timestamp,
                        name: metric.name.clone(),
                        value: metric.value,
                        step: metric.step as u64,
                        trial_id: Some(trial_id.clone()),
                        node_id: None,
                    }),
                    Event::MetricReported {
                        metric,
                        node_id,
                        trial_id,
                        ..
                    } => points.push(MetricPoint {
                        ts: metric.timestamp,
                        name: metric.name.clone(),
                        value: metric.value,
                        step: metric.step as u64,
                        trial_id: trial_id.clone(),
                        node_id: node_id.clone(),
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
        for env in self.envelopes()? {
            if let Event::HealthFlag {
                node_id,
                step,
                flag,
                detail,
                ..
            } = env.event.clone()
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

    /// Agent-level activity, aggregated per step node.
    ///
    /// Accounting rule: `AgentStepCompleted` totals are authoritative
    /// (they are cumulative — a resumed run re-counts its replayed
    /// effects). A `Suspended` cost stands in only until a later
    /// completion for the same node supersedes it, and turn counts seen
    /// on the wire (`AgentTurnStarted`) are used only for nodes that
    /// died without any accounting event at all.
    pub fn agentic_activity(&self) -> Result<AgenticActivity> {
        #[derive(Default)]
        struct Pending {
            turns: u64,
            input_tokens: u64,
            output_tokens: u64,
            duration_ms: u64,
        }
        let mut by_node: BTreeMap<String, AgentNodeActivity> = BTreeMap::new();
        let mut pending: BTreeMap<String, Pending> = BTreeMap::new();
        let mut observed_turns: BTreeMap<String, u64> = BTreeMap::new();

        for env in self.envelopes()? {
            match env.event.clone() {
                Event::AgentTurnStarted { node_id, turn, .. } => {
                    let seen = observed_turns.entry(node_id).or_default();
                    *seen = (*seen).max(turn as u64 + 1);
                }
                Event::EffectCompleted {
                    node_id,
                    effect,
                    replayed,
                    is_error,
                    ..
                } => {
                    let node = by_node.entry(node_id).or_default();
                    node.effects += 1;
                    *node.effects_by_label.entry(effect).or_default() += 1;
                    if replayed {
                        node.replayed += 1;
                    }
                    if is_error {
                        node.effect_errors += 1;
                    }
                }
                Event::ToolCalled {
                    node_id, is_error, ..
                } => {
                    let node = by_node.entry(node_id).or_default();
                    node.tool_calls += 1;
                    if is_error {
                        node.tool_errors += 1;
                    }
                }
                Event::Handoff { from, .. } => {
                    by_node.entry(from).or_default().handoffs_out += 1;
                }
                Event::Suspended {
                    node_id,
                    turns,
                    duration,
                    input_tokens,
                    output_tokens,
                    ..
                } => {
                    by_node.entry(node_id.clone()).or_default().suspensions += 1;
                    pending.insert(
                        node_id,
                        Pending {
                            turns: turns as u64,
                            input_tokens,
                            output_tokens,
                            duration_ms: duration.as_millis() as u64,
                        },
                    );
                }
                Event::AgentSpawned {
                    node_id, children, ..
                } => {
                    by_node.entry(node_id).or_default().spawned += children.len() as u64;
                }
                Event::AgentStepCompleted {
                    node_id,
                    turns,
                    duration,
                    input_tokens,
                    output_tokens,
                    failed,
                    ..
                } => {
                    let node = by_node.entry(node_id.clone()).or_default();
                    node.turns += turns as u64;
                    node.input_tokens += input_tokens;
                    node.output_tokens += output_tokens;
                    node.duration_ms += duration.as_millis() as u64;
                    if failed {
                        node.failures += 1;
                    } else {
                        node.completions += 1;
                    }
                    pending.remove(&node_id);
                }
                _ => {}
            }
        }

        for (node_id, p) in pending {
            let node = by_node.entry(node_id).or_default();
            node.turns += p.turns;
            node.input_tokens += p.input_tokens;
            node.output_tokens += p.output_tokens;
            node.duration_ms += p.duration_ms;
        }
        for (node_id, seen) in observed_turns {
            let node = by_node.entry(node_id).or_default();
            if node.turns == 0 {
                node.turns = seen;
            }
        }

        let mut totals = AgenticActivity::default();
        for node in by_node.values() {
            totals.turns += node.turns;
            totals.input_tokens += node.input_tokens;
            totals.output_tokens += node.output_tokens;
            totals.effects += node.effects;
            totals.replayed += node.replayed;
            totals.tool_calls += node.tool_calls;
            totals.steps_completed += node.completions;
            totals.steps_failed += node.failures;
            totals.suspensions += node.suspensions;
        }
        totals.by_node = by_node;
        Ok(totals)
    }

    /// Per-effect execution spans in event order — the gantt substrate
    /// for agent runs. Concurrent same-label effects within a turn are
    /// matched first-in-first-out, which is the order the driver
    /// reports completions in.
    pub fn agentic_timeline(&self) -> Result<Vec<EffectSpan>> {
        let mut spans: Vec<EffectSpan> = Vec::new();
        let mut open: BTreeMap<(String, usize, String), Vec<usize>> = BTreeMap::new();
        for env in self.envelopes()? {
            match env.event.clone() {
                Event::EffectRequested {
                    node_id,
                    turn,
                    effect,
                    ..
                } => {
                    open.entry((node_id.clone(), turn, effect.clone()))
                        .or_default()
                        .push(spans.len());
                    spans.push(EffectSpan {
                        node_id,
                        turn,
                        effect,
                        started_ts: Some(env.ts),
                        finished_ts: None,
                        duration_ms: None,
                        replayed: false,
                        is_error: false,
                        outcome: "running".into(),
                    });
                }
                Event::EffectCompleted {
                    node_id,
                    turn,
                    effect,
                    duration,
                    replayed,
                    is_error,
                    ..
                } => {
                    let key = (node_id.clone(), turn, effect.clone());
                    let idx = open
                        .get_mut(&key)
                        .filter(|v| !v.is_empty())
                        .map(|v| v.remove(0));
                    let span = match idx {
                        Some(i) => &mut spans[i],
                        None => {
                            spans.push(EffectSpan {
                                node_id,
                                turn,
                                effect,
                                started_ts: None,
                                finished_ts: None,
                                duration_ms: None,
                                replayed: false,
                                is_error: false,
                                outcome: String::new(),
                            });
                            spans.last_mut().expect("just pushed")
                        }
                    };
                    span.finished_ts = Some(env.ts);
                    span.duration_ms = Some(duration.as_millis() as u64);
                    span.replayed = replayed;
                    span.is_error = is_error;
                    span.outcome = "completed".into();
                }
                _ => {}
            }
        }
        Ok(spans)
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

    /// Self-contained SVG rendering of the run's graph with this run's
    /// overlay — no JavaScript, safe for notebook/report embedding.
    /// Errors if the run has no `graph.json` snapshot.
    pub fn to_svg(&self) -> Result<String> {
        let graph = self.graph()?.ok_or_else(|| {
            SomaError::Other(format!("run dir {} has no graph.json", self.dir.display()))
        })?;
        Ok(graph.to_svg_with(&self.overlay()?))
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
