---
title: Event System
description: Structured events at three levels for monitoring, visualization, and agent decision-making.
---

## Overview

Every execution in Soma produces a stream of structured events. Events are the **nervous system** of Soma -- they enable real-time monitoring, visualization, logging, and agent decision-making without coupling the runtime to any specific consumer.

Events are emitted at three hierarchical levels:

```
Study (optimization session)
  └── Trial (one hyperparameter evaluation)
       └── Run (one pipeline execution)
            └── Node events (per-filter)
```

## Event Types

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Event {
    // ══════════════════════════════════════════
    // Level 1: Graph execution (per run)
    // ══════════════════════════════════════════

    /// A graph run has started
    RunStarted {
        run_id: RunId,
        plan_summary: PlanSummary,
    },

    /// A filter node has started execution
    NodeStarted {
        run_id: RunId,
        node_id: NodeId,
        kind: FilterKind,
    },

    /// A filter node reports progress (0.0 to 1.0)
    NodeProgress {
        run_id: RunId,
        node_id: NodeId,
        progress: f32,
    },

    /// A filter node's result was loaded from cache
    NodeCacheHit {
        run_id: RunId,
        node_id: NodeId,
        key: CacheKey,
        tier: CacheTier,
        load_time: Duration,
    },

    /// A filter node completed successfully
    NodeCompleted {
        run_id: RunId,
        node_id: NodeId,
        duration: Duration,
        output_summary: String,
    },

    /// A filter node failed
    NodeFailed {
        run_id: RunId,
        node_id: NodeId,
        error: String,
    },

    /// The graph run completed
    RunCompleted {
        run_id: RunId,
        duration: Duration,
    },

    /// The graph run failed
    RunFailed {
        run_id: RunId,
        error: String,
    },

    // ══════════════════════════════════════════
    // Level 2: Trial execution (per hyperparameter set)
    // ══════════════════════════════════════════

    /// A new trial has started
    TrialStarted {
        study_id: StudyId,
        trial_id: TrialId,
        params: serde_json::Value,
    },

    /// A trial reports an intermediate metric (for pruning and live curves)
    TrialMetric {
        study_id: StudyId,
        trial_id: TrialId,
        metric: MetricRecord,
    },

    /// A trial was pruned (stopped early)
    TrialPruned {
        study_id: StudyId,
        trial_id: TrialId,
        step: usize,
        reason: String,
    },

    /// A trial completed successfully
    TrialCompleted {
        study_id: StudyId,
        trial_id: TrialId,
        final_metrics: Vec<MetricRecord>,
    },

    /// A trial failed
    TrialFailed {
        study_id: StudyId,
        trial_id: TrialId,
        error: String,
    },

    // ══════════════════════════════════════════
    // Level 3: Study execution (optimization session)
    // ══════════════════════════════════════════

    /// An optimization study has started
    StudyStarted {
        study_id: StudyId,
        name: String,
        total_trials: usize,
    },

    /// Study progress update
    StudyProgress {
        study_id: StudyId,
        completed: usize,
        total: usize,
        best_value: f64,
    },

    /// The best trial has been updated
    BestUpdated {
        study_id: StudyId,
        trial_id: TrialId,
        value: f64,
        params: serde_json::Value,
    },

    /// The Pareto front has changed (multi-objective)
    ParetoUpdated {
        study_id: StudyId,
        front_size: usize,
    },

    /// The study completed
    StudyCompleted {
        study_id: StudyId,
        best_trial_id: TrialId,
        best_value: f64,
    },

    // ══════════════════════════════════════════
    // Level 4: Population-Based Training
    // ══════════════════════════════════════════
    // GenerationStarted / GenerationCompleted / MemberExploited —
    // emitted by PbtRunner.

    // ══════════════════════════════════════════
    // Level 5: Training telemetry (native training loop)
    // ══════════════════════════════════════════

    /// A training epoch started / completed
    EpochStarted { run_id: RunId, epoch: usize, total_epochs: Option<usize> },
    EpochCompleted { run_id: RunId, epoch: usize, metrics: Vec<MetricRecord> },

    /// One optimizer step completed (coarse liveness marker)
    StepCompleted { run_id: RunId, step: usize, epoch: Option<usize> },

    /// A metric reported outside a trial (training loops, evaluation)
    MetricReported {
        run_id: RunId,
        metric: MetricRecord,
        node_id: Option<NodeId>,
        trial_id: Option<TrialId>,
    },

    /// A training-health diagnostic fired for a node
    /// (e.g. DEAD_CHANNELS, IGNORED_CHANNELS, LEAKAGE, NONFINITE)
    HealthFlag {
        run_id: RunId,
        node_id: NodeId,
        step: usize,
        flag: String,
        detail: String,
    },
}
```

## Event Bus

The bus offers two delivery paths with different guarantees:

- **Sinks** (`add_sink`) are invoked synchronously on the emitting
  thread *before* the broadcast — lossless and strictly ordered. This
  is how trackers persist events: a lagging disk never drops a line.
- **Subscribers** (`subscribe`) receive via a tokio broadcast channel —
  live but lossy under lag. Suitable for display and relays only.

```rust
pub struct EventBus {
    sender: broadcast::Sender<Event>,
    sinks: RwLock<Vec<Arc<dyn EventSink>>>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self { .. }

    /// Register a lossless sink, called synchronously on every emit
    pub fn add_sink(&self, sink: Arc<dyn EventSink>) { .. }

    /// Emit: sinks first (lossless), then all subscribers
    pub fn emit(&self, event: Event) -> usize { .. }

    /// Subscribe to receive events (lossy under lag)
    pub fn subscribe(&self) -> broadcast::Receiver<Event> { .. }
}

/// soma-core::tracking — implemented by any tracking backend
pub trait EventSink: Send + Sync {
    fn record(&self, event: &Event);
    fn flush(&self) {}
}
```

### Built-in Sink: JSONL Run Logs

`JsonlEventSink` (soma-runtime) appends every event as one JSON line to
`events.jsonl`, wrapped in an envelope `{seq, ts, ...event}` with a
monotonic sequence number, and tees metric-bearing events
(`TrialMetric`, `MetricReported`) into a flat `metrics.jsonl`.
`LocalTracker` owns a run directory (`.soma/runs/<run_id>/`) holding
those logs plus `manifest.json` (write-once) and `status.json`
(heartbeat). See the tracking design page for the full layout.

```rust
let tracker = LocalTracker::create(".soma", RunKind::Fit, "my-run")?;
bus.add_sink(tracker.sink());
// ... run ...
tracker.finalize(RunState::Completed)?;
```

A WebSocket relay for a live UI would be a broadcast subscriber, not a
sink — dropping frames under lag is acceptable there because the run
directory remains the source of truth.

## Events for Visualization

The three levels of events map directly to UI components. This mapping
is implemented today by the read-side stack (see
[Visualization](/soma/design/visualization/)): `RunReader` aggregates the
event log into node timings/cache activity/metric series,
`RunReader::overlay()` feeds `Graph::to_mermaid_with` for the annotated
DAG, and `soma.viz` renders the trial/study charts; `soma report`
packages all of it into one HTML file.

### Graph Level → DAG Visualization

```
Events used:
  NodeStarted   → highlight node as "running"
  NodeProgress  → show progress bar in node
  NodeCacheHit  → highlight node as "cached" with tier badge
  NodeCompleted → highlight node as "done" with duration
  NodeFailed    → highlight node as "error"
  RunCompleted  → show total duration

UI: Interactive DAG where nodes light up as they execute
```

### Trial Level → Learning Curves & Metrics

```
Events used:
  TrialStarted  → add new line to chart
  TrialMetric   → add point to line (step, value)
  TrialPruned   → mark line as pruned (dashed)
  TrialCompleted→ finalize line, show in table

UI: Real-time learning curves (loss/metric vs step/epoch)
    Updated live as trials report metrics
```

### Study Level → Optimization Dashboard

```
Events used:
  StudyProgress → update progress bar
  BestUpdated   → highlight best config in parallel coordinates
  ParetoUpdated → update Pareto front scatter plot
  StudyCompleted→ show final results, importance plot

UI components:
  - Parallel coordinates (params vs final metric)
  - Pareto front (multi-objective scatter)
  - Parameter importance (computed from completed trials)
  - Trial timeline (gantt chart of trial durations)
  - Progress bar (completed / total trials)
```

## Who Emits What

Filters never emit events directly; emission happens at the
orchestration layer:

- **Node/run events (level 1)** — node events are emitted by the
  executor and `LocalRunner`; the `RunStarted`/`RunCompleted`/
  `RunFailed` bracket is emitted by every entry point
  (`GraphSession::fit`/`run`, the Python `Graph.fit`/`run`, and the
  worker), sharing one `run_id` with the node events inside it.
  `NodeProgress` is reserved but currently has no emitter.
- **Trial/study events (levels 2–3)** — emitted by `StudyRunner`.
  Intermediate metrics flow through the trial handle
  (`TrialContext::report(name, value, step)`), which emits
  `TrialMetric` and consults the pruner.
- **PBT events (level 4)** — emitted by `PbtRunner`.
- **Training telemetry (level 5)** — emitted from the Python native
  training loop via the `Graph.emit_event` binding (epoch/step markers,
  `MetricReported`, `HealthFlag` from the gradient audit).

## Event Serialization

All events are serializable to JSON (via serde), enabling:

- **JSONL logging**: One JSON object per line for offline analysis
- **WebSocket streaming**: Real-time relay to browser UIs
- **Message queue publishing**: Integration with external monitoring systems
- **Agent consumption**: Structured input for autonomous decision-making

In a run directory each line carries the tracker envelope (`seq`, `ts`)
with the event's `event_type` tag flattened alongside its fields:

```json
{"seq":0,"ts":"2026-07-26T10:15:02Z","event_type":"NodeStarted","run_id":"run_001","node_id":"scaler","kind":"Trainable"}
{"seq":1,"ts":"2026-07-26T10:15:02Z","event_type":"NodeCacheHit","run_id":"run_001","node_id":"scaler","key":"abc123","tier":"Memory","load_time":0}
{"seq":2,"ts":"2026-07-26T10:15:03Z","event_type":"TrialMetric","study_id":"study_001","trial_id":"trial_042","metric":{"name":"f1","value":0.847,"step":15,"timestamp":"2026-07-26T10:15:03Z"}}
{"seq":3,"ts":"2026-07-26T10:15:03Z","event_type":"BestUpdated","study_id":"study_001","trial_id":"trial_042","value":0.847,"params":{"lr":0.003,"C":1.5}}
```
