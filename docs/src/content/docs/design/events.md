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
    // Level 1: Pipeline execution (per run)
    // ══════════════════════════════════════════

    /// A pipeline run has started
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

    /// The pipeline run completed
    RunCompleted {
        run_id: RunId,
        duration: Duration,
    },

    /// The pipeline run failed
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
        trial: Trial,
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
        trial: Trial,
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
        study: StudySummary,
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
        trial: Trial,
    },

    /// The Pareto front has changed (multi-objective)
    ParetoUpdated {
        study_id: StudyId,
        front: Vec<Trial>,
    },

    /// The study completed
    StudyCompleted {
        study_id: StudyId,
        best_trials: Vec<Trial>,
    },
}
```

## Event Bus

The runtime broadcasts events via an async channel. Multiple subscribers can listen concurrently:

```rust
pub struct EventBus {
    sender: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self { .. }

    /// Emit an event to all subscribers
    pub fn emit(&self, event: Event) { .. }

    /// Subscribe to receive events
    pub fn subscribe(&self) -> broadcast::Receiver<Event> { .. }
}
```

### Built-in Subscribers

```rust
// Console logger
let logger = ConsoleEventLogger::new(bus.subscribe());
tokio::spawn(logger.run());

// JSON file logger
let file_logger = JsonFileLogger::new(bus.subscribe(), "events.jsonl");
tokio::spawn(file_logger.run());

// WebSocket relay (for UI)
let ws = WebSocketRelay::new(bus.subscribe(), ws_sender);
tokio::spawn(ws.run());

// Agent subscriber (for autonomous decision-making)
let agent_inbox = AgentEventCollector::new(bus.subscribe());
tokio::spawn(agent_inbox.run());
```

## Events for Visualization

The three levels of events map directly to UI components:

### Pipeline Level → DAG Visualization

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

## Context: How Filters Emit Events

Filters don't emit events directly. They interact with the `Context` object:

```rust
pub struct Context {
    store: ContextStore,
    event_bus: EventBus,
    run_id: RunId,
    node_id: NodeId,
}

impl Context {
    /// Report a metric (emits TrialMetric event)
    pub fn report_metric(&self, name: &str, value: f64, step: usize) -> Result<()> {
        self.event_bus.emit(Event::TrialMetric {
            study_id: self.study_id,
            trial_id: self.trial_id,
            metric: MetricRecord {
                name: name.to_string(),
                value,
                step,
                timestamp: Utc::now(),
            },
        });

        // Check if pruner wants to stop this trial
        if self.pruner.should_prune(name, value, step) {
            return Err(SomaError::Pruned { step, reason: "below median".into() });
        }
        Ok(())
    }

    /// Report progress (emits NodeProgress event)
    pub fn report_progress(&self, progress: f32) {
        self.event_bus.emit(Event::NodeProgress {
            run_id: self.run_id,
            node_id: self.node_id,
            progress,
        });
    }

    /// Get input data from predecessor nodes
    pub fn input<T: FromValue>(&self) -> Result<T> { .. }

    /// Access the cache store
    pub fn cache(&self) -> &dyn CacheStore { .. }
}
```

## Event Serialization

All events are serializable to JSON (via serde), enabling:

- **JSONL logging**: One JSON object per line for offline analysis
- **WebSocket streaming**: Real-time relay to browser UIs
- **Message queue publishing**: Integration with external monitoring systems
- **Agent consumption**: Structured input for autonomous decision-making

```json
{"type":"NodeStarted","run_id":"run_001","node_id":"scaler","kind":"Trainable"}
{"type":"NodeCacheHit","run_id":"run_001","node_id":"scaler","key":"abc123","tier":"Memory","load_time_ms":0.2}
{"type":"TrialMetric","study_id":"study_001","trial_id":"trial_042","metric":{"name":"f1","value":0.847,"step":15}}
{"type":"BestUpdated","study_id":"study_001","trial":{"id":"trial_042","params":{"lr":0.003,"C":1.5}}}
```
