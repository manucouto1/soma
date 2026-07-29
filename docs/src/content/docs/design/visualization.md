---
title: Visualization
description: Architecture overlays, Plotly figures, HTML reports — and the three-layer design a future GUI reuses.
---

Soma renders what it already records: the run directory
([tracking](/design/tracking/)) is the single source of truth, and
every visualization — a terminal diagram, a notebook figure, an HTML
report, or a future web GUI — is a *reader* of those files. The
strategy has three layers, and GUI reuse is guaranteed at the **data
layer, not the chart layer**: charts are cheap to rewrite in any
framework; parsing event logs, tolerating torn lines, and aggregating
them into per-node timings is not.

## The three layers

### 1. Readers & aggregation (Rust, `soma-runtime`)

`RunReader` (`soma-runtime/src/tracking/reader.rs`) consumes one run
directory and produces chart-ready serde structs — the same shapes
serve PyO3, the CLI, and any future front-end:

| Method | Returns |
|---|---|
| `events()` | every parseable `EventEnvelope`, in log order (torn/unknown lines skipped; `seq` gaps reveal skips) |
| `node_timings()` | per-node execution spans: start/finish wall time (envelope `ts`), duration, outcome, cache tier |
| `cache_activity()` | hit/miss counts, total and per node |
| `metric_series(name)` | metric points from `metrics.jsonl` (event-log fallback) |
| `health_flags()` | `HealthFlag` events with wall time |
| `trial_timeline()` | trial lifetimes from `study.json` |
| `overlay()` | a `GraphOverlay` folding all of the above per node |
| `to_mermaid()` / `to_graphviz()` | the run's graph annotated with its overlay |

`list_runs(root)` scans `<root>/runs/*/` manifests; a `running` status
with a stale heartbeat (> 300 s) reports as `crashed`.

To make run grouping possible, every local execution path emits a
`RunStarted` / `RunCompleted` (or `RunFailed`) bracket sharing one
`run_id` with the node events inside it — previously only the remote
worker path did.

### 2. Graph overlays (Rust, `soma-core`)

`GraphOverlay` (`soma-core/src/viz.rs`) carries per-node execution
facts — status, total duration, cache tier, health flags — and
`Graph::to_mermaid_with` / `to_graphviz_with` fold them into the
rendering: a second label line (`1.2s · mem hit · ⚠ LEAKAGE`) plus a
status `classDef` per node. An empty overlay reproduces the plain
output byte-for-byte, and rendering stays a dependency-free
data→string transform (the overlay is computed elsewhere and passed
in).

```bash
soma graph <run_id> [--format mermaid|dot] [--no-overlay]
```

```text
graph LR
    scaler["scaler<br/>26ms"]
    model["model<br/>27ms · ⚠ DEAD_CHANNELS(2)"]
    scaler --> model
    classDef soma_completed fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
    classDef soma_flagged fill:#fff3e0,stroke:#ef6c00,stroke-width:3px,color:#e65100;
    class scaler soma_completed
    class model soma_flagged
```

### 3. Figures (Python, `soma.viz`, optional extra)

Plotly figures with Optuna-aligned names, installed as methods on
`Study` and `RunView`. The functions are thin (~20-line) skins over the
layer-1 aggregates, so a JS front-end can re-implement any of them
against the same data. They need the `viz` extra:

```bash
pip install 'somatize[viz]'   # plotly + pandas
```

```python
study.plot_optimization_history()   # objective/trial + best-so-far
study.plot_intermediate_values()    # learning curves, pruned dashed
study.plot_parallel_coordinate()    # params → objective, sequential ramp
study.plot_param_importances()      # |Spearman ρ| (fANOVA: future upgrade)
study.plot_timeline()               # trial gantt by state
study.plot_pareto_front()           # multi-objective front

run = soma.runs()[0]
run.plot_metrics()                  # logged metric curves
run.plot_gantt()                    # node spans — where wall time went
run.plot_health()                   # HealthFlag marks, node × step
run.plot_audit("out_grad.norm")     # gradient-audit series per filter
run.plot_channels("encoder")        # channel-correlation heatmap
run.plot_channel_evolution()        # eff. rank / max CKA over training

study.trials_dataframe()            # pandas projections (lazy import)
run.metrics_dataframe()
soma.experiments_dataframe()
```

Chart styling follows one system: a fixed-order, colorblind-validated
categorical palette; a single-hue sequential ramp for magnitude; a
blue↔gray↔red diverging scale for correlations; and a reserved status
set (`completed`/`cached`/`failed`/`running`/`pruned`) that matches the
mermaid/graphviz overlay colors, so a run reads the same in every
rendering.

## The HTML report

```bash
soma report <run_id|path> [-o report.html] [--inline] [--open]
```

One self-contained file per run: manifest header, the annotated DAG,
efficiency tiles (node compute, cache hits/misses), metric curves, the
node gantt, the full HPO section with trial table (for studies), and
the health section. `--inline` embeds plotly.js from the installed
package so the file opens with no network (the DAG ships as mermaid
source in that mode); the default uses pinned CDNs.

### Front-end data contract

Every dataset the report renders is embedded as
`<script type="application/json" id="soma-data-…">` blobs. **These ids
and shapes are the contract a future live GUI reads** — the report is
just their static packaging:

| Blob id | Shape (serde source) |
|---|---|
| `soma-data-info` | `RunInfo` |
| `soma-data-manifest` | `RunManifest` |
| `soma-data-overlay` | `GraphOverlay` |
| `soma-data-node-timings` | `Vec<NodeSpan>` |
| `soma-data-cache` | `CacheActivity` |
| `soma-data-metrics` | `Vec<MetricPoint>` |
| `soma-data-health-flags` | `Vec<HealthFlagRecord>` |
| `soma-data-trial-timeline` | `Vec<TrialSpan>` |

Charts are Plotly figure JSON under `soma-fig-<name>` ids
(`history`, `intermediate`, `parallel-coords`, `importances`,
`timeline`, `pareto`, `metrics`, `gantt`, `health`, `audit`,
`channels`).

## Timing semantics

Start events carry no timestamp by design: sinks are synchronous, so
the envelope `ts` written to `events.jsonl` **is** the wall clock of
emission. Consequences:

- Gantt/waterfall charts read the run directory, never the live
  (lossy, envelope-less) `on_event` callback.
- `NodeCompleted.duration` / `NodeCacheHit.load_time` are the precise
  per-execution durations; envelope deltas are the layout positions.

## Deferred

Documented, intentionally not built yet: `soma ui` (a live local
server tailing run dirs — every piece it needs now exists), fANOVA
importances, `NodeProgress`/`ParetoUpdated` emitters, historical
per-node cost from the persistent cache's `ActionResult.compute_ms`,
a Python-implementable `EventSink`, parquet compaction of metrics.
