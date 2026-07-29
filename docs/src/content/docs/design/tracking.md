---
title: Experiment Tracking
description: Run directories, event sinks, study persistence and training diagnostics on disk.
---

Soma tracks experiments in **run directories**: append-only logs plus
small atomic JSON files under `.soma/`. There is no database and no
server — a future web front-end (or a `wandb sync`-style uploader) is a
*reader* of these files, never a dependency of training. The design
borrows the patterns that survived in MLflow / W&B / Optuna / Sacred:
an event log as the source of truth, atomic summary files beside it,
pluggable sinks, and append-only resume semantics.

## Layout

```
.soma/
├── experiments.jsonl              # journal: one ExperimentRecord per line
│                                  # (FileKnowledgeBase; read by soma.experiments()
│                                  # and the MCP server's knowledge tools)
└── runs/
    └── study_20260726T101502_a3f1/
        ├── manifest.json          # write-once, atomic: schema_version, run_id,
        │                          # kind, name, created_at, soma/python version,
        │                          # hostname, git {sha, branch, dirty}, entrypoint,
        │                          # argv, cwd, seeds, tags, graph summary
        ├── status.json            # atomic rewrite: {state, updated_at,
        │                          # heartbeat_at, finished_at} — RUNNING with a
        │                          # stale heartbeat means the process died
        ├── graph.json             # serialized topology (nodes/edges)
        ├── graph.mmd              # Graph::to_mermaid() — draw the architecture
        ├── events.jsonl           # EVERY event, enveloped {seq, ts, ...event};
        │                          # lossless, ordered, tail-able by byte offset
        ├── metrics.jsonl          # flat tee of metric events:
        │                          # {ts, name, value, step, trial_id?, node_id?}
        ├── study.json             # full Study, atomically rewritten per trial
        │                          # (study runs only) — crash-safe resume
        ├── checkpoints/           # optional .somack bundles (user-driven)
        └── diagnostics/           # written by graph.gradient_audit(...)
            ├── audit_steps.jsonl  # per-filter per-step scalars (strict JSON)
            ├── report.json        # final aggregate: metrics + flags per filter
            └── channels/
                ├── index.jsonl    # {filter, step, file, keys, eff_rank, cka}
                └── <filter_id>/step_000050.safetensors
                                   # corr (C×C), act_abs_mean, act_zero_frac,
                                   # out_grad_abs_mean — numpy zero-copy load
```

## Delivery guarantees

`EventBus` has two delivery paths:

- **Sinks** (`add_sink`) are called synchronously on the emitting
  thread before the broadcast — lossless and strictly ordered. The
  tracker's `JsonlEventSink` lives here: a lagging disk never drops a
  line. Buffered writes flush every N events, on `finalize`, and when a
  study run returns.
- **Subscribers** (`subscribe` / Python `on_event`) receive through a
  tokio broadcast channel — live but lossy under lag. Display only.

Envelopes carry a monotonic `seq` so a reader can detect truncation and
resume incrementally; a torn final line (crash mid-write) is dropped on
read. `manifest.json`, `status.json` and `study.json` are written via
tmp-file + rename, so readers never observe partial JSON.

## Trackers

`Tracker` (soma-core) is the backend contract — `run_id`, `run_dir`,
`sink`, `save_manifest`, `save_artifact`, `save_study`, `heartbeat`,
`finalize`. `LocalTracker` (soma-runtime) is the file implementation;
a future `RemoteTracker` implements the same trait over HTTP, and a
sync daemon can replay `events.jsonl` to a server after the fact
(wandb's offline → sync model) with no format change.

## Python surface

```python
# Training run with diagnostics
with g.track_run("mos-baseline", tags=["mos"]) as run:
    with g.gradient_audit(channels=soma.ChannelConfig(
        groups={"encoder": {"audio": range(0, 64), "text": range(64, 128)}}
    )) as audit:
        for epoch in range(30):
            run.log_epoch(epoch, total=30)
            for x, y in batches:
                with g.context() as ctx:
                    g.zero_grad()
                    out, aux = g.forward(x)
                    g.backward(ctx, my_loss(out, y))   # audit + StepCompleted
                g.step(ctx)
            run.log("val_f1", evaluate(g), step=epoch)
    print(audit.report().pretty())

# Study — a study IS a run; follow or resume it from anywhere
study = g.study("mos-grid", strategy="grid", n_trials=4,
                objective=lambda m: 0.7 * m["val_f1"] - 0.3 * m["val_gap"],
                direction="maximize", pruning=("median", 3))
study.run(train, on_event=lambda e: print(e["event_type"]))

study = soma.Study.load(".soma/runs/study_20260726T101502_a3f1")
print(study.progress, study.best_trial)
study.run(train, resume=True)          # continues at trial N, no repeats

# The journal
for exp in soma.experiments():
    print(exp["name"], exp["metrics"], exp["tags"])
```

## Reading runs back

The read side mirrors the writer: `RunReader` in `soma-runtime`
aggregates one run directory into chart-ready shapes, exposed in
Python as `soma.runs()` / `soma.RunView` and on the CLI as
`soma runs` / `soma graph` / `soma report`:

```python
for run in soma.runs():                 # newest first; stale heartbeat ⇒ "crashed"
    print(run.id, run.state, run.name)

view = soma.RunView(".soma/runs/train_20260728T093011_9c2e")
view.events()          # enveloped {seq, ts, event_type, ...}, torn lines skipped
view.metric_series()   # metrics.jsonl (event-log fallback)
view.node_timings()    # per-node spans: wall times, durations, outcomes
view.cache_activity()  # hits/misses, per node
view.to_mermaid()      # graph.json + overlay: status colors, durations, flags
```

Readers never write. Wall-clock comes from the envelope `ts` (sinks
are synchronous, so it is the emission time); the live `on_event`
callback carries no envelope, which is why every timeline view reads
files. The visualization stack — overlays, `soma.viz` figures, the
HTML report and its embedded JSON data blobs (the future front-end's
contract) — is specified in [Visualization](/design/visualization/).

## Diagnostics captured

The gradient audit records, per node: activation stats, output-gradient
norms, parameter-gradient norms and the grad/param ratio each step;
with `channels=` enabled it adds per-channel dead fraction (dying-ReLU
criterion), the dormancy fraction β<sub>τ</sub> (Sokar et al., ICML
2023), *ignored* channels — alive in the forward pass but starved of
gradient (Pezeshki et al., NeurIPS 2021) — and, every `snapshot_every`
steps, a channel×channel correlation matrix, the effective rank of the
activation matrix (rank collapse, Dong et al. 2021) and minibatch
linear CKA between declared channel groups (Kornblith et al. 2019);
cross-group CKA above threshold flags `LEAKAGE`. Flags become
`HealthFlag` events and land in `report.json`, so a front-end can
overlay them on `graph.mmd`.

## Future extensions (schema already supports them)

- **Remote backend**: implement `Tracker` over HTTP, or replay
  `events.jsonl` after the fact.
- **ASHA / Hyperband pruning**: kill-based asynchronous successive
  halving fits the existing `trial.report()` + `should_prune()`
  contract; only rung bookkeeping is new.
- **fANOVA importance**: post-hoc over `study.json` — every trial keeps
  its full params and metrics.
- **Cross-run index**: a rebuildable `index.sqlite` over the manifests
  once run counts make directory scans slow (the MLflow FileStore
  lesson); never the source of truth.
- **Metric compaction**: fold long `metrics.jsonl` histories into
  parquet segments for analytics, keeping the live tail as JSONL.
- **Quantized dimensions**: an optional `step` on Float/Int search
  dimensions (ConfigSpace-style `q`).
- **Update/weight ratio**: per-step `‖Δw‖/‖w‖` (target ≈1e-3) needs a
  pre/post-optimizer-step parameter snapshot; today `grad_param_ratio`
  carries the same signal modulo the learning rate.
