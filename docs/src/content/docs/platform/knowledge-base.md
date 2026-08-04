---
title: Knowledge Base
description: The API over the experiment pool — what it stores, how to query it, and what it does not do.
---

This page documents the `KnowledgeBase` API. For *why* the pool is
shaped the way it is — fingerprints, derivation moves, the scoring
formula — see [Experiment Pool](/soma/design/experiment-pool/).

## What it is

A `KnowledgeBase` is a store of `ExperimentRecord`s: one per run, with
its conclusion, its parent, and the change that produced it. The default
backend is a JSONL file at `.soma/experiments.jsonl`, appended to
automatically whenever a tracked run or a study finishes successfully.

It answers:

- "What have I run that bears on this problem?"
- "What did I try from this starting point, and what came of it?"
- "What is different between these two runs?"
- "Which research lines are improving?"

## Backends

| Backend | Storage | When |
|---|---|---|
| `FileKnowledgeBase` | `experiments.jsonl`, append-only | The default. What `.soma/` gets. |
| `MemoryKnowledgeBase` | In-process `Vec` | Tests, and the MCP fallback when a project has no `.soma/`. |
| `ChronosKnowledgeBase` | ChronosVector `TemporalHnsw` | Feature-gated (`chronos`). See the caveat at the bottom. |

Only `record`, `all` and `len` are required of a backend. Search,
research lines, trends, trajectories, change points, lineage and ranked
retrieval all have default implementations over `all()`, so the
analytics live in one place rather than once per backend.

Methods return **owned** records. That costs a clone per hit and buys a
backend that can page or query a remote store, which a
reference-returning trait cannot have.

## Recording

Runs record themselves. `graph.track_run(...)` and `study.run(...)`
append a record on success, built by reading the run directory:

```python
with g.track_run("mos-baseline", params={"lr": 0.01}, tags=["mos"]) as run:
    for epoch in range(30):
        ...
        run.log("val_f1", evaluate(g), step=epoch)
# .soma/experiments.jsonl now has a line with the conclusion,
# the architecture fingerprint, the parent and the move.
```

Recording is best-effort throughout: it never fails a training run that
already produced its results.

To record work Soma did not execute, use the `record_experiment` MCP
tool or build an `ExperimentRecord` directly. To add a finding to a run
that already happened, use `soma.record_conclusion` — it appends an
amendment rather than rewriting the original line.

## Querying from Python

```python
import soma

soma.experiments()               # every record, as dicts
soma.experiments_dataframe()     # the same as a DataFrame (somatize[viz])
soma.head()                      # what the next run will descend from
soma.checkout(run_id)            # branch from an earlier run
soma.detach()                    # start a new line
soma.reindex()                   # rebuild the journal from .soma/runs/

soma.find_similar("dropout collapse", limit=3)   # ranked retrieval
soma.lineage(run_id)                             # ancestors + descendants
soma.diff(run_a, run_b)                          # works on siblings too
soma.record_conclusion(run_id, "what you learned")
```

From the CLI:

```bash
soma kb head                                    # what the next run descends from
soma kb checkout run_20260730T160239_c38a       # branch from an earlier run
soma kb detach                                  # start a new line
soma kb reindex                                 # rebuild the journal from run dirs
```

## Querying from Rust

```rust
use chrono::Utc;
use somatize_memory::{FileKnowledgeBase, KnowledgeBase, RetrievalQuery};

let kb = FileKnowledgeBase::open(".soma/experiments.jsonl")?;

// Ranked retrieval: text, structure, recency, importance.
let query = RetrievalQuery::new("dropout collapse", Utc::now()).with_limit(5);
for hit in kb.retrieve(&query)? {
    println!("{:.2}  {}", hit.score, hit.record.headline());
}

// The tree around one run, with the move on every edge.
if let Some(lineage) = kb.lineage("run_20260730T160239_c1d9")? {
    for node in &lineage.descendants {
        println!("{}{}", "  ".repeat(node.depth), node.record.name);
    }
}

// Line-level analytics.
for line in kb.research_lines()? {
    println!("{} — {} ({} experiments)", line.name, line.trend, line.experiments.len());
}
kb.trajectory("mos-baseline", "val_f1")?;
kb.change_points("mos-baseline", "val_f1", 0.05)?;
kb.promising_lines("val_f1")?;
```

### Staying current

A long-lived reader must refresh, or it answers from the snapshot it
loaded at startup:

```rust
let mut kb = FileKnowledgeBase::open(".soma/experiments.jsonl")?;
let new_records = kb.refresh()?;   // reads only the tail, by byte offset
```

The MCP server does this before every knowledge read. `refresh()` copes
with a half-written line (defers it) and with the file having been
replaced by `soma kb reindex` (reloads from scratch).

## Research lines

A research line groups an experiment with everything derived from it. A
run with no parent names its own line after itself (slugified); every
descendant **inherits** that name, however far the work drifts. That is
what makes line-level analytics — trend, trajectory, change points —
work on real data rather than on hand-assigned tags.

`Trend` is `Improving`, `Plateaued`, `Declining` or `Unknown`, computed
from the last three values of the line's first metric.

## MCP

Seven tools expose the pool to a model. See
[Experiment Pool](/soma/design/experiment-pool/#mcp-tools) for the full
table; in short: `kb_find_similar`, `kb_lineage`, `kb_diff`,
`kb_record_conclusion`, `kb_branch_from`, `kb_summarize_run`,
`kb_stats`.

```
Agent: "I want to try z-norm on the Coffee dataset"
  → kb_find_similar(query="z-norm Coffee")
  → "run_012 — completed in 3m 20s · val_f1=0.91
     ⚠ dead end — run_019 tried z-norm on top and lost 0.04
     run_dir: .soma/runs/run_012"
  → the agent reads the run dir and picks a different axis
```

## What this is not

Some things previous versions of this page described do not exist, and
should not be planned around:

- **There is no automatic embedding.** `ExperimentRecord.embedding` is
  an optional field and nothing populates it. `Embedder` is a trait with
  no implementation in Soma — a seam, not a feature. Text search is
  BM25.
- **There is no tiered hot/warm/cold storage.** The journal is one
  append-only file. The *computation cache* is tiered
  ([Caching](/soma/design/caching/)); the experiment pool is not.
- **There is no `velocity` or `acceleration`.** `ResearchLine` carries a
  categorical `trend` and the best metric value.
- **There is no `kb.compare()` or `lab.knowledge_base()`.** Use
  `kb_diff` (MCP) or `somatize_memory::derive` (Rust) to compare two
  records; `soma.experiments()` to read the pool from Python.
- **`ChronosKnowledgeBase` is not the default and is not semantic.** It
  is feature-gated behind `chronos`, and its encoder is feature hashing
  over words — not a learned embedding. Its `search` is therefore
  *weaker* than the BM25 default, not stronger. It becomes worth using
  when a real `Embedder` exists; until then, prefer
  `FileKnowledgeBase`.
