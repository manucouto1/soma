---
title: Experiment Pool
description: A searchable record of what has been tried, why, and what came of it — captured as a by-product of running.
---

## The idea

Most of the cost of research is repetition: re-running something that
was already tried, re-discovering a dead end, re-deriving a conclusion
that someone (often you, three months ago) already wrote down and lost.

The experiment pool is Soma's answer. Every tracked run leaves behind a
record of *what was run*, *what it descended from*, *what changed*, and
*what happened* — and all four are captured automatically, because Soma
already has them. The pool is then searchable, by a human or by a model
over MCP.

Four ideas from the literature, and the terminology used throughout the
code:

- **Experiment database** (Vanschoren; OpenML) — the pool itself: a
  queryable store of runs with their configurations and results.
- **Workflow-evolution provenance** (VisTrails) — the crucial shape.
  Nodes are workflows; **edges are the changes applied to the parent**.
  A tree of runs tells you what you ran. A tree of runs *plus the move
  on every edge* tells you what you tried, and whether it worked.
- **Case-Based Reasoning** 4R — Retrieve, Reuse, Revise, Retain. The
  MCP tools are named after these steps.
- **Memory-stream retrieval** (Generative Agents) — ranking by
  relevance × recency × importance, which Soma adapts (and makes
  additive; see below).

What Soma adds: the artifacts are a **by-product of executing**, not
something the user is asked to curate. Run directories, `graph.json`,
event logs, timings and diagnostics already exist. The content-addressed
cache gives every computation an exact identity. Nothing here asks the
user to describe their experiment twice.

## What a run leaves behind

```
.soma/
├── HEAD                          # which run the next one descends from
├── experiments.jsonl             # the pool: one ExperimentRecord per line
└── runs/<run_id>/
    ├── manifest.json             # identity, seeds, params, parent, git
    ├── status.json               # running | completed | failed (+ heartbeat)
    ├── graph.json                # the exact topology executed
    ├── graph.mmd                 # the same, for humans
    ├── fingerprint.json          # structural identity  ← new
    ├── events.jsonl              # every lifecycle event
    ├── metrics.jsonl             # the metric tee
    └── diagnostics/              # gradient audit, channels, modules
```

`graph.json`, `graph.mmd` and `fingerprint.json` are written by a single
writer — `Graph.begin_run` — because that is the one place where the
graph and the filter library are both in scope, and therefore the only
place that can stamp each node's config hash into the fingerprint.

## Architecture fingerprints

`fingerprint.json` answers two different questions with two different
fields, because conflating them makes both useless:

| Field | Question | Behavior |
|---|---|---|
| `digest` | "Is this the *exact* same architecture?" | SHA-256 over node ids, node kinds and edges. Sensitive to renaming — which is what makes it usable as a dedup key. |
| `nodes` / `edges` | "What was it, structurally?" | Node id → type token, plus edges by id. The diffable form: it is how `kb_diff` knows *which* node was swapped. |
| `node_config` | "Was the same node configured differently?" | Per-node filter config hash, from the cache-key machinery. |

`node_tokens()` and `edge_tokens()` derive **id-free** bags of type
tokens from those. Renaming `scaler` to `norm` changes the digest and
leaves the tokens untouched, so fuzzy matching survives refactoring:

```
structural_similarity = 0.6 · jaccard(node tokens) + 0.4 · jaccard(edge tokens)
```

using multiset Jaccard, so three stacked `Dense` layers do not look
identical to one. Deterministic and linear — deliberately not graph
isomorphism, which is both expensive and far too strict for "these two
look alike".

The canonical form excludes `edge.id`, `node.label` and `node.target`:
they are cosmetic or deployment detail, not architecture. `SubGraph`
nodes recurse **by digest**, so nesting terminates and the result never
depends on declaration order.

## Conclusions

`RunConclusion` is the join that did not exist: manifest + status + node
timings + cache activity + health flags + metrics + study + trial
timeline + graph + fingerprint + `diagnostics/report.json`, folded into
one answer to "what happened here?".

Its `headline` is **templated and deterministic** — the same run
directory always produces the same string, with no model in the loop.
That is what makes it safe to hash, snapshot-test and index:

```
completed in 4m 12s · val_f1=0.9125 · slowest encoder (2m 30s, 60% of compute) · cache 67% hits · flags: DEAD_CHANNELS×2
failed after 12.0s · error: shape mismatch: expected [32, 8] got [32, 16]
completed in 18m 04s · 40 trials (6 pruned), best val_f1=0.88
```

Everything it could not read becomes a `warning`, never an error. A run
that crashed before writing anything but its manifest still summarizes.

## Lineage: `.soma/HEAD`

A pool without edges is a list. Soma resolves a run's parent in four
steps, most explicit first:

1. the `parent=` argument to `track_run`,
2. `$SOMA_PARENT_RUN` (for schedulers and CI),
3. `.soma/HEAD`,
4. nothing — this run starts a new line.

HEAD advances automatically after every **successful** run, so a linear
session builds a linear lineage with no bookkeeping. A crash never
becomes the parent of everything after it.

```python
with g.track_run("baseline", params={"lr": 0.01}):
    ...                            # HEAD → baseline

with g.track_run("wider", params={"lr": 0.05}):
    ...                            # parent = baseline, HEAD → wider

soma.checkout(baseline_id)         # rewind
with g.track_run("deeper", params={"lr": 0.01, "depth": 4}):
    ...                            # a sibling of "wider", not a child
```

**Soma never infers a parent from timestamps.** "The run before this
one" is a different claim from "the run this one was derived from", and
a single false edge poisons every metric delta computed downstream of
it. A missing parent is recoverable; a wrong one is not.

## Derivation moves

The edge itself. Stored as a field of the **child** record — one node,
one edge, one append — so an edge can never be orphaned by a crash
between two writes.

```json
{
  "from": "run_a", "to": "run_b",
  "changes": [{"change": "ParamChanged", "key": "lr", "from": 0.01, "to": 0.05}],
  "metric_delta": {"val_f1": {"before": 0.81, "after": 0.87, "delta": 0.06}},
  "summary": "lr: 0.01 → 0.05 ⇒ val_f1 +0.06"
}
```

`Change` is `#[non_exhaustive]`: `NodeAdded`, `NodeRemoved`,
`NodeReplaced`, `NodeReconfigured`, `EdgeAdded`, `EdgeRemoved`,
`ParamChanged`, `ParamAdded`, `ParamRemoved`, `SearchSpaceChanged`,
`CodeChanged`, and **`Unspecified`** when the evidence to describe the
move is gone. Saying "something changed and I cannot say what" is worth
more than inventing a plausible diff.

Metric deltas are **signed, not judged**. Whether up is good depends on
the objective's direction, which the move does not presume to know.

## The record

`experiments.jsonl` is append-only, one `ExperimentRecord` per line. It
is the contract between whatever produces runs and whatever reasons
about them, so **every field added after the first release is
defaulted** and unknown fields are ignored: a journal written by a newer
Soma loads on an older one, and vice versa. A byte-exact legacy line
lives in the test suite as that promise.

| Field | Meaning |
|---|---|
| `id`, `name`, `timestamp`, `duration` | Identity. `id` is the run id. |
| `run_id`, `run_dir` | Where the raw artifacts are, so a reader can go look. |
| `pipeline_summary` | Topology in one line, read off `graph.json`. |
| `architecture` | The fingerprint. |
| `conclusion` | The `RunConclusion` above. |
| `parent`, `derivation` | The edge and the move on it. |
| `research_line` | Inherited from the parent; a root names its own. |
| `params`, `metrics`, `objective` | Configuration and results. |
| `hypothesis`, `notes` | What a human said about it. |
| `git`, `schema_version`, `kind`, `amends`, `embedding` | Provenance and extension points. |

`kind` is `experiment` or `amendment`. An amendment is a later note
attached to an existing record, appended as its own line — the journal
is never rewritten, so a conclusion added today cannot corrupt what was
recorded when the run happened.

### Recovering the journal

The run directories are the source of truth; the journal is an index.

```bash
soma kb reindex     # rebuild experiments.jsonl from .soma/runs/*
soma kb head        # what the next run will descend from
soma kb checkout <run_id>
soma kb detach
```

One operation covering migration (runs recorded before the pool
existed), backfill (a run whose journal line was lost) and disaster
recovery. It writes to a temp file and renames, so an interrupted
reindex leaves the previous journal intact.

## Retrieval

```
0.40 · lexical + 0.25 · structural + 0.15 · recency + 0.20 · importance
```

**Additive, not multiplicative.** A product lets any single term veto a
record: an experiment from last year scores near zero on recency and
therefore near zero overall — when a year-old dead end is exactly the
kind of thing the pool exists to surface. Terms that cannot apply (no
query architecture, no text) redistribute their weight over the rest, so
scores stay comparable across queries.

- **Lexical** — BM25 (`k1 = 1.2`, `b = 0.75`) over a document built by
  repeating each field by weight: name ×3, hypothesis ×3, headline ×2,
  tags ×2, pipeline ×2, derivation summary ×2, notes ×1. The tokenizer
  splits `snake_case`, `kebab-case`, dotted paths and both camelCase
  boundaries (`valF1` → `val f1`, `MoSHead` → `mo s head`). **No
  stemming**: experiment vocabulary is identifiers and acronyms, which
  stemmers only damage.
- **Structural** — `structural_similarity` between the query's
  architecture and the record's.
- **Recency** — exponential decay, 30-day half-life by default. Never
  reaches zero.
- **Importance** — rewards a recorded conclusion, a human hypothesis or
  note, and a deliberate derivation. **Floors at 0.6 for any failure,
  crash or regression that carries a conclusion**: not repeating a dead
  end saves as much as repeating a win.

Ranking is deterministic — `now` is a parameter, ties break on id, no
clock or RNG is read inside the scorer.

### Embeddings

`Embedder` is a trait Soma does not implement, so the pool needs no
model and no new dependencies. An implementation can be plugged in from
outside (a sentence-transformer behind the Python worker, an HTTP
endpoint). Every stored vector carries an `embedder_id`; a vector from a
different model is **left out of the scoring entirely** rather than
compared across embedding spaces, where cosine similarity is a number
but not a similarity.

## MCP tools

The protocol carries text and nothing renders it for us, so **the text
is the API**. Three rules hold across every result: it ends with a
`next:` line naming the follow-up calls, every experiment shows its
`run_dir` so a model can read the raw artifacts itself, and absence is
stated rather than left blank.

| Tool | CBR step | What it does |
|---|---|---|
| `kb_find_similar` | Retrieve | Rank past work against the problem at hand, by text and/or architecture. |
| `kb_lineage` | — | The tree around a run, with the move labelling every edge. |
| `kb_diff` | — | Two experiments compared: changes, metric deltas, and cost (duration, cache hits). |
| `kb_record_conclusion` | Retain | Append what you learned, as an amendment. |
| `kb_branch_from` | Revise | Point HEAD at a run so the next one branches from it. |
| `kb_summarize_run` | — | Summarize a run directory on demand; works on pre-pool runs. |
| `kb_stats` | — | Size, span, research lines, and honest coverage. |

Every knowledge read refreshes the journal first. An MCP server outlives
the training runs it is asked about; without that, it answers "no such
experiment" for a run that finished five minutes ago in another
terminal. `FileKnowledgeBase::refresh()` reads only from a byte offset,
which the append-only log makes safe.

`kb_stats` reports coverage — what fraction of records carry a
conclusion, a lineage, an architecture — because a pool that looks full
but has no edges cannot answer the questions the other tools promise.

## Not built yet

Deliberately deferred, with the seam left in place:

- **Warm-starting studies from the pool.** The hook already exists:
  `StudyRunner::run` replays completed trials into
  `Sampler::record_result`. What is missing is feeding it from a
  retrieval filtered by `architecture.digest` and a compatible
  objective — both of which the record now stores.
- **Dedup by cache key** ("did I already run exactly this?").
  `fingerprint.json`'s `node_config` plus a trial's params give an exact
  identity, so this is a lookup rather than a search.
- **ChronosVector as a vector index.** `ChronosKnowledgeBase` works and
  is tested, but its built-in feature-hashing encoder is not a semantic
  embedding, so its "semantic search" is weaker than the BM25 default.
  It earns its place once a real `Embedder` exists.
