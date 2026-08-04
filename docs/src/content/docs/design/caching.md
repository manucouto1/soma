---
title: Caching System
description: Persistent content-addressable caching — crash-resilient, deterministic, resolved at runtime.
---

## Overview

Soma memoizes computation with a **persistent, content-addressable cache**. Every `fit()` and `forward()` is an *action* identified by a deterministic key; completed actions are recorded on disk, so a crashed run — or a different investigation over the same data — reuses previous compute instead of redoing it.

By default every `Graph()` shares one cache at `$SOMA_CACHE_DIR` (or `~/.soma/cache`), tiered behind an in-memory LRU. Resume after a crash is not a separate mechanism: re-running simply recomputes keys and hits.

```python
g = Graph()                  # persistent tiered cache (default)
g = Graph(cache="memory")    # opt out: nothing persists
```

## Cache keys

Keys are computed **at runtime**, with the materialized data in hand (the compiler never sees the dataset, so it cannot resolve caching — a compile-time key could serve results from a different dataset):

- **State key** (fit results): `hash(config + x + y)` — labels are part of the key.
- **Output key** (forward results): `hash(config + state + input)`.
- With an experiment seed set, the seed is hashed into every key.

Downstream keys derive from the *content hashes of materialized inputs*, not from upstream provenance. This gives **early cutoff**: if an upstream filter's config changes but its output bytes are identical, everything downstream still hits.

## Filter identity

A filter's config hash must be stable across processes and machines, and change when behavior changes:

- **Rust filters** (`#[derive(SomaFilter)]`): type name + canonical-CBOR encoding of each non-`skip_hash` field (RFC 8949 deterministic encoding — sorted map keys, canonical NaN, no `-0.0`). `#[soma(cache_version = "…")]` folds an explicit version into the hash; `#[soma(deterministic = false)]` excludes forward outputs from caching.
- **Python filters**: qualified class name + canonical config (public attrs merged over `search()` defaults) + a code fingerprint resolved through a ladder:
  1. `_cache_version = "…"` class attribute (soundest — survives refactors),
  2. hash of `inspect.getsource(cls)` (default — editing the class invalidates; helper-module edits do **not**, declare those with `_cache_version`),
  3. cloudpickle hash with a loud `UserWarning` (last resort).

  Unhashable attributes raise `soma.CacheConfigError` — prefix them with `_` or define `__soma_config__()`. "Uncacheable" is always explicit, never a silent random key.

## Two-table store

The persistent store (`FsActionStore`) follows the Bazel action-cache/CAS split:

```text
$SOMA_CACHE_DIR/
  format.json                    {"version": 2}
  actions/<aa>/<key>.json        action records: output content hashes,
                                 compute cost, size, provenance, timestamps
  cas/b3/<aa>/<hash>.bin         SOMA1-encoded blobs, BLAKE3-addressed,
                                 deduplicated across actions
  pins/<name>                    GC roots
```

- **Commit protocol**: blobs first (idempotent temp+fsync+rename), action record renamed last — the record is the commit point, so a crash mid-write never leaves a record pointing at required-but-missing data. Multi-process safe on local filesystems.
- **Payloads** use the `SOMA1` binary codec (tensors as raw little-endian f64, ~1× raw size vs ~3× as JSON), hashed with BLAKE3 while encoding. Corrupt blobs are detected on read and treated as misses.

## Eviction (GC)

`soma cache gc --max-size 20G` evicts **blobs only, by value density** (`compute_ms × recency ÷ size`): a 100-byte state that took two days outlives a 10 GB intermediate that took two minutes. Action records are always retained, so an evicted entry is *regenerable* — the next run recomputes it and re-fills the same content address. Eviction degrades warm-ness, never correctness. `soma cache pin NAME KEY` marks GC roots.

```console
$ soma cache stats      # size, records, compute banked
$ soma cache verify     # blob integrity vs content hashes
$ soma cache purge-v1   # drop unreachable Phase-1 entries
```

## Seeds

Seeds are ordinary hashed inputs — each seed owns an independent cache line:

```python
g.fit(x, seed=42)                    # per-seed keys on a single graph

study = Study("exp", ..., seeds=[1, 2, 3, 4, 5])
def train(trial):
    torch.manual_seed(trial["seed"])  # wire it into your framework
    ...
```

`Study(seeds=[...])` runs every sampled config once per seed (trial params carry `"seed"`, recorded in `manifest.json`). A crash after 3 of 5 seeds resumes with 3 exact hits: the remaining seeds are independent, resumable trials.

Nondeterministic filters (`_deterministic = False` / `#[soma(deterministic = false)]`) are excluded from forward-output caching **unless** a seed is set — with a seed in the key, results vary across seeds but stay stable within one.

## The effect journal

Effectful nodes ([steps](/soma/design/agentic/)) do not use the output cache at all — a step's output is not a function of its input, the model is on the other end of it, so serving a recorded one would be a lie. What a step gets instead is a **journal** over the *effects* it performs, and the journal lives in the same two-table `FsActionStore` as everything above: effect results are action records pointing at CAS blobs, sharing the store, the commit protocol and the GC.

What the journal adds is the keying, and a distinction the filter cache cannot express:

| Effect kind | Key includes | Reused |
|---|---|---|
| Pure (a filter-only graph run) | the effect's content | across every run, like a filter |
| Impure (a model call, a tool) | run, node, turn, index, effect | only when replaying *that* run |

The second row is the point. Asking a model the same question twice is genuinely two events, so memoizing by content would freeze the first answer forever — the `_deterministic = False` foot-gun wearing a hat. Scoping the key to `(run, node, turn, index)` means a resumed run sees exactly what the original saw, while a fresh run asks afresh. Failures are never recorded: a transport error is not a result, and a replay retries it.

`Effect::Graph` — a step running a pipeline — sits on whichever side its content dictates: a **filter-only `forward`** is pure, because its nodes are deterministic and content-cached already; a sub-graph that **contains a step**, or any run in **`fit` mode**, is impure (the step calls a model; a replayed fit must re-write states, not serve a summary). This is decided by `Effect::is_pure()` in `soma-core/src/effect.rs`.

Two practical consequences: prompts land on disk under `$SOMA_CACHE_DIR` (set `journal = false` on a step's `StepMeta` for anything that must not persist), and GC marks impure blobs as expensive so a replay's record outlives an intermediate tensor of the same size — an evicted pure blob merely recomputes, an evicted impure one would make a replay diverge.

## Events

The executor emits `NodeCacheHit` / `NodeCacheMiss` (with the key) on the graph's event bus; tracking sinks persist them to `events.jsonl`, so a run directory shows exactly what was reused.
