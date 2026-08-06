---
title: Distribution — worker, coordinator, store
description: What crosses the wire — the worker protocol, the out-of-process Python daemon, worker placement, and the remote data-store backends.
---

Three crates handle everything that leaves the process: a worker daemon that
executes plans, a coordinator that places them, and the data-store backends that
move tensors without going through either.

One architectural invariant governs all of it, stated at
`soma-coordinator/src/registry.rs:26`: **the coordinator hands out an address and
steps aside.** `/submit` returns a worker and takes a lease; it does not proxy
the plan. Tensor payloads go client → worker directly.

The [notation](/soma/internals/map/) legend applies. `(!)` marks a documented
deviation, with the entry in the [Debt Register](/soma/internals/debt/).

---

## D5 · What crosses the wire

```
   client (soma-python / GraphSession)
       │
       │  1. POST /submit ──────────────▷ soma-coordinator
       │     ◁── worker address + lease      WorkerRegistry
       │                                     Arc<RwLock<HashMap<WorkerId, WorkerStatus>>>
       │                                     reaper every 10s
       │                                          ▲
       │                                          │ Register / Heartbeat (every 10s)
       │  2. WebSocket, direct                    │
       ▼                                          │
   «trait» Transport ──▷ WsTransport ─────────────┼──────▷ soma-worker
   soma-runtime/…       ws_transport.rs:17        │        server.rs (axum)
   /runner/remote.rs:18   on_own_runtime :42      │             │
                          « sync method,          │             ▼
                            own runtime »         │        Worker::execute_plan
                                                  │        worker.rs:275  (!) 324 lines
   msgpack to_vec_named  ──▷ SerializedPlan ──────┘             │
   protocol.rs:249            protocol.rs:209                   │
     ├─ protocol_version  (check_version :311)     ┌────────────┴───────────┐
     ├─ plan: ExecutionPlan                        ▼                        ▼
     ├─ input: InputSource {Inline | Reference}  EnvManager            PythonProcess
     ├─ filters: Vec<SerializedFilter>           env_manager.rs:38     python_process.rs:538
     │    « cloudpickle bytes, base64 »          venv/conda pooled       DAEMON_SCRIPT (!)
     ├─ mode: ExecutionMode {Fit | Forward}      by requirements hash    ~515 lines of Python
     └─ seed  « salts remote cache keys »        env_id_for :362        in a const &str
                                                                             │
   PlanResult {Success{OutputDelivery, states} | Failed}                     ▼
                                                                       SubprocessFilter
   StreamMessage {StreamBegin | ChunkData | StreamEnd                   ──▷ «trait» Filter
                 | ChunkResult | StreamComplete}                        python_process.rs:1025
     « drives the SAME StreamRun as local streaming;
       held with its Context in active_streams between messages »

   Large payloads bypass all of the above:
       DataStore ──▷ DataRef {Local | S3 | Zarr | Cached | Inline}
                     ↳ client PUTs, worker GETs, only the ref crosses the WS
```

---

## soma-worker (`somatize-worker`)

### Mandate

Execute a serialized plan in an isolated Python environment, and stream results
back. It owns the wire protocol, the environment manager, the out-of-process
Python daemon, and the axum server that fronts them.

`5 903 lines across 10 files · 0 traits defined · 1 error enum · deps: somatize-runtime, axum, tokio`

### Modules

| File | Lines | Owns |
|---|---|---|
| `soma-worker/src/worker.rs` | 1 196 | `Worker` and `execute_plan` `(!)` |
| `soma-worker/src/python_process.rs` | 1 083 | The embedded daemon script, `PythonProcess`, `SubprocessFilter` |
| `soma-worker/src/protocol.rs` | 964 | **The wire** |
| `soma-worker/src/server.rs` | 926 | The axum HTTP/WS server |
| `soma-worker/src/ws_transport.rs` | 547 | The client side — `impl Transport for WsTransport` |
| `soma-worker/src/env_manager.rs` | 479 | venv/conda provisioning |
| `soma-worker/src/bin/soma-worker.rs` | 303 | The clap CLI |
| `soma-worker/src/detect.rs` | 260 | `Capabilities::detect`, `ResourceLimits` |
| `soma-worker/src/error.rs` | 112 | `WorkerError` |
| `soma-worker/src/lib.rs` | 33 | Re-exports — `(!)` including `pub use protocol::*` |

### The wire protocol

`soma-worker/src/protocol.rs` is the crate's most important file: everything in
it is a compatibility surface.

Version: `PROTOCOL_VERSION: u32 = 1` (`:33`), `unversioned() -> 0` (`:37`),
`SerializedPlan::check_version` (`:311`), refused at `soma-worker/src/worker.rs:279`.

| Type | file:line | Shape |
|---|---|---|
| `SerializedPlan` | `soma-worker/src/protocol.rs:209` | `protocol_version` (default 0), `plan_id`, `plan: ExecutionPlan`, `input: Option<InputSource>`, `filters`, `mode`, **`seed`**, `metadata` |
| `SerializedFilter` | `:152` | `node_id`, `pickled_filter: Vec<u8>` (base64), `state`, `requirements`, `trainable`, `config_hash` |
| `InputSource` | `:88` | `#[non_exhaustive]`; `Inline{value}` \| `Reference{data_ref}`; `resolve()` at `:116` |
| `OutputDelivery` | `:560` | `#[non_exhaustive]`; `Inline{value}` \| `Reference{data_ref}` |
| `ExecutionMode` | `:192` | `#[non_exhaustive]`, default `Forward`; `Fit{y, batch_size}` \| `Forward` |
| `PlanResult` | `:585` | `Success{output, duration_ms, states}` \| `Failed{error, duration_ms}` |
| `StreamMessage` | `:620` | `#[non_exhaustive]`; `StreamBegin{…, plan: Box<SerializedPlan>}`, `ChunkData`, `StreamEnd`, `ChunkResult`, `StreamComplete` |
| `WorkerToCoordinator` | `:333` | 9 variants: `Register`, `Heartbeat`, `Event`, `PlanResult`, `JobProgress`, `JobResult`, `StateResult`, `Error`, `Ack`, `GradientsResult` |
| `CoordinatorToWorker` | `:484` | 10 variants: `Registered`, `AssignPlan`, `AssignPythonJob`, `CancelPlan` `(!)`, `StatusRequest`, `Ping`, `Shutdown`, `GetState`, `SetState`, `GetGradients`, `ApplyGradients` |
| `Capabilities` / `GpuInfo` / `LoadMetrics` | `:44`, `:59`, `:68` | What a worker advertises and reports |
| `PythonPipelineJob` / `PipelineFile` | `:451`, `:474` | The file-shipping job type |

**Framing** — `encode_frame` (`:249`) / `decode_frame` (`:256`) use msgpack
`to_vec_named`. The 10-line comment at `:236` records why: plain `to_vec`
produced arrays that `Value`'s adjacently-tagged enum could not read back, and
**two receivers swallowed the error** (`if let Ok`, `unwrap_or_default`). That
comment is the template for how this codebase records a fixed bug.

`InputSource::resolve` (`:116`) carries the same kind of note: it hard-errors when
a reference resolves nowhere, where it used to return `Value::Empty`.

### Types

| Item | file:line | Shape |
|---|---|---|
| `Worker` | `soma-worker/src/worker.rs:16` | **11 fields** `(!)`: id, capabilities, event bus, cache, catalog, *two* data stores, env manager, interpreter path |
| `PythonProcess` | `soma-worker/src/python_process.rs:538` | child, stdin, stdout, `node_ids`; `impl Drop` at `:970` |
| `SubprocessFilter` | `soma-worker/src/python_process.rs:982` | `Arc<Mutex<PythonProcess>>`, `node_id`, `trainable`, `config_hash` — `impl Filter` at `:1025` |
| `WsTransport` | `soma-worker/src/ws_transport.rs:17` | `{ address, token }` — connections opened per call, not held (`:63`); `impl Transport` at `:404` |
| `EnvManager` / `EnvType` / `EnvLockfile` | `soma-worker/src/env_manager.rs:38`, `:14`, `:25` | venv or conda, keyed by requirements hash |
| `ResourceLimits` | `soma-worker/src/detect.rs:19` | max cpus / memory / gpus / concurrent |
| `ShutdownSignal` | `soma-worker/src/server.rs:33` | Newtype over `Arc<tokio::sync::Notify>` |
| `WorkerError` | `soma-worker/src/error.rs:24` | `#[non_exhaustive]`: `Transport`, `Python`, `Env`, `Encoding`, `Concurrency`, `Remote`, `Io`, `Core` |

### Three mechanisms worth understanding

**Environments are pooled by content, not by plan.** `EnvManager::env_id_for`
(`soma-worker/src/env_manager.rs:362`) keys a venv by the hash of its
requirements. The comment at `soma-worker/src/worker.rs:315` records what this
fixed: one venv per plan meant an explosion of near-identical environments.
Updates are incremental — only what changed is installed, upgraded or removed.

**The Python daemon is a full anti-corruption layer.** `DAEMON_SCRIPT`
(`soma-worker/src/python_process.rs:19`) runs the user's filters in a separate
interpreter, communicating in JSON Lines over pipes. The GIL is fully isolated,
and `sys.stdout` is swapped to stderr at `:27` so a user's `print` cannot corrupt
the protocol. `SubprocessFilter` then satisfies the ordinary `Filter` trait by
delegating over that pipe — the executor cannot tell the difference. `(!)` The
script is ~515 lines of Python inside a Rust `const &str`, with no syntax check,
no lint and no test — [D-19](/soma/internals/debt/#d-19--two-embedded-python-interpreters-as-rust-string-constants).

**A synchronous `Transport` method is safe inside or outside tokio.**
`on_own_runtime` (`soma-worker/src/ws_transport.rs:42`) opens a scoped thread with
a fresh current-thread runtime. The doc at `:25` records that this replaced two
previous answers that contradicted each other — one assuming it was inside a
runtime, one assuming it was not.

Remote streaming drives **the same `StreamRun`** as local streaming. The worker
keeps the driver *and its `Context`* alive in `active_streams` between WebSocket
messages, and `SerializedPlan.seed` salts the remote cache keys. That is why a
streamed remote run and a streamed local run produce identical keys.

### Patterns

- **Message protocol with explicit versioning** — refusal on mismatch, not best-effort.
- **Out-of-process isolation (anti-corruption layer)** — the daemon script.
- **Remote proxy ×2** — `SubprocessFilter` proxies a `Filter` over a pipe; `WsTransport` proxies a `Transport` over a socket.
- **Content-addressed pooling** — `env_id_for`.
- **Sidecar runtime** — `on_own_runtime`.
- **Graceful shutdown token** — `ShutdownSignal`.
- **Fallback with degradation** — `fallback_config_hash` (`python_process.rs:1014`) for payloads from older coordinators. `(!)` Two other fallbacks in this crate are silent and should not be.
- **Builder** — `Worker::with_python` / `with_cache` / `with_data_store` / `with_temp_dir`.

### Debt

**High** — [D-02](/soma/internals/debt/#d-02--worker-and-worker-execute_plan) `Worker::execute_plan`, 324 lines across nine responsibilities ·
[D-25](/soma/internals/debt/#d-25--state-load-failure-silently-restarts-from-random-init) a failed state load silently restarts from random init

**Medium** — [D-19](/soma/internals/debt/#d-19--two-embedded-python-interpreters-as-rust-string-constants) the embedded interpreter ·
[D-24](/soma/internals/debt/#d-24--venv-provisioning-fails-into-the-system-interpreter) venv failure falls back to system python ·
[D-47](/soma/internals/debt/#d-47--a-cross-crate-contract-carried-by-an-environment-variable) `SOMA_LOCAL_PACKAGE`

**Low** — long functions: `handle_ws` 215 lines (`soma-worker/src/server.rs:353`),
`handle_stream_message` 163 (`:568`), `execute_python_job_with_progress` 141
(`:731`), `execute_streamed_from_store` 109 (`soma-worker/src/worker.rs:599`) ·
`CancelPlan` is in the protocol and answered with "not implemented"
(`soma-worker/src/server.rs:390`) · `pub use protocol::*`
(`soma-worker/src/lib.rs:29`) grows the public API with the protocol ·
interpreter mismatch is mitigated only by `$SOMA_PYTHON`
(`soma-worker/src/worker.rs:48`), with the failure mode described at `:30` and no
guard

**Notably correct**: this crate is the workspace's only substantial async code
and it handles the boundary properly — `spawn_blocking` at
`soma-worker/src/server.rs:366`, `:396`, `:450` and `:535`, and `on_own_runtime`
refusing to assume its context.

---

## soma-coordinator (`somatize-coordinator`)

### Mandate

Know which workers exist, whether they are alive, and which one should take the
next plan. Nothing more — it never sees a plan or a tensor.

`949 lines across 4 files · 0 traits · deps: somatize-worker (for the protocol vocabulary), axum`

**The tidiest crate in the workspace.**

### Types

#### `WorkerStatus` — `soma-coordinator/src/registry.rs:22`

`{ id, address, capabilities, load, active_plans, last_heartbeat, connected }`
with `has_capacity` (`:54`), `matches_tags` (`:59`), `is_alive(timeout_secs)`
(`:66`).

#### `WorkerRegistry` — `soma-coordinator/src/registry.rs:73`

```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<WorkerId, WorkerStatus>>>,
    heartbeat_timeout_secs: i64,
}
```

`#[derive(Debug, Clone)]` — cloning shares the map, so it is a **handle**, not a
value. API: `register` `:112`, `heartbeat` `:135`, `claim` `:149`, `release`
`:164`, `disconnect` `:176`, `remove` `:184`, `active_workers` `:190`, `get`
`:200`, `find_workers` `:206`, `total_count` `:214`, `active_count` `:219`,
`summary` `:224`, `prune_stale` `:247`.

Private lock helpers `read` (`:88`) and `write` (`:93`) use
`unwrap_or_else(|e| e.into_inner())` — deliberate poison tolerance, documented at
`:78`. Compare with [D-71](/soma/internals/debt/#d-71--four-policies-for-a-poisoned-mutex-three-of-them-silent),
where the same problem has four different answers.

### The server

`coordinator_router(registry, token)` (`soma-coordinator/src/server.rs:39`) also
spawns the 10-second reaper (`:42`). Routes at `:53`–`:60`:

```
GET  /health   GET /workers   GET /summary
POST /register POST /heartbeat POST /submit   POST /complete
```

`/submit` **places** — it returns a worker and takes a lease. `/complete`
releases it.

`check_auth` (`:86`) prefers `Authorization: Bearer`, still accepts `?token=`
with a deprecation warning at `:105`, and compares in **constant time**.

### Patterns

- **Registry / service discovery** with leases and a reaper task.
- **Shared-handle concurrency** — `Arc<RwLock<…>>` behind a `Clone` façade.
- **Graceful shutdown** — a ctrl-c future into `with_graceful_shutdown` (`soma-coordinator/src/bin/soma-coordinator.rs:63`).

### Debt

Two small observations only:

- [D-74](/soma/internals/debt/#d-74--the-coordinators-reaper-is-a-side-effect-of-building-a-router) — building a router spawns a background task, so `coordinator_router` is not idempotent in a test.
- `heartbeat_timeout_secs` has two independent defaults: the CLI flag (`soma-coordinator/src/bin/soma-coordinator.rs:37`) and the struct (`soma-coordinator/src/registry.rs:104`).

It also shares [D-18](/soma/internals/debt/#d-18--two-worker-capability-models-in-one-workspace)
with the compiler: two worker-capability models exist, and only this one is wired
to anything.

---

## soma-store (`somatize-store`)

### Mandate

Remote `DataStore` backends, feature-gated and **off by default**. Split out of
`soma-core` for one concrete reason, stated at `soma-store/src/lib.rs:5`: each
backend owns a `tokio::runtime::Runtime`, and `soma-core`'s rule is that
depending on it costs a caller nothing.

`1 285 lines across 3 files · 0 traits · 2 DataStore implementations · features: s3, zarr`

### Types

| Item | file:line | Shape |
|---|---|---|
| `S3DataStore` | `soma-store/src/s3.rs:24` | `config`, `bucket: Box<Bucket>`, `prefix`, `local_cache: PathBuf`, `rt: tokio::runtime::Runtime`. `new` takes **6 arguments** `(!)`; `from_env` |
| `ZarrStore` | `soma-store/src/zarr.rs:201` | `config`, `Arc<dyn ObjStore>`, `prefix`, `chunk_rows`, `local_cache`, `Mutex<ChunkLru>`, `rt`. `new` takes **7 arguments** `(!)` |
| `ZarrMeta` / `ChunkGrid` / `ChunkGridConfig` *(private)* | `soma-store/src/zarr.rs:78`, `:88`, `:94` | The Zarr v3 JSON shape |
| `ChunkLru` *(private)* | `soma-store/src/zarr.rs:147` | `VecDeque<(PathBuf, u64)>` + byte accounting |

`ZarrStore` is the only `DataStore` implementation anywhere that overrides
`get_rows` and `meta` (`soma-store/src/zarr.rs:658`) — the point of a chunked
backend is serving a row range without downloading the array, and the default
implementations do exactly the download the override avoids.

`byte_shuffle` / `compress_chunk` and their inverses handle the Zarr codec
pipeline; `append` (`soma-store/src/zarr.rs:532`) rewrites only the last partial
chunk and adds new full ones.

### Patterns

- **Strategy** — both are `DataStore` implementations, selected by `StorageConfig`.
- **Selective override of template defaults** — `ZarrStore` overriding `get_rows`/`meta` is the pattern working as intended.
- **Local read-through cache with LRU accounting** — `(!)` and it is broken, see below.

### Debt

- [D-31](/soma/internals/debt/#d-31--zarrstores-chunk-cache-is-write-only) — **High.** `key_from_path` (`soma-store/src/zarr.rs:515`) hashes the hex instead of decoding it, so the directory it writes to and the directory it reads from never coincide. The chunk cache is write-only, every `get` goes back to S3, and `remove` cleans up the wrong path so the cache grows without bound.
- `ZarrStore::append` is 101 lines of index arithmetic (`soma-store/src/zarr.rs:532`), with `first_new_chunk` (`:602`) mixing `div_ceil`, `min` and a subtraction, and no unit test exercising it.
- Seven- and six-argument constructors with no builder, five of the arguments `impl Into<String>` — trivially swappable at a call site.
- Two stores in one process means **two tokio runtimes** (`soma-store/src/s3.rs:78`, `soma-store/src/zarr.rs:238`). The split from `soma-core` solved the dependency problem, not the per-instance-runtime one.
- [D-82](/soma/internals/debt/#d-82--stale-instructions-in-feature-docs) — both module docs still tell the reader to enable these features on `soma-core`.
