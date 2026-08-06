---
title: Agentic Stack — llm, agent, memory, mcp
description: Trait and type reference for the four crates that turn a graph into an agent — provider access, the research loop, the experiment pool, and the MCP server.
---

These four crates sit on top of the runtime and share one shape: none of them
defines an execution model. `soma-llm` and `soma-agent` supply `Step`
implementations; `soma-memory` supplies a store; `soma-mcp` supplies a front end.
All of them re-enter the runtime through the same
[effect loop](/soma/internals/execution/#c-the-effectdriver-turn-loop).

The [notation](/soma/internals/map/) legend applies. `(!)` marks a documented
deviation, with the entry in the [Debt Register](/soma/internals/debt/).

---

## D4 · The effect loop

Where an agent's decision becomes work, and where the work becomes replayable.

```
   «trait» Step ──▷ poll(ctx) ──▷ [enum] Transition
                                    │
        ┌──────────┬────────────────┼──────────┬──────────┐
        ▼          ▼                ▼          ▼          ▼
     Await      Spawn             Goto      Suspend      Done
   Vec<Effect>  Vec<NodeSpec>    NodeId     reason       Value
        │          │                │          │          │
        │          │                └──▷ NodeOutcome::HandOff
        │          └──▷ recurses into EffectDriver::run per child
        ▼                                     └──▷ NodeOutcome::Paused
   EffectDriver::perform_all       effects/mod.rs:440
        │  one std::thread per effect
        ▼
   perform_one(journal, EffectSite{run,node,turn,index}, effect)   :519
        │
        ├──▷ EffectJournal::lookup ──▷ hit? replay, never re-run
        │        pure effect   → key = content
        │        impure effect → key = "sited" ‖ run ‖ node ‖ turn ‖ index
        │
        └──▷ handlers.iter().find(|h| h.handles(effect))    « chain of responsibility »
                 │
     ┌───────────┼────────────┬──────────────┐
     ▼           ▼            ▼              ▼
  LlmHandler  Toolbox    GraphHandler   SleepHandler
  soma-llm     soma-llm   soma-runtime   soma-runtime
   /lib.rs:205 /tools.rs  /effects/      /effects/
                :168       graph_handler  sleep_handler
     │           │          .rs:136        .rs:20
     ▼           ▼             │
  «trait»     «trait»          └──▷ runs a whole Graph, which may contain
  LlmProvider  Tool                 steps, which need another EffectDriver…
     ▲           ▲                  ↳ CYCLE, capped at MAX_GRAPH_DEPTH = 8
     │           ├─ FnTool
  OpenAiCompatible ├─ McpTool ──▷ McpClient ──▷ JSON-RPC over a child process
                   └─ PyToolAdapter  « a Python callable »
```

The two keying rules are the heart of it. A **filter memoizes by content**; a
**step journals by site**. A pure effect (a deterministic tool call) is keyed by
what it *is*, so any run can reuse it. An impure effect (a model call) is keyed
by *where and when* it happened, so a resumed run replays exactly the answer it
got the first time and never asks twice.

---

## soma-llm (`somatize-llm`)

### Mandate

Provider-agnostic model access, tools, and the three `Step` implementations that
use them. One HTTP client, a TOML-driven provider catalog, and retry logic that
lives in the transport rather than the domain.

`3 848 lines across 7 files · 2 traits · deps: somatize-core, somatize-runtime, reqwest (blocking)`

### Modules

| File | Lines | Owns |
|---|---|---|
| `soma-llm/src/steps.rs` | 1 108 | `ReactStep`, `LlmStep`, `JudgeStep`, `Verdict`, the schema-repair loop |
| `soma-llm/src/openai_compat.rs` | 851 | The single HTTP client; `Pushback` retry classifier; wire structs |
| `soma-llm/src/catalog.rs` | 693 | `Auth`, `Quirks`, `RetryPolicy`, `ProviderConfig`, `Catalog`, `split_model` |
| `soma-llm/src/lib.rs` | 382 | `ModelInfo`, `LlmProvider`, `Router`, `LlmHandler` |
| `soma-llm/src/mcp_client.rs` | 346 | `McpClient` — JSON-RPC over a child process |
| `soma-llm/src/tools.rs` | 340 | `Tool`, `FnTool`, `Toolbox`, `ToolOutcome`, `McpTool` |
| `soma-llm/src/error.rs` | 128 | `LlmError` |

### Public contracts

#### `LlmProvider` — `soma-llm/src/lib.rs:72`

```rust
pub trait LlmProvider: Send + Sync {
    fn id(&self) -> &str;
    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse>;   // blocking
    fn models(&self) -> Result<Vec<ModelInfo>> { Ok(Vec::new()) }  // provided
}
```

**Not async**, and neither is anything else in the workspace — `rg async_trait`
returns zero hits. The doc at `:76` gives the reason: "Blocking — the effect
driver runs these on threads."

One implementor: `OpenAiCompatible` (`soma-llm/src/openai_compat.rs:352`). Every
supported provider — ollama, HuggingFace, NVIDIA, Kimi, GLM, DeepSeek, Groq,
vLLM and the rest — is that one client plus a `ProviderConfig`, which is why
adding a provider is a TOML entry rather than a release.

#### `Tool` — `soma-llm/src/tools.rs:53`

```rust
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn call(&self, args: &Value) -> Result<ToolOutcome>;
}
```

| Implementor | file:line | Note |
|---|---|---|
| `FnTool<F>` | `soma-llm/src/tools.rs:62` | Wraps a closure |
| `McpTool` *(private)* | `soma-llm/src/tools.rs:212` | Delegates to an MCP server over `McpClient` |
| `PyToolAdapter` | `soma-python/src/agentic.rs:198` | Across the FFI — a Python callable |

### Types

| Item | file:line | Shape |
|---|---|---|
| `Router` | `soma-llm/src/lib.rs:93` | `providers: BTreeMap<String, Arc<dyn LlmProvider>>`, `default` — a registry + strategy |
| `LlmHandler` | `soma-llm/src/lib.rs:183` | `{ router }` — the `EffectHandler` for `Effect::Llm` |
| `ModelInfo` | `soma-llm/src/lib.rs:57` | `{ id, provider }`, `qualified() -> "provider/id"` |
| `Toolbox` | `soma-llm/src/tools.rs:92` | `tools: BTreeMap<String, Arc<dyn Tool>>` — the `EffectHandler` for `Effect::Tool` |
| `ToolOutcome` | `soma-llm/src/tools.rs` | Success or error **as data** — a failed tool is a message to the model, not an aborted run |
| `Auth` | `soma-llm/src/catalog.rs:34` | `#[non_exhaustive]`, tagged; includes bearer-from-env |
| `Quirks` | `soma-llm/src/catalog.rs:96` | `supports_tools`, `supports_json_schema`, `max_tokens_field`, `omit_empty_tools`, `system_as_message` |
| `RetryPolicy` | `soma-llm/src/catalog.rs:144` | `max_attempts`, `budget_secs`, …; `backoff(attempt, retry_after)` at `:195` |
| `ProviderConfig` | `soma-llm/src/catalog.rs:231` | base_url, auth, quirks, retry, headers, model_prefix, timeout; builders `local` / `hosted` / `with_*` |
| `Catalog` | `soma-llm/src/catalog.rs:374` | `builtin()` at `:387` hard-codes 12 providers `(!)`; `load()` at `:499` layers `$SOMA_PROVIDERS` and `~/.soma/providers.toml` |
| `LlmError` | `soma-llm/src/error.rs:29` | `#[non_exhaustive]`; `Provider`, `Mcp`, `Config`, `UnexpectedEffect`, `Io`, `Core` |
| `ReactStep` / `LlmStep` / `JudgeStep` | `soma-llm/src/steps.rs:214`, `:491`, `:586` | The three `Step` implementations |

### The two design decisions worth knowing

**Retries live in the HTTP client, not the step.** A 429 is transport, not
domain. `RetryPolicy` is a `ProviderConfig` field, so it is TOML-overridable per
provider: 408/425/429/5xx and transport errors retry, everything else is fatal.
`Retry-After` is honoured in both RFC-9110 forms — delta-seconds and IMF-fixdate,
the latter via a hand-rolled 43-line parser at `soma-llm/src/openai_compat.rs:76`
`(!)` — capped by `max_ms`, with exponential backoff and full jitter otherwise.
The wall-clock `budget_secs` is checked **before** sleeping. Giving up reports
the last failure plus the first when they differ. Retries deliberately do **not**
reach the `EventBus`.

**Structured output degrades rather than failing.** `LlmRequest.schema` plus
`Quirks::supports_json_schema` produce a real `response_format` when the endpoint
can enforce it and a system-prompt append when it cannot. `max_repairs=1` means
one violation buys one correction, quoted back. Validation is **structural and
permissive on purpose** (root type, `required`, property types) — an invented
violation would cost a real model call to "fix" a correct answer.

And one rule that shows up as a bug if you do not know it: `finish_reason:
length` and `content_filter` are **errors** in `ReactStep`, not empty replies
(`LlmResponse::reject_non_answers`, `soma-core/src/agentic/effect.rs`).

### Patterns

- **Registry + strategy** — `Router` over `Arc<dyn LlmProvider>`, `Toolbox` over `Arc<dyn Tool>`.
- **Chain of responsibility** — both are `EffectHandler`s.
- **Error-as-data** — `soma-llm/src/tools.rs:182`, `:197`: a tool failure becomes a message the model can read and retry, never an aborted run.
- **Configuration as data** — the provider catalog is TOML, layered over 12 built-in defaults.
- **Retry with classification** — `Pushback` + `classify` (`soma-llm/src/openai_compat.rs:41`) + `retrying` (`:123`).
- **Constrained generation** — schema-as-protocol, with a bounded repair loop (`soma-llm/src/steps.rs:174`).

### Debt

- [D-26](/soma/internals/debt/#d-26--judgestep-scores-an-unparseable-reply-as-00) `JudgeStep` scores an unparseable reply as 0.0 **(Medium)**
- [D-69](/soma/internals/debt/#d-69--one-mcp-server-serializes-every-tool-call) one `Mutex<Pipe>` per MCP server serializes all its tool calls
- `LlmStep` re-hashes `ReactStep`'s identity with a salt and rebuilds `meta()` by struct update — two `Step` names for one behaviour (`soma-llm/src/steps.rs:491`)
- `Catalog::builtin()` hard-codes 12 provider URLs in Rust while claiming the catalog is data (`soma-llm/src/lib.rs:19`)
- Long functions: `body()` 84 lines (`openai_compat.rs:230`), `retrying()` 74 (`:123`), `on_reply` 73 (`steps.rs:148`), `ReactStep::poll` 70 (`steps.rs:238`)
- A blocking `std::thread::sleep` in the retry loop (`openai_compat.rs:158`) parks an OS thread for up to `budget_secs`. Correct given `reqwest::blocking`, but worth knowing.

---

## soma-agent (`somatize-agent`)

### Mandate

One thing: the research loop, expressed as a `Step`. Propose an experiment →
`Effect::Graph` → read the metrics → propose again or conclude.

`620 lines across 3 files · 0 traits · 2 public types · deps: somatize-core, somatize-memory, somatize-llm`

**The cleanest crate in the workspace** — no function reaches 60 lines, and it
has no debt entry of its own.

### Types

#### `Action` — `soma-agent/src/action.rs:15`

```rust
#[serde(tag = "action", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Action {
    RunExperiment {
        name: String,
        research_line: String,
        hypothesis: String,                          // required, not Option
        params: BTreeMap<String, serde_json::Value>, // "<node>.<param>"
        #[serde(default)] parent: Option<String>,
    },
    Conclude { reason: String },
}
```

Two variants, deliberately. The doc at `soma-agent/src/action.rs:4` argues that
extra verbs would be "the model thinking, and thinking does not need a protocol".
`Action::response_schema()` (`:48`) returns the JSON Schema the model is
constrained to — the enum *is* the wire format.

`hypothesis` being required rather than `Option` is the same idea: an experiment
without a hypothesis is not an experiment.

#### `ResearchStep` — `soma-agent/src/research.rs:38`

```rust
#[derive(serde::Serialize, somatize_core::SomaStep)]
#[soma(cache_version = "soma-research-step-v1")]
pub struct ResearchStep {
    model: String,
    objective: String,
    pipeline: Graph,                // by value — composition of soma-core::Graph
    max_iterations: usize,
    seed: Vec<ExperimentRecord>,    // composition of soma-memory
}
```

This is the join point of the graph and memory subsystems — the only type in the
workspace that composes both.

`config_hash` is **derived by macro** (`:263`), and the comment at `:261` records
why that matters: the hand-written version omitted `pipeline`, so two research
loops over different graphs shared a journal.

The loop in `poll` (`:274`):

```
None                → Await(ask the model)
Llm(response)       → parse_action
                        Conclude      → Done
                        RunExperiment → Await(Effect::Graph)
anything else       → Await(ask again)
```

### Patterns

- **Event-sourced state** — `completed()` (`soma-agent/src/research.rs:87`) reconstructs the record list from `ctx.history` rather than holding it in a field, so replay is exact. Documented at `:82`.
- **Constrained generation / schema-as-protocol** — `Action::response_schema`.
- **Free-function extraction** — `read_metrics` (`:319`), `collect_numbers` (`:332`), which excludes arrays from metrics on purpose (`:352`).

Two small observations, neither worth a debt ID: `to_object` (`:313`) is a
`BTreeMap → serde_json::Map` copy that exists only to satisfy a signature, and
`collect_numbers` flattens nested results into dotted metric names with no depth
bound.

---

## soma-memory (`somatize-memory`)

### Mandate

The experiment pool: an append-only journal of what was tried, what happened, and
what changed relative to the parent — plus the retrieval that makes it useful.
Nodes are runs; edges are the derivation moves applied to the parent.

`3 746 lines across 7 files · 2 traits · deps: somatize-core (+ chronos-vector behind a feature)`

### Modules

| File | Lines | Owns |
|---|---|---|
| `soma-memory/src/retrieval.rs` | 979 | BM25 + cosine + structure + recency + importance ranking |
| `soma-memory/src/derivation.rs` | 684 | `Change` (12 variants), `DerivationMove`, `derive()` |
| `soma-memory/src/knowledge_base.rs` | 627 | `KnowledgeBase`, `MemoryKnowledgeBase`, `Lineage` |
| `soma-memory/src/record.rs` | 595 | `ExperimentRecord`, `RecordKind`, `ResearchLine`, `Trend`, `ChangePoint` |
| `soma-memory/src/file_kb.rs` | 489 | The JSONL append-only backend |
| `soma-memory/src/chronos_kb.rs` | 341 | Feature-gated `chronos` — a `TemporalHnsw` backend |
| `soma-memory/src/lib.rs` | 31 | Re-exports |

### Public contracts

#### `KnowledgeBase` — `soma-memory/src/knowledge_base.rs:50`

**Three required methods, eleven defaulted.** The clearest **template method** in
the workspace: a backend implements storage and inherits every analytic.

```rust
pub trait KnowledgeBase: Send + Sync {
    fn record(&mut self, experiment: ExperimentRecord) -> Result<()>;  // required
    fn all(&self) -> Result<Vec<ExperimentRecord>>;                    // required
    fn len(&self) -> usize;                                            // required

    fn refresh, is_empty, get, search, retrieve, lineage,
       experiments_in_line, research_lines, promising_lines,
       trajectory, change_points, children                             // all provided
}
```

| Implementor | file:line | Note |
|---|---|---|
| `MemoryKnowledgeBase` | `soma-memory/src/knowledge_base.rs:336` | `Vec<ExperimentRecord>` |
| `FileKnowledgeBase` | `soma-memory/src/file_kb.rs:127` | **Decorator** over the in-memory one, adding a durable JSONL log with byte-offset `refresh` |
| `ChronosKnowledgeBase` | `soma-memory/src/chronos_kb.rs:162` | `cfg(feature = "chronos")` — HNSW vector index |

`Box<dyn KnowledgeBase>` appears in exactly one place workspace-wide:
`soma-mcp/src/context.rs:13`.

`(!)` Every defaulted query calls `self.all()?`, which clones the entire record
vector — see [Debt](#debt-3) below.

#### `Embedder` — `soma-memory/src/retrieval.rs:64`

```rust
pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}
```

**Zero implementors**, and the doc at `:55` says so explicitly: it is a seam for
an outside plug-in. Its `id` is stored on `Embedding.embedder_id` so vectors from
two different models are never compared — a small design detail that prevents a
whole class of silent wrongness.

### Types

| Item | file:line | Shape |
|---|---|---|
| `ExperimentRecord` | `soma-memory/src/record.rs:53` | **26 fields.** v1 core: id, name, hypothesis, pipeline_summary, params, metrics, timestamp, duration, parent, research_line, tags, notes. v2 additions, all `#[serde(default)]`: schema_version, kind, run_id, run_dir, `ArchitectureFingerprint`, objective, `RunConclusion`, `DerivationMove`, `GitInfo`, amends, embedding. **12 `with_*` builders** — the model the other wide structs should follow |
| `RECORD_SCHEMA_VERSION = 2` | `soma-memory/src/record.rs:20` | Format version |
| `RecordKind` | `soma-memory/src/record.rs:26` | `#[non_exhaustive]`; `Experiment \| Amendment \| Other(#[serde(other)])` |
| `Change` | `soma-memory/src/derivation.rs:31` | `#[non_exhaustive]`, **12 variants**: node added/removed/replaced/reconfigured, edge added/removed, param changed/added/removed, search-space changed, code changed, unspecified |
| `DerivationMove` | `soma-memory/src/derivation.rs:175` | `{ from, to, changes, metric_delta, summary }` |
| `MetricDelta` | `soma-memory/src/derivation.rs:161` | `{ before, after, delta }` — signed on purpose |
| `RetrievalQuery` | `soma-memory/src/retrieval.rs:98` | text, architecture, now, half_life_days, embedding, limit, research_line, tags |
| `ScoreComponents` | `soma-memory/src/retrieval.rs:163` | `{ lexical, structural, recency, importance }`, each in `[0, 1]` |
| `ScoredRecord` | `soma-memory/src/retrieval.rs:177` | `{ record, score, components }` + `why()` at `:188` |
| `Lineage` / `LineageNode` | `soma-memory/src/knowledge_base.rs:33`, `:24` | Focus + ancestors + descendants with depth |
| `ResearchLine` / `Trend` / `ChangePoint` | `soma-memory/src/record.rs:355`, `:370`, `:394` | Line analytics |

### The retrieval formula

Additive, and every term is inspectable through `ScoredRecord::why()`:

```
0.40 · BM25  +  0.25 · structural  +  0.15 · recency  +  0.20 · importance
```

with **importance floored at 0.6 for failures that carry a conclusion**
(`soma-memory/src/retrieval.rs:340`). That floor is the design decision: dead ends
must stay retrievable, or the pool only remembers what worked and the agent
repeats every mistake.

Parent resolution is equally deliberate: `parent=` → `$SOMA_PARENT_RUN` →
`.soma/HEAD` → none. `HEAD` advances only on success, and a parent is **never**
inferred from timestamps.

### Patterns

- **Template method** — 3 required, 11 defaulted. → [Patterns](/soma/internals/patterns/#template-method)
- **Decorator** — `FileKnowledgeBase` wraps `MemoryKnowledgeBase` and adds durability with an incremental byte-offset refresh (`soma-memory/src/file_kb.rs:8`).
- **Event sourcing with amendments** — `RecordKind::Amendment` plus the `amends` field. Nothing is ever rewritten.
- **Schema versioning** — `RECORD_SCHEMA_VERSION`, `legacy_schema_version()` (`record.rs:128`), `#[serde(other)]`.
- **Explaining scorer** — the score decomposes into named terms rather than being an opaque number.
- **Pure-function diff** — `derive()` (`derivation.rs:209`) is pure and deterministic, used both at capture time and by on-demand `kb_diff`, so a recorded derivation and a live one cannot disagree.
- **Builder** — 12 `with_*` on `ExperimentRecord`.

### Debt {#debt-3}

- **O(n) materialization behind every defaulted query** — `soma-memory/src/knowledge_base.rs:76` (`get`), `:89`, `:95`, `:101`, `:105`, `:120`, `:139`, `:146`, `:186` all call `self.all()?`. With `FileKnowledgeBase` the journal is already in memory, so this is a `Vec` clone per call rather than I/O — but `kb.get(a)` followed by `kb.get(b)` in `soma-python/src/readers.rs:203` clones it twice.
- `rank()` (`soma-memory/src/retrieval.rs:200`) is 91 lines combining five scoring concerns.
- `ChronosKnowledgeBase` adds a `semantic_search` (`soma-memory/src/chronos_kb.rs:112`) with no counterpart in the trait, so a caller holding `Box<dyn KnowledgeBase>` cannot reach it.
- ~150 lines of hand-rolled IR primitives (`tokenize` `:467`, `Bm25Index` `:499`) inside a domain crate.
- `Lineage::root()` (`soma-memory/src/knowledge_base.rs:44`) returns the focus when there are no ancestors — correct, but it silently covers a *corrupt* parent chain the same way.

---

## soma-mcp (`somatize-mcp`)

### Mandate

Expose the project to a model over MCP: 20 tools spanning source access, graph
execution, and the experiment pool. It defines no trait and owns no domain
logic — it is a front controller over `soma-memory`, the filesystem, and a Python
subprocess.

`3 267 lines across 9 files · 0 traits · deps: somatize-memory, somatize-core`

### Modules

| File | Lines | Owns |
|---|---|---|
| `soma-mcp/src/render.rs` | 1 039 | **19 pure `&T -> String` functions** — the crate's best structural decision |
| `soma-mcp/src/context.rs` | 571 | `SomaContext` + 13 tool handlers `(!)` |
| `soma-mcp/src/tools/mod.rs` | 507 | `all_tools()` (349 lines of inline JSON schemas) + `dispatch` `(!)` |
| `soma-mcp/src/exec.rs` | 370 | The embedded Python `DRIVER` + `GraphRunner` `(!)` |
| `soma-mcp/src/server.rs` | 258 | The stdio JSON-RPC loop |
| `soma-mcp/src/tools/knowledge.rs` | 245 | The 7 `kb_*` pool tools `(!)` |
| `soma-mcp/src/protocol.rs` | 230 | JSON-RPC + MCP DTOs |

### The tool surface

`dispatch(ctx, tool_name, params)` at `soma-mcp/src/tools/mod.rs:362` is a
21-arm string match:

```
kb_find_similar | kb_lineage | kb_diff | kb_record_conclusion
| kb_branch_from | kb_summarize_run | kb_stats        → tools::knowledge::*

list_filters | read_filter_source | write_filter_source
run_pipeline | run_study
record_experiment | query_knowledge_base | get_trajectory
| get_change_points | list_research_lines | promising_lines
create_research_line | generate_report                → SomaContext::*

_ => ToolCallResult::error("Unknown tool: {tool_name}")
```

`reads_knowledge(tool_name)` at `:400` is a **second** string predicate deciding
whether to refresh the knowledge base first. `(!)` Adding a knowledge tool
therefore means editing three literals — the schema, the dispatch arm, and the
refresh predicate. Contract tests catch two of the three:
`soma-mcp/src/tools/mod.rs:418` asserts every defined tool dispatches, `:453`
asserts the count is 20, `:494` asserts the refresh policy.

### `run_pipeline` and `run_study` actually execute

A model supplies nodes (`module.Class`, `path.py:Class`, or a bare name found in
the files `list_filters` returns), edges and an input. `GraphRunner`
(`soma-mcp/src/exec.rs:256`) generates a driver that builds a `soma.Graph` and
runs it in a Python subprocess rooted at the project, with `$SOMA_PYTHON`
choosing the interpreter. Runs are tracked by default, so `kb_summarize_run` can
read them back.

Two details that are easy to get wrong and are handled:

- A config value written `{"__search__": {…}}` becomes a **class-level** search dimension — that is where `FilterMeta` looks, and it is the whole difference between `run_pipeline` and `run_study`.
- The driver writes its result to a **file, never stdout**, because a `print` inside a user's filter must not corrupt the reply.

`(!)` This is unsandboxed execution of project code reachable from a model
(`soma-mcp/src/exec.rs:16`, `sys.path.insert(0, os.getcwd())` at `:31`). The file
argues it is no worse than the pre-existing `write_filter_source`, which is true,
and it is still worth knowing.

### The rendering layer

`soma-mcp/src/render.rs` is 19 pure functions with zero I/O, and **the MCP text
IS the API**: every result ends with a `next:` line and a `run_dir:`. That is a
deliberate contract — the model reads prose, so the prose is the interface, and
keeping it in pure functions means it can be tested without a server.

### Patterns

- **Front controller / command dispatch** — `dispatch`.
- **Facade** — `SomaContext` over memory + filesystem + subprocess.
- **Presentation layer separated from logic** — `render.rs`.
- **Error-as-data** — every handler returns `ToolCallResult::error(…)` rather than a JSON-RPC error, so the model can read and retry. Documented at `soma-mcp/src/context.rs:100`.
- **Out-of-process execution** — `GraphRunner` + the embedded driver, mirroring `soma-worker`'s daemon.
- **Contract tests over the dispatcher** — three of them, catching most of the three-literal problem above.

### Debt {#debt-4}

- [D-16](/soma/internals/debt/#d-16--two-knowledge-base-front-ends-already-divergent) duplicated KB front-end, **already divergent** (limit clamped to 50 here, 100 in Python) **(Medium)**
- [D-19](/soma/internals/debt/#d-19--two-embedded-python-interpreters-as-rust-string-constants) the second embedded Python interpreter — 225 lines in a `const &str` **(Medium)**
- [D-08](/soma/internals/debt/#d-08--somacontext-mixes-three-subsystems) `SomaContext` mixes three subsystems
- [D-28](/soma/internals/debt/#d-28--unwrap-on-the-json-rpc-hot-path) two `unwrap`s on the JSON-RPC hot path
- `all_tools` is 349 lines of inline JSON schemas in one function (`soma-mcp/src/tools/mod.rs:13`)
- `soma-mcp/src/server.rs:60` answers a JSON-RPC **notification**, which the spec forbids; the comment at `:61` acknowledges it
- `GraphRunner::run` returns `Result<Value, String>` (`soma-mcp/src/exec.rs:281`) — one of only two stringly-typed error channels in the workspace
