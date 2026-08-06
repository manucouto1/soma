---
title: Symbol Index
description: Every public trait, struct and enum in the workspace, A to Z, with its crate, its definition site, and the page that describes it.
---

**278 public types**: 29 traits, 178 structs, 71 enums. This page exists to be
searched — if you have a symbol name and want to know where it lives and what it
is for, start here and follow the link.

Ten of these are marked **pyclass**. They are `pub(crate)` in Rust and public to
Python, which makes them API in the way that matters: `PyGraph` is what a user
writes as `soma.Graph`.

Excluded: anything inside a `#[cfg(test)]` module, and genuinely internal types
even where the reference pages discuss them — `PlanCtx`, `StreamNode`,
`ChunkLru`, `McpTool`, `StepRuntime`, `Behaviour`, `StepSpec`, `StoreConfig`,
`ScheduleState`, `Bm25Index`, and the four FFI bridges (`PyFilterBridge`,
`PyStepBridge`, `PyToolAdapter`, `PyPbtExecutor`). Look for those on the crate's
page.

## The 29 traits, by role

The traits are the architecture. Everything else is a type that participates in
one of them.

| Role | Traits |
|---|---|
| **Node behaviour** | `Filter`, `Step`, `AsAny` |
| **Compilation** | `NodeRegistry` |
| **Execution** | `Runner`, `ForwardStrategy`, `Transport` |
| **Caching & storage** | `CacheStore`, `ActionCache`, `BlobStore`, `DataStore`, `StateStore` |
| **Effects** | `EffectHandler` |
| **Search & tuning** | `Sampler`, `Pruner`, `Searchable`, `TrialExecutor`, `PbtExecutor`, `StudyIo` |
| **Distributed training** | `StrategyContext`, `StrategyExecutor`, `GradientAggregator`, `StateAggregator` |
| **Observability** | `EventSink`, `Tracker` |
| **Models & tools** | `LlmProvider`, `Tool` |
| **Knowledge** | `KnowledgeBase`, `Embedder` |

`Searchable` and `StudyIo` are the only two that are not object-safe.
`Embedder` has zero implementors, deliberately — it is a seam for an outside
plug-in (`soma-memory/src/retrieval.rs:55`).

## Keeping this page true

The table below is **generated**. After adding, removing or moving a public type:

```bash
cd docs && node scripts/gen-symbol-index.mjs
```

Everything above this heading is hand-written and preserved; only the table is
replaced. `npm run check` verifies every row against the source, so a stale index
fails the build rather than quietly misleading a reader.

## A–Z

### A

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Action`](/soma/internals/agentic/#soma-agent-somatize-agent) | enum | `soma-agent` | `soma-agent/src/action.rs:15` |
| [`ActionCache`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/action.rs:134` |
| [`ActionResult`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/action.rs:110` |
| [`AgentCost`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/summary.rs:121` |
| [`AgenticActivity`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:149` |
| [`AgentNodeActivity`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:181` |
| [`ArchitectureFingerprint`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/fingerprint.rs:36` |
| [`AsAny`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/any.rs:13` |
| [`Assignment`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/scheduler.rs:57` |
| [`Auth`](/soma/internals/agentic/#soma-llm-somatize-llm) | enum | `soma-llm` | `soma-llm/src/catalog.rs:34` |

### B

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Batched`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/forward.rs:99` |
| [`BayesianSampler`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/sampler/bayesian.rs:16` |
| [`BlobStore`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/action.rs:143` |

### C

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`CacheActivity`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:89` |
| [`CacheKey`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/cache.rs:18` |
| [`CacheStore`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/cache.rs:204` |
| [`CacheTier`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/cache.rs:150` |
| [`Capabilities`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:44` |
| [`Catalog`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/catalog.rs:374` |
| [`Change`](/soma/internals/agentic/#soma-memory-somatize-memory) | enum | `soma-memory` | `soma-memory/src/derivation.rs:31` |
| [`ChangePoint`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/record.rs:394` |
| [`ChronosKnowledgeBase`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/chronos_kb.rs:24` |
| [`ClientSelection`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:161` |
| [`CommunicationProtocol`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:119` |
| [`CompileMode`](/soma/internals/execution/#soma-compiler-somatize-compiler) | enum | `soma-compiler` | `soma-compiler/src/compiler.rs:17` |
| [`Compiler`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/compiler.rs:202` |
| [`CompileResult`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/compiler.rs:50` |
| [`CompositeObjective`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/study.rs:69` |
| [`ContentBlock`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/message.rs:59` |
| [`ContentHash`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/action.rs:52` |
| [`ContentItem`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:167` |
| [`Context`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executor.rs:124` |
| [`CoordinatorToWorker`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:484` |

### D

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`DataRef`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/store/mod.rs:98` |
| [`DataStore`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/store/mod.rs:208` |
| [`DataTransfer`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/scheduler.rs:124` |
| [`DataType`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/schema.rs:12` |
| [`DerivationMove`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/derivation.rs:175` |
| [`Diagnostic`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/compiler.rs:28` |
| [`DiagnosticLevel`](/soma/internals/execution/#soma-compiler-somatize-compiler) | enum | `soma-compiler` | `soma-compiler/src/compiler.rs:40` |
| [`Dimension`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/schema.rs:115` |
| [`Direction`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/study.rs:15` |
| [`Distribution`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/filter.rs:53` |
| [`DistributionPlan`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/scheduler.rs:93` |

### E

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Edge`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/graph.rs:229` |
| [`EdgeKind`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/graph.rs:220` |
| [`EdgeRef`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/fingerprint.rs:60` |
| [`Effect`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/effect.rs:35` |
| [`EffectDriver`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/effects/mod.rs:57` |
| [`EffectHandler`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/effect.rs:262` |
| [`EffectJournal`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/effects/journal.rs:51` |
| [`EffectResult`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/effect.rs:278` |
| [`EffectSite`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/effects/journal.rs:36` |
| [`EffectSpan`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:218` |
| [`Embedder`](/soma/internals/agentic/#soma-memory-somatize-memory) | «trait» | `soma-memory` | `soma-memory/src/retrieval.rs:64` |
| [`Embedding`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/record.rs:43` |
| [`EntryMeta`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/cache.rs:185` |
| [`EnvLockfile`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/env_manager.rs:25` |
| [`EnvManager`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/env_manager.rs:38` |
| [`EnvType`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/env_manager.rs:14` |
| [`Event`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/event.rs:54` |
| [`EventBus`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/event_bus.rs:22` |
| [`EventEnvelope`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/tracking.rs:225` |
| [`EventSink`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/tracking.rs:243` |
| [`ExecutionMode`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:192` |
| [`ExecutionPlan`](/soma/internals/execution/#soma-compiler-somatize-compiler) | enum | `soma-compiler` | `soma-compiler/src/plan.rs:19` |
| [`ExperimentRecord`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/record.rs:53` |
| [`ExploitStrategy`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:180` |
| [`ExploreStrategy`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:200` |

### F

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`FederatedAggregation`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:136` |
| [`FileKnowledgeBase`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/file_kb.rs:25` |
| [`Filter`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/filter.rs:120` |
| [`FilterKind`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/filter.rs:16` |
| [`FilterMeta`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/filter.rs:73` |
| [`FlagCount`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/summary.rs:76` |
| [`FnPbtExecutor`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/pbt.rs:63` |
| [`FnTool`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/tools.rs:62` |
| [`FnTrialExecutor`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/study.rs:144` |
| [`ForwardEnv`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/forward.rs:25` |
| [`ForwardStrategy`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/forward.rs:40` |
| [`FsActionStore`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/cache/fs_store.rs:40` |

### G

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`GcPolicy`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/cache/gc.rs:29` |
| [`GcReport`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/cache/gc.rs:48` |
| [`GitInfo`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/tracking.rs:58` |
| [`GpuInfo`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:59` |
| [`GradientAggregation`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:89` |
| [`GradientAggregator`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/strategy.rs:132` |
| [`Graph`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/graph.rs:293` |
| [`GraphEffectMode`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/effect.rs:134` |
| [`GraphHandler`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/effects/graph_handler.rs:47` |
| [`GraphInfo`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executor.rs:28` |
| [`GraphOverlay`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/viz.rs:53` |
| [`GraphRunner`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/exec.rs:256` |
| [`GraphSession`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/graph_session.rs:38` |
| [`GraphSummaryInfo`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/tracking.rs:74` |
| [`GridSampler`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/sampler/mod.rs:54` |

### H

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`HashAlgo`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/action.rs:32` |
| [`HealthFlagRecord`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:130` |

### I

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`InitializeResult`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:92` |
| [`InputSource`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:88` |

### J

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`JoinPolicy`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/effect.rs:482` |
| [`JsonlEventSink`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/jsonl_sink.rs:19` |
| [`JsonRpcError`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:52` |
| [`JsonRpcRequest`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:7` |
| [`JsonRpcResponse`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:32` |
| [`JudgeStep`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/steps.rs:527` |

### K

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`KnowledgeBase`](/soma/internals/agentic/#soma-memory-somatize-memory) | «trait» | `soma-memory` | `soma-memory/src/knowledge_base.rs:50` |

### L

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Lineage`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/knowledge_base.rs:33` |
| [`LineageNode`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/knowledge_base.rs:24` |
| [`LlmError`](/soma/internals/agentic/#soma-llm-somatize-llm) | enum | `soma-llm` | `soma-llm/src/error.rs:29` |
| [`LlmHandler`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/lib.rs:183` |
| [`LlmProvider`](/soma/internals/agentic/#soma-llm-somatize-llm) | «trait» | `soma-llm` | `soma-llm/src/lib.rs:72` |
| [`LlmRequest`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/effect.rs:144` |
| [`LlmResponse`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/effect.rs:330` |
| [`LlmStep`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/steps.rs:459` |
| [`LoadMetrics`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:68` |
| [`LocalCache`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/cache/local.rs:15` |
| [`LocalDataStore`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/store/mod.rs:241` |
| [`LocalRunner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/runner/local.rs:16` |
| [`LocalTracker`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/local_tracker.rs:27` |
| [`LoopCondition`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/control.rs:28` |
| [`LoopSignal`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/control.rs:46` |

### M

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`McpClient`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/mcp_client.rs:29` |
| [`MedianPruner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/pruner.rs:33` |
| [`MemoryCache`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/cache/memory.rs:16` |
| [`MemoryKnowledgeBase`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/knowledge_base.rs:318` |
| [`MemoryStateStore`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/state.rs:46` |
| [`Message`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/message.rs:137` |
| [`Messages`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/message.rs:189` |
| [`MetricDelta`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/derivation.rs:161` |
| [`MetricPoint`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:111` |
| [`MetricRecord`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/event.rs:24` |
| [`ModelInfo`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/lib.rs:57` |

### N

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Node`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/graph.rs:68` |
| [`NodeCacheCounts`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:100` |
| [`NodeCatalog`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/node_catalog.rs:79` |
| [`NodeCost`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/summary.rs:65` |
| [`NodeImpl`](/soma/internals/execution/#soma-runtime-somatize-runtime) | enum | `soma-runtime` | `soma-runtime/src/node_catalog.rs:37` |
| [`NodeKind`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/graph.rs:26` |
| [`NodeMeta`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/node.rs:72` |
| [`NodeOutcome`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/node.rs:44` |
| [`NodeOverlay`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/viz.rs:33` |
| [`NodeRegistry`](/soma/internals/execution/#soma-compiler-somatize-compiler) | «trait» | `soma-compiler` | `soma-compiler/src/compiler.rs:65` |
| [`NodeSpan`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:66` |
| [`NodeSpec`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/effect.rs:450` |
| [`NodeStatus`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/viz.rs:20` |

### O

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Objective`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/study.rs:36` |
| [`OpenAiCompatible`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/openai_compat.rs:188` |
| [`Origin`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/cache.rs:161` |
| [`OutputDelivery`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:560` |

### P

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Partition`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/strategy.rs:107` |
| [`PbtConfig`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/pbt.rs:22` |
| [`PbtExecutor`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/executors/pbt.rs:55` |
| [`PbtRunner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/pbt.rs:84` |
| [`PercentilePruner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/pruner.rs:106` |
| [`Phase`](/soma/internals/execution/#soma-compiler-somatize-compiler) | enum | `soma-compiler` | `soma-compiler/src/scheduler.rs:76` |
| [`PipelineFile`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:474` |
| [`PlanPhase`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/scheduler.rs:109` |
| [`PlanResult`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:585` |
| [`PlanSummary`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/event.rs:37` |
| [`PopulationMember`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/pbt.rs:41` |
| [`ProviderConfig`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/catalog.rs:231` |
| [`Pruner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/pruner.rs:10` |
| [`PruningStrategy`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/study.rs:179` |
| [`PyAgent`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/agentic.rs:283` |
| [`PyGraph`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/graph.rs:28` |
| [`PyJudge`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/agentic.rs:422` |
| [`PyPbt`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/pbt.rs:35` |
| [`PyRun`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/run.rs:13` |
| [`PyStepCtx`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/agentic.rs:506` |
| [`PyStudy`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/study.rs:173` |
| [`PythonPipelineJob`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:451` |
| [`PythonProcess`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/python_process.rs:538` |
| [`PyTool`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/agentic.rs:47` |
| [`PyTrial`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/study.rs:112` |
| [`PyWorker`](/soma/internals/python/) | struct · pyclass | `soma-python` | `soma-python/src/worker.rs:54` |

### Q

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Quirks`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/catalog.rs:96` |

### R

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`RandomSampler`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/sampler/mod.rs:175` |
| [`ReactStep`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/steps.rs:32` |
| [`RecordKind`](/soma/internals/agentic/#soma-memory-somatize-memory) | enum | `soma-memory` | `soma-memory/src/record.rs:26` |
| [`RemoteRunner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/runner/remote.rs:73` |
| [`RemoteTarget`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/filter.rs:64` |
| [`ResearchLine`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/record.rs:355` |
| [`ResearchStep`](/soma/internals/agentic/#soma-agent-somatize-agent) | struct | `soma-agent` | `soma-agent/src/research.rs:38` |
| [`ResourceLimits`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/detect.rs:19` |
| [`RetrievalQuery`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/retrieval.rs:98` |
| [`RetryPolicy`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/catalog.rs:144` |
| [`Role`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/message.rs:22` |
| [`Router`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/lib.rs:93` |
| [`RunConclusion`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/summary.rs:157` |
| [`RunContext`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/runner/mod.rs:32` |
| [`RunInfo`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:42` |
| [`RunKind`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/tracking.rs:27` |
| [`RunManifest`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/tracking.rs:93` |
| [`RunMode`](/soma/internals/execution/#soma-runtime-somatize-runtime) | enum | `soma-runtime` | `soma-runtime/src/executor.rs:92` |
| [`Runner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/runner/mod.rs:124` |
| [`RunOutcome`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/summary.rs:27` |
| [`RunReader`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:36` |
| [`RunState`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/tracking.rs:46` |
| [`RunStatus`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/tracking.rs:193` |
| [`RunSummary`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/summary.rs:331` |

### S

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`S3DataStore`](/soma/internals/distribution/#soma-store-somatize-store) | struct | `soma-store` | `soma-store/src/s3.rs:24` |
| [`Sampler`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/sampler/mod.rs:22` |
| [`Scalarizer`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/study.rs:48` |
| [`Scale`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/search.rs:17` |
| [`Schema`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/schema.rs:104` |
| [`ScoreComponents`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/retrieval.rs:163` |
| [`ScoredRecord`](/soma/internals/agentic/#soma-memory-somatize-memory) | struct | `soma-memory` | `soma-memory/src/retrieval.rs:177` |
| [`Searchable`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/search.rs:321` |
| [`SearchDimension`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/search.rs:36` |
| [`SearchSpace`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/search.rs:172` |
| [`SearchStrategy`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/study.rs:113` |
| [`SerializedFilter`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:152` |
| [`SerializedPlan`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/protocol.rs:209` |
| [`ServerCapabilities`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:110` |
| [`ServerInfo`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:127` |
| [`ShutdownSignal`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/server.rs:33` |
| [`SimpleNodeRegistry`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/compiler.rs:91` |
| [`SleepHandler`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/effects/sleep_handler.rs:18` |
| [`SomaContext`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/context.rs:9` |
| [`SomaError`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/error.rs:14` |
| [`Standard`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/forward.rs:46` |
| [`StateAggregator`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/strategy.rs:139` |
| [`StateStore`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/state.rs:24` |
| [`Step`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/step.rs:250` |
| [`StepCtx`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/step.rs:115` |
| [`StepMeta`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/step.rs:177` |
| [`StopReason`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/effect.rs:392` |
| [`StorageConfig`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/store/mod.rs:161` |
| [`StoreMeta`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/store/mod.rs:21` |
| [`StrategyContext`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/strategy.rs:33` |
| [`StrategyExecutor`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/strategy.rs:120` |
| [`Stream`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/forward.rs:58` |
| [`StreamFormat`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/store/mod.rs:145` |
| [`StreamMessage`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:620` |
| [`StreamMode`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/filter.rs:37` |
| [`StreamOutput`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/stream.rs:283` |
| [`StreamRun`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/stream.rs:73` |
| [`Study`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/study.rs:319` |
| [`StudyIo`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/study_io.rs:19` |
| [`StudyRunner`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/study.rs:160` |
| [`SubprocessFilter`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/python_process.rs:982` |
| [`SuspendReason`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/effect.rs:507` |

### T

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`TieredCache`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/cache/tiered.rs:11` |
| [`Tool`](/soma/internals/agentic/#soma-llm-somatize-llm) | «trait» | `soma-llm` | `soma-llm/src/tools.rs:53` |
| [`Toolbox`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/tools.rs:92` |
| [`ToolCallResult`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:149` |
| [`ToolOutcome`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/tools.rs:26` |
| [`ToolsCapability`](/soma/internals/agentic/#soma-mcp-somatize-mcp) | struct | `soma-mcp` | `soma-mcp/src/protocol.rs:117` |
| [`ToolSpec`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/tool.rs:16` |
| [`Tracker`](/soma/internals/foundation/#soma-core-somatize-core) | «trait» | `soma-core` | `soma-core/src/tracking.rs:255` |
| [`TrainingStrategy`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/strategy.rs:22` |
| [`Transition`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/step.rs:43` |
| [`Transport`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/runner/remote.rs:18` |
| [`TransportContext`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/strategy.rs:541` |
| [`Trend`](/soma/internals/agentic/#soma-memory-somatize-memory) | enum | `soma-memory` | `soma-memory/src/record.rs:370` |
| [`Trial`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/study.rs:235` |
| [`TrialContext`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/executors/study.rs:48` |
| [`TrialExecutor`](/soma/internals/execution/#soma-runtime-somatize-runtime) | «trait» | `soma-runtime` | `soma-runtime/src/executors/study.rs:133` |
| [`TrialMetricHistory`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/pruner.rs:25` |
| [`TrialOutcome`](/soma/internals/execution/#soma-runtime-somatize-runtime) | enum | `soma-runtime` | `soma-runtime/src/executors/study.rs:24` |
| [`TrialSpan`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/tracking/reader.rs:241` |
| [`TrialState`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/study.rs:210` |
| [`TrialSummary`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/summary.rs:138` |

### U

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Usage`](/soma/internals/foundation/#soma-core-somatize-core) | struct | `soma-core` | `soma-core/src/effect.rs:411` |

### V

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Value`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/value.rs:15` |
| [`ValueStatus`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/virtual_value.rs:65` |
| [`Verdict`](/soma/internals/agentic/#soma-llm-somatize-llm) | struct | `soma-llm` | `soma-llm/src/steps.rs:508` |
| [`VirtualValue`](/soma/internals/foundation/#soma-core-somatize-core) | enum | `soma-core` | `soma-core/src/virtual_value.rs:26` |

### W

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`Worker`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/worker.rs:16` |
| [`WorkerError`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/error.rs:24` |
| [`WorkerIdentity`](/soma/internals/execution/#soma-runtime-somatize-runtime) | struct | `soma-runtime` | `soma-runtime/src/strategy.rs:557` |
| [`WorkerInfo`](/soma/internals/execution/#soma-compiler-somatize-compiler) | struct | `soma-compiler` | `soma-compiler/src/scheduler.rs:15` |
| [`WorkerRegistry`](/soma/internals/distribution/#soma-coordinator-somatize-coordinator) | struct | `soma-coordinator` | `soma-coordinator/src/registry.rs:73` |
| [`WorkerStatus`](/soma/internals/distribution/#soma-coordinator-somatize-coordinator) | struct | `soma-coordinator` | `soma-coordinator/src/registry.rs:22` |
| [`WorkerToCoordinator`](/soma/internals/distribution/#soma-worker-somatize-worker) | enum | `soma-worker` | `soma-worker/src/protocol.rs:333` |
| [`WsTransport`](/soma/internals/distribution/#soma-worker-somatize-worker) | struct | `soma-worker` | `soma-worker/src/ws_transport.rs:17` |

### Z

| Symbol | Kind | Crate | Defined at |
|---|---|---|---|
| [`ZarrStore`](/soma/internals/distribution/#soma-store-somatize-store) | struct | `soma-store` | `soma-store/src/zarr.rs:201` |
