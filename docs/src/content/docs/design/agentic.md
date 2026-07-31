---
title: Agentic Graphs
description: Effectful nodes, data-dependent control flow, and why an agentic flow is just a Soma graph.
---

An agentic flow in Soma is a graph whose nodes are *effectful* and whose control flow depends on data. That is the whole design. There is no second engine, no agent DSL, and no catalog of agent node types — because everything Soma already does (schema validation, content-addressed caching, search spaces, studies, lineage) is exactly what agentic flows are missing everywhere else.

This page explains the shape, then the evidence for it.

## The shape

Four pieces on top of the existing runtime:

| Piece | What it is | Where |
|---|---|---|
| `Step` | An effectful node: `poll()` returns a `Transition` | `soma-core/src/step.rs` |
| `Effect` | What a step asks the world for: LLM, tool, graph, sleep | `soma-core/src/effect.rs` |
| Effect journal | Record-once, replay-on-resume, over the existing cache | `soma-runtime/src/effects/` |
| Provider layer | OpenAI-compatible access to ~12 providers, as data | `soma-llm/` |

### `Step`: a synchronous trait over an asynchronous world

```rust
pub trait Step: Send + Sync {
    fn config_hash(&self) -> CacheKey;
    fn meta(&self) -> StepMeta;
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub enum Transition {
    Await(Vec<Effect>),                 // run these, then poll me again
    Spawn { specs, join },              // dynamic fan-out
    Goto { target, carry },             // hand control to another node
    Suspend { reason },                 // wait for a human, or for later
    Done(Value),
}
```

`poll` is synchronous and cheap. It decides; a *driver* performs. Every other Rust agent framework makes this trait `async` and colours the whole runtime with it. Keeping it synchronous buys three things:

1. **The Python bridge stays simple.** Python decides, Rust performs. No holding the GIL across I/O.
2. **The journal is trivial.** A step's behaviour is a pure function of its inputs and the effect results it has been handed. Feed it recorded results and it takes the identical path.
3. **Steps compose with filters.** Both are just nodes.

A step also has no state of its own. `StepCtx` carries `results` (this turn) and `history` (every previous turn), and anything a step accumulates it derives from those. This is not stylistic: derive the history and a replay reconstructs *exactly* the history the original run had, because it replays exactly the same results. Keep it in a field and the two can drift.

### Six structural node kinds

```rust
pub enum NodeKind {
    Filter { .. },     // deterministic, cacheable by content
    Step { .. },       // effectful, journaled
    SubGraph { .. },
    Loop { max_iterations, until },
    Branch { arms },
    Map { .. },
}
```

Everything else — LLM, tool, retriever, judge, aggregator, human, memory, orchestrator, panel — is a filter or a step in the registry, and every *pattern* is a function returning a graph. See [`soma.agentic`](#the-patterns-are-functions).

### Writing a step in Python

A step is any object with `poll(ctx)`, duck-typed exactly the way a filter's `forward` is. It returns one of five transitions, as plain dicts built by helpers, so what crosses into Rust is data rather than a class hierarchy:

```python
from soma.agentic import Done, Await, Spawn, Goto, Suspend, Run, Llm

class Fanout:
    _cache_version = "1"

    def poll(self, ctx):
        if ctx.turn == 0:
            return Spawn([Run("worker", task) for task in ctx.input])
        return Done([r["output"] for r in ctx.results])

g.node("fanout", Fanout())
g.register_step("worker", Worker())   # spawnable, and not a root
```

`Spawn` is the reason this exists. The width of a fan-out is often a property of the data — a plan with two tasks wants two workers and one with nine wants nine — and that is the one thing a static topology cannot say. `register_step` puts a step in the library without adding a node, because a node with no edges is a root and would also run once on the graph's own input.

`Goto` needs a declared control edge (`g.handoff(a, b)`); handing control somewhere the graph never said it could is an error rather than a silent jump. `ctx` carries `input`, `turn`, `results` and `history` and nothing else on purpose: a step that accumulates rebuilds from `history`, because replay feeds back identical results and a field on `self` would drift from them.

`poll` runs under the GIL on whichever thread the driver is using, so Python steps fan out for I/O concurrency, not for CPU parallelism.

### Control flow, and the rule that governs it

> An unreadable signal is an error, never a default.

A loop's stop condition and a branch's arm selector are read from a designated node's output. If that output carries no signal the run fails with a message naming the node, rather than silently exhausting a hundred iterations or running arm zero.

The compiler resolves both at compile time:

- **Ownership by dominance.** A loop owns its body-entry nodes and everything they dominate; a branch owns each arm's entry and its dominated subgraph. Without that exclusion, the body would be emitted twice — once inside the loop, once after it.
- **Declared arms.** `Branch` carries the labels its condition may produce. A declared arm with no edge, and an edge labelling an undeclared arm, are both compile errors.
- **Resolved conditions.** `LoopCondition::BodyTerminal` becomes `WhenSignaled(node)` at compile time; a body with several terminals is an error, not a race.

Two semantics are worth stating because they are easy to get wrong:

**A branch passes its input through.** The selector is control, not data. Leaving the label in place would hand the chosen agent the string `"billing"` instead of the customer's question. If the request needs transforming, put a filter *before* the branch.

**A loop carries.** The loop node's value is seeded from its input, then replaced after each pass by its `carry_from` node's output. That is what the next iteration reads. `carry_from` is separate from `until` on purpose: what a loop carries and what tells it to stop are different questions, and a fixed-round debate has the first without the second.

### The journal

Effects are keyed two ways:

```rust
if effect.is_pure() {
    // content-addressed: shared across runs forever
    CacheKey::from_parts(&[b"soma-journal-v1", b"pure", &effect_key])
} else {
    // sited: replay-only, tied to this run's position
    CacheKey::from_parts(&[b"soma-journal-v1", b"sited",
        run_id, node_id, &turn, &index, &effect_key])
}
```

A pure effect (a deterministic tool) memoizes like a filter. An impure one (a model call) is recorded once for *this* run and replayed on resume, never re-served to a different run. Failures are not recorded — a replay retries them, because a transport error is not a result.

This is the durable-execution discipline Temporal and Restate established, over the two-table `ActionStore` Soma already had. It is what makes a nondeterministic run reproducible, and therefore comparable to its parent in the experiment pool.

### The patterns are functions

```python
from soma.agentic import react, route, refine, debate, board, parallel_vote, orchestrate

refine(worker=Draft(), judge=soma.Judge(model="ollama/qwen2.5", rubric="..."), max_rounds=4)
route(classifier, {"billing": BillingAgent(), "tech": TechAgent(), "default": Escalate()})
debate([alice, bob], rounds=3, judge=critic)
board([solver, solver, solver], rounds=2)
```

Each returns an ordinary `Graph`, built from the same `node`, `connect`, `branch` and `loop` anyone can call. Adding a pattern is adding a function.

`board` is the one worth reading as an argument rather than a convenience. It is the multi-agent debate of [Du et al. (ICML 2024)](https://arxiv.org/abs/2305.14325): a panel answers independently, a chair reads every answer and records a decision, and the next round shows the panel what the chair recorded — the summarizer variant that paper introduces for larger panels, which is what makes the chair a moderator rather than a tallying clerk. The loop is `brief → members → chair`, the chair also reads the brief so the round after it still knows the question, and the chair's `done` is the exit condition: a panel that has converged stops instead of buying the rounds it was allowed.

The default chair, `MajorityVote`, is a stateless filter rather than a model — the aggregator the paper actually closes a debate with, costing no tokens and keeping the model as the only stochastic part of the flow. Pass a `Judge` or an agent instead to have a model moderate. And because the result is a graph, "five members or three, two rounds or four" is a `search_space()` dimension rather than an opinion; notebook 14 replicates the paper's GSM8K ordering and then searches that space.

### Providers are data

```toml
[providers.ollama]
base_url = "http://localhost:11434/v1"
auth = { type = "none" }
```

Ollama, HuggingFace's router, NVIDIA NIM, Kimi, GLM, DeepSeek, Mistral, Groq, Together, OpenRouter and vLLM all speak `POST /chat/completions`. One client covers them; a catalog entry plus a `Quirks` record covers the differences (`max_tokens` vs `max_completion_tokens`, whether an empty tool list is tolerated, whether the system prompt is a message). Models are addressed `provider/model`, or bare when the graph declares a default with `use_provider`. New provider, new TOML entry — usually no code.

The built-ins ship configured but unreachable: each hosted one names the environment variable holding its key, and reads it only when you actually call it. A missing variable is an error at that point rather than an empty `Bearer` header and a confusing 401.

| Provider | Endpoint | Reads |
|---|---|---|
| `ollama` | `$OLLAMA_HOST` + `/v1`, else `localhost:11434` | — |
| `vllm` | `$VLLM_URL`, else `localhost:8000/v1` | — |
| `nvidia` | `integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` |
| `hf` | `router.huggingface.co/v1` | `HF_TOKEN` |
| `groq`, `kimi`, `glm`, `deepseek`, … | vendor endpoint | `GROQ_API_KEY`, `MOONSHOT_API_KEY`, `ZHIPU_API_KEY`, … |

To change any of it, write `~/.soma/providers.toml` (or point `$SOMA_PROVIDERS` at a file). An entry reusing a built-in's name replaces it wholesale, which is how you repoint `ollama` at another host or read a key from a variable you already have under a different name:

```toml
[providers.nvidia]
base_url = "https://integrate.api.nvidia.com/v1"
auth = { type = "bearer", env = "MY_EXISTING_NVIDIA_VAR" }
```

Note that `provider/model` splits only on the *first* segment, and only when it names a known provider — so `nvidia/meta/llama-3.3-70b-instruct` resolves to the `nvidia` provider asking for `meta/llama-3.3-70b-instruct`, and a bare `meta-llama/Llama-3-70B` stays whole.

## Why this shape

### Node taxonomies converge on about seven things

Across LangGraph, CrewAI, AutoGen/AG2, the OpenAI Agents SDK, Airflow 3, Dagster, Prefect 3, Temporal, OpenClaw and OpenFang, the "node types" fall into two groups that should not be mixed:

- **Structural**, which the engine must understand to schedule: sequence, parallel, conditional, loop, dynamic fan-out, suspension, subgraph. About seven, in all of them.
- **Behavioural**, which it does not: LLM, tool, retriever, judge, merge, accumulator, human, memory, orchestrator, panel, debate.

Airflow, Dagster and Prefect settled this a decade ago — operators are library, the scheduler understands dependencies plus `expand()` plus branching. The counter-example is instructive: a production agentic engine we examined has 24 closed-enum node variants, of which the `Tool` node returns only an error, the human-in-the-loop node always fails, and the fail-edges are unimplemented — all three still in the frontend catalog and the documentation. Every new type was a core change, so eventually nobody changed the core.

### ~79% of multi-agent failures are contract failures

[MAST](https://arxiv.org/abs/2503.13657) (NeurIPS 2025) annotated 1600+ traces across 7 frameworks at κ=0.88: **41.8% specification/design failures, 36.9% inter-agent misalignment** (context lost in handoffs, incompatible formats), 21.3% verification failures. Roughly four fifths are contract problems, not model problems.

Soma is one of very few runtimes with a `Schema` (dtype + shape) and compile-time validation. Extending `DataType` with `Text`, `Messages` and `Json(schema)` and typing the branch arms turns a large part of that 36.9% into a compile error. The branch pass-through described above closes another slice of it by construction.

### The five composable workflow patterns

Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) names five: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. The first three are static topology — Soma had them. Orchestrator-workers needs dynamic fan-out. Evaluator-optimizer is a loop with a data-dependent exit, which is structurally the `StudyRunner` Soma already had tested.

### Agentic graphs as optimizable objects

This is the part that is not another LangGraph.

- [GPTSwarm](https://proceedings.mlr.press/v235/zhuge24a.html) (ICML 2024) treats LLM agents as computational graphs and optimizes both node prompts and **edge connectivity**.
- [ADAS](https://openreview.net/pdf?id=D01WR1yVW2) searches at program level over workflow structures.
- [AFlow](https://arxiv.org/abs/2410.10762) (ICLR 2025) runs MCTS over the workflow space with execution feedback — up to +57% on complex tasks.

All three need three things that everyone else builds by hand and Soma already had: a declarative search space, a runner with sampling and pruning, and a record of what was tried with lineage. So:

```python
g.node("writer", soma.Agent(
    model=soma.search(choices=["ollama/qwen2.5", "kimi/kimi-k2"]),
    system=soma.search(choices=["be terse", "be detailed"]),
))
g.optional("retriever", "critic")        # the edge is a dimension too

study = g.study("shape-and-prompt", strategy="tpe", n_trials=40,
                objectives=[("score", "maximize")])
```

An agent's constructor arguments are its hyperparameters, so the space is declared where the value goes. A filter declares its space as a class attribute; both land in the same `search_space()`, and a `Study` cannot tell them apart. Cutting an edge sets it aside whole, so restoring it restores the graph byte-identically — a trial that changes the topology must leave the next trial starting from the same place.

Median pruning is not optional here. A study over an agentic graph spends real money, and the report shows tokens per trial next to the metric.

## The serialization contract

A graph's JSON is the interface for anything outside this process: a visual
editor, another language, a run directory, a worker. `graph_json()` produces
it, and `begin_run` writes the same bytes to `graph.json` in every tracked
run directory.

```json
{
  "nodes": [
    {"id": "draft",  "label": "draft",
     "kind": {"type": "Step", "step_name": "Agent"}},
    {"id": "critic", "label": "critic",
     "kind": {"type": "Step", "step_name": "Judge"}},
    {"id": "refine", "label": "refine",
     "kind": {"type": "Loop", "max_iterations": 3,
              "until": {"type": "WhenSignaled", "node": "critic"}}},
    {"id": "router", "label": "router",
     "kind": {"type": "Branch", "arms": ["a", "default"]}}
  ],
  "edges": [
    {"id": "e_0", "source": "draft",  "target": "critic", "kind": "Data",    "label": null},
    {"id": "e_1", "source": "refine", "target": "draft",  "kind": "Control", "label": null},
    {"id": "e_2", "source": "router", "target": "a",      "kind": "Control", "label": "a"}
  ],
  "strategy": {"type": "Local"}
}
```

What the shape guarantees:

- **`kind` is adjacently tagged everywhere.** A `NodeKind` is `{"type": ...}`
  plus its fields; `LoopCondition` is `{"type": ..., "node": ...}`. Internal
  tagging was tried and could not represent `WhenSignaled` at all — serde
  cannot put an internal tag on a newtype variant wrapping a string, so a
  graph containing any resolved loop failed to serialize, run directory and
  all.
- **Structure is in the graph; behaviour is not.** A node names its filter or
  its step kind. What that filter *does* lives in the registry, keyed by node
  id. An editor can rewire a graph without being able to run it.
- **Control edges carry the structure.** A loop owns what its control edges
  reach; a branch arm's label is the edge's `label`. Everything the compiler
  needs to reconstruct ownership is in `edges`, nowhere else.
- **Enums are `#[non_exhaustive]`,** so a reader must tolerate an unknown
  `type` rather than assuming the set is closed.
- **Node ids are stable and meaningful.** They are what events, run
  directories, cache keys, search-space dimensions (`"<node>.<param>"`) and
  optional edges (`"edge:<source>-><target>"`) are all keyed by.

This is a stable contract, and deliberately so: it is what a visual editor
built elsewhere needs in order to read, edit and hand back a Soma graph
without depending on Soma's internals.

## Consequences worth knowing

**Prompts land in the cache directory.** The journal records what was sent. Before putting credentials in a system prompt, set `journal = false` on the step's `StepMeta`.

**A step has no fit phase.** A graph containing only steps needs no `fit()` before `forward()`.

**A branch condition may be a step.** An LLM router is the common case, so the executor dispatches the condition node to the step library when it is registered there.

**A failed pipeline is a result.** `Effect::Graph` returns `EffectResult::Failed` rather than erroring: a configuration that will not run is a finding, and ending the run would discard everything learned before it.

## Sources

- [Why Do Multi-Agent LLM Systems Fail? (MAST, NeurIPS 2025)](https://arxiv.org/abs/2503.13657)
- [Building Effective AI Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [GPTSwarm: Language Agents as Optimizable Graphs (ICML 2024)](https://proceedings.mlr.press/v235/zhuge24a.html)
- [Automated Design of Agentic Systems](https://openreview.net/pdf?id=D01WR1yVW2)
- [AFlow: Automating Agentic Workflow Generation (ICLR 2025)](https://arxiv.org/abs/2410.10762)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Control Flow Primitives — LangGraph](https://deepwiki.com/langchain-ai/langgraph/3.5-control-flow-primitives)
- [Agent Workflows Are Rediscovering Durable Execution](https://nittikkin.medium.com/agent-workflows-are-rediscovering-durable-execution-be110661ed8c)
- [Model Context Protocol](https://modelcontextprotocol.io)
