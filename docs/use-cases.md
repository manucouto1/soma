# Use cases

The project moves in vertical slices. Every use case reaches all the way to
Python, and is considered closed when it answers every guarantee on its
questionnaire.

---

## CU1 — Creating a graph

```python
g = soma_next.Graph()
g.node("clean", Clean())
g.edge("clean", "vectorize")
```

Status: **closed**. 16 tests in Rust, 13 in Python.

### The design decision that comes first

Before `node()` you have to answer **what a node is**. In the original that
question was answered with `NodeKind` (5 structural variants), `NodeMeta`
(metadata common to filters and steps, with `cacheable`/`deterministic` as data
rather than an `if is_step`) and a `NodeCatalog` that is the single registry. It
is a reasonable answer; it is not the only one, and it is not inherited.

What is worth looking at in the original before deciding, because they are scars
from real mistakes: `soma-core/src/graph/node.rs` (172 lines) and its tests
`graph_node.rs` — `a_filter_keeps_its_caching_contract`,
`a_step_is_not_output_cacheable`, `schemas_survive_both_directions`.

### Questionnaire (from `soma-core/tests/unit/graph*.rs`)

**Construction**
- [x] an empty graph is valid
- [x] a single-node graph is valid
- [x] a node can be added with an explicit id
- [x] a node can be added without an id and the system gives it one (snake_case of the class)
- [x] adding the same thing twice does not duplicate — **decided**: identity = id,
      and the derived id suffixes `_2`, `_3`. Two identical filters are two nodes;
      deduplicating by content is a caching decision, not a topology one, and that
      use case does not exist
- [x] a linear pipeline has the structure it says it has

**Topology queries**
- [x] roots and leaves
- [x] a node's predecessors and successors
- [x] topological order of a linear chain
- [x] topological order with parallel branches
- [x] a cycle is an error — **decided**: when the edge is added, not when it is walked

**Validation** — **decided**: there is no `validate()`. The constructors return
`Result` and the invariant holds at all times, so an invalid `Graph` is not a
value that exists. What it buys: `topological_sort()` does not return `Result`,
because it cannot fail.
- [x] duplicate ids are rejected, in `add_node`
- [x] an edge to a non-existent node is rejected, in `add_edge`

**Deferred to later use cases** — it is in the same test files, do not drag it
into CU1: serialization (`graph_serde_roundtrip`), rendering (`to_mermaid*`,
`to_text`, overlays), control nodes (`loop_and_branch_nodes`, `subgraph_node`,
all of `graph_control.rs`), and the `Filter`/`Step` contract (`graph_filter.rs`,
`graph_step.rs`).

### Decisions taken in CU1

1. **The core's `Graph` is topology only.** Ids and edges. What a node does is
   none of its business, because creating a graph does not need to know. The
   id → Python object map lives in `python/`. That is why `core` depends on
   nothing.
2. **Errors at insertion, not in a `validate()`.** There is no instant at which
   the graph is malformed.
3. **DAG by construction.** The cycle is rejected in `add_edge`. *Risk taken*: if
   a future use case needed back edges, it would have to be revisited — in the
   original, loops are nodes, not backward edges, so the bet is that it is not
   needed.
4. **`NodeId` is a type, not a `String`.** There are more ids coming.
5. **O(n) where it could be O(1).** Adjacency is computed on the fly. The code
   reads at a glance and no use case has asked for anything else.

### What did NOT go in, and why

`target=` in `node()`. I wrote it as sugar for creating the edge in the same
step, and checking it against the original turned up that there `target` is
**not an edge**: it is the supervision target the optimizer's `step()` reads.
Reusing the name with another meaning is exactly the kind of thing that makes a
system incomprehensible, so out it went: it is not CU1 and it has no consumer
today.

---

## CU2 — Executing a graph of filters

```python
g.node("add", Add(1))
g.node("double", Double())
g.edge("add", "double")
g.forward(41)          # → 84.0
```

Status: **closed**. 29 tests in Rust, 25 in Python.

### The starting decision

**The engine goes in Rust, Python is a wrapper.** It is your decision and it has
a consequence worth keeping in mind: it forces `Value` to exist already. If the
core executes, the data has to have a shape Rust understands.

The four roles, which are easy to confuse:

| piece | role | where |
|---|---|---|
| `Graph` | the **structure** | `core/src/graph.rs` |
| `Catalog` | the **store** of implementations | `core/src/filter.rs` |
| `Filter` | the **contract** of an executable unit | `core/src/filter.rs` |
| `Graph::run` | the **engine** | `core/src/execution.rs` |

### Decisions taken

1. **`Value` with four variants**: `Null`, `Text`, `Bytes`, `Tensor`. There is no
   `Json` because it would require `serde_json` and the core depends on nothing;
   there is no opaque `Object` because it is only good for sending something down
   a wire and there is no wire. The conversion error says what is missing rather
   than inventing a representation.
2. **`Filter` has one method**: `forward(&Value) -> Result<Value, FilterError>`.
   No `fit` (training is another use case), no `config_hash` (caching), no `meta`
   (compiler), no `composite_fit` (autograd).
3. **No `state` parameter.** The original passes `state` on every `forward`, even
   to stateless filters. State arrives with `fit`.
4. **`Send + Sync` on the trait** — and it is not decoration: PyO3 requires a
   `#[pyclass]` to be `Send`, the graph carries the catalog inside, and the bound
   climbs to the trait. The compiler found it on its own.
5. **An object without `forward` fails when registered**, not halfway through a
   run.
6. **A bool does not cross the boundary.** `True` as the tensor `1.0` is the kind
   of silent conversion nobody understands later.

### The two decisions the engine was NOT taking

Both were settled in CU4. `Fanin` and `ManyLeaves` no longer exist as errors.

## CU3 — The shape of the execution, and steps

```python
g.step("agent", Agent())            # an object with poll(ctx)
g.forward(x, driver=MyDriver())     # whoever serves what it asks for
g.plan()                            # how it is going to be walked
```

Status: **closed**. 39 tests in Rust, 38 in Python.

### The question: are there several ways of executing a graph?

Yes — local, remote, by turns, in parallel. But the original's answer is **not a
trait of executors**: it is an enum of ten variants (`Sequence | Parallel |
Execute | Step | Loop | Branch | Remote | Composite | Stream | Empty`) and a
single `match`. `Remote` is not another executor: it is a variant that *wraps* a
sub-plan.

It is the same principle we had already found — variation as data, not as a
subtype — and it has a concrete advantage over a bare function or a trait: when
`Parallel` was added halfway through this use case, the compiler pointed at the
single place that had to decide what to do with it. A wildcard arm would have
said nothing.

### The missing piece: compiling

Between the structure and the engine there is now a step:
`compile(&Graph, &Catalog) -> Plan`. It decides the shape, and along the way
**everything structural is detected before anything executes**. The engine no
longer works out where each node's input comes from: the plan says so.

It needs the catalog because the shape depends on what each node is — a filter is
called once, a step is driven by turns.

### Decisions taken

1. **`Plan` is an enum**, not a trait. Closed, exhaustive, no wildcards.
2. **`Parallel` means "the branches do not depend on each other"**, not "it runs
   on threads". Spreading them out is a decision that does not change the result
   and nobody has asked for it.
3. **A fan produces a `Value::List`** with each branch's output, in order.
4. **`Executor` is a type**, not a bare function: executing needs context (today
   the store and the driver; tomorrow a cache and events). That "tomorrow" is
   what the original calls `GraphSession`.
5. **What a step asks for is opaque to the core**: a `Value` the `Driver`
   interprets. That is why there are no LLMs, no tools, no effect log — that is
   library and persistence, not the contract.
6. **`Transition` has two variants**: `Done` and `Await`. `Spawn`, `Goto` and
   `Suspend` will arrive with their own use case. No `#[non_exhaustive]`, on
   purpose.
7. **A cap of 64 turns.** A step that does not finish is a bug in the step; the
   cap makes it show up as a named error and not as a stalled process.
8. **`Value` loses `Tensor` and gains `Number` and `List`.** Nobody was producing
   a shaped tensor, and the round trip to Python has to be symmetric: what goes
   in as a list comes out as a list.

### What did NOT go in

**`Plan::Remote`.** There is no transport, so it would be a variant nobody can
execute. What the enum buys is precisely that adding it the day there is a worker
is one more variant, and that the compiler points at every place that has to
decide.

> **CU4 note**: the `Plan::Parallel` and the `Fanin`/`ManyLeaves` errors
> described above no longer exist. See below for why.

## CU4 — Fans in both directions

```python
Graph.somatize(Left().named("left") | Right().named("right")) >> Mean()
# `Mean` receives {"left": …, "right": …}
```

Status: **closed**. 46 tests in Rust, 52 in Python.

### The question: where does aggregation live?

The original answers it **twice**, and both answers teach something:

- **On the edge** (forward): it joins what arrives into a `serde_json::Map` keyed
  by the source node, and the aggregator is an ordinary node — `MajorityVote` is
  a `Filter`.
- **In training** (federated): `FederatedAggregation::{FedAvg, FedProx, FedYogi}`
  and `GradientAggregation::{AllReduce, ParameterServer, …}` — enums of
  algorithms, wrapped in `StateAggregator`/`GradientAggregator` traits with
  **exactly one implementor each: the enum itself**. Both are on the orphan-trait
  list. When they enumerated FL's real algorithms they got an enum; the trait on
  top bought nothing.

And the trap to spot: **federated aggregation is not fan-in.** In FedAvg what is
averaged are the states of N workers when a round closes — there is no edge and
no predecessors there. It is an operation inside `fit`, and it will arrive with
it.

### Decisions taken

1. **There is no `Aggregator` trait.** An aggregator is a filter that reads a map.
   `Mean`, `MajorityVote`, `Concat`, `WeightedMean` are library.
2. **`Value::Map`, ordered.** A `HashMap` iterates differently in each process, so
   flattening it to a list would give a different order every time and the
   content hash — once the cache arrives — would be useless. The pairs follow the
   edges' declaration order, which is also what mirrors a Python `dict`: the
   round trip gives the same dict.
3. **Both directions have the same shape.** Several inputs → a map keyed by each
   source. Several leaves → a map keyed by each leaf. A diamond comes back round.
4. **The weight travels with the value.** FedAvg weights by each client's
   samples; neither a list nor a map of raw outputs gives that weight. Each
   branch produces something like `{"update": …, "n": 128}` — another independent
   reason to have `Value::Map`.

### What was removed

**`Plan::Parallel`**, added in CU3. It broke on fan-in: on a diamond both
branches claimed the join node and it executed twice. The right shape is for
**every step to carry where its input comes from** (`Execute { node, from }`).
With that the plan stays self-contained — the engine does not look at the graph
again — and the fans fall out without any special variant. `Parallel` will come
back when it means something it does not mean today: spreading across threads.

The `CompileError::Fanin` and `ManyLeaves` errors went with it. `CompileError` is
left with a single variant.

## CU5 — The DSL

```python
from soma_next import Filter, Graph

g = Graph.somatize(Source() >> (Left().named("left") | Right().named("right")) >> Mean())
g.forward(0)
```

```rust
let (graph, catalog) = (filter("source", Add(1.0))
    >> (filter("left", Add(10.0)) | filter("right", Add(100.0)))
    >> filter("join", Mean))
.somatize()?;
```

Status: **closed**. Same tests, plus `build.rs` and `test_dsl.py`.

`>>` chains, `|` opens branches, and a `>>` after some open branches closes them
— which is CU4's fan-in: the node on the right receives the map. The expression
above *is* the diamond.

### Decisions taken

1. **The same syntax in both languages.** In Rust it falls out of implementing
   `std::ops::Shr` and `BitOr` on a type of our own; no macro needed
   (`macro_rules!` would give syntax the operators do not, but that is not needed
   here). The precedence matches too: `>>` binds tighter than `|`, so the
   branches go in parentheses in both.
2. **It is called `somatize`, which is the project's verb.** In Python it is a
   classmethod of `Graph`; in Rust, a method of `Wire`, because it returns **two**
   things — the structure and the store — and neither contains the other.
3. **There is a Python class on top of the Rust one.** `somatize` walks an
   expression of Python objects, so it cannot be in Rust; and a `#[pyclass]` is an
   immutable type, so it cannot be hung on it at import time either. It is
   declared in a subclass body, which is also the only thing `help()`, an IDE and
   mypy can see. It is the same structure as the original Soma (`soma/_graph.py`),
   and for the same reasons.
4. **`Filter` and `Step` are abstract, and inheritance is what decides.** Each
   requires its method with `@abstractmethod`, so a `class X(Filter)` without
   `forward` cannot even be instantiated, and `isinstance` is the only question
   the DSL asks.

   It was a correction: they were born as empty mixins that asked by duck typing
   whether the object *had* `poll`. That let two ugly things through — a
   `class X(Step)` with only `forward` ended up registered as a filter without a
   warning, and an object with both methods gave three different answers
   depending on whether it came in through `node()`, through `step()` or through
   the DSL. The names promised a contract nobody enforced.

   `node()` and `step()` are still the lower door and accept an outside object
   that inherits from nothing, because there the type is chosen by the caller.
   What they do not accept is the contradiction: `node()` with something that
   inherits from `Step` is an error that says which call is the right one.
5. **`Wire` does not materialize until `somatize`.** It records where you enter
   and where you leave, plus the lists of nodes and edges. That way joining two
   pieces is concatenating lists rather than merging two graphs, and a repeated
   id is caught at the end, once.
6. **The DSL is nothing but `node` and `edge`.** There is a test that builds the
   same graph both ways and compares nodes, edges and plan.

## CU6 — A single contract

```rust
pub trait Node: Send + Sync {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError>;
}
```

Status: **closed**. 48 tests in Rust, 61 in Python.

### The question: why two types?

The difference between a filter and a step was a single thing: whether it can
finish on its own. But **that was already said in `Transition`** — a filter is a
node that always answers `Done`. Having two traits duplicated in the type system
a distinction that lived in the return value, and with it propagated upwards the
obligation to know which one each node was: catalog, plan, engine, errors,
adapters, DSL. **35 places.**

Two more alternatives were tried before deciding, and both were rejected for
concrete reasons:

- **A sugar trait with a blanket impl** (`impl<T: Filter> Node for T`). Compiled:
  `error[E0034]` — with two traits in scope the name `forward` is ambiguous *even
  when the arities differ*, because Rust resolves the name before the arguments.
  And `error[E0119]` — a type that implements `Filter` can no longer implement
  `Node` by hand, so a node could not evolve from always finishing to asking for
  a turn without being rewritten entirely.
- **State as a continuation** (`Pending { requests, resume }`). Simpler on the
  surface — it carries the whole `Ctx` — but it breaks deterministic replay: the
  log relies on `forward` being deterministic given `(turn, results)`, and
  resuming a continuation would require serializing a `Box<dyn Node>`. The
  typestate variant dies sooner: the `Catalog` is a heterogeneous map that erases
  the type parameter, and it does not cross into Python at all.

### Decisions taken

1. **One trait, one method, `forward` in both languages.** Without the second
   trait there is no name ambiguity, so there is no need to call it `advance` in
   Rust.
2. **`Pure` is a struct, not a trait.** Sugar for wrapping a function. It obeys
   our own rule: if you cannot name two implementors, it is a struct. And it
   reintroduces neither of the two errors above.
3. **`input` travels apart from the context.** A node that finishes on the first
   turn never looks at `ctx`; it should not have to cross a struct to reach the
   only thing it cares about.
4. **In Python `Filter` and `Step` stay, as a façade.** They are the two
   convenient calling conventions — `forward(x)` returns a value,
   `forward(x, ctx)` returns a transition — and inheritance still decides which is
   used. Underneath there is a single contract. The separation stopped being the
   system's and became the door's.

### What disappeared

`trait Step`, `FilterError`, `StepError`, `StepCtx`, `NodeImpl`, `Plan::Step`,
`RunError::{Filter, Step, WrongKind}`, `insert_filter`/`insert_step`,
`run_filter`/`drive_step`, and `PyStep` as a separate adapter.

`RunError` goes from 7 variants to 5. `Plan` from 4 to 3. `compile` no longer
needs the catalog to know what kind each node is — only to check that there is
one.

### What was gained, and was not in the plan

**A node can evolve.** It starts always returning `Done`, and the day it needs to
consult something an `Await` branch is added to the same body, without changing
type or registration. With two traits that was `error[E0119]`. There is a test.

## CU7 — The same mechanism in both languages

```python
from soma_next import Await, Done, Graph, Node

class Clean(Node):
    def forward(self, x, ctx):
        return Done(x.strip())

class Ask(Node):
    def forward(self, x, ctx):
        if ctx.turn == 0:
            return Await([f"and {x}?"])
        return Done(ctx.results[0])

Graph.somatize(Clean() >> Ask()).forward("  hello  ", driver=MyDriver())
```

Status: **closed**. 47 tests in Rust, 56 in Python.

### What CU6 left undone

CU6 unified the core but left Python with two classes (`Filter` and `Step`) and
two calling conventions. It was an asymmetry with no reason: if underneath there
is one contract, up top there is no reason to pick a door.

### Decisions taken

1. **A single `Node` class in Python**, with `forward(input, ctx)` returning a
   transition. `Filter` and `Step` disappear, and with them `g.step()`,
   `kind_of`, `ensure_kind` and `Graph`'s two overrides.
2. **`Ctx`, `Done` and `Await` are `#[pyclass]`es**, not dictionaries. They are
   the core's own concepts crossing the seam, so the adapter recognizes them by
   type instead of guessing from a dict's keys, and `ctx.turn` reads as it does
   in Rust rather than `ctx["turn"]`.
3. **One adapter, not two.** `PyFilterNode` and `PyStepNode` merge into `PyNode`.
4. **`Pure` is gone.** It was sugar in the core for wrapping a function, and each
   implementation decides how it transitions without needing a shortcut.

### The price, said plainly

A node that only transforms now writes `return Done(x.strip())` and accepts a
`ctx` it does not look at. It is more ceremony than `return x.strip()`, and it is
deliberate: it buys that **there are not two ways of writing a node**, that the
DSL has a single door, and that a node can gain a turn by adding a branch rather
than by changing class.

## CU8 — A value that crosses without being converted

```python
class Layer(Node, nn.Module):
    def __init__(self, m):
        nn.Module.__init__(self); self.m = m
    def forward(self, x, ctx):
        return Done(Opaque(self.m(x)))

g = Graph.somatize(Layer(l1) >> Layer(nn.ReLU()) >> Layer(l2))
y = g.forward(Opaque(x))
y.pow(2).sum().backward()      # crosses all three nodes
```

Status: **closed**. 53 tests in Rust, 64 in Python.

### The problem

`Value` is a **conversion** boundary, and some values do not survive being
converted. The case that motivated it: a torch tensor mid-autograd-graph.
Measured — round-tripping it through lists gives `requires_grad = False,
grad_fn = None`. The gradient graph breaks.

### The decision

One variant, and only one:

```rust
Opaque(Arc<dyn Any + Send + Sync>)
```

It is not a `PyObject` because the core does not depend on PyO3 and is not going
to start. `Arc<dyn Any + Send + Sync>` lets the Python crate store a `Py<PyAny>`
inside and retrieve it with `downcast_ref`, without the core knowing there is
Python behind it.

**What the variant means, and where everything else follows from**: this value
only exists in this process and in this run.

| property | consequence | correct? |
|---|---|---|
| not hashed by content | the node is not memoized | yes — memoizing a tensor mid-autograd would be a bug |
| not serialized | that subgraph does not travel to another machine | yes — that is why the original sends gradients over the wire, not the graph |
| compared only by identity (`Arc::ptr_eq`) | two wrappers of the same object are distinct | it is the only thing the core can assert |

The boundaries of the future cache and of remote execution end up **visible in
the type** rather than being a rule somebody has to remember.

### It is asked for by hand, on purpose

`Opaque(x)` gets written. Making an unknown object turn opaque by itself was
rejected: the honesty of "a `set` does not cross" would be lost, and a hole
everything fits through becomes the default path — leaving the graph without
cache, without schemas and without distribution all at once, with nobody
noticing.

A **registry of opaque types** that `soma_next.torch` would fill on import was
also rejected: it adds mutable global state and a dependency on import order, to
save one word.

The node that receives it sees it **unwrapped**, so it is only written on
returning (and once at the graph's input).

### Limitations, measured

- **`torch.compile` does not fuse across nodes.** Three nodes → 3 graphs and 2
  breaks; the same thing without Rust → 1 graph, 0 breaks. It is correct (the
  backward pass gets through) but **the node is the compilation unit**. User
  mitigation, one line: `torch.compile(my_module)` inside the node.
- **No content cache on those edges.** For training that is right; for inference
  it is a real loss. It is recovered by converting on purpose at the edge:
  `Done(y.detach().tolist())`.
- **Out of the schemas' reach** when they arrive: there is no dtype and no shape.
- **The GIL** serializes dispatch per node; torch releases it during kernels.

### The pattern, tested end to end

`python/tests/test_pipeline_torch.py` assembles a four-node pipeline —
lemmatizer (no gradients) → encoder → bottleneck → LSTM classifier — and
**trains** it: 12,571 parameters, the loss drops from 1.09 to 0.005 in 40 steps.
It is in the tests in full because besides checking, it documents the pattern.

Three things it teaches, and they are not obvious:

- **The two regimes coexist.** The lemmatizer returns text, which crosses
  converted; the three nodes with parameters return `Opaque`. The boundary falls
  by itself where the gradient graph begins, without declaring it.
- **The node holds the modules, it does not inherit from `nn.Module`.**
  Inheriting registers the parameters on its own, but breaks calling the node as
  a module: our `forward` carries `ctx` and torch calls it without one
  (`TypeError`). Verified.
- **The training loop goes outside**, and the line that collects the parameters
  by walking `g.nodes()` is exactly the pain a `soma_next.torch.parameters(g)`
  would erase. It is in plain sight so the decision is made with the example in
  front of you.

### What did NOT go in

`soma_next.torch` — `module()`, `parameters()`, the training loop — is left for
when it is clear how it should work. **The core provides the hole; whoever knows
what goes in it is a library**, and that separation is what let this be closed
without deciding that.

## What comes next, undecided (16 August 2026)

The discussion was left open here, with **CU12 in doubt**. "Micro-batches" covers
three different problems that have neither the same owner nor the same value:

| problem | what solves it | whose it is | consumer today? |
|---|---|---|---|
| the batch does not fit in memory | splitting it and accumulating gradients | the **Trainer**'s, five lines | yes, and it is 80% of cases |
| the bubble: `cuda:1` idle while `cuda:0` computes | chaining micro-batches | the **graph**'s | doubtful |
| bounding the live activations | a real 1F1B scheduler | **nobody's**, and that is the problem | no |

The two reasons for the doubt:

**The bubble may already not exist.** CUDA launches asynchronously: a
micro-batch loop on the host already overlaps the devices without a scheduler,
because nothing synchronizes along the way — `Opaque` wraps the tensor and there
is no `.item()` in the seam. What a scheduler would add **has to be measured
before it is written**, and with a single GPU here it cannot be measured.

**Real 1F1B is not ours.** Its value is not the bubble but bounding how many
micro-batches have their activations live, and for that the backward passes have
to be interleaved with the forwards. The backward pass is fired by the Trainer,
not by the engine. A 1F1B scheduler would require the plan to know about the
backward pass — i.e. putting training inside the graph, which is exactly what
CU11 decided against. The version that does fit the levels — micro-batches
forward only — is the one with the least value.

### The three candidates

- **A. Gradient accumulation in the Trainer.** The real problem of the 80%,
  level 2, zero Rust. Honest and small; it teaches nothing new about the design.
- **B. A local worker: `Plan::Remote` targeting a process on this same machine.**
  ← *recommended.* It is the only one with a benefit **measurable here and now**:
  two Python nodes in a wave serialize against the GIL, and `test_waves.py`
  already documents that as a known limitation; in two processes, they do not.
  All Rust — the `Transport` trait, the serialization, `Opaque` as a
  `CompileError` — the first new trait since CU2, and it prepares CU13 without
  needing a second machine.
- **C. What does not fit on the GPU**: being able to ask that two structurally
  parallel branches do not overlap, and to release what nobody reads any more.
  All Rust, and it touches the most delicate parts: the compiler and the engine.

## Following use cases (not opened)

The order comes from the August 2026 literature review, and the content of the
last two changed when CU11 closed: the separation into three levels — graph,
training run, study — reorders which mechanism solves what.

What actually happened with the two next ones, since this list was written:
**CU12 was candidate B** — the worker and `Plan::Remote` — and **CU13 was the
cache**, which was not on the list at all: it came out of the store the worker's
`have`/`want` had been waiting for. Micro-batches did not open, and they are
still where the grain per item joins them.

- ~~CU12 — micro-batches~~: overlapping inside a branch (GPipe, 1F1B). **Graph
  level**: it is one forward from the inside. Still open, now with `.mapped()`
  waiting on it
- ~~CU13 — `Plan::Remote`~~: **done as CU12**. Transport (the only new trait,
  `Transport`), and `Opaque` crossing the wire. Only for spreading **one graph**
  across hosts — a network that does not fit on one machine — which is model
  parallel and **split learning** at once. Spreading *whole training runs* is
  another thing and does not need it
- CU14 — federated: `map` over the clients and `reduce` with FedAvg, at the study
  level. FedAvg, FedProx and company are library — **functions**, not nodes. This
  is where the question of what a training run exports as state has to be
  answered
- *(candidate)* releasing what nobody reads any more, and being able to ask that
  two structurally parallel branches do not overlap. Both are born of "it does
  not fit on the GPU" and both are at the graph level
- *(candidate, with an entry condition)* **training from Rust**. It is not opened
  until there is a consumer, and the consumer has a name: a federated client that
  trains **without a CPython loaded**. Researched on 16 August 2026, and these
  four results save having to work it out again:
  - `tch::Tensor` is `Send` but **not `Sync`** (`unsafe impl Send for Tensor` in
    `wrappers/tensor.rs`, and no `Sync` anywhere in the crate), so it **does not
    fit in `Value::Opaque`**, whose bound is `Arc<dyn Any + Send + Sync>`. `tch`
    is ruled out short of wrapping every tensor in a `Mutex`
  - `candle_core::Tensor` **is** `Send + Sync` — it carries an
    `Arc<RwLock<Storage>>` inside, and its own code explains they chose the
    `RwLock` for exactly that — so it would fit today **without touching the
    core**. Verified by compiling
  - the limit that does not move: **a graph is all-Python or all-Rust for the
    tensors**. An `Opaque` put there by Python carries a `PyObject` inside, and a
    Rust node doing `downcast_ref::<candle::Tensor>()` would get `None`. There is
    no cheap bridge: converting for real would mean copying the raw data and
    losing the autograd graph, which is what `Opaque` prevents
  - the order, if the day comes: **first a Rust node with parameters**, then the
    collection, and the Trainer last. Never starting with the Trainer

## CU9 — Branches run at the same time

```python
g = Graph.somatize(
    Source()
    >> ((Encoder() >> Bottleneck()) | (Other() >> Other2()))
    >> Join()
)
g.plan()      # Sequence([Execute, Wave([Sequence, Sequence]), Execute])
g.forward(x)  # both branches, on two threads, start to finish
```

Status: **closed**. 88 tests in Rust, 86 in Python.

### The question: what does a wave group?

`Plan::Parallel` was added in CU3 and removed in CU4 because it broke on the
diamond: its branches overlapped — both claimed the join node — and it executed
twice. It was said then that it would come back "when it means something it does
not mean today: spreading across threads". This is that day, and the question
left to answer is **what goes inside each branch**.

Two answers were tried and the first was rejected with a counterexample:

- **By topological level** (Kahn by levels). Each wave is an antichain, and its
  members are lone steps. It is correct and no node can be duplicated. But with
  `a >> (b >> b2 >> b3 | c >> c2) >> d` you get
  `Seq([a, Wave([b,c]), Wave([b2,c2]), b3, d])`: **lockstep**. `b2` does not
  start until `c` finishes even though it does not depend on it, and `c2`
  finishes and sits watching while `b3` runs alone. Worse for what is coming:
  torch's device is *thread-local*, so a branch that hops threads on every wave
  cannot set it once.

- **By branch**, which is what stayed.
  `Seq([a, Wave([Seq([b,b2,b3]), Seq([c,c2])]), d])`. One thread per branch,
  start to finish, and a single join.

### The missing piece: decompose, do not flatten

`compile` stops walking the topological order and **recovers the tree**. And it
has to recover it from the graph, not from the expression: decision 6 of CU5 says
that the same graph built with `node()`/`edge()` in a loop gives the same plan,
and a loop has no tree. **The DSL expression is the oracle, not the source.**

Four cases, and the order matters:

| case | yields |
|---|---|
| no nodes | `Empty` |
| one node | `Execute` |
| the subgraph splits into connected components | `Wave`, one branch per component |
| there is a **series cut** | `Sequence` of the two sides |
| no cut | flat sequence: it is not series-parallel |

A **series cut** `(A, B)` is what a `>>` does: the crossing edges run from **all**
the sinks of `A` to **all** the sources of `B`, and from nowhere else. Both halves
of the check are needed — without the first, an edge leaving an interior node
passes as good.

Testing the **prefixes of a topological order** sufficed, and it is provable: in
a serial composition every node of `A` reaches a sink of `A`, every sink of `A`
has an edge to every source of `B`, and every node of `B` is reachable from a
source of `B`. Therefore every node of `A` precedes every node of `B` in *any*
topological order. There is no need to enumerate subsets.

### The rule that was rejected along the way

Before the series cut, cutting at a **barrier node** was tried — one such that
`ancestors(x) ∪ {x} ∪ descendants(x)` was everything. It only gets it right when
the join is a single node. The counterexample, which is in the tests:

```
(a >> a2 | b) >> (c | d)

with barrier →  Seq([ Wave([a, b]), a2, Wave([c, d]) ])   ← splits the branch
correct      →  Seq([ Wave([Seq([a,a2]), b]), Wave([c,d]) ])
```

### Decisions taken

1. **`Wave(Vec<Plan>)`, not `Vec<Execute>`.** A branch is a whole plan. The
   restrictive shape was simpler to execute, but cannot express a branch of
   several nodes, which is exactly the case.
2. **A wave means "they are launched at the same time"**, not "they are
   independent". That was CU4's lesson: a variant that only describes structure
   buys nothing.
3. **The branches are connected components**, so they are disjoint by
   construction and no node can appear in two. The bug that killed `Parallel`
   cannot come back: there is a test that checks it over a battery of topologies.
4. **`std::thread::scope`, with no dependencies.** It lends out `&Catalog` and
   `&Driver` without wrapping them. The bounds that allow it — `Node: Send + Sync`,
   `Driver: Send + Sync` — have been there since CU2 for another reason: PyO3
   requires `Send` on a pyclass. Rayon would have been the obvious answer and the
   worst one.
5. **Each branch copies what was produced and returns its own**; the parent
   merges on joining. Copying is cheap because a `Value` clones by `Arc`, and in
   exchange there is not a single lock.
6. **The error is the first declared branch's**, not that of whichever failed
   earlier on the clock. If two branches break at once, which arrives first is a
   race and the message cannot depend on it.
7. **A panic inside a branch is not swallowed**: it propagates with
   `resume_unwind` after `scope` has waited on the others.
8. **A linear chain compiles to the previous plan, identically.** It is the
   regression that matters most: everything closed from CU2 to CU8 is a chain.
9. **What is not series-parallel is walked in sequence**, as before. It is
   neither a failure nor a warning: it is what there was.

### The fortunate boundary

**The image of the DSL is exactly the series-parallel graphs.** `>>` composes
serially by connecting all terminals to all heads, `|` composes in parallel with
disjoint union, and there is no third operation.

The minimal pattern that is not series-parallel is the "N" — `a→c`, `a→d`, `b→d`
— and it **cannot be written with `>>` and `|`**. Getting to it requires
`node()`/`edge()`. So the line explains itself in one sentence: *if you wrote it
with the DSL, it parallelizes; if not, it parallelizes where it can*. That there
are DAGs without a tree is a theorem, not a gap in the algorithm — see Valdes,
Tarjan and Lawler, "The recognition of series parallel digraphs", SIAM J. Comput.
11(2), 1982.

### The GIL, which is where this hangs

`Graph.forward` held the GIL while the engine ran. The moment a wave spawns
threads that call a Python object's `forward`, those threads block asking for it
and **the whole process freezes** — not even a `join(timeout=…)` on the main
thread returns, because it needs the GIL too. The fix is one line,
`py.allow_threads`, and its test has to live **in another process**: a hang like
that cannot be caught by anything inside.

What `allow_threads` does not fix, and must not be confused with the above: two
pure Python nodes in the same wave **interleave, but do not overlap**. The wave
puts both in flight — which is why a rendezvous between them resolves, and the
test that proves it has no driver in the way — but the GIL means only one runs at
any instant, so there is no time to gain. Time is gained when the work releases
the GIL: torch in its dispatch, waiting on a network driver, I/O.

And a distinction that costs dearly if lost: the busy-driver test —
`two_branches_can_keep_the_driver_busy_at_the_same_time` — is not what proves the
branches are concurrent. The rendezvous tests already prove that, with no driver.
What that test measures is that the **shared** driver is not the bottleneck: it
is lent as `&dyn Driver` to both threads and serves two requests at once.

### Questionnaire

**Decomposition** (`core/tests/unit/plan.rs`)
- [x] empty, one node, and a linear chain identical to the previous one
- [x] output fan, input fan and diamond, each with its wave
- [x] `a >> (b >> b2 >> b3 | c >> c2) >> d` gives a wave of two sequences
- [x] `(a >> a2 | b) >> (c | d)` does not split the branch — the barrier counterexample
- [x] `(a | b) >> (c | d)` is two waves, not one of four
- [x] a wave inside another wave's branch
- [x] two unrelated graphs, each long, are two branches
- [x] the N is walked in sequence, and does not spoil the parallelism beside it

**Invariants, over a battery of ten topologies**
- [x] no node executes twice or is left out
- [x] the order the plan dictates respects every edge
- [x] every step declares exactly its predecessors in the graph
- [x] a wave's branches share no node
- [x] the same graph always compiles the same

**The oracle** (`core/tests/unit/build.rs`)
- [x] seven DSL expressions, and their plan is the tree that was written

**Execution** (`core/tests/unit/execution.rs`)
- [x] the **real** execution order respects the edges, with threads in the way
- [x] a whole branch runs on the same thread, and two branches on different threads
- [x] two and three branches really do run at once — without sleeping: they agree
      to meet, and were they sequential the first would exhaust the deadline
- [x] the diamond's result is the same spread out as in a row
- [x] what a branch produces inside reaches whoever reads it
- [x] two failing branches always give the first declared one's error
- [x] a panic inside a branch is not swallowed
- [x] two branches can keep the driver busy at the same time
- [x] a wave that is the whole plan returns the map of its leaves

**Python** (`python/tests/test_waves.py`)
- [x] the engine releases the GIL — in another process, with a deadline
- [x] two and three branches really run at once, **without a driver**: the
      rendezvous resolves, therefore both are in flight
- [x] two pure Python nodes give the right result even though the GIL interleaves
      them — the price, said plainly
- [x] the DSL with branches gives the same plan as `node()`/`edge()`
- [x] threads, real order, failures and the N, as in Rust

### What did NOT go in

**The device.** `Plan::Execute` still does not say *where*. This slice is what
enables it — one branch per thread is what makes pinning a device per branch mean
something — but `.on("cuda:1")` is the next use case.

**Micro-batches.** Overlapping inside a branch, not across branches, is another
problem and another variant.

**Spreading a wave across processes.** It needs transport, and `Opaque` does not
cross a wire. That is the other next use case.

## CU10 — Where a node runs

```python
g = Graph.somatize(
    Tokenize()
    >> (Encoder().on("cuda:0") | Other().on("cuda:1"))
    >> Join()
)
g.devices()   # {"encoder": "cuda:0", "other": "cuda:1"}
g.plan()      # the usual one: the plan says when, not where
```

Status: **closed**. 110 tests in Rust, 119 in Python.

### The question: where does the device live?

The first design put it in the plan, as `Plan::On { device, inner }` wrapping a
subplan — the same shape `Plan::Remote` is going to have. It was rejected for a
reason that knocks it down entirely: **the plan determines the execution order
and the concurrency, and placing changes neither**. They are two different axes
and putting them in the same type ties them together needlessly.

Taking it out of the plan paid off immediately:

- `plan.rs` is not touched in the whole use case.
- The rule for collapsing contiguous runs of the same device disappears, and it
  was the most fragile part of the design: the one part that had to be canonical
  and could stop being so.
- "Placing does not change the plan" stops being something to check and becomes
  true by construction, because `compile` does not see the placement. The test is
  still written, but as a warning for the day someone tries to put it there.

Also rejected, and why:

| where | why not |
|---|---|
| in the `Node` (`fn device(&self)`) | it puts an **orchestration** decision inside the implementation's contract. The node does not choose where it runs; and it also becomes invisible: it cannot be printed or reasoned about |
| in the `Graph` | a `Graph` is topology only. And the engine does not look at it — every plan step has been self-contained since CU3 — so it would have forced passing the graph to the `Executor` |
| in the `Catalog` | it was the runner-up: the engine already has it to hand. It loses because the catalog is the half that is **not** data, and a placement is. When a subgraph travels to another machine, the placement travels with it and the implementations do not |
| a generic `Metadata` | it is the generic name for `Placement`, and generic is paid for dearly: an `id → dict` sack cannot hold a **typed** `Device` |

### Decisions taken

**1. `Placement` is a type of its own, and it is given to the engine the way the
driver is.** That leaves a fourth orthogonal fact, which is what the question
made visible:

| piece | answers |
|---|---|
| `Graph` | **what** exists and how it connects |
| `Catalog` | **who** executes it |
| `Placement` | **where** |
| `Plan` | **when**, and with what concurrency |

It fits what `Executor` had already written about itself: "executing needs
context — today the store and the driver — and tomorrow it will need more".

**2. `Device` is an enum, not a shape-validated `String`.** The argument that
decided it is not exhaustiveness but this one: **with an enum, a typo is an error
at declaration time**. `.on("cude:0")` fails where it was written; a
`Device(String)` that only checked the shape would accept it and the failure
would surface inside torch halfway through a run.

The cost of the vocabulary becoming ours can be paid because **the core does not
`match` on a `Device` anywhere else**: it decides nothing based on which one it
is, it only carries it. Adding a variant is three lines — the enum, a `FromStr`
arm and a `Display` arm — and nowhere else stops compiling.

**3. The `cuda` index is mandatory.** In torch, bare `"cuda"` means "the current
GPU", which is thread state. To whoever is placing, "the current one" is not a
placement: `.on("cuda")` is rejected asking for `cuda:0`. An ambiguous
declaration cannot be written.

**4. `meta` goes in as a variant.** It is the only device that lets us prove end
to end that a placement arrives and is obeyed **on any machine**. The development
machine has a single GPU, so without `meta` half the questionnaire would depend
on the hardware.

**5. Unplaced ≠ placed on `cpu`.** The first is "wherever it already is", the
second is an order to move. That is why `Placement::of` returns an `Option`
rather than a default `Cpu`.

**6. The device arrives via the `Ctx`, and the one that obeys is the node.** It
is the consequence of the core not knowing what a GPU is: its role is to carry
the declaration to the point of execution. `ctx.device` arrives written the way
torch writes it — `"cuda:0"` — so it can be handed to `.to()` without
translating.

**7. `.on()` in the DSL, `place()` with the id, and a single door.** `.on()` is
handed out to the leaves without a place and **the innermost one wins**:
`(a.on("cuda:0") >> b).on("cuda:1")` leaves `a` on 0 and `b` on 1. But `.on()`
needs the object inside an expression, and there are two cases where only the id
is left: the graph built in a loop, and — the one that really matters — the
placement decided **afterwards**, from whatever is on the machine:

```python
for i, nid in enumerate(g.nodes()):
    g.place(nid, f"cuda:{i % torch.cuda.device_count()}")
```

They are not two paths: `.on()` ends up calling `place()`, so the validation is
written once and the DSL inherits it. No orphan id is possible — `.on()` only
names nodes of its own `Wire`, and `place()` validates against the graph.

### What `.on()` is **not**

`.on("cuda:1")` is not `torch.cuda.set_device(1)`. For a node to compute on a GPU
three things have to happen, and the ambient context only affects the third:

| what | how | when |
|---|---|---|
| the **parameters** are there | `module.to(dev)` | once |
| the **input** is there | `x.to(dev)` | every forward |
| what is **created inside** is born there | explicit `device=` | every forward |

The counterexample was already in the repo: `test_pipeline_torch.py` creates the
index tensor with `torch.tensor(rows)` inside the `forward`. With the `Embedding`
moved to cuda, that blows up with *"Expected all tensors to be on the same
device"*, and no `set_device` fixes it.

Hence obeying is the node's job, and the pattern is written by hand — it is five
lines, and until they repeat three times there is nothing to pull out into a base
class:

```python
def forward(self, x, ctx):
    if ctx.device:
        if self.placed != ctx.device:
            self.lin.to(ctx.device)   # the parameters, once
            self.placed = ctx.device
        x = x.to(ctx.device)          # the input, every time
    return Done(Opaque(self.lin(x)))
```

### The postcondition, which is what prevents the silence

A node that ignores its `ctx.device` would run in the wrong place without anyone
noticing, and that is exactly what this project does not tolerate. From outside
there is only one thing to look at: **where what it returned ended up**. If it
does not match, it is a named error:

```
node `encoder` failed: it declared `cuda:0` but returned a value on `cpu`
```

Whatever has a `.device` to look at is checked — a tensor, loose or inside an
`Opaque`. A placed node that returns a list of strings is not checked, and
placing it did not make much sense anyway.

**The case it flags without it being an error**: a node that runs on the GPU and
deliberately finishes with a `.cpu()`. It is accepted knowingly — it is the rare
case, the message says exactly what happened, and the alternative was silence.

### Why this came after the waves

A wave's branch runs whole on one thread (CU9's decision), so a device per branch
means something. The other way round did not work: grouping by topological level
would have made a branch hop threads, and torch's device is thread-local.

And what makes the whole use case cheap: `.to()` between devices is
**differentiable**, so autograd crosses the hop and `Opaque` has not had to
change a line. There is a test: two layers, one on `cuda:0` and one on `cpu`,
training end to end.

### Questionnaire

**Rust** (`core/tests/unit/device.rs`, `placement.rs`, `execution.rs`)
- [x] `cpu`, `cuda:N` and `meta` parse, and the round trip gives the same thing
- [x] `cude:0` is an unknown kind; `cuda` asks for an index; `cuda:`, `cuda:x`,
      `cuda:1:2`, `cpu:0` and `""` are not shaped like a device
- [x] `.on()` spreads over the whole piece and the innermost one wins
- [x] each branch of a `|` in its own place, and what is unplaced stays unplaced
- [x] the node sees its own and only its own — nobody catches the neighbour's
- [x] a wave's branches see different devices, each on its own thread
- [x] placing changes neither the plan, nor the graph, nor what it produces

**Python** (`python/tests/test_device.py`)
- [x] `.on()` and `place()` give the same graph, and `.named` and `.on` commute
- [x] placing afterwards in a loop, and replacing overwrites the previous one
- [x] placing a node that does not exist fails, and each bad name with its warning
- [x] `ctx.device` arrives, and shows up in the `Ctx`'s `repr`
- [x] the postcondition fires, and says which node — without torch, with any old
      object that knows how to say where it is
- [x] with torch: `meta` end to end without hardware; a node that ignores its
      device is caught
- [x] with a GPU: `cuda:0` → `cpu` in the same graph, and the backward pass
      crosses the hop while training

### What did NOT go in

**Choosing the device automatically.** Balancing, "auto", looking at how much
memory is left: that is a policy, and there is nobody asking for it yet.

**Splitting a node across devices**, and `g.to("cuda")` for the whole graph.

**`soma_next.torch`.** The pattern for obeying a placement is written by hand in
the test, which is where it is documented until it repeats.

**Generalizing `Placement` to "a place, local or remote"** to get ahead of CU13.
When `Remote` arrives, `Placement` will be there to grow — or not; that gets
decided then.

### Measured, not asserted

With a single GPU on the development machine, `cuda:1` does not exist: spreading
across two GPUs can be **declared** and cannot be **executed** here. The tests say
so in their names rather than leaving it implicit.

And CU9's warning still stands: do not justify this with a benchmark of two
branches on two GPUs. CUDA launches asynchronously and the two already overlap
when executed in sequence; what the waves buy is **host** time.

## CU11 — Training, outside the graph

```python
from soma_next.torch import Trainer, parameters

g = Graph.somatize(Encoder().on("cuda:0") >> Head().on("cuda:0"))
t = Trainer(g, objective=cross_entropy,
            optimizer=torch.optim.Adam(parameters(g), lr=1e-3))

t.fit(data, epochs=10)    # the sugar
t.step(batch)             # the primitive
```

Status: **closed**. 110 tests in Rust, 136 in Python, and **zero new lines in
`core/`** — the first use case that does not touch the core.

### The question: does the training loop go inside the graph?

No, and there are two independent reasons.

**The first is in the node contract.** `forward(input, ctx) → Done | Await`
describes **one step**: it executes once per run, it has a budget of 64 turns,
and `run()` has no partial recovery. A training run lasts an afternoon, mutates
its own state, emits metrics continuously and fails in ways one wants to recover
from. **The graph operates at the scale of a `forward`; a training run operates
at the scale of an afternoon.**

The original tried it: its node trait carries `fn fit(&self, x, y)`. The bill
shows in its own tests — `soma-worker`, `soma-compiler`, `soma-runtime` and
`soma-agent` all implement an empty `fit` just to be able to exist. It is the
same tax CU6 removed on the filter/step axis.

**The second is that a graph describes a network, and searching is a family of
networks.** That graph is precisely the artifact CU13 serializes and sends; one
carrying five configurations inside would be lying about the architecture.

### The three levels

| level | what it is | scale | what it spreads |
|---|---|---|---|
| the graph | a network | one `forward` | slices of a forward: waves, `Placement`, and `Remote` in its day |
| `Trainer` | one training run | an afternoon | nothing; it repeats forwards |
| a study | N training runs | an experiment | whole runs |

And the rule that holds it up: **no level knows the one above exists.** The graph
does not know it is being trained; the trainer does not know there are other
trainers. Composition between levels is composition of **functions**, not of
graphs.

### Level 3 has no type, and that is on purpose

```python
study = {lr: Trainer(net(), ..., lr).fit(data) for lr in (1e-4, 1e-2)}
best = min(study, key=lambda lr: study[lr].loss)
```

> A graph earns its keep when there are **dependencies** to declare. N
> independent training runs have none: they are a list. Modelling a list as a
> graph is paying a DAG's price without using it.

The alternative was even designed — the N configurations as branches of a `|`,
with a node that picks the best — and rejected. And so was its clever variant,
"one graph, one plan, **N catalogs**", which falls over something concrete:
`Catalog` is `Clone`, but it clones `Arc`s. The N replicas would share the node
objects, i.e. **the weights**, and the five configurations would train the same
model, giving results that look good. Each replica has to be built — and once you
build it, you no longer have one graph with N plans, you have N graphs. There is
a test.

### Decisions taken

**1. The Trainer receives the graph; never `g.fit(...)`.** That way the same
graph is trained three ways without touching it, and it remains the artifact that
travels.

**2. It lives in `soma_next.torch`.** Loss, `backward()` and optimizer are torch;
writing it neutrally would ask for a `Backend` with a single implementor. The core
does not learn what training is, and that is the sign the separation holds.

**3. `step(batch)` is the primitive; `fit(data, epochs)` is sugar.** It is what
avoids the god-trainer path: early stopping, odd schedules, federated rounds and
PBT are a `while` the user writes over `step`, not a growing list of options and
callbacks. A federated round is `for _ in range(k): t.step(batch)`.

**4. Parameters are collected by duck typing, and a graph without them fails when
building the Trainer.** It asks for `.parameters()` and skips whoever lacks it — a
lemmatizer does not train and does not stop being a node for it. Putting it in the
contract would be the original's `fit` all over again. And since duck typing fails
quietly — a graph without parameters trains nothing and shows a flat loss — the
empty list blows up at construction, just like CU10's postcondition.

**5. They come without repeats, by identity.** Two nodes can share a module —
tied weights between embedding and output — and then the same `Parameter` comes
out twice.

**6. The optimizer is built by the caller.** No `optimizer="adam"`, which would
end up being a name registry. The only thing checked is that the optimizer and the
graph **share some parameter**: sharing none has no innocent reading. Covering
only a part is legitimate and passes — freezing the encoder and training the head
is exactly that.

**7. The data is an iterable of `(input, target)`.** A `DataLoader` is one.
Deliberately rejected: **data as a source node**, because a node produces *one*
value per execution; to be a stream it would have to remember where it is, and
then two executions of the same graph stop giving the same thing.

**8. The loss is a callable, not a node.** It is not part of the network: it is
swapped without touching the model and at inference time it does not exist. As a
node, the graph's output would be a scalar and the graph would only be good for
training.

### What the GPU test found

The target does not cross the graph. The **input** does, and each node moves it to
its device because that is what a placed node does; the **target** goes straight
to the loss, so nobody moved it. With the last layer on `cuda:0`, the output comes
from there, the target is still on the cpu and torch stops the training with
*"expected all tensors to be on the same device"*.

The only one that sees both sides fixes it. And the target is moved, not the
output: bringing the output to the cpu would drag the backward pass back over the
wire at every step.

### Questionnaire

**Python** (`python/tests/test_trainer.py`)
- [x] `parameters(g)` collects from every node that has any and skips those that
      do not; in declaration order; without repeating a shared module
- [x] a graph without parameters fails **when building the Trainer**
- [x] an optimizer from another graph fails; freezing a part passes
- [x] training brings the loss down, and `fit` gives the same as the hand-written loop
- [x] the weights the optimizer updates are the ones the graph uses
- [x] training does not change the graph: `nodes()`, `edges()`, `plan()` and
      `devices()` identical before and after
- [x] an input that is not a tensor crosses as always
- [x] **two nets from the same factory do not share weights**
- [x] the hyperparameter search, as a list comprehension
- [x] with a GPU: the optimizer still points at the weights after the node moves;
      the target goes to meet the output; and the two layers on different devices
      train

### What did NOT go in

Checkpoints and resumption · callbacks and early stopping, which are a `while`
over `step` · metrics beyond the loss · schedulers · gradient accumulation · the
study as a type · and **exporting or loading a model's state**, which is CU14's
question and was not needed here: training locally extracts no state.

### What this did to the plan

- **The state question stops blocking.** It is no longer "is a node's state a
  `Value`?" in the core, but "what does a training run export?" at level 2, and it
  gets answered in CU14 with the case in front of us.
- **CU13 splits in two.** Spreading *one graph* across hosts is `Plan::Remote`,
  with dependencies halfway through the forward. Spreading *training runs* — HPO,
  federated, data parallel — is "execute this whole thing over there", level 3,
  and does not need it. The original had them in the same enum: `ModelParallel`
  next to `DataParallel`, `Federated` and `PopulationBased`.
- **CU14 changes shape.** "A federated round is a graph" is withdrawn: it is `map`
  and `reduce`. A graph would only be justified by non-flat topologies —
  hierarchical federation, gossip — and that day has not come.
- **`soma_next.torch` stops being pending** and opens here.

---

## CU12 — A slice that runs in another process

```python
# A. the worker is your own code, already on that machine
g = Graph.somatize(Encode() >> Classify().at("gpu-box"))
g.forward(x, workers={"gpu-box": Worker.at("gpu-box:7000")})

# B. the worker is a bare node: `pip install soma-next` and nothing else
#    python -m soma_next.worker --listen 0.0.0.0:7000
g.forward(x, workers={"gpu-box": Worker.at("gpu-box:7000",
                                           mode="network", send=["my_package"])})
```

Status: **closed**. A crate of its own, `soma-next-transport`, and the generic
worker in `soma_next.worker`. 74 tests in the transport crate today — 49 of the
worker, 22 of the protocol, 3 of the artifact — plus `test_remote.py` (37),
`test_integration.py` (17), `test_manifest.py` (13) and `test_fingerprint.py`
(13) in Python.

It comes in two halves. The first is in the core and adds three things: `Host` —
**a name**, and not a `Device`, which is a place inside a machine — `Plan::Remote`,
and the `Transport` hole. The second is the crate that fills it, and it lives
outside the core for the same reason `python/` does: here there are child
processes, sockets and a byte format, three things a core has no business
knowing.

The benefit is measurable here, with one machine and no cluster: **two Python
nodes in a wave interleave but do not overlap**, because of the GIL, and
`test_waves.py` already said so as a known limitation. In two processes they
overlap, and there is a test that makes them prove it by meeting in a file.

### The question: who resolves the implementations?

A plan travels; a `Catalog` does not — an `Arc<dyn Node>` has no way of crossing
a wire. So somebody over there has to turn a name into something executable, and
there are exactly two answers:

| | where it gets the code | who it is for |
|---|---|---|
| `Serving::own` | it brings it: the same binary, the same clone | you control the infrastructure |
| `Serving::provisioned` | the client sends it, in an `Artifact` | a bare node, and you do not |

**Two constructors and not an optional parameter**, because they reject
different things: offering the first an artifact is an error, and not offering
the second one is too. An `Option` would have turned both refusals into a branch
somebody forgets.

### What travels, and what deliberately does not

The **plan**, the **values the slice reads and does not produce**, and the
**placement** — three things that are data. And, for an empty worker, an
**artifact this crate does not look at**, which is where the nodes **and the
driver** ride: whoever packs one packs the other.

Not the catalog as such. Not the **environment**: that `torch` is installed over
there is the business of whoever stood the worker up, and putting it in cost the
original soma 420 lines of environment manager and a hot `pip install`. And not
a `Value::Opaque`, which points into the process that made it — it fails at
encoding time, with the host in front of you.

> `Host` is a **name** and `Device` is a place inside a machine, and they are
> independent: `.on("cuda:0")` and `.at("gpu-box")` can be written in either
> order. What a name resolves to is said by whoever executes, in
> `forward(workers=…)`, so the same graph spreads across two processes here or
> two machines there without touching a line of what was declared.

### The two strategies, asked by `kind` and not tried in a chain

| `mode=` here | `kind` on the wire | what travels | what the worker supplies | size |
|---|---|---|---|---|
| `"project"` *(default)* | `project` | names, versions and state | **the code**, from its own clone | 48 bytes for two nodes |
| `"network"` | `pickle` | the code as well, via `cloudpickle` | nothing | megabytes |

Two vocabularies for the two sides of the same choice, and they are not the same
word: what you write is what the artifact is **for** — a worker on the network
that has nothing — and what travels is what it **is**, a pickle. `send=` names
the modules of yours that have to go inside it, because `cloudpickle` serializes
by reference anything importable, which leaves out exactly the case a generic
worker exists for.

`project` is what you want when the worker runs in a clone of the project: tens
of bytes per node, no coupling between interpreters, and it **checks the
version**. `pickle` removes all friction when the worker does not have your code
at all, and pays for it by demanding that both interpreters look very much alike.

**Versioning, which `project` needs and gets almost for free.** A class's
fingerprint is the hash of its **AST**, plus transitively whatever of yours that
code names — a helper in the same module, a base class, a module constant. Not
comments and not docstrings: comparing text would make versioning noise. It stops
at what is installed, at what has no source, and at `soma_next` itself, which
goes into the client's identity instead so that upgrading the library does not
invalidate every class at once.

Where the check lands is the good part: `pickle`'s own `find_class`. The whole
policy is one method and `pickle` does the rest, nested objects included.
`--strict` *(the default)* stops with both versions in front of you; `--lucky`
runs whatever it has and says so on `stderr` — running a different version in
silence is what gets discovered three days later.

### Decisions taken

**1. The name is announced before the artifact is sent.** `Hello { runtime,
offering }` carries the artifact's **id**, not the artifact. The saving is real —
a pickled catalog with weights is megabytes and asking "do you have
`sha256:abc…`?" is forty bytes — but it is not the point. The point is that the
day there is a store, a worker answers `Ready` off its own shelf and **the
protocol does not change a line**. It is git's `have`/`want` and docker's layer
exchange. (That day came in CU13.)

**2. The client identifies itself in the greeting.** This is the original's
lesson written as a method: its worker chose an interpreter with `$SOMA_PYTHON`
or `python3` from the `PATH`, and a pickled filter rebuilt by an interpreter that
is not close enough comes back as the class's `__dict__` instead of an instance —
surfacing as `'dict' object is not callable` from inside a subprocess, with
nothing pointing at the version gap. Refusing on connect, with both runtimes in
front of you, is cheaper than anything afterwards.

**3. An artifact's `id` is set by whoever produces it.** Hashing the bytes here
would be the natural thing and it is wrong: without interpreting the content
there is no criterion for saying when two artifacts are the same one, and two
pickles of one catalog can differ byte for byte. Whoever produces it knows what
identifies it; here we compare strings.

**4. The driver travels with the nodes**, in the same artifact and by the same
strategy. A node that answers `Await` has to be served **where it runs**.
Declared versus injected is about the **graph** — a node is in it, a driver is
not — and not about how either one gets there.

**5. One thread per conversation.** See below; it is what the tests found.

**6. A worker holds one catalog.** A second client provisioning it with a
different artifact would pull it out from under the first, and an id present in
both would silently run the wrong implementation. Each session remembers what it
greeted with and is told so, rather than executing somebody else's nodes. Checked
at the **job** and not at the greeting, because that is where it can go wrong and
there is no race to lose there.

**7. Chunks with a length in front, and a cap.** A pipe has no messages, it has
bytes. The cap is not a limit of the format — the length is a `u32` — it is a
safety net: four ASCII characters on the worker's `stdout` read as a length of
between 500 MB and 2 GB, and without it a stray `print()` is a hung process with
no message.

**8. There is no authentication, and there will not be.** Whoever reaches that
port runs code on that machine as that user — `pickle` artifacts are opened with
`cloudpickle.loads` and `project` ones resolve classes out of that clone. That is
`ssh`'s job and `srun`'s, not a framework's, and it is said plainly in
`soma_next.worker`'s own docstring rather than left to be discovered.

### What the tests found

**Serving one conversation at a time deadlocked.** Two branches of a wave open
two connections; the second sits in the `accept` queue and the first does not
release its own until its `forward` finishes. The integration test caught it,
which is where it had to show. The lesson was not "do not serialize" but
**serialize at message granularity, not session granularity**.

**A failing test that had stood a worker up hung `cargo test` for ten minutes.**
The kill was the last line of each test, and a test that fails never reaches its
last line — so the orphan kept the test binary's inherited `stderr` open and
cargo waited on that pipe instead of reporting the failure. Twenty milliseconds
of failure, six hundred seconds of wait. It is a `Drop` now.

**A stray `print()` in a user's node is on the wire.** `stdout` **is** the
protocol, so there is not one `println!` in a worker and talking goes to
`stderr`. In Python this is more dangerous, because the `print` can be in a
library on import, and it is why the frame cap exists.

### Questionnaire

**The core's seam** (`core/tests/unit/execution.rs`, with a double that never
leaves its seat)
- [x] what a slice reads and does not produce is what travels with it, and no more
- [x] the placement travels; the host half does not, having already done its job
- [x] a host nobody resolves is **not executed here just in case**
- [x] what comes back is merged as if it had been produced here

**The transport** (`transport/tests/unit/`)
- [x] a node placed away really runs in another process, and the same graph
      undistributed runs here
- [x] a whole wave on one worker goes in a single trip, and two workers really do
      run at the same time
- [x] the node over there sees the device it was given here
- [x] an opaque does not leave this process, in either direction
- [x] a failure over there comes back with the host and the reason
- [x] something printed on the worker's `stdout` is reported and does not hang
- [x] the artifact is sent only once, and cannot be swapped once the session is
      open
- [x] a runtime it does not accept, a kind it does not know and a broken artifact
      are each rejected saying which
- [x] a standing worker serves whoever connects, outlives the client leaving, and
      keeps its catalog from one client to the next
- [x] two simultaneous connections against the same standing worker

**Python** (`test_remote.py`, `test_integration.py`, `test_manifest.py`,
`test_fingerprint.py`)
- [x] `.at()` sends the whole piece, the innermost one wins, and the order of
      `.on()` and `.at()` does not matter
- [x] a worker with the project receives **only names and state**, and the code
      does not go over the wire
- [x] a worker without the project says so instead of guessing
- [x] another version of the code stops with `--strict` and is reported with
      `--lucky`
- [x] the driver's state travels with it, and it runs over there and not here
- [x] one driver serves both sides of the same run
- [x] the fingerprint changes with a helper, a base class or a module constant,
      and not with a comment or a docstring

### The cluster, which came later and is where this is really tested

Every test above runs its workers as **subprocesses of the test**. That proves
the protocol and cannot prove the rest: those workers share this filesystem, this
interpreter and these installed packages, so "the worker cannot import your
module" is a trick with `sys.path` and "another version of the code" is the file
rewritten underneath.

`docker/compose.yaml` and `python/tests/cluster/` are the same thing without the
tricks: four containers, each with **the wheel and nothing else**, and the client
outside. What only becomes provable there:

| | as subprocesses | as containers |
|---|---|---|
| the worker has none of your code | a trick with `sys.path` | it really has none |
| another version of it | the file rewritten | another mount, another image |
| two hosts | two processes on the loopback | two network namespaces |
| a store shared between workers | a common `/tmp` | a volume between machines |
| a worker with a GPU | — | a device, and torch, in one of them |

It is opt-in — `SOMA_CLUSTER=1 python -m pytest tests/cluster -q` — because the
first build takes minutes, and the suite has to keep taking seconds.

Writing it turned up something missing: `Serving::store` and `Serving::keeping`
existed in Rust and **were not reachable from the shipped worker**. The generic
worker now takes `--store DIR`, which is what lets two containers share a shelf:
one keeps what a node produced, and the other reads it.

### Training across the cut, which is where a wire really shows

What crosses a wire is the **value**, not the graph that made it. So a node with
parameters that runs on another machine gets no gradient here, and — this is the
part worth writing down — **nothing failed** when that happened: the loss came
down, because whatever was downstream of it was learning, and half the net never
moved. Silently wrong numbers of the same family as the cached prefix, and with
nobody asking the question.

The question is asked now, and at the level that can ask it. After the first
`backward()`, if the optimizer is about to update a parameter that received no
gradient, `Trainer` stops with `NoGradient` and names the node. It is more
general than the case that prompted it: a slice on another host, an output read
back from a store, a branch the loss never reads — one symptom, one check.

And **`.at()` is not a refusal to train**, which is why the check is about
gradients and not about hosts. The far half can perfectly well train itself:
that is [**split learning**](https://arxiv.org/abs/1812.00564), and it works
today without a line of framework:

```python
class SplitPart(Node):
    def forward(self, msg, ctx):
        if msg["kind"] == "forward":
            self.held = self.lin(tensor(msg["value"])).relu()   # the graph stays HERE
            return Done({"value": self.held.detach().tolist()})
        self.opt.zero_grad()
        self.held.backward(tensor(msg["value"]))                # dL/da, off the wire
        self.opt.step()
```

Three things that were already there make it fall out: a worker **keeps its
catalog**, so the node object survives between calls and its activation stays
alive on the far side; a node is **one contract**, so it dispatches on its input
instead of needing a kind of its own; and a gradient is a tensor like any other,
so it crosses as data. There is a test that trains the far half over a real
container and a control that shows it is not the near half doing all the work.

What it is not, yet: it takes two round trips the caller drives, `self.held` is
hidden state nobody checks the alternation of, activations cross as lists of
floats rather than tensor bytes — the codec exists, the wire still refuses an
`Opaque` — and the two halves never overlap. Making it a concept of the framework
is a use case of its own, and it turns on one question: whether a worker stops
being dumb and starts holding an optimizer.

### What did NOT go in

**Authentication and encryption**, on purpose and for good (decision 8) ·
**installing the environment**, which is what cost the original 420 lines ·
**scheduling**: which host gets what is declared, not decided — there is no
placement policy and no load balancing · **retrying a failed slice**, because a
node that already ran half of itself is not idempotent and nobody has said what
it means to run it again · **a protocol version**, since both sides are the same
binary from the same `cargo build`; the day they stop being so, the place for one
is the `Hello`, which already negotiates the runtime · and **a store**, which was
the next slice and is where the `have`/`want` finally got its `have`.

---

## CU13 — What is remembered, and what is not computed twice

```python
from soma_next.torch import Trainer, freeze, parameters

g = Graph.somatize(Encoder().frozen().cached() >> Head())

# training: the Trainer settles what was declared, and keeps what was declared kept
Trainer(g, objective=cross_entropy, optimizer=Adam(head_only, lr=1e-3),
        store="/scratch/soma").fit(data, epochs=20)

# inference: settle it yourself, and the second run does not touch the encoder
freeze(g)
g.forward(Opaque(x), store="/scratch/soma")
```

Status: **closed**. 185 tests in the core, 33 in the store, 73 in the transport
and 249 in Python.

The case it exists for is labchain's
([SoftwareX, S2352711026000373](https://www.sciencedirect.com/science/article/pii/S2352711026000373)):
an expensive, settled node — an encoder, an embedding — under a head that
changes twenty times in an afternoon. What has to be true is that changing the
head **does not touch the name of what is underneath it**.

### The question: where does the hash go?

labchain hangs it **inside the data**: an `XYData` object carrying the value and
its hashes. It has no choice — it has no engine, the pipeline is the user's code,
and the only place left to put a hash is the datum itself.

Here there is an engine, and `walk` already carries `produced` everywhere. So the
key goes **beside it**, in a parallel table with the same life cycle: the same
copy into a wave's branches, the same retention in `resume`, the same merge from
what came back over a wire. In exchange, `Value` grows no wrapper that every
`forward` would have to unwrap, and the `Node` contract does not change by one
character.

> The engine is the only one that sees every edge. Anything that wants to travel
> along an edge without being a value belongs to the engine, not to the value.

### The key: a Merkle hash over the recipe, not over the data

```text
key(root) = H(the input, by its content)     ← the only place data is hashed
key(node) = H(identity, state, salt, the keys of its predecessors)
identity  = the name of the class            ← not the fingerprint of the code
```

From the root down they are hashes of hashes, and that is the whole point: **the
key is known before anything runs**. Naming what a node will produce does not
cost a byte of the data it will produce it from, and changing the classifier does
not touch the name of the embeddings under it.

Three things deliberately left out of the key:

- **The fingerprint of the code.** In it, a cosmetic refactor would invalidate
  half a store in silence. It is written *beside* the value and compared on a
  hit, which turns the same event into a line on `stderr` you can act on. The
  window it leaves open is narrow and known: two classes of the same name with
  different bodies share a key, and the fingerprint is what says so out loud.
- **The device.** Where something ran is not what it is. What the key cannot see,
  the user says with `.cached(salt="a100-fp16")`.
- **The graph.** A key names a node's output, not the run it happened in: that is
  what makes two graphs share what they have in common.

### The prefix rule, which is one line and has two reasons

> A node's output can be kept if **nothing upstream of it can change** — itself
> included.

Freezing the node alone is not enough: freezing layer 3 of 5 does not stop the
gradient crossing it towards layers 1 and 2. Two independent arguments land on
the same line, and that is why it is a check and not a warning:

- what is restored from a store is a **leaf**. The backward pass stops there and
  everything above it quietly stops training;
- the digest of the state is in the key, so a node that still trains gets a new
  key every step. It never hits; it only fills the store.

At inference the whole prefix is settled by definition, so all of it is
cacheable. The question is asked by `cacheable(graph, memory)` **before the first
node runs** — the engine never sees a graph, so it cannot be the one to ask.

### Declaring and obeying, for the fourth time

`Memory` is the **fifth fact**, and like the other four it is inert data:

| piece | answers |
|---|---|
| `Graph` | **what** exists |
| `Catalog` | **who** executes it |
| `Placement` | **where** |
| `Plan` | **when** |
| `Memory` | **what is remembered** |

The core defines *frozen* as "this node's state does not change while the graph
runs" — a statement about **cache validity**, not about gradients, which it still
knows nothing about. `soma_next.torch.freeze` is what makes it true, with
`requires_grad_(False)`, exactly as a node and not the core is what moves a
tensor to a GPU. And the digest of the weights is paid for **there**, once,
because settling is the moment that makes both halves true at the same time.

### The fourth hole, and the first that is a decorator

`Keeper` joins `Node`, `Driver` and `Transport`: the core provides the hole,
whoever knows what goes in it is a library. Here it is doubly true — hashing is
`sha256` and keeping is a directory, and the core has no dependencies at all.

**Driver serves, Transport carries, Keeper keeps.**

What fills it is `soma_next_store::Cache`, and in Python there is a second one in
front: `Packing` turns every `Opaque` into bytes on the way in and back on the
way out, so the store sees maps and bytes and never learns Python exists.

### Decisions taken

**1. The key travels beside `produced`, not inside `Value`.** See above. It also
means a node that declares nothing pays nothing, and that without a keeper the
table is not even computed.

**2. `.cached()` is opt-in, and not declaring it does not break the chain.** A
node with no cache still gets a key and still passes it on. Otherwise declaring
the cache node by node would be declaring it for the whole graph.

**3. A keeper that fails never kills the run.** It is said on `stderr` and the
value is recomputed. A cache is an optimization, and one that can kill a run at
hour three is not one. Same criterion the worker already applies to a store it
cannot reach.

**4. The frontier of `Opaque` moves rather than disappearing.** From "an opaque
does not travel" to "an opaque nobody registered a codec for does not travel",
which is the more precise of the two. `codec(kind, type, dump=, load=)` is the
register; `soma_next.torch` fills in the tensor's **on being imported**, so a
graph that keeps tensors and never imports it keeps nothing and says why on
`stderr`. Importing `torch` is not enough and is not meant to be: registering it
from `soma_next` would mean importing torch for everyone who does not have it.
What comes back is a **leaf**, and there is a test that says so, because it will
look like a bug the first time somebody sees it.

**5. A tensor comes back on the cpu, and `weights_only=True`.** A store shared
between machines that only reads back where it was written is not shared at all;
and one that unpickles arbitrary objects is a way in. Whoever receives it moves
it, which is what a placed node already does with its input.

**6. Values and artifacts live in the same store, under two namespaces.**
`value:<key>` and `artifact:<kind>:<id>`. Two questions — a catalog that is not
sent twice, a node that is not run twice — one directory, and what keeps them
apart is the name.

**7. Declared versus injected, for the fifth time.** `Memory` is declared, like
the `Placement`: it belongs to whoever wrote the graph, and it **travels**. A
`Keeper` is injected, like the `Driver` and the `Transport`: it belongs to
whoever runs, and it does not. They were one builder call for a while — "neither
is any use alone" — and that was false: a coordinator that keeps nothing itself
still has to tell a worker what the nodes are, or the one side that *does* keep
things can name none of them. `remembering(&Memory)` and `keeping(&dyn Keeper)`,
and there is a test where this side keeps nothing at all.

**8. Whoever obeys is the only one who can check that obedience happened.** The
core cannot tell a node with **no state to settle** — a tokenizer — from one
whose weights **nobody has hashed yet**: both arrive as a state of `None`. Left
alone, that is the one failure a cache must not have — two checkpoints of the
same class under one name, the wrong tensor back, no error and no warning. So
Python asks it, by the same duck it asks for `parameters()`, wherever a cache is
declared: something with state, settled, and no digest, refuses to run and says
to call `soma_next.torch.freeze(g)`. There is a test that reproduces the bad hit
and one that shows two checkpoints settled at the same digest **are** one name,
because that is what says the digest is what the key believes.

### Questionnaire

**The core** (`core/tests/unit/{execution,build,memory}.rs`)
- [x] what is kept is not computed again, and what is kept under that name is
      what was produced
- [x] a different input is a different name, and the node runs
- [x] what is above names what is below: another state upstream, another name
- [x] the fingerprint of the code is **not** part of the name, and what is
      written beside the value is what produced it
- [x] a node that keeps nothing still passes its name on
- [x] an `Opaque` root leaves everything below nameless, and it is not an error
- [x] nothing is named without a keeper
- [x] `cacheable` names the cached node **and** the ancestor that can still change
- [x] the names a slice brings are not the names it gives, and both cross a
      `Cargo` and come back in an `Outcome`

**The store** (`store/tests/unit/cache.rs`)
- [x] the pieces of a recipe cannot run into each other: `["ab","c"]` and
      `["a","bc"]` are two names
- [x] the same recipe is the same name every time; only a root is named by its
      content
- [x] a batch answers in the order it was asked, holes included
- [x] a name nobody kept is a miss and not a failure
- [x] a kept value is findable by looking at what is in the store
- [x] the same bytes under two names are stored once

**The worker** (`transport/tests/unit/worker.rs`)
- [x] what a worker already kept is not run again **over there** — and the same
      worker without a keeper runs it every time
- [x] the name of what ran over there comes back

**Python** (`python/tests/{test_cache,test_freeze}.py`)
- [x] changing the head does not recompute the embedding, with real tensors
- [x] a cache over something that can still change is refused, naming both
- [x] the salt is another name, and the innermost one wins
- [x] declaring is not obeying: until `freeze`, the weights still ask for a
      gradient
- [x] the same weights are the same state; other weights are another state
- [x] `Trainer` obeys whatever was declared, and the optimizer still points at
      the same objects
- [x] a different fingerprint says so on `stderr` and **uses** what is kept
- [x] an opaque nobody can write down is said and is not fatal
- [x] what comes back is a leaf
- [x] a node declared settled that **nobody settled** refuses to run, naming the
      class whose two checkpoints would collide — and it is asked with or
      without a store
- [x] two checkpoints settled at the same digest are one name, and at two
      digests are two
- [x] a node that answers `parameters()` and not `state_dict()` is settled all
      the same

### What did NOT go in

**The grain per item** (`.mapped()`): a node that maps over items, with a key per
item instead of per node. It is designed and unwritten — `Key` is public and
there is no `Keys` yet, on purpose: a variant nobody can construct is worse than
a variant that arrives late, and the day it arrives every `match` stops compiling
and somebody decides. It opens together with micro-batches, because it is the
same question.

Also out: **`.overwrite(times=1)`**, which is a policy of the *run* and lives in
the executor, not in what is kept · **the queryable index** — what do I have,
from which run, from when — which is a SQLite derived from the records and
throwaway, and making it the truth would mean a single writer over NFS · a
**strict mode** for the fingerprint (`.cached(strict=True)`) · and **S3**, which
arrives the day there is a MinIO to point at, through OpenDAL and as another
configuration rather than another implementation.

### What reviewing this slice found

Three things, and they went in before it was called closed. Written down because
two of them were **wrong numbers**, not rough edges:

- a `.frozen()` declared and never obeyed gave a **false hit**: two checkpoints
  of the same class, the same input, and the second run got the first one's
  tensor back. Reproduced with a script before being fixed; it is decision 8;
- `Memory` and `Keeper` went in one builder call, so a coordinator with no store
  of its own sent an empty table and a worker that *did* have one could name
  nothing. It is decision 7;
- `cacheable` was asked only when a `store=` was given, so a `.cached()` in the
  wrong place stayed silent until somebody added a directory to the call —
  possibly in production. It is asked wherever a cache is declared.

And two more, found writing the worked example that is now `test_pretraining.py`
— which is the argument for writing one:

- **`Trainer` had no `store=`.** `step` called `graph.forward(input)` and nothing
  else, so the one case this whole slice exists for could not go through the
  `Trainer` at all;
- **a class's fingerprint changed when an unrelated global was defined.**
  `_names` read `code.co_names`, which mixes global loads with **attribute
  names**: `self.model` puts `model` in there, and a module with a global called
  `model` had its *value* hashed into the version of a class that never named it.
  It is CU12's code, and it is not cosmetic — with `project` and `--strict` a
  worker refuses to run over a mismatch that does not exist.

---

## CU14 — Training the half that is not here

```python
from soma_next.torch import Split, Trainer, parameters

class Body(Node):                        # a node. It knows nothing about any of this
    def __init__(self):
        self.lin = nn.Linear(8, 6)

    def forward(self, x, ctx):
        return Done(Opaque(self.lin(x).relu()))

    def parameters(self):
        return list(self.lin.parameters())

g = Graph.somatize(Body().at("gpu") >> Head())
trains = {"body": Split(SGD, lr=0.1)}    # what trains it, where it runs

Trainer(g, objective=cross_entropy,
        optimizer=SGD(parameters(g, without=trains), lr=0.1),   # the half that is here
        trains=trains,                                          # the half that is not
        workers={"gpu": Worker.at("node3:7000")}).fit(data, epochs=10)
```

Status: **closed**.

### The question CU12 left open

> Making it a concept of the framework is a use case of its own, and it turns on
> one question: whether a worker stops being dumb and starts holding an
> optimizer.

It does not. **The worker holds nothing new: it holds a catalog, as always, and
what went into it is a trainer.** Dumb means *does not decide*, not *cannot do* —
it is told what to do and knows how, exactly as it is told a `Device` and moves
itself. Nothing in the worker, the protocol or the core learns that training
exists.

And the node holds nothing new either, which is the second answer and the one
that took a redesign to get to. The first version of this slice had the node
inherit a mixin and grow a `learn`: it worked, it was measured, and it was
wrong — *how I am trained* is a fact of a training run, and a node is the scale
of one `forward`. It is the same mistake CU11 rejected as `fit` in the contract,
one step disguised.

### What is not negotiable, and what is

The optimizer has to point at the tensors that execute, and those live on the
machine the node runs on: the client's copy of the weights is other objects in
another process. Something has to be **there**, and it has to survive between the
call that produces the activation and the call that brings the gradient. That is
physics, and no reshuffling of responsibilities moves it.

What is negotiable is who writes that something and where it is declared. Here it
is written by nobody — it is `Split`, or whatever else is put in the `Learning`
hole — and it is declared by whoever trains, in one dict.

### Two positions in the graph, one object

`around` puts the trainer on both sides of the node it trains, and the stage
machinery does the rest:

```text
    …  →  body:in  →  body:computes  →  body   →  …
          the trainer   the node        the trainer
          leafs the     computes        keeps the activation,
          input                         gives out what it let go of
```

Both positions are the **same object**, and `pickle` keeps that: two entries of
one catalog, one trainer, one optimizer, pointing at the weights that are there.
The id the graph knew stays with the **last** position, because what the rest of
the graph calls `body` is what `body` gives out — so nobody downstream is told
anything and their fan-in maps are keyed as they always were.

The backward pass is the transpose of the stage, and only what takes a gradient
is transposed — which is the trainer and never the node. What sits between them
is walked **through**: the gradient a trainer gives back is owed to whoever fed
the chain that reached it.

### What cuts a graph, and why it is not declared in the graph

A pass stops being one `forward` where the chain that joins the output to the
input breaks, and that happens for two reasons that look different and are the
same one: the value **crossed a cable** — what arrives on the other side is data,
not the graph that produced it — or **somebody trained the node that produced
it**, and a trainer lets go of the activation by construction.

```text
trained(n)  said by whoever trains, and never asked of the node
where(n)    hosts().get(n), None meaning here
cut(p, c)   (where(p), trained(p)) != (where(c), trained(c))
level(n)    0 with no predecessors, else max(level(p) + cut(p, n))
stage k     the nodes with level(n) == k
```

Where a thing **runs** is declared in the graph and is nobody else's business;
which of them is **trained where it runs** is a fact of the training run, and the
same graph is the same graph either way. So `stages` is told and a node is never
asked — and grouping by the pair is what makes local greedy and split learning
the same path through the code.

Three properties fall out, and they are what make the backward pass
demonstrable: every cut edge crosses a stage boundary, no cut edge stays inside a
stage, and no edge goes backwards — so the stages in reverse are a valid order to
walk back through.

A stage is **not uniform in host on purpose**: `A.at("a") | B.at("b")` is one
stage and a single `forward`, and the plan of that stage still has the `Wave`
with both `Remote`s inside it. The waves are kept by not being clever about them.

### One hole, four techniques

`learn(signal, ctx)` receives `dL/d(what the node produced)` and gives back
`dL/d(what it was given)`. `Split` is the one that ships; the other three are a
subclass each, written in the tests as a user would write them.

| technique | what crosses | the control that says it is real |
|---|---|---|
| split learning | activations out, gradients back | the same run with nothing handed back leaves it where it was |
| local greedy | nothing | whatever is above it gets no gradient, and `NoGradient` names it |
| forward-forward | nothing | the goodness separates; with the rate at zero, it does not |
| synthetic gradients | nothing | `‖ĝ − g‖` gets closer; with the guesser frozen it is exactly 1, every step |

What tells a backward message from an input is an **envelope**, on the precedent
of `__soma_opaque__`: a reserved key and a cheap check before anything is built.
One key and not two, because unlike a packed opaque there is no kind to carry. In
a learning pass every value on every edge is an envelope, so a **map** of
envelopes is not a fan-in of inputs but a fan-in of gradients — and an envelope
carrying nothing is how "no gradient for you this step" is said, which is what a
technique that gives none back answers.

### The evidence

Bit for bit against the same net trained in one piece: the same weights, the same
batches, ten steps, `==` on the losses and `torch.equal` on the weights
afterwards — and again with the far half in **another process**, where it also
comes out identical. Everything in between — `tolist` on a float32, a detached
leaf, an optimizer of its own with the same rule — is the same operations in the
same order.

And in the cluster, against a real container with a GPU, the hand-written loop
CU12 left behind and `Trainer.step` side by side, producing the same losses. It
is the strongest thing a framework can be asked to show: **it changed who writes
the loop, not the arithmetic** — and the node it trained is a plain `Slab` that
does not know any of it happened.

### Decisions taken

**1. The trainer travels and lives in the catalog.** It is the one thing a worker
keeps between calls, so it is where anything that has to survive between them
goes. Reaching it is a `forward` like everything else, because that is the only
channel a worker has — and what matters is that it is **not the user's node and
not written by the user**: it has the same standing as `Held` and `Tap`.

**2. The optimizer is built at first use, never in `__init__`.** `pickle` does
not call `__init__`, and being rebuilt on another machine is this object's normal
life. What travels is a **factory** — `Split(SGD, lr=0.1)` keeps a `partial` —
and the optimizer is built there, over the parameters that are there. Everything
a rebuilt object reads is a class attribute, for the same reason.

**3. Two positions and not one.** The trainer has to be *before* the node, so the
input becomes a leaf a gradient can be asked of, and *after* it, to keep the
activation. One position cannot do: in front it never sees the output, and behind
it the input was already built inside the node and there is no leaf to go back
to.

**4. It is driven in stages when something is trained where it runs, and only
then.** Not "when the graph is cut": a lone trained node is a single stage and
still needs its backward over the transpose, while a slice on another host with
nobody training it needs none of this — and for that one `step` is, line for
line, the one it always was.

**5. The wire is not touched.** Activations and gradients cross as lists of
floats. It is the known bill of this slice.

**6. A piece of a graph provisions the whole graph.** `Graph.provision` came out
of `forward` and came out **public**, because the graph is not the `Trainer`'s:
whoever runs one in pieces has to be able to do it with no trainer in sight and
no rule to remember. A worker has **one** catalog, and half of one is a different
catalog — refused mid-session by an open one, and swallowed in silence by a
worker that has not greeted yet, taking with it every live activation and every
optimizer state over there. Both measured against a real worker before anything
was changed.

**7. The same weights in two optimizers is refused.** Where a trained node runs
may well be here, and then this side's optimizer and its trainer would both move
it every step: two updates for one gradient, which is a worse loss and not a
wrong one — the kind nobody notices. `parameters(g, without=trains)` is how they
come out, and holding them anyway stops the `Trainer` from being built.

**8. The gradient check comes after the gradients are handed back.** With a
trained node in the middle, everything above it is an orphan until the stage
behind it answers; asking any earlier calls the whole near half orphaned. And
what is trained elsewhere is never an orphan: `NoGradient` knows, because it was
told.

**9. `.cached()` only in the first stage.** A root's key comes from the input it
was handed, and after a cut the roots of a stage are holds handed nothing: two
different batches would be kept under one name. Refused in `__init__`, with that
as the reason.

**10. `.frozen()` and being trained are a contradiction**, refused before the
first step: one says the state does not change while the graph runs, the other
changes it every step.

### CU12's debt, paid on the way

An **intermediate `Opaque` did not survive a remote stretch of two nodes**: an
answer carries back the output of *every* node it ran, and one that cannot leave
the process refused the whole message. It made `(A() >> B()).at("w1")` with
tensors unwritable — and this slice needs exactly that, three nodes on one host
with something live between them.

The fix is one line each way and it is where the rule belongs: a worker sends
back what **can** travel (`Outcome::travelling`), and whoever reads what stayed
behind is told so by name (`RunError::Lost`), rather than the walk finding a hole
where a predecessor should be. `last` is not filtered — the slice's own value has
a reader here by definition, so refusing it is the honest answer.

### Questionnaire

**Cutting the graph** (`python/tests/test_stage.py`, no torch anywhere in it)
- [x] another host cuts, a node somebody trains cuts, and nobody saying so does
      not
- [x] two hosts side by side are **one** stage, and the wave is still inside it
- [x] a fan lands the join after the deepest branch, and coming back here cuts
      again
- [x] a hold is named after the real producer, so the fan-in map is the one the
      whole graph gave
- [x] a node that feeds inside **and** outside comes back, which is what the tap
      is for
- [x] holds and taps are never placed, and a stage keeps everything that was said
      about its nodes
- [x] every cut edge crosses a boundary, none stays inside one, and no edge goes
      backwards — over five topologies
- [x] `around` puts two positions and leaves the id on the last, the company
      stands where the node stands, and what was said about a node stays with the
      node
- [x] the transpose keeps only what takes a gradient and walks up **through** the
      rest
- [x] a stage provisions the **whole** graph and not its half
- [x] the stages run to what the whole graph runs to

**The trainer** (`python/tests/test_learning.py`)
- [x] the input becomes a leaf and the node never finds out
- [x] an ordinary value is the activation, an envelope is a gradient, and a map
      of them is summed
- [x] what it gives back is `dL/d(what the node was given)`, checked by hand on
      `y = wx`
- [x] it lets go of the chain that produced its input: nothing above it gets a
      gradient
- [x] the activation is let go of after each `learn`, and a gradient with none is
      `OutOfStep`
- [x] the optimizer is built at first use over the node's parameters, and a
      trainer rebuilt by `pickle` still trains — with **one** copy of the node
      between its two positions
- [x] thirty steps with a head above it: the loss comes down, its weights move,
      and the graph is the same graph afterwards
- [x] the same net in another process, bit for bit against this one, and with the
      far rate at zero the loss comes down less
- [x] the same weights in two optimizers is refused
- [x] the four techniques, each with the control in the table above

**Driving it** (`python/tests/test_trainer.py`)
- [x] a cut graph trains to the same numbers as the whole one, weights included
- [x] it is driven in stages only when something is trained where it runs
- [x] what is above a trained node gets its gradient through it, and nobody
      stands still
- [x] a graph where everything is trained elsewhere takes no optimizer here
- [x] `.cached()` after a cut, `trains` naming somebody who is not there,
      settled-and-trained, and a node that does not say what its parameters are:
      all refused when the trainer is built

**The wire** (`core`, `transport`, and the cluster)
- [x] an opaque read only over there does not stop the slice — and reading it
      from here names both ends
- [x] an outcome leaves behind what cannot travel and keeps what it answers with
- [x] the hand-written loop and `Trainer.step` come out with the same losses,
      against a real container
- [x] the transpose reaches the catalog the worker has live

### What did NOT go in

**FedAvg**, which is CU15 and is what a training run *exports* rather than what it
does · **`Opaque` over the wire**: activations still cross as floats, and the
codec that would fix it exists — the wire is what refuses. *(Closed afterwards,
below.)* · **micro-batches**,
which open together with the grain per item (`.mapped()`) · **a trained node with
two producers**, which would owe a different gradient to each and is refused in as
many words, because routing one gradient per edge is not something the transpose
alone does · and **`resume` exposed to Python**.

---

## After CU14 — `Opaque` over the wire

The pending CU14 wrote down, closed before starting CU15. Not a use case of its
own: no new call shape, nothing new to declare. What changes is what a wire will
carry, and it is measured rather than argued.

Status: **closed**. 188 in core, 33 in store, 80 in transport, 340 in Python, 18
in containers.

### It does not move the frontier, it moves what falls on which side

`Value::travels` does not change and stays true: what comes out of a codec
**does** travel, being maps and bytes. A codec does not relax the limit — it
turns a value that could not cross into one that can, before anybody asks. The
frontier goes from "the variant" to "the variant nobody registered a codec for",
which is the more precise statement of the two, and it is CU13's sentence about a
store said again about a socket.

### Not a fifth hole

The first shape this was given had `Serving` taking a new trait and `CLAUDE.md`
going from four holes to five. That was wrong, and the tree already said so:
**`transport` has a hole of its own**, `Provision`, filled from `python/`, and
nobody counts it among the core's. The core's four are what *the core* provides
and does not fill. This one is about the wire's alphabet, and the wire belongs to
`transport`.

So: one trait in `transport`, one implementor in `python/`, and `core` untouched.

| | on the way out | on the way back |
|---|---|---|
| the client, `Worker::packing` | the input and what is known, packed | what came back, unpacked |
| the worker, `Serving::packing` | what it produced, packed | the input and what is known, unpacked |

Packing happens **before** a message is built, so the refusal in
`Answer::to_bytes` is untouched and still guards: by the time it looks, whatever
had a codec is already bytes.

### What executing it said that the design did not

- **Packing goes before `travelling`.** `Outcome::travelling` drops what does not
  travel, and a tensor with a codec does travel — asking first leaves behind
  exactly what this exists to carry.
- **Failing is not the same in the two directions.** Going out, a value nobody
  can write down is an error: somebody over there is waiting for it. Coming back,
  it stays where it ran and is named by `RunError::Lost`, which is CU14's rule
  unchanged. And unlike a `Keeper`, a codec that fails **does** stop the run: a
  cache that cannot answer recomputes, a wire has nothing to fall back on.
- **A worker never imports `soma_next.torch`.** It starts empty, and the nodes
  that arrive may never mention `torch` while a tensor goes past them all the
  same. Importing it on standing up works and costs two seconds per worker —
  most of a suite that stands up twenty. So it is summoned at the moment it is
  known to be needed: something written by that codec arrived and nothing here
  reads it. Both ways round, because a kind is named after its type.
- **Two things were slower than the wire**, and neither was the wire. An artifact
  taken as `Vec<u8>` is read out of a `bytes` one element at a time — 10 MB/s,
  413 ms on a 4 MB artifact, three times a step. And `Value::Bytes` went through
  serde as a **sequence**, one element per byte, so a megabyte of tensor was a
  million integers. Both were older than this slice and only became visible
  because it made the wire fast enough for them to show.

### The consumer, which is what makes it worth anything

Two `_data`s, one per direction, and the one in `_learning.py` said it out loud:
*"this slice does not touch the wire, so activations and gradients cross as
floats"*. Both are now a tensor wrapped to cross, `detach` included — the graph
does not cross a wire and never did, so letting go of it here says out loud what
the wire does anyway.

| a step, half of it in another process | floats | bytes | |
|---|---|---|---|
| 64×256 | 103.9 ms | 11.1 ms | **9.4×** |
| 256×1024 | 1051.7 ms | 69.9 ms | **15.1×** |

Worth more than the number: **the same node is handed the same shape wherever it
runs**, which is the whole argument of `.at()`. Until now a node that worked here
was handed a list of floats the day somebody placed its producer elsewhere.

### Questionnaire

**The seam** (`transport/tests/unit/worker.rs`, no Python in it)
- [x] with a codec, an opaque produced over there comes back what it was
- [x] and one bound for over there arrives as what it was
- [x] one the codec cannot write and the slice answers with is refused **in the
      codec's own words, which are the far end's**
- [x] one it cannot write and nobody asked for stays where it ran
- [x] a worker that does not pack hands its node what it was sent, and the
      failure is quiet — which is why nothing installs one end without the other

**From Python** (`python/tests/test_remote.py`)
- [x] an opaque nobody can write down still does not leave, and says which type
      it was and how to say so
- [x] one produced over there that nobody can write down stays there
- [x] a tensor crosses whole and is a tensor over there — asked of the object
      itself, since a list of floats has no `shape`
- [x] one produced over there comes back a tensor, equal
- [x] what a node is handed on the far side is a `Tensor` and not a `list`

**Nothing else moved**
- [x] the 18 container tests, the bit-for-bit cross-process training of CU14, and
      everything CU13 keeps
- [x] bytes written as a list of numbers are still read as bytes, so a store
      written before this still opens — and what is written now is the size of
      the data and not twice it

---

## After CU14 — A group of steps, and one update

CU12's candidate A, written down as *gradient accumulation* and left for last
because it is small. The other pending closed before CU15.

Status: **closed**. 353 in Python; nothing else moved.

```python
Trainer(g, objective=cross_entropy, optimizer=..., every=4)
```

Four steps, one update, and the loss of each divided by four so that the four
together pull exactly as one step over the four batches would. The idiom
everybody writes by hand, written once. The loss `step` gives back is **whole**:
divided for the backward pass and not for whoever is reading, or a history would
change shape with `every`.

### Said here, for the third time

`.at()` says where, `trains` says who trains whom, and `every` says how many
steps make an update — all three are facts of **this training run** and none of
them is a fact of the graph. The graph is the scale of one `forward`; a group of
four steps is not something a `forward` could have an opinion about.

And the same rule `trains` already follows: a technique that named its own
(`Split(SGD, lr=0.1, every=8)`) **wins**, and the trainer's is the default for
whoever did not say. Two numbers is then something somebody meant rather than
something nobody noticed — which was the objection worth answering, because the
number is the user's to choose and choosing it per node is a thing somebody may
want.

### The far side is never told which step it is on

Only how many make a group, and it counts its own `learn` calls from the same
start. Told once, at the only moment there is one number: **before the trainer
that travels is packed**. No message, no round trip, no field in the protocol —
the same answer CU14 found when it asked where the trainer lives.

Out of phase by one and the two optimizers would move on different steps, so the
test that says it is the CU14 bar: the same run bit for bit with the far half in
another process, with `every=3`, plus the control that a group of three is not a
run of groups of one.

### What executing it said

- **The counter is the framework's, what to do about it is the technique's.**
  `Learning.forward` ticks; `learn` asks `opens()` and `closes()` and never has
  to remember to tick anything. One thing less for whoever fills the hole.
- **A group of one is the line it was**, on both sides — `opens` and `closes` are
  both true every step, the loss is not divided, and `Split`'s three movements
  fall back together. The 340 tests that existed before pass without one moving,
  which is the only acceptable answer for a default.
- **Closing a group across a cut costs a pass and not a step.** No forward, no
  gradient: the fact that the group is over goes the road a gradient goes, in an
  envelope carrying nothing. Only what takes a gradient is transposed, so every
  hold of a transposed stage feeds a trainer directly and reaches all of them and
  nothing else.
- **There is no `bool` on an edge**, so the marker carries nothing and its
  presence is the fact. The closed set of variants stopped one being invented as
  `1.0`, which is what that rule is for.

### Questionnaire

**The group** (`python/tests/test_trainer.py`)
- [x] a group of steps comes out where one step over all of them does
- [x] the optimizer moves once per group and not once per step
- [x] a group of one is what there was before it could be said, loss for loss and
      weight for weight
- [x] the loss it gives back is the one the objective said, not the divided one
- [x] a group the epoch ended in the middle of is still a group, and the epoch
      does not leave one open
- [x] closing a group that is not open does nothing and says so
- [x] a group is a whole number of steps and at least one

**Across a cut** (`test_trainer.py` in one process, `test_learning.py` in two)
- [x] a cut graph accumulates in step with the whole one, weights included
- [x] the one trained beside the node does not move until the group closes either
- [x] closing a group across a cut reaches it, with no step and no gradient
- [x] a technique that names its own group wins over the trainer's
- [x] **in another process**: a group of three is the same group on both sides,
      bit for bit — and a group of three is not a run of groups of one

**That the tests say something**
- [x] eight deliberate mutations — not scaling, closing every step, not counting,
      the far side stepping always, the far side not counting, not telling it the
      number, ignoring being told — every one caught

---

## CU15 — What a training run exports

```python
for _ in range(rounds):
    for client in clients:
        client.fit(client.data)
    average = fedavg([client.export() for client in clients])
    for client in clients:
        client.load(average)
```

Status: **closed**. 410 tests in Python, 40 in the store.

### The question CU11 put off, and how it changed shape on the way

CU11 asked *is a node's state a `Value`?* and put it off. It came back as a
better question — **what does a training run export?** — and the answer is the
smallest one that is true: its weights, node by node, `{node_id: {key: tensor}}`.

By **the same two ducks** the rest of the project asks by: a `state_dict` by
name, `parameters()` in order, and a node with neither has no weights and does
not stop being a node. It has to be the same two `state_digest` and
`Graph._check_it_was_obeyed` use, or a node could be told to settle and then have
no way of being exported — one state, two questions, and a project that answers
them differently in two places.

### A `for` over a list, and it stays one

A federated round has **no dependencies to declare**: the clients do not read
each other and the order they run in is nobody's business. A graph earns its keep
when there are dependencies, so this is not one — the original put training runs
and graph slices in the same enum, and that was the mistake the three levels
exist to avoid.

FedAvg, FedProx, FedYogi and SCAFFOLD differ in **arithmetic**. That is what a
function is for, so `fedavg` is one. The day a topology stops being flat —
hierarchical, gossip — that day it is a graph, and not before.

### Three things it had to decide

- **What is trained where it runs cannot be exported from here.** Those weights
  are on the other machine; what is here is the copy that was sent, and it never
  learnt anything. Handing it back would be silent, which is the only way this
  could go wrong, so it is refused with the node named. `trains` on its own is
  fine — running elsewhere is the half that matters.
- **The optimizer's state is not in it.** Momentum is a client's own, and
  averaging it is not what averaging weights means.
- **What is not a number you can halve is not halved.** A `num_batches_tracked`
  is a count and the mean of two counts is not a count; the first one's stands.
  Every implementation of this does it and none of them says so out loud.

### The test that was written twice

The demonstration has to be a **control**, not a loss going down: a net says that
just as loudly when only a third of it is learning.

The first attempt gave each client one class. Cross-entropy then pushes its own
logit up for ever, each client diverges alone, and the average of three diverged
nets is a lesson about learning rates — measured, a loss of 2.7e8 against 4.5 for
the client that stayed home. **The task has to be the same for everyone**; only
where they draw their inputs from may differ. Three clients, one corner of the
input space each, a fixed teacher they are all trying to learn, and a fourth
trained alone on its corner for exactly as long. The average reads the union
better.

### Questionnaire

**What it exports** (`python/tests/test_federated.py`)
- [x] the weights node by node, and a node with none is simply not in there
- [x] a node that says what its weights are called is asked by name, and one that
      does not is asked in order
- [x] what comes out is a snapshot and not a view
- [x] it goes back in where it came from
- [x] the optimizer's state is not in it

**What it refuses**
- [x] loading something this graph does not have, by name
- [x] loading a weight of the wrong shape, with both shapes
- [x] and a refusal leaves the net as it was rather than half loaded
- [x] exporting or loading what is trained **where it runs**, which is not here
- [x] but one trained here is exported like any other

**Putting several together**
- [x] the average of one is that one, and of two is halfway between
- [x] `sizes` is what it weighs by, and ten times the data pulls ten times
- [x] what is not a number you can halve is not halved
- [x] two different networks, the same node with a different shape, nothing at
      all, and the wrong number of sizes: all refused before anything is computed
- [x] three clients that each see a corner average into one that reads the union
      better than the one that stayed home
- [x] and a round leaves every client at the same weights
- [x] three mutations — returning the first export, ignoring `sizes`, averaging
      the integers — every one caught

### The store, opened by hand

The second piece, and what makes the first one reach another machine. Until now
the store was a **string** you handed the engine — `forward(store=...)` — a place
it kept things in and nobody else could open. That is enough while the only thing
kept is what the engine decided to keep, and it stops being enough the moment a
training run has weights of its own to write down.

```python
store.keep("round/3", trainer.export())
trainer.load(store.recall("round/3"))
```

`Store(directory)` has the four the Rust trait has, one for one, dealing in
**bytes** — `put`, `get`, `bind`, `resolve`, and `bound()` to look — and two
more dealing in **values**, tensors included, by the codecs. The two are
`Keeper`'s vocabulary and not new, and they are there because the thing this was
opened for is a map of tensors: without them everybody writes their own
`torch.save`, and whoever gets `weights_only` wrong writes a way into a shared
directory.

Two things it found by being used:

- **An export's keys were integers.** The positional duck numbered them, and the
  one thing a map that crosses anything here may have for a key is text. Found by
  handing an export to a `Store`, which is what it was built for.
- **An edge and a store do not want the same rule.** On an edge a bare tensor is
  refused and that is a feature — the mistake has two right answers, convert it
  or say `Opaque` and mean it, and refusing makes the cost of the first visible.
  In a store it is bytes either way, so the refusal defends nothing. One private
  `Unknown::{Refused, Wrapped}` is the whole difference.

**The two tests that matter run a real second interpreter**: it trains, writes
what it learnt, and this one reads it back to the same weights — a federated
round's client half with nothing between the two but a folder both can see. And
twice over: the same weights written by two processes are one blob, which is what
makes a round that changed nothing free.

### Claiming, which is how work gets handed out

The third piece. `bind` **replaces**, and that is right for what it is for — a
name is a question and its answer can be refreshed. It is exactly wrong for
handing out work: two processes given the same round would both bind it, both do
it, and nobody would do the next one.

```python
me = store.put(f"{gethostname()}/{getpid()}".encode())
if store.claim(f"round/{r}/client/{k}", me):
    ...
```

One operation that either takes the name or finds it taken, on the trait and
**with no default**: one written out of `resolve` and `bind` would be a race with
a doc comment on it. `link` and not `rename` is the whole difference — a rename
replaces, and `link` fails when the name is taken. It is also the one that has
always been trusted over NFS, where `O_EXCL` has not, and a network folder is
where this is going to live.

**The tests really race.** Eight threads in Rust, eight **processes** in Python,
because that is the case this exists for: Slurm tasks on a folder they all
mounted. And not just that one wins — that the one told it won is the one written
down, or the winner does the work with somebody else's name on it. Against the
mutant written as `resolve` then `bind`, every other test still passes and these
say seven of the eight racers were told they had it.

### The barrier, with nobody in charge

The last piece, and the half the study's design never had: trials are independent
and rounds are not.

```python
for r in range(rounds):
    trainer.fit(my_data)
    trainer.load(gather(store, trainer.export(), run="cifar", round=r,
                        clients=4, mine=int(os.environ["SLURM_PROCID"])))
```

The same script on every machine. The obvious answer is a coordinator, and a
coordinator is a process that has to stay alive — and a run that hangs over a
weekend when it does not. Instead: **whoever finds the round complete claims the
averaging**, and exactly one can win that, because that is what a claim is. One
client does a little more work than the rest, once a round, and nobody babysits
anything.

A round, on disk:

```text
<run>/round/<r>/client/<k>    what client k learnt   (its size in the record)
<run>/round/<r>/averaging     who is doing the mean  (claimed, so exactly one)
<run>/round/<r>/average       the mean               (what everybody leaves with)
```

**The deadline says who is missing by name**, and in two messages rather than
one, because they are not the same thing: somebody missing is a client that never
came; **nobody** missing is worse — the round is complete, so whoever claimed the
averaging died holding it, and no one else will try. That is what a claim is,
said from the other side.

Waiting longer is a number. Going on without them is a **policy**, and not this
function's to make: `fedavg` takes whatever list it is handed.

### Questionnaire, the distributed half

**The store, opened by hand** (`python/tests/test_store.py`)
- [x] bytes by what they are, names that point at them, and both directions of
      each
- [x] a value with tensors in it goes in and comes out alive, and a bare tensor
      is kept although it would not cross an edge
- [x] something nobody registered a codec for says which type it was
- [x] a training run written down by **another interpreter** is read back here to
      the same weights, and two processes that wrote the same weights wrote them
      once

**Claiming** (`store/tests/unit/local.rs`, `test_store.py`)
- [x] a name nobody has can be claimed and one somebody has cannot
- [x] what `bind` replaces, `claim` refuses
- [x] eight threads and eight **processes** on one name: exactly one wins, and
      the one told it won is the one written down
- [x] a claim leaves nothing behind in the temporaries
- [x] against the mutant written as `resolve` then `bind`, seven of the eight
      racers were told they had it

**The round** (`python/tests/test_round.py`)
- [x] the only client of a round averages it itself, and publishes it
- [x] a client that arrives after the average finds it and does not wait
- [x] two runs sharing a directory are two runs, and so are two rounds of one
- [x] the deadline says which clients are missing — and says a different thing
      when the round is complete and the average never came
- [x] the size travels in the record; all of them saying it or none
- [x] **four processes, one folder, two rounds**: all four leave with the same
      number, and exactly two averagings happen, which nobody arranged
- [x] a client that never starts stops the others **by name**
- [x] four mutations — `bind` for `claim`, never waiting, never publishing, not
      weighing — every one caught

### What is NOT in it yet

**FedProx, FedYogi, SCAFFOLD**, the same shape with different arithmetic ·
**secure aggregation** · **partial rounds**, which is a policy and belongs to
whoever is running one · and **what a round is worth remembering by**, which is
where the digest of a state stops being only a cache key.

None of it went through the graph's transport, and that was the point: no
`Plan::Remote`, no port, no protocol. A folder, and Slurm hands the work out.

---

## CU16 — The grain of an item

```python
embed.named("embed").frozen().mapped().cached()
Trainer(g, objective=..., optimizer=..., micro=4)
```

Status: **closed**. 196 in core, 40 in store, 83 in transport, 428 in Python.

The two things the plan had written down as opening together. They do not, and
finding out why was most of the design.

### The question they share, and where it splits them

**What is an item, and who can see them.** Two facts settle it:

1. The engine sees items only inside a `Value::List`. Everything else is one
   thing.
2. And the case that matters is exactly what it cannot see: a batch of images is
   `Opaque(tensor[128, …])` and its items are rows. That is not an oversight of
   the API — it is the whole point of `Opaque`.

Caching item by item has to **name** each item, and a name comes from content:
the same document in another list has to be the same item, or labchain's argument
is gone. So the engine has to be able to look. A micro-batch names nothing — and
at level 2 the batch belongs to the caller, who hands it in, so `torch.chunk`
reaches it and the core never learns what an item is.

**They separate.** One is `_trainer.py`; the other is the engine.

### An item's name is its content, not its place

The cheap design is `key(item i) = H(key of the whole thing, i)` — free, and
worth nothing. It hits exactly when the whole-value cache of CU13 already hits,
and misses exactly when it misses: add one document and the root's content hash
changes, and with it all thousand item keys. The middle case — fifty new
documents on a thousand old ones — is the only thing per-item grain buys, and it
requires each item's name to depend on **its own** content.

The cost is not a toll to dodge: it *is* the feature. And it is far smaller than
it first looked — the engine hashes text, bytes and numbers itself, and the case
this exists for has text at the root. Only opaque items cost a codec's write.

### Where the chain starts is where content is hashed

If what is above already names each item, these are built out of those and
nothing is hashed. If it does not — a root, or a list from a node nobody mapped —
each item is hashed by itself. And what reads a mapped node **without** mapping
is named after the whole list, so changing one item makes it run again. Which of
those a list of names collapses into is `Keeper::combine`'s to decide, not the
core's.

### Three ways to see an item, and why (A)

- **(A)** an item is a list element, and `.mapped()` refuses an opaque — nothing
  new in the core, and the node stacks what it is handed. **Chosen.**
- **(B)** a third hole that knows how to cut an opaque. More powerful; costs a
  hole. And what would justify it — the cost — is inherent to the idea rather
  than to the choice, so it buys **convenience, not capability**. It fits on top
  later if (A)'s ergonomics prove wrong.
- **(C)** the node cuts, told which items are wanted. The `Device` pattern
  exactly, and **it does not reach**: the node can cut what the engine cannot
  hash.

### What executing it said

- **`torch.chunk` gives at most what it is asked for.** Six rows into four is
  three pieces, and a group counting four while three run never closes — the
  optimizer stops moving, and across a cut the far side counts what it sees and
  the two fall out of step in silence. So a batch that does not divide is
  refused, which also makes accumulation's "equal pieces" assumption true rather
  than stated.
- **`memory_in` is written out one fact at a time, and that is a hole with a name
  on it.** A fifth thing to remember that is not added there does not fail: it
  stops being true on the other side of the wire. A mapped node would go on
  answering the same thing while its cache quietly lost its grain. Found by going
  to look, and there is a test that crosses now.
- **A branch of mine went in the bin.** I wrote a cut for a batch that is a map
  of tensors and then found such a batch does not cross an edge today with or
  without it. The project's first rule, applied to my own code.
- **`Keeper::recall` has taken a slice since the first day**, and its docstring
  says it is not for symmetry. This is what it was for.

### Questionnaire

**Cutting a batch** (`python/tests/test_trainer.py`)
- [x] a batch in pieces comes out where the whole one does
- [x] the optimizer still moves once a step, and `every` and `micro` multiply
- [x] the loss it gives back is still the number the whole batch would have said
- [x] a batch that does not divide is refused with both numbers and the flag that
      fixes it; so is something it cannot cut, and halves that do not line up
- [x] across a cut, the far side counts the **pieces**

**The grain of an item** (`core/tests/unit/execution.rs`, `test_cache.py`)
- [x] a node that maps answers one for each item, and does so with nobody keeping
      anything at all
- [x] the second run of the same list looks at nothing
- [x] **a new item among old ones is the only one looked at**
- [x] an item is named after itself and not after where it sits: the same four
      shuffled are the same four
- [x] what reads a mapped node without mapping is named after the whole list, so
      one item changing makes it run again
- [x] what is not a list, an answer with the wrong number of items, and a mapped
      node with two producers: refused where it happens with the node named
- [x] what maps still maps **on the other side of a wire**
- [x] three mutations — naming an item after its position, running all of them,
      losing the order on the way back — every one caught

### What is NOT in it yet

**(B), the hole that cuts an opaque**, which goes in the day `.mapped()` over a
list of wrapped rows proves too clumsy to live with · **a mapped node with two
producers**, which needs one gradient — one name — per edge and is refused rather
than guessed · and **a compile-time refusal** for it: today it is refused when the
map reaches the node, which names it but names it late.

---

## CU17 — Level 3: where to look, how to cut, and when to give up

```python
from soma_next.study import Partition

for train, test in Partition.stratified(5).folds(len(y), classes=y.tolist()):
    trainer.fit(data[train], epochs=10)
    scores.append(evaluate(g, data[test]))
```

The first piece of the level the vision calls **Study**: hyper-parameter search,
cross-validation, and whatever else is N training runs rather than one. Opened on
21 August 2026, and **not closed**: what is in is the cut. The sampler, the
pruner and how a run is asked to stop are still open.

### The question that came first, and it was not about folds

Whether "as much as possible in Rust" means the **loop** too. It does not, and
the original measured it without meaning to:

| trait | shape | implementors |
|---|---|---|
| `Sampler` | `sample(space, i) -> params` | Bayesian, Grid, Random |
| `Pruner` | `should_prune(metric, step, history) -> verdict` | Median, Percentile |
| `TrialExecutor` | `execute_trial(params, ctx) -> outcome` | `FnTrialExecutor<F>` |

The first two **return a decision**: data in, data out, and three and two real
implementors respectively. The third **calls back out**, and its only implementor
is a closure wrapper. `TrialExecutor` is not an abstraction, it is the loop
leaking: the step that trains is torch, so a loop written in Rust has to return
to Python for it, and the trait is the hole it goes through.

So the line is not drawn by language but by shape:

> Rust keeps everything that is pure, deterministic and hashable. The loop stays
> in Python, where torch is. **No callback crosses**: Rust returns decisions,
> Python acts on them.

Which is the same answer CU11 and CU15 gave — level 3 has no type, a federated
round is a `for` — reached this time from the other side.

### The decision the layer discussion left

A **layer** is a rule about *direction*; a **hole** is a rule about *width*. The
original obeyed the first — its arrows all point down — and still ended up
unreadable, because a wide crossing is paid for either with everybody importing
everything below (`soma-runtime`, 24.592 lines, imported by five crates) or with
a trait per crossing. `StudyIo`, whose only implementor is `Study`, is what a
layer boundary looks like when it manufactures its own abstraction.

Hence `study/`: a crate with **no dependencies at all**, not even the core's. A
partition is arithmetic over indices and does not know what a graph is.

### `Partition`, and why five variants and not sklearn's fifteen

Stratifying and grouping look like two axes crossed with every scheme, and that
cross product is where `KFold`, `StratifiedKFold`, `GroupKFold`,
`StratifiedGroupKFold`, `ShuffleSplit`, `StratifiedShuffleSplit`,
`GroupShuffleSplit`… come from. They are not different algorithms:

- **stratifying** is a k-fold inside each class, the folds concatenated
- **grouping** is a k-fold over the groups, the samples following theirs

So the scheme is named and the rest is parameters. `LeaveOneOut` is `kfold(n)`. A
holdout of one part in `k` is fold 0 of a k-fold. Purged and embargoed
cross-validation are `time_series(k, gap=…)`. A variant that is a parameter is a
name you have to remember for nothing.

### Decisions taken

- **Each scheme is a type with its own `folds`; the enum is only the family.**
  `KFold { k: 5, shuffle: None }.folds(&samples)` when you know which cut you
  want, `Partition` when the scheme arrives as data. The enum forwards —
  `Self::KFold(cut) => cut.folds(samples)` — so the dispatch is static either
  way and there is a test that going through it cuts exactly the same.
- **The family is an enum, and the first reason I gave was wrong.** I said a
  trait "does not deserialize"; that is true of a *type-erased* trait, not of a
  trait. With static dispatch each scheme serializes and hashes perfectly well.
  The three that survive the correction:
  - **The name is structural, not agreed.** A cut is part of a cache key (CU13).
    With a trait the name is supplied by the implementor, and two that collide —
    or one that changes between versions — hand back the wrong fold **in
    silence**. Derived, it cannot happen, and there is a test that says so as a
    property.
  - **Static dispatch needs the type when it compiles, and here it is in the
    data.** `#[pyclass]` cannot be generic, and a partition read back from a
    trial record has its type inside the JSON. Without the enum that is a
    `match` on strings — the same match, minus exhaustiveness — written once per
    consumer instead of once here.
  - **A new scheme stops compiling in three places and the compiler lists
    them.** With a trait it compiles, and what you forgot is the registration.
- **It is called `Partition`, not `Split`.** `soma_next.torch.Split` is already
  split learning. Two alike names for two unrelated things is how a framework
  stops being readable, and this one was caught before it was written.
- **Indices and keys in, indices out. Never a tensor.** Stratifying does not want
  the labels, it wants the classes *as numbers*; turning `y` into them is one
  line where `y` already lives. That contract is the whole reason this can be
  Rust while the core never learns what a dataset is.
- **Keys decide nothing; the scheme does.** A variant that needs classes and is
  not given them fails; one that does not need them and is handed them ignores
  them without complaining. The asymmetry lets one `Samples` be cut several ways
  to compare them, and makes stratifying *by accident* impossible.
- **What cannot be honoured is an error, not a warning.** A class with fewer
  members than folds, fewer groups than folds, a gap that eats the first
  training set. sklearn warns and carries on, which leaves a result you cannot
  tell from a good one.
- **`shuffle: Option<u64>`.** The seed both switches shuffling on and makes it
  repeatable, so "shuffled but not reproducible" cannot be written down.
  Fisher-Yates over splitmix64, ten lines rather than a dependency: the seed has
  to mean the same thing on every machine that reads the same record, which
  rules out whatever `rand` defaults to this year.
- **Both sides come out ascending.** The shuffle decides *who* is in a fold,
  never the order they are listed in.
- **No `Explicit { folds }` escape hatch.** Three lines the day someone needs it,
  and until then a variant with no consumer.

### Questionnaire (from sklearn, because the original has none)

`grep -ri 'kfold|cross.?valid|stratif'` over the original: **zero hits**. This is
the first piece written with no old version pulling at it.

**The cut** (`study/tests/unit/partition.rs`, `test_partition.py`)
- [x] k folds are a partition: every sample held out exactly once, never held out
      and training at once
- [x] what does not divide is spread one at a time (10 over 3 is 4-3-3)
- [x] the same seed gives the same cut on any machine; a different one does not
- [x] stratifying keeps every class's share in every fold
- [x] grouping never puts a group on both sides, and places the heaviest first so
      the folds stay comparable
- [x] both at once keeps the groups whole and the classes as even as that allows
- [x] time series never trains on its own future, and a `gap` drops what sits
      between — the one scheme that is deliberately **not** a partition, because
      the first block has nothing before it to learn from
- [x] `LeaveOneOut` is `k = n` and not a variant
- [x] keys that are spare change nothing; keys that are missing name the call
      that supplies them
- [x] every refusal happens before a single index comes out
- [x] two cuts that differ are written down differently
- [x] going through the enum cuts exactly the same as not going through it, and a
      scheme writes itself the same wrapped or not

### The pruner, and the question it was going to force

The one piece expected to touch level 2: a pruner needs a training run that can
be **stopped from outside**, and there was no such call. There still is not, and
there is not going to be — `Trainer.step` was already documented as the
primitive and `fit` as sugar over it, "whatever does not fit in an epoch loop is
written as a `while` over this". So a pruner **stops nothing**:

```python
for epoch in range(50):
    reported.append(trainer.fit(data, epochs=1).loss)
    if why := pruner.verdict(reported, finished):
        break
```

It answers, and the loop stops calling. **Zero lines in level 2**, and the
`Trainer` never finds out there was a pruner in the room — there is a test that
says exactly that. Anything else would have been a callback crossing the
boundary, which is what the original's `TrialExecutor` turned out to be.

### Three schemes, and they differ in what they judge against

| scheme | judged against | needs other trials |
|---|---|---|
| `Percentile` | **the others** at the same step | yes |
| `Threshold` | **a constant** already known to be hopeless | no |
| `Patience` | **itself**: it has stopped improving | no |

`Median` is not a fourth — it is `Percentile { p: 50 }`, and the original having
both is the same "a scheme that is a parameter" that gave sklearn fifteen ways
of cutting. `median()` is a constructor.

**Successive halving and Hyperband are deliberately out.** They are not verdicts
on a trial, they are a way of handing budget out across the whole population.
That is the shape of the loop, and the loop belongs to whoever writes it.

### More decisions taken

- **`Goal` is told, never inferred.** Nothing in a number says whether it should
  go up or down, so it lives on the piece that compares — a pruner without a
  direction is a state that cannot be written. `min`/`max`, and a typo is caught
  where it was typed rather than becoming a search that optimised backwards.
- **What is not a number is pruned by every scheme, warmup or no warmup.** A
  `NaN` loss does not recover, and the epochs spent finding that out are the
  cheapest a pruner can save.
- **`Percentile` compares each trial's best so far, not its latest value.** One
  bad epoch is noise; a run that already touched a good number has shown it can.
- **`p` is the share that is *kept*** — smaller prunes more, optuna's way round.
  Written the other way in the first draft, and it was a test that found it.
- **`Patience.steps` is a `NonZeroUsize`.** Zero patience would prune every trial
  at its first report, improvement or not: made impossible rather than validated.
- **`Reason` is structured, not a string.** "How many were pruned, and for which
  of the three reasons" is the question you ask of a search that pruned too much.

### Questionnaire — the pruner

**The schemes** (`study/tests/unit/pruner/`, `test_pruner.py`)
- [x] the median drops what is behind the finished trials and keeps what is not,
      and a trial that ties is not pruned for tying
- [x] `warmup` buys a slow starter its epochs; `startup` stops the first trial to
      finish becoming the bar; only the trials that got this far have a say
- [x] it compares the best so far, so a run that touched a good number survives a
      bad epoch
- [x] maximizing is the same thing read from the other end
- [x] a threshold works with no other trial at all, which is where a diverged
      configuration costs most
- [x] patience prunes a trial the field has no complaint about, and a delta stops
      noise from looking like progress
- [x] what diverged goes under all three, inside the warmup
- [x] going through the enum judges exactly the same as not going through it
- [x] every reason says enough to act on without the curve in front of you

**And the point of it** (`test_pruner.py`)
- [x] **a pruned trial simply stops being stepped** — no callback, no flag, no
      `trainer.stop()`
- [x] one that is holding its own runs to the end, same loop, same trainer

### The sampler, and the decision the original did not take

Three schemes again, and again what tells them apart is **what each one looks
at**:

| scheme | looks at | runs out | derivable from the index |
|---|---|---|---|
| `Grid` | the space's shape | **yes** | yes |
| `Random` | nothing | no | yes |
| `Tpe` | what already happened | no | **no** |

The column that matters is the last one. The original's `Sampler` took
`&mut self` and had a `prepare` to build its state up front; this one takes
neither, so a grid's combination is arithmetic on the index and a random point
comes from `(seed, trial)`. Asking for trial 7 twice gives the same answer, and
**asking for it without having asked for the first six gives the same answer
too**.

That is not tidiness. It is what lets a study spread over a shared folder work
with **nobody in charge**: `claim` hands a machine the number 7 and it derives
the point on its own — the same shape as CU15's federated round, and for the
same reason. `Tpe` is the honest exception and says so in its own docstring: it
is guided, so it depends on what the asking machine had seen, and a study spread
over four machines gets a different search than one in a single process.

`Grid` running out is the other half of it: `ask` answering `None` is how a `for`
stops **without being told a number**, which is the one thing a level-3 loop
would otherwise need a `Study` type to hold.

### More decisions taken

- **`log` is a property of the knob, not a transform the caller applies.** Drawn
  linearly, four fifths of `1e-5..1e-1` sits above `0.02` and a search never sees
  a small learning rate at all. A logarithmic range starting at zero is refused
  where it is written rather than becoming a `-inf` inside a draw.
- **A `Point` is a mapping in Python and a name in the record.** `build(**point)`
  works, and `str(point)` is `lr=0.001,batch=32` — derived from the values in the
  space's order, so two machines that never spoke file a configuration
  identically. It is half of what a trial's cache key will be.
- **`Tpe` keeps an option nobody tried reachable**, by counting one imaginary
  observation of each. A search that can never revisit a discarded option cannot
  recover from three unlucky trials.
- **A trial that scored `NaN` is dropped, not counted as terrible.** Counted, it
  would drag the good/bad split about; dropped, the proposal does not move at
  all, and there is a test that says so.
- **The generator is ours again**, splitmix64 as in the folds and for the same
  reason: a seed has to mean the same thing on every machine that reads the same
  record, which is not something `rand` promises across versions.

### Questionnaire — the sampler

**The space** (`study/tests/unit/space.rs`, `test_sampler.py`)
- [x] the knobs keep declaration order, which is what a grid and a name depend on
- [x] a duplicate name, an empty choice, a reversed range and a logarithmic range
      starting at zero are all refused where they were written
- [x] a space is built up and every call gives back a new one

**The schemes** (`study/tests/unit/sampler/`, `test_sampler.py`)
- [x] a grid walks every combination exactly once, takes **both ends** of a
      range, takes a narrow `int` whole, and then answers `None`
- [x] the same seed and index give the same point however it is asked for —
      including out of order and after everything else
- [x] what is drawn stays inside every knob, and a logarithmic one spreads over
      the decades rather than over the line
- [x] tpe concentrates where the good trials were, prefers the option they chose,
      and keeps one nobody tried reachable
- [x] before it has anything to learn from it is **exactly** the random one, seed
      for seed
- [x] maximizing looks at the other end of the scores
- [x] two of the three ignore what finished, which is why there are three
- [x] going through the enum asks exactly the same as not going through it

### What is NOT in it yet

**Naming a dataset by its content**, which is what a fold's cache key needs —
`(dataset, partition, i)` — and which CV is now the consumer for · **recording**
what was tried and what was pruned, which wants the store and not this crate ·
**conditional dimensions**, a knob that only exists when another took a
particular value, which needs a consumer before it needs a design · and **the
loop itself**, which is a `for` and will stay one.
