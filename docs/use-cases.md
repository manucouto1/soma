# Use cases

The project moves in vertical slices. Every use case reaches all the way to
Python, and is considered closed when it answers every guarantee on its
questionnaire.

**How to read it.** Each section records a decision at the moment it was taken.
The argument is the point — code moved on, the reasoning did not — so what a
section says is what was true when it closed, with three exceptions written down
once here rather than patched into every section that predates them:

- `workers={"gpu-box": Worker.at(…)}` in `Graph.forward` and in `Trainer` became
  `broker=Broker.embedded({…})` in CU28. The shape of what is handed over is the
  same; what it resolves to is no longer a connection.
- the crate called `transport` moved out to `soma-fabric/wire`, unchanged, after
  CU27. Paths in the questionnaires point at where the tests are today.
- `Transition`, `Await` and `Driver` — a node that suspends and something that
  serves what it asked for — were removed after CU18. A node is a function.
  Sections written before that still show them; *After CU18* says why they went.

---

## What each slice settled

| | | what it settled |
|---|---|---|
| CU1 | creating a graph | topology only, and errors at insertion rather than in a `validate()` |
| CU2 | executing one | the engine is Rust, `Value` has a closed set of variants |
| CU3 | the shape of it | `Plan` is an enum, and `compile` is the step between structure and engine |
| CU4 | fans both ways | aggregation is a node reading a map; there is no `Aggregator` trait |
| CU5 | the DSL | `>>` and `\|` in both languages, and `somatize` is the verb |
| CU6 & CU7 | one contract | one `Node` with one `forward`, in both languages |
| CU8 | `Opaque` | a value that only exists in this process, asked for by hand |
| CU9 | waves | one thread per branch, decomposed as a tree and not flattened |
| CU10 | the device | `Placement` is a fact of its own; the plan says *when*, not *where* |
| CU11 | training | outside the graph, and the **three levels** the rest rests on |
| CU12 | a worker | `Plan::Remote`, the `Transport` hole, and what does not travel |
| CU13 | the cache | a Merkle key over the recipe, and the `Keeper` hole |
| CU14 | training the far half | a trainer travels and stands beside the node |
| *after* | `Opaque` on the wire | a codec in front of it, measured at 9–15× |
| *after* | `every=N` | a group of steps is one update |
| CU15 | federated | export is weights node by node, `fedavg` is a function, a round is a `for` |
| CU16 | the grain of an item | an item's name is its content; micro-batches are level 2 |
| CU17 | level 3 | `Partition`, `Pruner`, `Sampler` — pure, and no callback crosses |
| CU18 | a study in a folder | a trial is a number, `claim` settles it, the state *is* the queue |
| *after* | a node is a function | `Transition` and `Driver` had no tenant and went |
| CU19 | a graph drawn | observability is **three** things, and this is the first |
| *after* | a bucket | the second `Store`, and `claim` is a conditional PUT |
| CU20 | the record | the `Watcher` hole; facts meet in the record, not in Rust |
| *after* | a run drawn | one drawing function, live and read back |
| CU21 | the diagnosis | an opinion about the record, reproducible without training again |
| CU22 | before a step | a probe is one recorded `forward`; what separates is a runaway |
| *after* | a block is a box | repeats, lanes, and an arrow that left the drawing |
| CU23 | the fleet | no registry; the record turned the other way up |
| CU24 | the data | a source is a **node**, and it is handed a coordinate |
| CU25 | not running what nobody needs | names are knowable before anything runs |
| CU26 | what an edit did | two sets of names, and findings rather than buckets |
| CU27 | what a node was built with | the declaration goes in the key |
| *after* | the wire leaves | `transport` → `soma-fabric/wire`, and why |
| CU28 | a broker | which broker is a URL; ask eagerly, connect lazily |
| CU29 | provenance | what cannot be recovered is written down when it is known |

---

## CU1 — Creating a graph

```python
g = somatize.Graph()
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
from real mistakes: `soma-legacy/soma-core/src/graph/node.rs` (172 lines) and its tests
`graph_node.rs` — `a_filter_keeps_its_caching_contract`,
`a_step_is_not_output_cacheable`, `schemas_survive_both_directions`.

### Questionnaire (from `soma-legacy/soma-core/tests/unit/graph*.rs`)

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
| `Graph` | the **structure** | `soma-core/src/graph.rs` |
| `Catalog` | the **store** of implementations | `soma-core/src/filter.rs` |
| `Filter` | the **contract** of an executable unit | `soma-core/src/filter.rs` |
| `Graph::run` | the **engine** | `soma-core/src/execution.rs` |

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

## CU3 — The shape of the execution

```python
g.plan()      # how it is going to be walked
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
a variant is added, the compiler points at the single place that has to decide
what to do with it. A wildcard arm would say nothing.

### The missing piece: compiling

Between the structure and the engine there is now a step:
`compile(&Graph, &Catalog) -> Plan`. It decides the shape, and along the way
**everything structural is detected before anything executes**. The engine no
longer works out where each node's input comes from: the plan says so.

### Decisions taken

1. **`Plan` is an enum**, not a trait. Closed, exhaustive, no wildcards.
2. **`Executor` is a type**, not a bare function: executing needs context (today
   the store; tomorrow a cache and events). That "tomorrow" is what the original
   calls `GraphSession`.
3. **`Value` loses `Tensor` and gains `Number` and `List`.** Nobody was producing
   a shaped tensor, and the round trip to Python has to be symmetric: what goes
   in as a list comes out as a list.

### What did NOT go in

**`Plan::Remote`.** There is no transport, so it would be a variant nobody can
execute. What the enum buys is precisely that adding it the day there is a worker
is one more variant, and that the compiler points at every place that has to
decide. *(It arrived in CU12.)*

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

### What was removed, and the lesson in it

A `Plan::Parallel` variant that meant *these branches do not depend on each
other* broke on the diamond: both branches claimed the join node and it executed
twice. The right shape is for **every step to carry where its input comes from**
(`Execute { node, from }`). With that the plan stays self-contained — the engine
does not look at the graph again — and the fans fall out with no special variant
at all.

So the variant went, and with it the `CompileError::Fanin` and `ManyLeaves`
errors, leaving `CompileError` with one. **A variant that only describes
structure buys nothing**; parallelism comes back in CU9 meaning something it did
not mean here — running at the same time.

## CU5 — The DSL

```python
from somatize import Filter, Graph

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
4. **The base class is abstract, and inheritance is what decides.** It requires
   its method with `@abstractmethod`, so a subclass without `forward` cannot even
   be instantiated, and `isinstance` is the only question the DSL asks. It was a
   correction: the bases were born as empty mixins that asked by duck typing
   whether the object *had* a method, and an object could get three different
   answers depending on which door it came in through. **The names promised a
   contract nobody enforced.** `node()` stays the lower door and accepts an
   outside object that inherits from nothing, because there the type is the
   caller's to choose.
5. **`Wire` does not materialize until `somatize`.** It records where you enter
   and where you leave, plus the lists of nodes and edges. That way joining two
   pieces is concatenating lists rather than merging two graphs, and a repeated
   id is caught at the end, once.
6. **The DSL is nothing but `node` and `edge`.** There is a test that builds the
   same graph both ways and compares nodes, edges and plan.

## CU6 & CU7 — A single contract, and the same one in both languages

```rust
pub trait Node: Send + Sync {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError>;
}
```

Status: **closed**. 47 tests in Rust, 56 in Python. Two use cases and one
argument: the core stopped having a `Filter` and a `Step` in CU6, and Python
stopped having them in CU7.

### The question: why two types?

The difference between a filter and a step was a single thing: whether it can
finish on its own. But that was already said in the **return value** — a filter
is a node that always answers `Done`. Having two traits duplicated in the type
system a distinction that lived somewhere else, and with it propagated upwards
the obligation to know which one each node was: catalog, plan, engine, errors,
adapters, DSL. **35 places.**

Two alternatives were tried before deciding, and both were rejected for concrete
reasons worth keeping:

- **A sugar trait with a blanket impl** (`impl<T: Filter> Node for T`). Compiled:
  `error[E0034]` — with two traits in scope the name `forward` is ambiguous *even
  when the arities differ*, because Rust resolves the name before the arguments.
  And `error[E0119]` — a type that implements `Filter` can no longer implement
  `Node` by hand, so a node could not evolve from always finishing to asking for
  a turn without being rewritten entirely.
- **State as a continuation** (`Pending { requests, resume }`). Simpler on the
  surface, but it breaks deterministic replay, and resuming would require
  serializing a `Box<dyn Node>`. The typestate variant dies sooner: the `Catalog`
  is a heterogeneous map that erases the type parameter, and it does not cross
  into Python at all.

### Decisions taken

1. **One trait, one method, `forward` in both languages.** Without the second
   trait there is no name ambiguity, so there is no need to call it `advance` in
   Rust — and no reason for Python to keep two doors either. `Filter`, `Step`,
   `g.step()`, `kind_of` and `Graph`'s two overrides all go.
2. **`input` travels apart from the context.** A node that never looks at `ctx`
   should not have to cross a struct to reach the only thing it cares about.
3. **`Ctx` is a `#[pyclass]`**, not a dictionary. It is a core concept crossing
   the seam, so the adapter recognizes it by type instead of guessing from a
   dict's keys, and `ctx.device` reads as it does in Rust.
4. **One adapter, not two.** `PyFilterNode` and `PyStepNode` merge into `PyNode`.

### What disappeared

`trait Step`, `FilterError`, `StepError`, `StepCtx`, `NodeImpl`, `Plan::Step`,
`RunError::{Filter, Step, WrongKind}`, `insert_filter`/`insert_step`,
`run_filter`/`drive_step`, `PyStep`, `Pure`. `RunError` goes from 7 variants to
5, `Plan` from 4 to 3, and `compile` no longer needs the catalog to know what
kind each node is — only to check that there is one.

### The conclusion outlived its own mechanism

CU6's argument was that the distinction lived in the return value and not in the
type. *After CU18* removed the return value's variants too — and the conclusion
did not need them: with one shape there is nothing left for the distinction to
live in. The diagnosis was right and the mechanism it rested on was incidental,
which is the most a design argument can hope for.


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

A **registry of opaque types** that `somatize.torch` would fill on import was
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

`soma-python/tests/test_pipeline_torch.py` assembles a four-node pipeline —
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
  by walking `g.nodes()` is exactly the pain a `somatize.torch.parameters(g)`
  would erase. It is in plain sight so the decision is made with the example in
  front of you.

### What did NOT go in

`somatize.torch` — `module()`, `parameters()`, the training loop — is left for
when it is clear how it should work. **The core provides the hole; whoever knows
what goes in it is a library**, and that separation is what let this be closed
without deciding that. *(It opens in CU11.)*

## An aside: micro-batches, and what became of the open questions

The plan was left open here on 16 August 2026, with what was then called CU12 in
doubt. It is worth keeping because the doubt was right: **"micro-batches" covers
three problems that have neither the same owner nor the same value.**

| problem | what solves it | whose it is | consumer? |
|---|---|---|---|
| the batch does not fit in memory | splitting it and accumulating gradients | the **Trainer**'s, five lines | yes, and it is 80% of cases |
| the bubble: `cuda:1` idle while `cuda:0` computes | chaining micro-batches | the **graph**'s | doubtful |
| bounding the live activations | a real 1F1B scheduler | **nobody's**, and that is the problem | no |

**The bubble may already not exist.** CUDA launches asynchronously, so a
micro-batch loop on the host already overlaps the devices without a scheduler —
nothing synchronizes along the way, since `Opaque` wraps the tensor and there is
no `.item()` in the seam. What a scheduler would add has to be measured before it
is written.

**Real 1F1B is not ours.** Its value is bounding how many micro-batches have
their activations live, and for that the backward passes have to be interleaved
with the forwards. The backward pass is fired by the Trainer, not by the engine,
so a 1F1B scheduler would require the plan to know about the backward pass — i.e.
putting training inside the graph, which is exactly what CU11 decided against.

What happened next: the first row is the Trainer's `every=` (*After CU14*), the
second and third never opened, and the local worker that was the third candidate
became **CU12**, whose benefit was measurable with one machine — two Python nodes
in a wave serialize against the GIL; in two processes they do not.

### Training from Rust, researched and never opened

It waits on a consumer with a name: a federated client that trains **without a
CPython loaded**. Four results from 16 August 2026, kept so the work is not done
twice:

- `tch::Tensor` is `Send` but **not `Sync`**, so it does not fit in
  `Value::Opaque`, whose bound is `Arc<dyn Any + Send + Sync>`. `tch` is ruled
  out short of wrapping every tensor in a `Mutex`.
- `candle_core::Tensor` **is** `Send + Sync` — an `Arc<RwLock<Storage>>` inside,
  and its own code says the `RwLock` was chosen for exactly that — so it would
  fit today without touching the core. Verified by compiling.
- the limit that does not move: **a graph is all-Python or all-Rust for the
  tensors**. An `Opaque` put there by Python carries a `PyObject`, and a Rust
  node doing `downcast_ref::<candle::Tensor>()` gets `None`. Converting for real
  would mean copying the raw data and losing the autograd graph, which is what
  `Opaque` exists to prevent.
- the order, if the day comes: **first a Rust node with parameters**, then the
  collection, and the Trainer last. Never starting with the Trainer.


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

**Decomposition** (`soma-core/tests/unit/plan.rs`)
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

**The oracle** (`soma-core/tests/unit/build.rs`)
- [x] seven DSL expressions, and their plan is the tree that was written

**Execution** (`soma-core/tests/unit/execution.rs`)
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

**Python** (`soma-python/tests/test_waves.py`)
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

**Rust** (`soma-core/tests/unit/device.rs`, `placement.rs`, `execution.rs`)
- [x] `cpu`, `cuda:N` and `meta` parse, and the round trip gives the same thing
- [x] `cude:0` is an unknown kind; `cuda` asks for an index; `cuda:`, `cuda:x`,
      `cuda:1:2`, `cpu:0` and `""` are not shaped like a device
- [x] `.on()` spreads over the whole piece and the innermost one wins
- [x] each branch of a `|` in its own place, and what is unplaced stays unplaced
- [x] the node sees its own and only its own — nobody catches the neighbour's
- [x] a wave's branches see different devices, each on its own thread
- [x] placing changes neither the plan, nor the graph, nor what it produces

**Python** (`soma-python/tests/test_device.py`)
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

**`somatize.torch`.** The pattern for obeying a placement is written by hand in
the test, which is where it is documented until it repeats. *(It opens in CU11.)*

**Generalizing `Placement` to "a place, local or remote"** to get ahead of a
worker. *(CU12 decided against it: a `Host` is a name and a `Device` is a place
inside a machine, and they are independent.)*

### Measured, not asserted

With a single GPU on the development machine, `cuda:1` does not exist: spreading
across two GPUs can be **declared** and cannot be **executed** here. The tests say
so in their names rather than leaving it implicit.

And CU9's warning still stands: do not justify this with a benchmark of two
branches on two GPUs. CUDA launches asynchronously and the two already overlap
when executed in sequence; what the waves buy is **host** time.

## CU11 — Training, outside the graph

```python
from somatize.torch import Trainer, parameters

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

**2. It lives in `somatize.torch`.** Loss, `backward()` and optimizer are torch;
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

**Python** (`soma-python/tests/test_trainer.py`)
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
study as a type · and **exporting or loading a model's state**, which is CU15's
question and was not needed here: training locally extracts no state.

### The question it reshaped

**The state question stops blocking**, and it stops being the core's. It is no
longer *is a node's state a `Value`?* but *what does a training run export?* at
level 2 — answered in CU15, with the case in front of us.

The three levels also split what the original had joined. Spreading *one graph*
across hosts has dependencies halfway through the forward and needs
`Plan::Remote`; spreading *whole training runs* — HPO, federated, data parallel
— is "execute this thing over there", level 3, and needs none of it. The
original had them in one enum, `ModelParallel` beside `DataParallel`,
`Federated` and `PopulationBased`.

---

## CU12 — A slice that runs in another process

```python
# A. the worker is your own code, already on that machine
g = Graph.somatize(Encode() >> Classify().at("gpu-box"))
g.forward(x, workers={"gpu-box": Worker.at("gpu-box:7000")})

# B. the worker is a bare node: `pip install somatize` and nothing else
#    python -m somatize.worker --listen 0.0.0.0:7000
g.forward(x, workers={"gpu-box": Worker.at("gpu-box:7000",
                                           mode="network", send=["my_package"])})
```

Status: **closed**. A crate of its own — `transport`, which left for
`soma-fabric/wire` after CU27 — and the generic worker in `somatize.worker`. 74
tests in it — 49 of the worker, 22 of the protocol, 3 of the artifact — plus
`test_remote.py` (37),
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
at what is installed, at what has no source, and at `somatize` itself, which
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
`somatize.worker`'s own docstring rather than left to be discovered.

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

**The core's seam** (`soma-core/tests/unit/execution.rs`, with a double that never
leaves its seat)
- [x] what a slice reads and does not produce is what travels with it, and no more
- [x] the placement travels; the host half does not, having already done its job
- [x] a host nobody resolves is **not executed here just in case**
- [x] what comes back is merged as if it had been produced here

**The transport** (`soma-fabric/wire/tests/unit/`)
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

`soma-python/tests/cluster/` is the same thing without the tricks: four containers
declared in its own `docker/compose.yaml`, each with **the wheel and nothing
else**, and the client outside. What only becomes provable there:

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
moved. Silently wrong numbers, with nobody asking the question.

The question is asked now, and at the level that can ask it. After the first
`backward()`, if the optimizer is about to update a parameter that received no
gradient, `Trainer` stops with `NoGradient` and names the node. It is more
general than the case that prompted it: a slice on another host, an output read
back from a store, a branch the loss never reads — one symptom, one check.

And **`.at()` is not a refusal to train**, which is why the check is about
gradients and not about hosts. The far half can perfectly well train itself —
that is [**split learning**](https://arxiv.org/abs/1812.00564) — and it worked
here with no framework at all: a node told which half of the pass it was in,
branching on it, keeping its activation between the two calls.

Three things that were already there made it fall out: a worker **keeps its
catalog**, so the node object survives between calls and its activation stays
alive on the far side; a node is **one contract**, so it dispatches on its input
instead of needing a kind of its own; and a gradient is a tensor like any other,
so it crosses as data. Those three are why CU14 could take it over without a new
variant anywhere — the mechanism was already right, it was the **user** who was
being asked to drive it. Nobody writes that node any more; it survives in the
tests as the **control**, because the strongest thing a framework can show is the
loop it replaced, running beside it, producing the same losses step for step.


### What did NOT go in

**Authentication and encryption**, on purpose and for good (decision 8) ·
**installing the environment**, which is what cost the original 420 lines ·
**scheduling**: which host gets what is declared, not decided — there is no
placement policy and no load balancing · **retrying a failed slice**, because a
node that already ran half of itself is not idempotent and nobody has said what
it means to run it again · **a protocol version**, since both sides are the same
binary from the same `cargo build`; the day they stop being so, the place for one
is the `Hello`, which already negotiates the runtime · and **a store**, which is
CU13 and where the `have`/`want` finally got its `have`.

---

## CU13 — What is remembered, and what is not computed twice

```python
from somatize.torch import Trainer, freeze, parameters

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
knows nothing about. `somatize.torch.freeze` is what makes it true, with
`requires_grad_(False)`, exactly as a node and not the core is what moves a
tensor to a GPU. And the digest of the weights is paid for **there**, once,
because settling is the moment that makes both halves true at the same time.

### The fourth hole, and the first that is a decorator

`Keeper` joins `Node`, `Driver` and `Transport`: the core provides the hole,
whoever knows what goes in it is a library. Here it is doubly true — hashing is
`sha256` and keeping is a directory, and the core has no dependencies at all.

**Driver serves, Transport carries, Keeper keeps.**

What fills it is `somatize_store::Cache`, and in Python there is a second one in
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
register; `somatize.torch` fills in the tensor's **on being imported**, so a
graph that keeps tensors and never imports it keeps nothing and says why on
`stderr`. Importing `torch` is not enough and is not meant to be: registering it
from `somatize` would mean importing torch for everyone who does not have it.
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
to call `somatize.torch.freeze(g)`. There is a test that reproduces the bad hit
and one that shows two checkpoints settled at the same digest **are** one name,
because that is what says the digest is what the key believes.

### Questionnaire

**The core** (`soma-core/tests/unit/{execution,build,memory}.rs`)
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

**The store** (`soma-store/tests/unit/cache.rs`)
- [x] the pieces of a recipe cannot run into each other: `["ab","c"]` and
      `["a","bc"]` are two names
- [x] the same recipe is the same name every time; only a root is named by its
      content
- [x] a batch answers in the order it was asked, holes included
- [x] a name nobody kept is a miss and not a failure
- [x] a kept value is findable by looking at what is in the store
- [x] the same bytes under two names are stored once

**The worker** (`soma-fabric/wire/tests/unit/worker.rs`)
- [x] what a worker already kept is not run again **over there** — and the same
      worker without a keeper runs it every time
- [x] the name of what ran over there comes back

**Python** (`soma-python/tests/{test_cache,test_freeze}.py`)
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
a variant that arrives late. *(CU16 wrote it, and found that it and micro-batches
are **not** the same question.)*

Also out: **`.overwrite(times=1)`**, which is a policy of the *run* and lives in
the executor, not in what is kept · **the queryable index** — what do I have,
from which run, from when — which is a SQLite derived from the records and
throwaway, and making it the truth would mean a single writer over NFS · a
**strict mode** for the fingerprint (`.cached(strict=True)`) · and **S3**, which
arrives the day there is a MinIO to point at, through OpenDAL and as another
configuration rather than another implementation. *(It arrived after CU19, and
that last prediction was wrong on both halves — see there.)*

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
from somatize.torch import Split, Trainer, parameters

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

**Cutting the graph** (`soma-python/tests/test_stage.py`, no torch anywhere in it)
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

**The trainer** (`soma-python/tests/test_learning.py`)
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

**Driving it** (`soma-python/tests/test_trainer.py`)
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
- **A worker never imports `somatize.torch`.** It starts empty, and the nodes
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

**The seam** (`soma-fabric/wire/tests/unit/worker.rs`, no Python in it)
- [x] with a codec, an opaque produced over there comes back what it was
- [x] and one bound for over there arrives as what it was
- [x] one the codec cannot write and the slice answers with is refused **in the
      codec's own words, which are the far end's**
- [x] one it cannot write and nobody asked for stays where it ran
- [x] a worker that does not pack hands its node what it was sent, and the
      failure is quiet — which is why nothing installs one end without the other

**From Python** (`soma-python/tests/test_remote.py`)
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

The last of the three micro-batch problems — *gradient accumulation* — left for
last because it is small. The other pending closed before CU15.

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

**The group** (`soma-python/tests/test_trainer.py`)
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

**What it exports** (`soma-python/tests/test_federated.py`)
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

**The store, opened by hand** (`soma-python/tests/test_store.py`)
- [x] bytes by what they are, names that point at them, and both directions of
      each
- [x] a value with tensors in it goes in and comes out alive, and a bare tensor
      is kept although it would not cross an edge
- [x] something nobody registered a codec for says which type it was
- [x] a training run written down by **another interpreter** is read back here to
      the same weights, and two processes that wrote the same weights wrote them
      once

**Claiming** (`soma-store/tests/unit/local.rs`, `test_store.py`)
- [x] a name nobody has can be claimed and one somebody has cannot
- [x] what `bind` replaces, `claim` refuses
- [x] eight threads and eight **processes** on one name: exactly one wins, and
      the one told it won is the one written down
- [x] a claim leaves nothing behind in the temporaries
- [x] against the mutant written as `resolve` then `bind`, seven of the eight
      racers were told they had it

**The round** (`soma-python/tests/test_round.py`)
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

**Cutting a batch** (`soma-python/tests/test_trainer.py`)
- [x] a batch in pieces comes out where the whole one does
- [x] the optimizer still moves once a step, and `every` and `micro` multiply
- [x] the loss it gives back is still the number the whole batch would have said
- [x] a batch that does not divide is refused with both numbers and the flag that
      fixes it; so is something it cannot cut, and halves that do not line up
- [x] across a cut, the far side counts the **pieces**

**The grain of an item** (`soma-core/tests/unit/execution.rs`, `test_cache.py`)
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
from somatize.study import Partition

for train, test in Partition.stratified(5).folds(len(y), classes=y.tolist()):
    trainer.fit(data[train], epochs=10)
    scores.append(evaluate(g, data[test]))
```

Status: **closed**, in three passes over the same shape — the cut first, then the
pruner, then the sampler. Opened 21 August 2026.

The level the vision calls **Study**: hyper-parameter search, cross-validation,
and whatever else is N training runs rather than one. Three families, each an
enum of structs: `Partition` says where to cut, `Pruner` when to give up, and
`Sampler` where to look. What joins them is CU18.

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
- **It is called `Partition`, not `Split`.** `somatize.torch.Split` is already
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

**The cut** (`soma-study/tests/unit/partition.rs`, `test_partition.py`)
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

**The schemes** (`soma-study/tests/unit/pruner/`, `test_pruner.py`)
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

**The space** (`soma-study/tests/unit/space.rs`, `test_sampler.py`)
- [x] the knobs keep declaration order, which is what a grid and a name depend on
- [x] a duplicate name, an empty choice, a reversed range and a logarithmic range
      starting at zero are all refused where they were written
- [x] a space is built up and every call gives back a new one

**The schemes** (`soma-study/tests/unit/sampler/`, `test_sampler.py`)
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
`(dataset, partition, i)` — and which CV is now the consumer for *(CU24)* ·
**recording** what was tried and what was pruned, which wants the store and not
this crate *(CU18)* ·
**conditional dimensions**, a knob that only exists when another took a
particular value, which needs a consumer before it needs a design · and **the
loop itself**, which is a `for` and will stay one.

## CU18 — A study handed out of a folder

```python
# the same script on every machine; Slurm gives out `me`
for trial in range(100):
    point = sampler.ask(space, trial, finished(store, space, study="spam"))
    if not take(store, point, study="spam", trial=trial, me=me):
        continue                                   # somebody else has that one
    ...
    report(store, point, drawn, study="spam", trial=trial, me=me, state="done")
```

CU17 left the three families and nothing that joined them: `Sampler` said where
to look, `Pruner` when to stop, and there was no way for two machines to search
one space together. This is that, and it is the **first half**: the guided
sampler spread over machines is still open, for a reason given below. Opened on
22 August 2026.

### The answer was already in the building

`soma-coordinator` in the original is 959 lines and does not contain the word
`trial` once: it hands out **plans**, not trials. The study never distributed,
and the reason is mechanical — `StudyIo::save()` writes the whole `Study` as one
JSON. One file, one writer. Two machines calling `save()` overwrite each other,
which is why the trait has exactly one implementor.

The `EventBus` is not the answer either. It is a `tokio::broadcast` inside one
process, and its own docstring says the subscriber path is *"lossy under lag.
Display / relay only"*. Three different things get called a bus and they have
opposite semantics:

| | losing a message means |
|---|---|
| observation fan-out | nothing |
| a work queue | a trial never runs |
| a durable log | both, plus a service to operate |

> **What cannot be lost goes in the store. What only deserves looking at goes on
> the bus. The store is the truth; the bus is a view of it.**

And the work queue was already here, and is better than a queue: `claim` is an
atomic `link`. No server, and **no message can be lost because there are no
messages** — the state *is* the queue, exactly-once by construction. The one
thing a real queue would add is a visibility timeout.

The deployment argument runs the opposite way to the expected one, too. On
Slurm/HPC/NFS the directory wins outright because there is no service to deploy;
on a platform there is no directory but there is S3 with conditional writes
(`If-None-Match`), which **is** `claim`. The directory is not a limitation to
outgrow, it is an implementation of `Store`.

### Handing out work costs nothing because nothing is handed out

A trial is a number. `ask` is a function of that number and not of what was asked
before, so a machine that claims trial 7 works out where to look on its own
without replaying six and without asking anybody. That is the property CU17 built
the samplers around, and this is what it was for.

`Sampler.tpe` is the honest exception, and it is why CU18 is only half done: two
machines asking at the same moment see the same history and propose neighbouring
points. That is the known cost of parallel Bayesian optimisation — *constant
liar*, penalising trials in flight — and there is none of it here yet.

### What a trial is, on disk

```text
<study>/trial/<n>/<attempt>
```

In the **record**, which a scan already carries: `state`, `point`, `score`,
`who`. In the **blob**, for whoever wants the detail: the whole curve and why it
stopped.

The split is not tidiness, it is the cost model, and the tests measure it with a
store that counts:

| reader | what it does | cost |
|---|---|---|
| `finished`, for a sampler | one scan; point and score are both in the record | **zero fetches** |
| `curves`, for a pruner | the same scan, then the blobs | one fetch per trial |

**One record rewritten as it goes, and not five events.** The original's
`TrialStarted`/`TrialMetric`/`TrialPruned`/`TrialCompleted`/`TrialFailed` are the
*diff* of this record. From a state the events derive; from a lossy stream the
state does not.

The `<attempt>` segment has no reader yet and is paid for anyway, with `0`.
`claim` is a link, so a trial whose machine died stays claimed for ever and
rescuing it with a plain write would be a race; a retry is a claim of the next
attempt and whoever reads keeps the highest. It is paid now because **the name is
the one part of the design that cannot be refactored later**: changing it means
migrating directories belonging to people with studies running.

### `Space::read`, and why it is a method of the space

A record keeps the configuration as text beside the score, which is what makes
the sampler's history one scan. Reading it back needs the knobs in front of it:
`batch=64` on its own does not say whether 64 is a whole number or an option
spelt `"64"`. Nothing in the text can settle that, so reading is not something a
`Point` can do for itself.

And what could not be read back is refused **where it was typed** — a knob name
or a choice option carrying a `,` or an `=`. Caught at the read it would be too
late: by then which knob was meant is gone.

### A pruned trial is not a configuration that scored badly

Its score is real and it is **not** comparable with a finished one: it was
measured after fewer epochs. A sampler handed it as an ordinary result learns
that a region is bad when all that happened is that it was cut short. So
`finished` returns what ran to the end, and pruned trials stay visible in
`trials` where a notebook can see them.

### The end-to-end case, and the two things it found

`tests/cluster/test_searching.py` is the first test of level 3 with a real
pipeline under it — real SMS messages out of the SMS Spam Collection, a graph of
preprocessing → embedding → classifier cut across containers, and the study
itself cut across machines.
Two distributions at once, and they are not the same distribution:

- **the graph** by `.at()` — tokenising on a worker with 193 MB and *no torch in
  it at all*, the embedding on the one that has it, trained over there by a
  `Split` while the classifier stays with the loop;
- **the study** by `claim` — processes over one directory, each deriving its own
  configuration from the index and pruning against curves the others drew.

Two things came out of writing it that no unit test was ever going to say:

**A worker holds one catalog.** Two machines running different graphs against the
same worker is the second of them being told to reconnect. That is not a bug to
route around — a machine searching a space needs a worker of its own, exactly as
it would on Slurm. It costs a container, not an image.

**A cut graph cannot be scored on held-out data from here.** `embed` is trained
where it runs, so `export` refuses to hand back a copy that never learnt
anything — the right refusal. So the number a study of a cut graph compares is
the one the loop produces. The day a held-out score is wanted, what has to travel
is the **scoring**, the same way the trainer travelled in CU14.

And one that was simply broken and nobody had noticed: the worker image never
copied `study/`, so the cluster images had not been rebuildable since CU17.

### Questionnaire

**Handing out the work** (`test_study.py`, `tests/cluster/test_searching.py`)
- [x] a trial somebody claimed is not claimable twice, and the loser goes on
- [x] four processes over one directory run every trial exactly once
- [x] and what they searched is what one machine alone would have searched
- [x] two studies sharing a directory are two studies
- [x] whatever else is in the store is not a trial

**What a scan costs**
- [x] the history a sampler wants comes back with **zero** fetches
- [x] the curves a pruner wants cost one fetch per trial
- [x] a machine that ran none of them rebuilds the whole history

**Reading a point back** (`soma-study/tests/unit/space.rs`)
- [x] every point the space can produce survives being written down
- [x] the space is what says whether `64` is a number or a word
- [x] a record written against another space is refused and not half read
- [x] a name or an option with a `,` or an `=` is refused where it was typed

**The states, which are not the same state**
- [x] a pruned trial is not a configuration that scored badly
- [x] the curve is watchable while it is still being drawn
- [x] a retry is the next attempt and whoever reads keeps the higher

**Which way is better, which the number does not say** (`test_study.py`)
- [x] a score carries the direction it was searched in
- [x] a trial claimed and never reported still says which way it looked
- [x] a study nobody told answers `None` rather than guessing `min`
- [x] changing your mind halfway is the newest record and not a vote
- [x] a typo is caught where it was typed, before anything is written

**The real case** (`tests/cluster/test_searching.py`)
- [x] something it tried actually learnt to tell spam from ham
- [x] the configurations are not all the same one
- [x] what was given up on stopped early, what was not ran to the end
- [x] the preprocessing ran where there is no torch at all
- [x] the embedding was trained on the machine that has it

### What is NOT in it yet

**a bus**, which earns its place
in observability and not in coordination, and would turn every crate it touches
async for no subscriber · **a held-out score for a cut graph**, which is scoring
that travels · and **retries**, which have a name on disk and no reader.

## After CU18 — A node is a function

```rust
pub trait Node: Send + Sync {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError>;
}
```

`Transition` and `Driver` are gone. Decided and done on 22 August 2026, and the
argument is short enough to fit here.

### What they were, and what they cost

`Transition::Await(requests)` let a node **suspend**: it said what it needed, an
injected `Driver` served it, and the node was asked again with the answers in
`ctx.results`. It was the seam of an agentic layer — `driver.rs` said so in its
own docstring, *"that ignorance is what keeps the agentic layer out of the
core"*.

| | |
|---|---|
| `Done(...)` written by hand across the repo | **164** |
| places that returned `Await` | **14** |
| ...of those, outside the tests | **0** |
| core surface | `Transition`, `Driver`, `ctx.turn`, `ctx.results`, `MAX_TURNS`, 3 `RunError` variants, the turn loop |
| plumbing beyond the core | 12 files — bindings, transport, worker |

Every node anybody wrote paid for the wrapper, and after eighteen use cases half
the enum had **no tenant**. The project's first rule is that nothing is written
without a real consumer today; this was the largest exception in it.

### The three things only suspension bought, honestly

1. **The engine could bound a runaway loop.** `MAX_TURNS = 64`, a node that
   cannot stop failing instead of hanging. Real.
2. **The engine saw each turn.** A base for traces, replay, checkpoints — and
   **nothing used it**, not one hook.
3. **A suspended node is resumable**: a value and a turn number. The big
   theoretical justification, and **the design did not deliver it** — a node that
   accumulates does so in its own object, which is serialized nowhere. A durable
   agent was not checkpointable however much you wanted one.

One works, one is unused, one is not delivered.

### What was kept, which is what makes it reversible

**`Ctx` stays.** Not `turn`, not `results` — the type, carrying `device`. It is
the **channel** by which whoever executes hands a node what it knows, and the
day an agentic layer wants something injected it goes there and **no node
signature changes**. That was the one strong argument for keeping `Await` — cheap
now, breaking to add back — and keeping the channel dissolves it.

The agentic layer goes as **concrete nodes** instead: one that wraps a model call
and its retries, one that routes, one that orchestrates several agents, each
holding whatever client it takes. Retries in particular belong there and the repo
had already said so — CU12 refused them at the engine because *"a node that
already ran half of itself is not idempotent and nobody has said what it means to
run it again"*.

### What is lost, said as a decision and not a side effect

**The engine no longer bounds anything.** A node that does not return does not
return, the same way a function that does not return does not. It could not
honestly do better: it has no way to tell a loop that will not end from work
that is slow, and guessing wrong either way is worse than not guessing.

And **whatever a node keeps is entirely its own business** — which was already
true, and is now the only thing that is true. The catalog holds the node, not a
copy per run, so state outlives a `forward`: that is what lets an activation stay
alive across a cut and what `Split` rests on, and it is a trap from the other
side. There is a test in both languages saying so.

`RunError` goes from 11 variants to 8. `advance()` goes from 38 lines to 5. An
artifact carries **the nodes** and not `(nodes, driver)`, so `provide()` returns
a dict.

### And the other half of CU18: a guided sampler that knows what is in flight

A guided sampler spread over machines is worse than random until it knows what
the others are holding, and the record was shaped so that knowing costs one scan
and no fetches — `state = running` sits beside `point`.

The plan was to need no Rust: hand the sampler the in-flight points with a
made-up bad score, which is *constant liar* as the literature has it, and let it
avoid them. **Measured, that backfires.** `Tpe` sizes the pile it imitates as a
share of everything handed over, so one more point raises the quota and promotes
a trial out of the bad pile into the good one — and when that trial sits in the
same region as the one in flight, the warning pulls the search **towards** it.
Of two hundred proposals, one landed on the occupied region without the warning
and thirty-nine with it.

So `ask` takes `&[(Point, Option<f64>)]` and **an absent score means running**.
Nothing is made up: those points go in the pile to keep away from and do not vote
on how big the other pile is. Four of the five schemes ignore the argument
entirely, as they already ignored the history.

`abandoned` is the other half, and it decides nothing. Liveness here is not "does
it answer" — nothing asks — but "is it still writing", and `report` was already
paying for that heartbeat. It is measured against the newest write in the study
and **not against this machine's clock**, because two machines sharing a folder
are two clocks that disagree by minutes.

---


## CU19 — A graph draws itself

```python
g = Graph.somatize(
    Tokenize().at("worker1").mapped()
    >> Embed().on("cuda:0").cached()
    >> (Svm().named("classic") | Net().on("cuda:1").frozen())
    >> Vote()
)

g                  # in a notebook, the figure: what runs at once, and what leaves
g.figure()         # the same, as a plotly Figure, to show or to compose
```

The first slice of the layer Manu calls *practically the most important part of
the framework*: a researcher opens a notebook, declares a graph, prints it and
**sees** the architecture with the decisions on it. Opened and closed on
22 August 2026.

Before it there was nothing to see anywhere: no `Instant`, no `elapsed`, no
`logging` in `core/`, `transport/`, `store/` or `study/` — ten loose `eprintln!`
and level 3's trial record, which had been observability all along under another
name.

### Observability is three things, and the original made them one

Splitting them is the whole design, and the rest of the layer rests on it:

| | what it is | needs |
|---|---|---|
| the declaration, **drawn** | a graph can be drawn having never run | nothing |
| the record of what **happened** | facts; the residue of running | a store |
| the **diagnosis** | *an opinion about the facts*, with arguable thresholds | the record |

The original keeps all three in one `enum Event` of **37 variants**, so
`NodeStarted` — a fact — sits beside `HealthFlag` — a judgement about a fact —
and both beside seven agentic variants whose mechanism this project has since
removed. Underneath: a `Tracker` trait with **one** implementor and an
`EventSink` with **one**.

> **The invariant that makes the split real, and it is a test and not an
> aspiration: a diagnosis has to be reproducible from the stored record, without
> training again.** An alarm that can only be raised live means the layers are
> tangled.

CU19 is the first row only. It touches no event, no bus and no store, and that
is what let it be written without deciding anything about the other two.

### What is drawn is the plan, and not the graph

Because the plan is where the decisions show: a `Wave` is what runs at once, a
`Remote` is what crosses to another machine. A bare list of edges says neither.

| in the plan | on the figure |
|---|---|
| `Execute` | a box, filled by the device it runs on |
| `Sequence` | its children stacked, top to bottom |
| `Wave` | its children side by side, inside a frame |
| `Remote` | a frame labelled with the host |
| `Empty` | an empty figure that says so, not an exception |

`Sequence` gets no frame: top-to-bottom is already how a figure is read, and the
root is always one — a box around everything is a border.

And **the layout needs no heuristic**, because `Plan` is a *tree* and not a DAG:
one pass upwards asking each subtree its size, one pass downwards handing out
positions. No crossing minimisation, no Sugiyama, nothing to tune.

### The arrows are not decoration

`decompose` is a real series-parallel decomposition — components become a `Wave`,
a series cut becomes a `Sequence` — but it has a way out at the bottom, and it
says so itself:

```rust
let Some(cut) = series_cut(graph, nodes) else {
    // No cut, no tree: walked in sequence, as before waves existed.
    return Plan::Sequence(nodes.iter().map(|node| step(graph, node)).collect());
};
```

A graph that is **not** series-parallel — only reachable through `node()`/`edge()`,
never through the DSL — falls back to a flat `Sequence`, and there the nesting
stops saying who feeds whom. The `N` is the case: `a→c`, `a→d`, `b→d` comes out
as four boxes in a column, with `a` above `b` although neither reads the other.

> **The boxes say *when*. The arrows, from each step's `from`, say *what feeds
> what*.** For a graph built with `>>` and `|` the two agree; for the other one
> the arrows are all there is, and a figure without them would be a lie.

### One table of colours, and what a fill is allowed to mean

The fill says **where a node runs** and nothing else. Cached, frozen and mapped
are badges in the label: three facts do not fit in one colour, and inventing a
precedence between them would only hide two of the three.

The table is looked up with `[]` and never with `.get(…, default)`. The original
left the reason written down — the same five strings lived in four tables, two of
them ending in a catch-all arm meaning *flagged*, so a typo came out as the alarm
colour instead of failing. Here a typo raises.

### Two things the wall taught, one of them the oracle's

The original wrote **its own SVG renderer** for one reason: a notebook sanitises
`<script>`, and mermaid needs a JS runtime. The same wall stands in front of
plotly, and the way through it is the one plotly already uses — a figure reaches
a cell through the **mimebundle** it publishes, not through hand-written HTML. So
`Graph._repr_mimebundle_` delegates to the figure's, and there is no
`_repr_html_` here at all.

The second is plotly's: `Figure._repr_mimebundle_()` answers `{}` when no
renderer is configured, which is what happens outside a notebook. An empty bundle
has to become `None`, or the cell shows neither a figure nor a `repr`.

### What the oracle does not have

**The original does not draw placement at all** — not the device, not the host,
not the worker. Its `NodeOverlay` is `{status, duration_ms, cache_tier, flags,
sublabel}` and that is the lot. What was asked for here has no answer over there
and was decided here. What *was* taken, because it is knowledge and not design:
one palette for every renderer; a diagram stops being readable past **80 nodes**,
which is measured and not guessed; and an escaping test with a node called
`<script>`.

### Two gaps closed on the way

`mapped` had a **setter and no reader** from Python, while `frozen`, `cached`,
`devices`, `hosts`, `identities` and `fingerprints` all had one — hence
`mapped_nodes()`, a list and not a dict because mapping carries nothing beside
it. And the plan only came out as a `Debug` string; `plan_json()` hands it over
as **data**, because parsing a `Debug` to find out what runs beside what is how
a renderer starts lying. `plan()` stays: it answers to a person, and to the tests
that already compare its text.

The core still does not learn to draw. It is asked for the fact.

### Questionnaire

**Where the boxes land** (`test_figure.py`, over `boxes`, which is pure)
- [x] a wave puts its branches side by side, and a sequence stacks its steps
- [x] a wave is framed and a sequence is not
- [x] a remote frames its slice, says the host, and contains what it framed

**That the figure does not lie**
- [x] the `N` has no series cut, falls back to a flat `Sequence`, and its three
      edges are drawn anyway
- [x] every edge of a DSL graph is drawn too
- [x] an arrow crosses into a remote slice

**What is on it**
- [x] device, host, salt, state, mapping and fingerprint are all on the hover
- [x] a mapped node is marked in its box
- [x] a node named after its class does not say so twice; one named by hand does

**What it costs**
- [x] drawing runs nothing — a node that would raise if it ran is drawn quietly
- [x] an empty graph is a statement and not an exception
- [x] a node called `<script>` never reaches the page as live HTML
- [x] without plotly the notebook falls back to text, and `figure()` says what to
      install
- [x] a graph too big to read is not drawn on its own, and still is if asked

### What is NOT in it yet

**An overlay** — what happened while it ran — which costs this nothing: the
original left it written that an empty overlay has to give a byte-identical
drawing. *(CU20 records the facts; CU21 puts them on the figure.)* · **the plan
drawn while it runs**, which is the same thing said live · **a text fallback** for a graph past the limit, where the
original had one and here the notebook just shows the `repr` · and **anything at
all about a worker's health**, which needs something to report it first.

## After CU19 — A store that is a bucket

```python
store = Store.on_bucket("http://minio:9000", "soma")   # S3, MinIO, R2
```

```rust
let store = Bucket::at(endpoint, "soma", "us-east-1", UrlStyle::Path, credentials)?;
```

Not a use case: the second implementor of a trait that had one, written on
22 August 2026 and closed on the 23rd, out of a question — **what happens when the workers and whoever
launched them share no disk?** It went in front of CU20 because the answer
decides where the record of what happened can live.

### Three uses of a store, and only one of them demands sharing

| what for | who opens it | does it need a shared disk? |
|---|---|---|
| the cache of a `.cached()` node | each worker, its own | **no** — it degrades to a miss |
| artifacts, so a catalog is not sent twice | each worker, its own | **no** — it degrades to resending |
| federated rounds (CU14) and a study (CU18) | everybody, the same one | **yes, and there was no alternative** |

The store **does not travel**: it is one more thing that was lent, like the
catalog, and whoever brings a worker up hands it one with `--store`. The keys
come out of the content, so two stores are two hit rates and never two answers.
Only the third row was actually stuck, and it is the row where `claim` hands out
work.

### A bucket and a directory are the same store, deliberately

The same split, from the same `Digest::path`, and the **same JSON** inside. So a
directory can be moved onto a bucket with `aws s3 sync` and back, and neither end
has to know. Naming an object after the digest **of the name** rather than after
the name settles what a key may contain, which on a filesystem was already
settled the same way.

Two things are genuinely different:

- **`claim` is a conditional PUT.** On a filesystem it is a hard link, which
  fails when the name is taken; here it is `If-None-Match: *`, the same promise
  from the other side, and the signature covers the header.
- **A scan costs a round trip per name.** `bound()` lists and then reads, fanned
  out sixteen at a time — because S3 has no *"give me these forty objects"*, only
  forty round trips that need not happen one after another. The way out, when it
  starts hurting, is the one `Store::bound` already names: an index built from
  the records, which can be thrown away.

### The two round trips on the way in, which are not optional

`Bucket::at` writes a probe key twice before it hands the store over.

An S3-compatible endpoint that accepts `If-None-Match: *` and **writes anyway**
— some do — makes every `claim` answer `true`. Every machine then takes every
trial, and nothing anywhere says so: no error, no warning, just a study whose
numbers are quietly four times the work and one machine's answer. That failure is
invisible by construction, so it is refused at the door instead of discovered
later. There is a test with a server that answers `200` to everything, checking
that the store is refused **and that the refusal names the reason**.

Measured against MinIO: four processes competing for forty numbers took each of
them exactly once, and a scan of the forty came back in 0.09 s.

### What CU13 predicted, and what it got wrong

> *"and **S3**, which arrives the day there is a MinIO to point at, through
> OpenDAL and as another configuration rather than another implementation."*

Both halves. It is **another implementation** — `soma-store/src/s3.rs`, brother of
`local.rs` — because a configuration would have meant one type with two
behaviours and a conditional write that is a link in one branch and a header in
the other. And it is not OpenDAL: what an object store abstraction abstracts over
is exactly the operation that had to be honest here.

The note in `local.rs` that said HTTP belonged in another crate is rewritten. What
it was protecting is given instead by a **feature `s3`, off by default** — the
same shape as the core's optional `serde`, and `cargo build -p soma-store`
still compiles with neither.

### The dependency, which was measured before it was chosen

`rusty-s3` signs and does not fetch — **sans-IO** — with `ureq` doing the
blocking request. `Cargo.lock` went from **44 packages to 133**, nearly all of it
TLS plus a date library and an XML parser.

An official SDK would have brought tokio, and **that would have made `Store`
async**: every caller of `resolve` in the engine, in `serve`, and every level-3
function in Python. That is the same objection that has kept a bus out of this
repo twice, and it is the reason a signing library beat the obvious choice.

### What writing the second one found

The contract could not be seen until something other than `Local` had to keep it.
`soma-store/tests/unit/contract.rs` is every assertion written against `&dyn Store`
and run against each — a directory always, a bucket when `SOMA_S3` says there is
one. It has no counterpart in `src/`, the same way `study`'s `invariants` has
none: what it covers is the trait.

It found the drift immediately: `record` and `read_record` existed **twice**, and
one used `to_vec_pretty` while the other used `to_vec`. Nothing would have failed
— the two records would simply have stopped being each other's. They live in
`store.rs` now, next to `Bound`, and that is what makes the sentence above true
rather than aspirational.

### Nothing above it learned a new word

`PyStore` holds a `Box<dyn Store>`. `take`, `report`, `finished`, `curves`,
`trials`, `in_flight` and `gather` did not change a line: they never asked what
kind of store they had. Level 3 uses it **by duck** — no annotation, no
`isinstance` — and `test_bucket.py` runs the same small study over a directory
and over a bucket and compares the two histories, which is the assertion that
this stays true.

### Questionnaire

**The contract, on every implementor there is** (`contract.rs`, `&dyn Store`)
- [x] the same bytes are the same digest however often they are written
- [x] what was never put is absent and not an error
- [x] a name points at bytes and carries what was said about it
- [x] binding the same name again replaces it
- [x] a claimed name cannot be claimed twice, and the first one keeps it
- [x] a claim does not overwrite what `bind` put there
- [x] many are answered in the order they were asked, with the gaps where asked
- [x] a scan finds what was bound, in an order two scans agree on
- [x] a record that is not a record is corrupt and not missing

**The bucket, and the one thing only it can get wrong**
- [x] an endpoint that ignores the condition is refused, naming the reason

**And from Python** (`test_bucket.py`, opt-in on `SOMA_S3`)
- [x] a store opened against nothing says so instead of being found out later
- [x] credentials come from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` when none
      are given, and the message names the one that was missing
- [x] bytes, names, meta and `claim` answer as a directory's do
- [x] a map of tensors is kept and recalled — a codec is a fact about the value
- [x] the same study over a directory and over a bucket gives the same history
- [x] a trial another machine is holding is visible before it finishes

## CU20 — The record of what happened

```python
g.forward(x, watching=print)                     # in a notebook
g.forward(x, watching=Recorder(store))           # kept
Trainer(g, objective=..., optimizer=..., watching=[Recorder(store), draw])
```

```text
{'fact': 'ran',      'node': 'tokenize', 'took_us': '4120', 'host': 'worker1'}
{'fact': 'recalled', 'node': 'embed',    'key': 'sha256:9c…'}
{'fact': 'left',     'host': 'worker1',  'took_us': '5330'}
{'fact': 'finished', 'took_us': '9210'}
{'fact': 'loss',     'value': '0.2517'}
```

The second of the three things CU19 split observability into. Before it the
engine measured nothing at all — not one `Instant`, not one `elapsed`, not one
line of `logging` in `core/`, `transport/`, `store/` or `study/`.

### The requirement that decided the design

Not "a log". **That a researcher training in a notebook, with half the graph on
other machines, keeps being told what is going on**: that the nodes reached the
workers, that the data got there, that nothing failed, that the profiling is
reasonable, that the training is progressing. Live, and not at the end.

That kills the cheap answer immediately: `run()` cannot return the facts,
because a curve drawn after the run is a report and not a view. So there is a
fifth hole, and it is injected exactly like the fourth.

```rust
pub trait Watcher: Send + Sync {
    fn saw(&self, fact: &Fact);
}
```

### Emitting is synchronous; delivering is not the core's problem

`saw` is called from the walk and returns. What the implementor does then —
write it, drop it, push it onto a channel another thread drains into a figure —
is where anything asynchronous belongs.

That is what lets *live* cost no runtime. An `async` here would be `async` in
every caller of the engine and would drag `Store` with it, which is the
objection that has twice kept a bus out of this repo. **The core still has no
dependencies and no executor.**

A wave calls it from several threads at once, hence `Send + Sync` — and hence
the order facts arrive in is not the order they happened in. Nothing here
pretends otherwise, and the engine will not serialize a run to make a log tidy.

### An enum of facts is not the original's mistake

The original's `enum Event` has **37 variants**, and the number is not what is
wrong with it. The project's own rule says an enum is right when the set is
closed and you know it, and that the compiler keeping count as variants are
added is the point of one. What is wrong is that those 37 are **three
vocabularies in one**: `NodeStarted` is a fact, `HealthFlag` is an opinion about
facts, and seven of them belonged to a layer this project has since removed.

So each level keeps its own, in its own language:

| level | vocabulary | where |
|---|---|---|
| the engine | `ran`, `failed`, `recalled`, `kept`, `items`, `left`, `finished`, `broke` | `soma-core/src/fact.rs` |
| a training run | `loss`, `updated` | `somatize.torch`, where the loss is |
| a study | a trial's record | on disk, since CU18 |

> **They do not meet in Rust. They meet in the record.**

The joint is `Fact::flattened`: a fact is **emitted as an enum** and **written as
a name and text-to-text pairs**, which is the shape `Meta` already had. What is
typed stays typed where the compiler helps; what crosses to another vocabulary
crosses as the flattest thing there is. Level 2 produces that shape directly
from Python and lands in the same record, and the core never learns what a loss
is.

The same shape is what reaches a notebook, which is worth more than it sounds:
**what you print is what you would find in the store.**

### What happens on another machine comes back down the connection that is open

The finding that made this cheap: `Worker::dispatch` sends `Work` and then
blocks in `recv` waiting for `Done`. That blocked read is exactly the moment the
worker has something to say and nobody is listening.

```text
→ Work { … }
← Saw(fact)                     any number, and not the end
  | Done { … } | Failed(why)
```

`Answer` gained one non-terminal variant; reading one answer became reading
until one is terminal; and `attend` hands its writer to the worker's own
`Executor`, whose watcher does nothing but put the fact back on the socket. **No
port, no second connection, no thread, no async, no bus.**

> **Where a connection is open, facts come back down it. Where there is none,
> they go to the store and whoever wants them scans.**

That is not a second design. It is the rule CU18 was already following: a study
handed out of a folder has no connection, so it scans. The transport already
tells the two cases apart.

A **relay attributes nothing**. The worker emits exactly what it would emit at
home, and the client wraps what arrives in `Fact::Elsewhere { host }` — because
the host's *name* is the graph's, and a worker does not know what it is called.
Flattening turns the nesting back into a `host` field, so a slice that crossed
two machines comes out with its route in order and the reader gets columns
rather than a tree.

### One record per `forward`

Five nodes trained ten thousand steps are fifty thousand node executions. A
record each is fifty thousand writes and a scan nobody can afford; one for the
whole run has no step 500 in it. The `forward` is the unit the engine actually
has, and `Fact::Finished` is emitted by exactly the walk that is one.

```text
run/<id>/<n>
```

`<study>/trial/<n>/<attempt>` with a different noun: the level above, and a
number. In the **record**, so a scan answers with no fetches: `run`, `forward`,
`took_us`, `state = ok|broke`, `nodes`. In the **blob**: every fact, flattened,
in the order it arrived.

Three things fall out of the split rather than being decided:

- **Durations, never instants.** A `took` measured on another machine means
  something; a wall clock from another machine is two clocks that disagree,
  which is the problem CU18 solved by comparing writers with writers. *When*
  something was written is the store's, and it stamps it.
- **How often it is written is a policy of the run**, not of the record — the
  same ruling that kept `.overwrite(times=1)` out of CU13. Ten thousand objects
  in a bucket are fixed by flushing in segments, and that is the writer's
  business.
- **The client writes it**, with the local and the remote in one record, because
  the remote already arrived. A worker that dies leaves behind what it did send,
  plus the failure.

### A loss arrives after the forward it belongs to

The one thing that needed a real decision rather than a derivation. A loss is
computed **after** the `forward` that produced it has ended, so a recorder that
only knew how to open records would file every loss one step late and every
curve would be off by one.

There is no guessing, because the two vocabularies come through different doors:
`saw` is the engine's and a terminal fact closes a record; `said` is everybody
else's and goes into the one that closed last, rewriting it. A store already
does that — a name is a question and its answer can be refreshed — and it is
what a trial's record has done since CU18.

### Reading it back, which is a price list

A record written and never read is not a record. `somatize.record` is the other
half, and — like `gather` and `take` — it is **functions over a `Store`**: what
is being read is a folder, and a class around one would be the store with a
longer name.

| call | what it answers | what it costs |
|---|---|---|
| `runs(store)` | what is in here at all | one scan |
| `forwards(store, run=…)` | step by step: state, time, nodes, loss | one scan |
| `curve(store, run=…)` | the series somebody plots | one scan\* |
| `facts(store, run=…, forward=n)` | everything one step did | one fetch |
| `nodes(store, run=…, last=N)` | who spent the time, added up | a fetch per `forward` |

Everything a progress view asks for is on the free side, and the per-node
breakdown — the expensive one — is asked once rather than once a step.

\* **only for what the recorder was told to summarise.** That is the one thing
this made the writer learn:

```python
Recorder(store, run="tuesday", summarising=["loss"])
```

Named kinds go into the record itself as `<kind>.<field>` and not only into the
blob. It is the lesson CU18 already paid for — a trial keeps its score beside
its configuration so a sampler rebuilds a history with one scan and no fetches —
and the same question is asked of every training curve ever drawn: ten thousand
losses read one blob at a time is ten thousand round trips, and the number
wanted from each is one. Which kinds those are is the **caller's**, so the store
still does not learn what a loss is.

Without it `curve` still answers, and `curve_costs` says which of the two it
did. A reader that is quietly a thousand times slower is worse than one that
says so.

### Live and read back are two paths, and that is not a duplication

While a run is going, what you want arrives at `watching=` and costs nothing to
get. When it is over — or when it is **another machine's**, where there is no
connection at all — a scan is the only thing there is. Both answer in the same
shape, because a fact read back is the very dict a watcher was handed:
`Fact::flattened`, once, for everybody.

### What is not in it

**`ctx.saw(...)`** — a node speaking for itself. The engine cannot see a
gradient norm and the node can, and `Ctx` is already *"where whoever executes
hands a node what it knows"*, so the mirror of it changes no node's signature.
It is deferred because today it has neither a tenant nor a vocabulary, which is
`Driver`'s mistake in miniature — and because deferring it is **cheap**: it is a
field in one file, where the name in the store was the part that could not be
refactored later. It is what CU21 opens with, and what a **remote** trainer needs
before it can say anything about its own loss.

Also out: **the overlay**, which CU19 predicted would arrive here — it does not.
What arrives is the thing an overlay is made of, and turning a record into one is
a reader over these facts rather than a change to either end. Also out: a bus,
still deferred and still not refused · and any judgement whatsoever about what
any of these numbers mean, which is the whole of CU21.

### Questionnaire

**The vocabulary** (`soma-core/tests/unit/fact.rs`)
- [x] a node that ran says which one, how long, and where — and nothing about a
      device nobody declared
- [x] a duration is whole microseconds and not a float
- [x] what happened elsewhere comes out as a `host` field and not as a tree
- [x] a fact that crossed two machines keeps its route in order
- [x] the two facts that end a run say so, and a node failing is not one of them

**What the engine says** (`soma-core/tests/unit/watcher.rs`)
- [x] a run nobody watches behaves exactly as it did
- [x] every node that ran is said so, in the order it ran
- [x] a run ends with exactly one fact that says it is over — an empty plan too
- [x] a node that failed says which one **before** the run stops
- [x] a run that could not finish is still closed
- [x] a hit and a miss are two different facts, and a hit does not say a node ran
- [x] a mapped node says how many items it did not have to compute
- [x] what ran over there arrives here saying where it ran
- [x] the round trip is its own fact
- [x] a slice that went away says nothing about finishing

**Over a real process** (`soma-fabric/wire/tests/unit/worker.rs`)
- [x] what a real worker saw comes back saying it was that worker
- [x] a fact arrives **while the work is still going** and not with the answer —
      checked against a node that takes 300 ms, because batched or live the
      facts are the same facts and only *when* differs

**Written down** (`soma-store/tests/unit/recorder.rs`)
- [x] nothing is written until the `forward` is over
- [x] a scan says how it went without reading a single blob
- [x] the detail is in the blob, in the order it arrived
- [x] each `forward` is its own record, numbered from zero
- [x] one that broke says so where a scan can see it
- [x] a run can be given the name it already has, and gets one if not
- [x] what level 2 says lands in the `forward` it belongs to, rewriting it
- [x] and the next `forward` still starts a new record
- [x] rewriting says the same thing about the same facts

**Read back** (`soma-python/tests/test_record.py`)
- [x] a store says which runs it holds, and what is not a run is not read as one
- [x] a store nobody recorded into says so rather than failing
- [x] every `forward` comes back in order, with its numbers as numbers
- [x] one that broke is visible without reading its blob
- [x] a summarised loss is read with one scan, and without summarising it is
      still read and says it cost more
- [x] anything a fact carries can be a curve, `took_us` included
- [x] the facts of one `forward` are exactly what was seen live
- [x] a `forward` that is not there is nothing and not a failure
- [x] who spent the time is added up across `forward`s, slowest first
- [x] only the last N can be asked for, because each costs a fetch
- [x] a node that was read back is not averaged as a fast one

**From Python** (`soma-python/tests/test_watching.py`)
- [x] a fact is a `dict` of text, and what is printed is what is written
- [x] a list of watchers is told, and something that is not callable is refused
- [x] a recorder nobody named is still findable
- [x] what ran on a real worker comes back saying which host
- [x] a training step says the loss and when it moved
- [x] a loss lands in the `forward` it belongs to
- [x] a group of steps moves once and says so once

## After CU20 — A run, drawn

```python
live = Live()                                   # in a cell, on its own
t = Trainer(g, objective=..., optimizer=...,
            watching=[Recorder(store, summarising=["loss"]), live])

progress(store, run="tuesday")                  # afterwards, or another machine's
spent(store, run="tuesday", last=200)           # where the time went
```

The presentation half, asked for in the same breath as the reader: *"un módulo
encargado de leer y presentar al usuario toda la información, de forma
independiente y agregada"*, and plotly for it, because plotly can redraw in
place.

### The same figure from two sources, which is the point

`progress` reads a store and `Live` is handed facts as they happen, and they
fill **one drawing function**. They can, because a fact read back is the very
dict a watcher was given — `Fact::flattened`, once, for everybody. What that
buys is not tidiness: a live view and a report that are written twice are two
things that slowly stop agreeing, and the one you are watching at three in the
morning is the one that is wrong.

`Live` holds one row per `forward` and not one per fact, so watching a run for an
afternoon costs what the run is long and not what it is wide.

### Dark, and one table for both figures

CU19 wrote the rule about the graph — one table, looked up with `[]` and never
with `.get(…, default)`, because the original kept the same strings in four
tables and a typo came out as the alarm colour. The moment there was a second
figure the same rule applied one level up, so the table moved to
`somatize._theme` and **the graph moved with it**: a library whose graph is
light and whose curves are dark is two libraries.

The discipline survives the move intact: **one fact per channel**. Hue says
where a node ran or which series it is, never good-or-bad. The only red on any
of these figures marks a `forward` that broke, which is a fact in the record and
not an opinion about one. Whatever CU21 decides is *unhealthy* will need a
channel of its own, and it does not get to recolour these.

### The smooth line is a mean, and that is not a detail

A spline drawn through measured values invents the values between them, and an
overshoot on a loss curve dips below a minimum that never happened. The rule
about figures here has been the same since CU19 — they may simplify and may not
lie — so the bold line is a **rolling mean**, which is a stated transformation,
and the raw series stays underneath it thin and faint. Nothing is hidden by the
smoothing: what was measured is on the figure, and what is easy to read admits
to being an average.

**Centred and not trailing.** A trailing mean is the same curve shifted right,
and drawn on top of the raw series that shift reads as the smoothing disagreeing
with the measurement. Nothing is being predicted — every point of the run is
already in hand. It is computed off prefix sums, because a live view redraws it
on every step and a window of five hundred over ten thousand points done the
obvious way is five million additions a frame.

A `forward` with no loss said about it is a **gap** and not a zero: zero on a
loss curve reads as the best result of the run.

### An edge drawn over a node says something that is not true

CU19 left this: *"the boxes say **when**, the arrows say **what feeds what**"* —
and where the nesting stops saying who feeds whom, the arrows are all there is.
They were drawn straight, so in exactly that case — the `N`, a flat `Sequence` —
`a→c` passed **through** `b`, which reads as an edge into `b`.

Now an edge that would cross a box it does not belong to goes **around**:
outside everything, down, and in through the side of what reads it. Outside
everything rather than outside the boxes in the way, because a lane threaded
between two of them is a lane that will cross a third the next time the layout
changes. Whether it would cross is asked exactly, with a slab test — sampling
the segment would miss a thin box, and a figure that is *usually* honest is the
kind of thing nobody ever finds.

And **one lane per edge**: three edges that all have to go around shared one line
in the first version, so the figure stopped saying there were three.

### The study, drawn — and the figures belong in the library

`table`, `influence` and `coordinates` in `somatize.study`, with `importance`
beside the other readers. They were written in a notebook first and that was
wrong: a figure hand-rolled in an example is a figure with no tests, no shared
palette, and a second copy the day somebody wants it elsewhere.

`importance` is **Spearman's ρ** — a rank correlation, so it says *this knob
orders the results* and not *this knob is worth these many points*. It is what
the original actually has: its documentation names fANOVA and says it was
deferred, and it never arrived. Thirty lines of plain Python, so it is not a
dependency. Ranks and not values, so a knob searched in log needs no special
case, and `0.0` where a knob never varied — no evidence, which is not the same
as no effect.

`coordinates` draws every finished trial as a **curve** and not a polyline,
which meant not using plotly's `Parcoords` at all: it only draws straight
segments. What that costs is its brushing; what it buys is that a trial reads as
one continuous thing, which is what makes a bundle visible as a bundle. A curve
claims nothing — a point exists only where it crosses an axis, and it crosses at
the value it has — but it is still drawn gently, because a spline bulging past
the top of an axis reads as a value beyond its range even when it means nothing.

`goal` decides which end of the colour scale is good and it is **read from the
study, never guessed**: getting it backwards is the quietest lie a figure can
tell — everything is drawn, nothing raises, and the region you read as promising
is the one to stay away from. `table` sorting the wrong way round is the same
lie with a different label, since it says *best first* either way.

It was a parameter with a default of `min`, which is the guess in another
place. So `report` writes the direction into the record beside the score — the
number does not say which way is better and neither did anything else a reader
could reach — and both figures read it, with `goal=` left as an override for a
study run before it was written down. When nobody says at all, they part
company: `table` gives up the claim and falls back to the order the trials ran
in, saying so in its title, and `coordinates` raises, because a colour scale has
two ends and drawing one is saying which of them is good.

The cost is one word per record and it is **denormalised on purpose** — the
direction belongs to the study, not the trial. A name of its own would cost a
fetch and a missing case, against a normalisation nobody was going to query;
and writing it per trial records what was meant **at the time**, so a study
whose direction changed halfway does not retell the old trials.

And in all three, pruned and finished are not ranked together. `table` shows
both with their state; the other two use only what ran to the end, for the same
reason `finished` leaves pruned trials out.

### Questionnaire

**Edges that would cross a node** (`soma-python/tests/test_figure.py`)
- [x] an edge with something in the way is routed around it
- [x] one with nothing in the way is still a straight arrow
- [x] three routed edges do not share one lane
- [x] a routed edge runs outside every box
- [x] a segment is tested against a box exactly, not by sampling

**Which knob mattered** (`soma-python/tests/test_study_figure.py`)
- [x] a knob that decides the score comes out near one
- [x] one that never varied is zero, because that is no evidence
- [x] a study with nothing to compare says nothing rather than guessing
- [x] a pruned trial does not vote
- [x] the biggest comes first

**The study, drawn**
- [x] the table shows the pruned ones too, and says which
- [x] it has a column per knob in the space
- [x] the influence bars are the numbers `importance` gives
- [x] every finished trial is a curve, and a pruned one is not
- [x] a knob searched over orders of magnitude gets a log axis
- [x] the goal decides which end of the scale is good
- [x] a study nobody has finished is a statement and not an exception
- [x] the table reads the direction the study recorded, both ways round
- [x] a caller overrides the record for a study that predates it
- [x] a table that does not know gives up the claim instead of guessing
- [x] a study that never said raises rather than drawing it backwards
- [x] a direction nobody recognises is refused by the figure as well

**One figure, two sources** (`soma-python/tests/test_record_figure.py`)
- [x] live and read back draw the same series, point for point
- [x] a live view keeps one row per `forward` and not one per fact

**The smoothing, which is where a figure could start lying**
- [x] the smoothed line stays inside what was measured
- [x] the mean is centred and not trailing
- [x] asking for no smoothing gives back what was measured
- [x] a `forward` with no loss is a gap and not a zero

**What the figure says happened**
- [x] a `forward` that broke is marked, and only then is it in the legend
- [x] the title says what the figure is showing
- [x] a node is coloured by where it ran and by nothing else

**One product**
- [x] the graph and the run are drawn from the same table
- [x] without plotly, drawing says how to get it
- [x] a live view outside a notebook reads as nothing, like the graph's figure

## CU21 — The diagnosis, which says it is an opinion

```python
from somatize.health import diagnose, overlaid, alerts
from somatize.torch import Trainer, Audit

t = Trainer(g, objective=..., optimizer=..., auditing=Audit(every=10, inside=True),
            watching=[Recorder(store, summarising=["loss"])])
t.fit(data, epochs=5)

found = diagnose(store, run=t.run)            # read back, nothing runs again
alerts(found)                                 # the loud one, cards in a cell
overlaid(g, store, run=t.run, inside=True)    # and where, on the graph
```

The third of the three things CU19 split observability into. The first was the
declaration drawn, the second the record of what happened, and this one is
**neither**: it is an opinion about the record.

### A crate with no dependencies at all, not even the core's

`health/` takes numbers and gives back flags. It does not measure, it has no
clock, it never touches a store. That is what turns CU19's invariant from an
aspiration into a test:

> a diagnosis has to be reproducible from the stored record, without training
> again.

Change a bound and ask again. The record has not moved, so an argument about a
threshold costs a scan instead of an afternoon of GPU. It is the shape `study/`
already had: pure, deterministic, hashable, and the loop that owns a tensor
stays in Python.

### The taxonomy is inherited, and how it reads is the knowledge

`DEAD` and `SATURATED` read the **maximum** over a window and never the mean. A
layer that dies one step in four is dead, and an average hides it. **Dormant is
not dead** — two findings, not one bound with two names.

Three come from the literature. `STALLED` and `OVERSTEPPING` from the
update-to-weight ratio, which the original measured and never said anything
about, and which lands a healthy layer at about `1e-3`. `LOSING_PLASTICITY` is a
**conjunction** on purpose: weights growing, or units going quiet, one at a time,
is a network that is training.

### A threshold that was measured and did not survive it

`NARROWING` is in the vocabulary and **off by default**. The published monitor's
certificate is the deviation from a healthy baseline, and one run has none.
Measured: healthy runs sit at 0.69–0.71 and a destabilised one at 0.43–0.86,
which overlap. The measurement is in `soma-health/tests/narrowing.py`, the metric is
recorded and drawn, and the alarm was not invented. `Thresholds` is data, so
whoever does have a baseline sets the bound and gets the finding.

### Measuring is level 2's, and thresholds never go near it

`Trainer(..., auditing=True)` hooks the nodes and emits `health` facts through
the same `watching=` CU20 built. A threshold baked into the measurement would
make disagreeing with it cost another training run.

`Audit(inside=True)` looks **inside** a node, because a node is often a whole
architecture and *this node is unhealthy* is not an answer when it is twenty
layers. Findings are keyed `node.path.to.submodule`, and the audit's scope is
**the same scope the drawing uses** — which is what makes *what is measured has a
box* true rather than hopeful.

### A node is opened up, and an architecture is a graph

`architecture(g, x)` traces what a node is made of: `fx` where it can, because it
sees the operations that are **not** modules and a residual connection is exactly
one; a real forward where it cannot, **saying so**, because a residual that is
missing looks like a residual that is not there.

`g.figure(inside=...)` draws the node's box as a **frame** — the shape a `Wave`
and a `Remote` already are — and lays the inside out by what feeds what, so a
skip runs down a gutter and enters from the side. The rules that make it
readable: a **kind** decides the silhouette, not a class name; a composite
everybody recognises is one box and `depth=` opens it; blocks that are the same
block collapse to `×N`; the **shape is written on the layer**, because that is
the only thing that makes a bottleneck a picture; and every number **says what
it is** — `4 batch · 16 steps · 24 dim`, never `4×16×24`.

Two of those grew after CU22 — see *After CU22 — a block is a box*.

Findings are coloured by **family** — numeric, signal, activation, step,
capacity, data — with a legend of the ones on the figure. Six alarms that all
look the same are one alarm.

### Where is a question the graph answers

Health gets a **channel of its own**: the fill goes on saying where a node runs
and the outline turns red. On a graph spread over three machines, *where does
this run* is the answer somebody came for, and taking the fill for a second fact
would have cost it.

`gantt` is the timeline. Every fact carries how far into the `forward` it began,
so a `Wave` draws as overlapping bars and a remote slice sits inside the round
trip it arrived under. An offset into a slice is a fact **about the slice**; two
wall clocks would not have composed.

### And a third question, which is not about the network at all

`somatize.data.contribution` shuffles one input and scores again; the drop is
what that input was worth. `health` asks whether a network is **learning**; this
asks whether it is learning **what you meant**, which no amount of looking at a
gradient will ever say.

It exists because of a real project: symptom channels for a mental-health
condition, months spent on the architecture, and the predictive signal was in the
self-disclosure and not in the presence of symptoms. `IGNORED_INPUT` is the
finding that would have said so in an afternoon; `SOLE_RELIANCE` is the other end
of the same worry.

**Shuffled and not zeroed**: a zero is a value, and what is being asked about is
the correspondence with the answer.

### What is not in it

The **static** half, before a GPU is spent: signal propagation and dynamical
isometry at initialisation, where a normalisation layer is missing, and the
zero-cost proxies. Deferred rather than refused, and with a caveat already
written down — synflow correlates 0.76 with parameter count, which is close to
saying it measures size. If it does not separate when measured, it ships off with
its measurement beside it, the way `NARROWING` did. *(CU22, and that is exactly
what happened to it.)*

### Questionnaire

**The verdict** (`soma-health/tests/unit/verdict.rs`)
- [x] a gradient too small to train on says so, and one too big to step on, and
      it cannot be both
- [x] a layer that dies one step in four is dead, and one that is merely sparse
      is not
- [x] a layer pinned where the derivative is nothing says so
- [x] a node moving too little next to its own weights says so, and one moving
      so much it forgets where it was
- [x] a healthy ratio is near a thousandth and says nothing
- [x] dead channels are counted and are not the same as a dead layer
- [x] a channel alive and never asked for is its own finding
- [x] two groups carrying the same information leak
- [x] a collapsing update says nothing at the default bound, **because it is
      off** — and says so for whoever has a baseline to set it against
- [x] an update that is merely low rank all along is not narrowing
- [x] losing plasticity needs all three signs at once, and any one alone is
      ordinary
- [x] what stops a run is read first
- [x] the same numbers answer differently under other thresholds
- [x] a node nobody measured is not called healthy and is not flagged

**The vocabulary** (`soma-health/tests/unit/flag.rs`)
- [x] a flag that counts something says how many, and its name is stable
      whatever it counts
- [x] every flag says what to do about it

**The data** (`soma-health/tests/unit/leaning.rs`)
- [x] shares add up to one, so they read as how much of what matters
- [x] an input the model is not using says so
- [x] two inputs that share the work say nothing
- [x] one input carrying everything is worth knowing before it goes missing
- [x] a model that loses nothing whatever you take away is using none of it
- [x] an input the model does better without keeps its negative
- [x] one input alone says nothing, because there is nothing to compare
- [x] the bounds are data here too

**Measured and read back** (`soma-python/tests/test_health.py`)
- [x] **a diagnosis is taken from the record and not from the run**
- [x] the same record answers differently under other thresholds
- [x] a threshold nobody has is refused by name
- [x] a deep sigmoid stack starves its early layers
- [x] the update ratio lands a healthy layer near a thousandth
- [x] a block whose relu cuts everything off is dead, and one pinned at the far
      end of its range is saturated
- [x] a gradient too big to step on explodes
- [x] a healthy shallow stack raises nothing
- [x] a node with no weights is not diagnosed at all
- [x] a run that is not audited says nothing about health
- [x] a cadence measures fewer steps and says the same kind of thing
- [x] **auditing does not change what the network computes**

**What a node is made of** (`soma-python/tests/test_health.py`)
- [x] a node says what it is made of
- [x] a skip connection is an edge and not an order
- [x] a bottleneck is visible in the shapes
- [x] a module `fx` cannot trace is still drawn, and says how
- [x] what is drawn is a superset of what is measured
- [x] a composite everybody recognises is one box, and `depth` counts composites
      opened and not names
- [x] blocks that are the same block collapse to one and a count
- [x] what comes after a stack is not adopted by its last block
- [x] a tensor nobody holds cannot invent an edge
- [x] a shape says what each of its numbers is, and something that did not change
      the shape keeps the names
- [x] a recurrent cell says its output and not its hidden state

**Drawn** (`soma-python/tests/test_figure.py`)
- [x] a node with an inside becomes a frame around it, and one without is drawn
      exactly as before
- [x] a layer is drawn by what it is and not by its name
- [x] a layer that narrows is drawn **wide at the top**
- [x] what feeds what decides the rows, and a skip jumps one
- [x] every layer sits inside the box it belongs to, named after the node it is in
- [x] the shape is written on the layer, because a bottleneck is shapes
- [x] a flag on a layer marks that layer and not the others
- [x] the overlay marks the node without taking the fill
- [x] a branch of a wave can be opened too

**The data layer** (`soma-python/tests/test_data.py`)
- [x] an input the model is not using is found in one afternoon
- [x] and the shares say how lopsided it is
- [x] two channels that both carry it say nothing
- [x] **nothing is trained and nothing is changed**
- [x] shuffling keeps the channel and breaks only what it lines up with
- [x] an opaque is unwrapped and wrapped again
- [x] something that is not a batch is left alone
- [x] only the inputs asked for are tried
- [x] no data is nothing and not a failure

## CU22 — What can be said before a step is taken

```python
from somatize.torch import probe, proxies
from somatize.health import diagnose, overlaid, profile

probe(g, x, watching=Recorder(store, run="before"))   # one forward, nothing trained
diagnose(store, run="before")                         # {"trunk.4": ["MISSING_NORMALISATION"]}
overlaid(g, store, run="before", inside=...)          # and where, on the graph
profile(store, run="before", of="jacobian_gain")      # the vanishing picture, with no optimizer

proxies(candidate, x)                                 # and scoring one without training it
```

The static half of CU21, and the half that costs seconds rather than an
afternoon. CU21 asked whether a network **is** learning, which needs it to have
been learning. This asks whether it **can**.

### A probe is one `forward` that was recorded and never trained

Not a turn of phrase for the record's benefit. It is literally `run/<id>/0`,
written through the same `Watcher` a `Trainer` writes through, holding `health`
facts under the same `node.path.to.submodule` keys. Which is why the whole third
row of observability reads a probe with **no new code at all**: `diagnose`,
`seen`, `history`, `profile`, `flags`, `where`, `overlaid` and `alerts` never
asked what made a record, and now they do not have to learn.

That is CU20's decision to write a fact as `(kind, pairs)` — rather than as a
type each level shares — being paid back. A probe is a new *producer*, not a new
vocabulary, and the invariant comes free:

> a diagnosis has to be reproducible from the stored record, without training
> again — and here, without ever having trained.

### The three numbers, and why none of them is a gradient norm

| what | how | why not something else |
|---|---|---|
| `signal_gain` | the scale here against the last normalisation upstream | the drift is geometric, so what matters is the ratio and never the size |
| `jacobian_gain` | `sqrt(E‖Jᵀv‖²)` from here to the output | the backward signal, scale-free, so it means the same thing at every depth |
| `jacobian_spread` | `s_max / s_rms` of the sketch `JᵀV` | isometry is a claim about the spectrum's *shape*, and a mean cannot see it |

There is deliberately **no `grad_norm`**. At initialisation there is no loss, so
a parameter gradient would have to be taken against a target somebody made up,
and the number would land in the very field the audit fills from a real loss —
at a different scale, to be judged by the same bound. Two things under one name
is how a threshold quietly stops meaning anything. The backward direction is
`jacobian_gain`, which needs no target because it is a ratio.

Two forwards and `k` backwards, and the `k` are **over the whole network**
rather than `k` per layer: every layer reads its own `Jᵀv` off the same pass.
That is what puts this in front of a training run rather than instead of one.

The second forward is `architecture`'s, and it is what decides *which* layers
are measured. Walking the modules here instead would be a forward cheaper and
would break the invariant the whole row rests on — **every layer that can carry
a flag has a box** — because a module walk and what the figure draws stop being
the same set once `fx` has had its say. It was written the cheap way first, and
a `TransformerEncoderLayer` inside a `Sequential` is what said so.

### `MISSING_NORMALISATION`, and both halves of it are load-bearing

The structural half lives in the **measurement** and not beside the bound. Where
the last normalisation is is structure; resetting the reference there is not a
threshold, and a normalisation reports no gain of its own because changing the
scale is its job. So the conjunction is baked in, the shape `LOSING_PLASTICITY`
already has: drifting alone is a network that is fine, and having no
normalisation alone is a network that is fine.

Measured, and the numbers are in `soma-health/tests/normalisation.py`. Everything
that trained sat at **2.81** or below and everything that did not was at **100**
or above, so a decade sits between them with 3.6x of margin below and 10x above
— a decade because the drift is geometric and the useful signal is an order of
magnitude, never a percentage. The row that makes it a conjunction rather than a
lint is the badly-initialised stack **with** normalisation: it drifts 2.81x and
trains, and structure alone would have flagged it along with every plain stack
that trained best.

**It fires only upwards, and that is the measurement's doing rather than the
design's.** A plain stack whose signal arrives five ten-thousandths of the size
it went in trained as well as the healthy one, the two ranges overlapping. Adam
is scale-invariant per parameter, so a signal that shrank does not stop a step
being taken. There is no lower bound and that is a finding, not an omission.

The false positive it had to stay quiet on was already written down:
`examples/07-a-real-architecture.ipynb` dropped the normalisation from a
three-block residual trunk and the un-normalised version scored *better*. An
unnormalised residual trunk grows like the square root of its depth — eighty
blocks reach 4x, and it would take some five hundred to trip a decade.

### The isometry half raises nothing, and the reason generalises

Both Jacobian numbers **rank** and neither **separates**, which is not the same
thing and only one of them is a flag. At criticality the nine rows come out
almost in order — orthogonal-`tanh` at a spread of 1.10 with the best loss,
`he-relu` at 1.88 with the worst. Walking the gain off criticality is what
breaks it: the worst network that still trains reads a first-layer gain of
**1.41** and the best that does not reads **1.95**, and a factor of 1.4 is where
the sampling landed rather than a bound. The spread inverts outright — 1.87
trains and 1.76 does not, so the failing network has the *tighter* spectrum.

So they are recorded and drawn and neither raises anything, which is what
`NARROWING` established as the thing to do. See `soma-health/tests/isometry.py`.

And there is a rule under all three measurements, worth more than any of them:

> **What separates is a runaway. What ranks is a proxy.**

The forward scale separates because it is geometric — it either stays put or
leaves by decades, and there is nothing in between to be wrong about. The
backward numbers vary continuously with how well a network turns out, and
something continuous is a ranking. A ranking belongs at level 3, where a number
only ever means something next to another candidate's.

### A proxy is not a `Flag`, and it never was

`synflow` of one network is a number with no meaning. It only means something
next to another network's, which is level 3 — where a study is a `for` loop and
there is no type at all — and not the vocabulary of a diagnosis, which is about
*this* network. So `somatize.torch.proxies` is a **cheap objective** the loop
scores with instead of training, it takes a `Graph` the way `probe` and
`architecture` do, and no `Flag` ever comes out of it.

Which leaves one question, and it is not *does it correlate with the score*:

> **Does it beat counting parameters?**

Size is free. `soma-health/tests/proxies.py` asks it of all five over twenty-four
candidates, and the answer is not the one the scores alone would give:

| | ρ vs score | ρ vs parameters | beats counting by |
|---|---|---|---|
| **parameters** | **0.59** | — | the baseline, and it costs nothing |
| `snip` | 0.61 | **-0.02** | **+0.02** |
| `naswot` | **0.69** | **0.97** | +0.10 |
| `zen` | 0.45 | -0.39 | -0.14 |
| `grasp` | -0.08 | 0.57 | -0.67 |
| `synflow` | **-0.16** | 0.42 | -0.75 |

`naswot` scores highest and is the least interesting: at 0.97 with parameter
count it **is** size, with noise on top. `snip` is the only one that beats
counting *and* is uncorrelated with it — two hundredths, but two hundredths of
something orthogonal to what size already says, which is the only kind of gain a
proxy can honestly claim. And `synflow` comes out **worse than nothing**, because
on this family it reads *depth*: a `relu` stack at depth eight scores 12 to 21
where the same widths at depth two score 7 to 9, and depth is what hurts here.
The published 0.76 is on NAS-Bench-201, where depth and size move together.

So the library ships **all five and picks none**. Which proxy is worth anything
depends on the family being searched, and that is a question with a cheap answer
rather than a default somebody has to discover is wrong.

### And the notebook

`examples/08-before-a-step-is-taken.ipynb`: a candidate drawn, probed in a third
of a second, the flag on the figure, and the fix that makes it quiet — followed
by training all three for real, because **an opinion that is never checked is a
habit**. The one the probe flagged lands on the floor; the one it stayed quiet
about trains.

It also does something the measurements could not do on their own: it runs the
proxies over six candidates and gets `synflow` at **+0.11 against the baseline**,
where the twenty-four-candidate measurement has it at **-0.75**. Same proxy,
same code, different family — here depth helps and `synflow` reads depth. That
is the argument for shipping all five and picking none, and it is much better
made by two tables that disagree than by a paragraph.

### What is not in it

A probe of a graph whose slices run **elsewhere**. The hooks are registered
here and a remote slice runs its own forward, so it contributes nothing — said
out loud, because a node quietly absent from a diagnosis reads exactly like a
healthy one. Doing better means a probe that travels the way a trainer does,
which is CU14's shape and a slice of its own.

And two debts older than this one, both still open: CU20's dump policy by
segments, which the docs assert and the code does not do, and a worker that does
not write its own record.

### Questionnaire

**The verdict** (`soma-health/tests/unit/verdict.rs`)
- [x] a signal growing where nothing normalises it says so
- [x] a signal that shrank says nothing at all, **because that is what was
      measured** and not because nobody looked
- [x] the drift a residual trunk has anyway is not a finding
- [x] and whoever normalises differently moves the bound
- [x] a probe that measured no signal is not called healthy

**Before a step is taken** (`soma-python/tests/test_health.py`)
- [x] **a probe is one `forward` that was recorded and never trained**
- [x] nothing is trained and no weight moves
- [x] a signal growing where nothing normalises it is found before a step
- [x] and the same stack normalised says nothing about it
- [x] a signal that shrank says nothing, because that is what was measured
- [x] a normalisation resets what the gain is measured from
- [x] **everything a probe measures has a box**, at every depth
- [x] the backward signal falls away with depth before an optimizer exists
- [x] every flag a probe raises says what to do about it
- [x] a node holding no modules is not probed
- [x] and a node whose modules never ran is said out loud

**Scoring a candidate** (`soma-python/tests/test_proxies.py`)
- [x] three of them never see a label
- [x] and with a loss every one of them answers
- [x] one that reads a loss and was given none says which
- [x] something that is not a proxy is refused by name
- [x] nothing is trained and no weight moves
- [x] `synflow` puts the signs back
- [x] no gradient is left hanging off the candidate
- [x] **a proxy is a ranking and says nothing about one network**
- [x] the bigger network scores higher on `synflow`, which is the caveat written
      as a test rather than as a footnote
- [x] a batch of one has nothing to tell apart

## After CU22 — A block is a box, and a lane is inside the picture

Three things about the figure, found by looking at one: `examples/07`'s `In[6]`,
which is the cell that opens a composite.

### An arrow that left the drawing

A routed edge takes a lane **outside every box** — a lane threaded between two
of them is a lane that will cross a third the next time the layout moves — and
the canvas was measured from the boxes. So the lane sat two pixels past the
axis, and the arrows from the outer branches left the picture on one side and
came back on the other.

The reason it survived a suite with a figure test in it is worth more than the
fix: a routed edge is a `path` shape, and a `path` has no `x0`/`y0`. **Every
range check reads `x0`, so every range check skipped the only shapes that can
leave the canvas.** The test now parses the path.

### A repeated block is a frame, not a word on every layer

Four encoder layers opened up were eight boxes each saying `×4`: the count said
eight times, and the block itself said none. Now a block of **two or more**
layers is a frame around them with `TransformerEncoderLayer ×4` on it, and a
block that is a single layer keeps its `×N` inline — a frame around one box says
nothing a word could not.

An edge that comes **down** into a block ends on the block rather than on the
layer inside it, because the frame's header is where the count is written and an
arrow through a label reads as neither. A skip comes in through the **side**,
never touches the header, and goes on to the layer it really feeds: saying *into
the block* there would lose the one thing a skip is about.

### What runs several of itself at once

`4 heads` on a `MultiheadAttention`, drawn with plates behind the box. Read off
`num_heads` and **never inferred**, and never drawn as separate boxes: torch
packs the heads into one `in_proj_weight` and a reshape, so `fx` sees one
operation and a hook sees one module. Four boxes wired together would be a graph
nobody built — the same rule that makes a traced residual say *how* it was found.

The plates are capped at two however many lanes there are, because eight plates
are a smudge and what says *eight* is the word. They go **downwards**, since the
first layer of a block has its `×N` immediately above it.

### Why this was copied and not adopted

`torchview` does both of these already: `expand_nested=True` puts an expanded
submodule in a dashed graphviz cluster, and `roll=True` collapses recursively
used modules. Which is the confirmation that the shape is right, and the reason
to take the idea rather than the tool.

What none of them has is the half this figure exists for. Netron, torchview and
`visualtorch` draw a module tree; not one of them draws **placement** — no
device, no host, no wave, no remote — and none has somewhere to put a health
overlay. The original soma did not either, which is why CU19 had to invent it.
Adopting graphviz would cost a system binary, lose plotly's hover and the live
view, and still leave the wave frames, the remote frames and the health channel
to be written.

### Questionnaire

**Drawn** (`soma-python/tests/test_figure.py`)
- [x] **an edge routed around the drawing is still in the drawing**, and the
      arrowheads too
- [x] a repeated block is a frame around its layers with the count on it, and
      not the count on each of them
- [x] and the frame holds every layer of the block and nothing else
- [x] a layer that runs identical lanes is drawn with them behind it
- [x] one lane is not several and draws nothing extra

**What a node is made of** (`soma-python/tests/test_health.py`)
- [x] a repeated block of several layers puts its count on the block
- [x] and a block that is one layer keeps its count inline
- [x] **how many lanes a layer runs is read and never inferred**
- [x] something that runs one lane says nothing about lanes

## CU23 — Workers and jobs, live

```python
from somatize.record import fleet, machines

fleet(store, run="tuesday")        # what each machine did, and what it says it is
machines(store, run="tuesday")     # drawn: working against waited on
```

The last row of the observability plan CU19 opened, and the first thing to
decide about it was what **not** to build.

### There is no registry, and there was never a place for one

A machine does work here in exactly two ways, and each already answers *is it
alive* without anybody keeping a list:

| | who knows | already there |
|---|---|---|
| a worker serving slices (`.at("worker1")`) | the client, which is talking to it now | `left` with the whole round trip, and a `host` on every fact from over there |
| a machine claiming trials from a folder | the store | `in_flight`, `abandoned`, and *is it still writing* measured writer against writer |

There is no third case. The original keeps a coordinator with a `WorkerStatus`,
a `last_heartbeat` and a thirty-second timeout — three missed beats at ten
seconds — and it needs one **because it has a coordinator**. CU15 removed that
on purpose, and CU18 answered liveness the other way: not *does it answer* but
*is it still writing*, against the newest write and never against a local clock.

### The record, turned the other way up

What was missing is not state, it is a **view**. The record is written run →
`forward` → node and *where* is an attribute, so nobody could ask what a machine
is doing. `fleet` inverts it at the price `nodes` already costs.

The column that earns it is `waiting_us`: the round trip **minus** what actually
ran over there — the wire, the queue and the codec. Neither half of that
subtraction belongs to a node, so no per-node view can produce it, and it is the
answer to *was sending it worth it*. Against a real pair of workers it says the
thing immediately: sixty microseconds of work behind 1.2 seconds of round trip
is a slice that should have stayed here.

### And the half no record can derive

How loaded a machine is, how much memory is left, how long it has been up.
Nobody on this end can work that out. So the worker says it, and three decisions
fall out of rules that were already written:

**It is a level of its own.** A load average is not a fact about a graph, and a
variant for it in the core's `Fact` would be the engine learning what a machine
is — the mistake that keeps `loss` out of the core. So the vocabulary lives in
`transport/`, where a host is already a thing.

**It crosses flat.** `Fact::Said { kind, pairs }` is a **carrier and not a
vocabulary**: what the core learns is that other levels exist and one of them
may be speaking from another machine. `(kind, pairs)` is the shape CU20 named as
where the levels meet, and it is what `flattened` already produces.

Which turned out to cost **nothing on the wire**. `Answer::Saw` already carries
a `Fact`, the client already relays one straight to its watcher, and the engine
already wraps whatever comes back in `Elsewhere` — so a reading **arrives saying
which host it came from** without one line attributing it. No message was added
to the protocol and no trait grew a method.

**It is read and never judged.** No bound anywhere near it. Whether 0.9 busy is
trouble is an opinion at a threshold, and this library keeps those in `health/`
where they can be argued with against a record that has already been written.

The reading is taken **before** the slice and not after: one taken when the work
is over is a reading of a machine that has just stopped, and the question is
what it was like while it was asked.

### The idle machine, which is the one you came to see

A worker only speaks down a wire when somebody gives it work, so the machine
sitting there doing nothing would not be in the picture at all — and that is the
one a fleet view exists for. `python -m somatize.worker --store DIR --reporting
SECONDS` is the clock.

It goes to the **store** and not down the connection, and the pipe was already
decided rather than open: CU20's rule is *where a connection is open, facts come
back down it; where there is none, they go to the store*, and an idle worker's
connection is one nobody is reading. That is measured against the code rather
than preferred — the client only reads the socket inside `say`, so a worker
beating while idle writes into a buffer nobody drains: it blocks on the write,
stops being able to accept the next job, and what the client eventually reads is
the **oldest** beats. Which is the worst available answer to *is it alive now*.

**One name per machine, rewritten**, and not one object per reading. That is
CU18's shape and it buys two things: a store that does not grow while a worker
sits there, and liveness for free — the store stamps every write, so `quiet_s`
is a scan with no fetches, measured **writer against writer** and never against
a local clock.

And the name it files under is what the machine calls **itself**, a hostname and
a process, because `w1` is the graph's word and *a worker does not know it*.
Which is a fact about this design and not a wrinkle: the two names only ever
meet on a reading that came down a wire, where the client attributed it, and
`fleet` joins them there. A machine that wrote and was never asked for anything
is in the fleet under its own name with nothing sent to it — which is exactly
what it is.

`examples/09-a-fleet.ipynb` is the picture of it: three workers, two runs, and
the bars flip. A cheap slice is 9.4 ms of which almost all is waiting; the same
shape with a slice worth sending is 810.7 ms of which almost all is work.
Nothing about the figure changed between them.

### What is not in it

**This machine.** `here` says nothing about itself. It is the one you can look at
with `top`, and inventing a row nobody had to send is not worth the line.

### Questionnaire

**The fleet** (`soma-python/tests/test_record.py`)
- [x] a run says what each machine did
- [x] and what it was waited on for
- [x] a machine nobody sent anything to is not in it — there is no registry
- [x] the fleet is drawn working against waited on
- [x] **a machine says what only it can say**
- [x] and it arrives saying which host, without anybody attributing it
- [x] a run with nobody else in it says nothing about machines
- [x] **a worker nobody is using still says it is there**
- [x] and the name the graph gave it is joined on
- [x] a machine that wrote and was never asked is there under its own name
- [x] how quiet a machine is is measured against the other writers

**The reading** (`soma-fabric/wire/tests/unit/machine.rs`)
- [x] a reading crosses as a flat fact and not as a variant of its own
- [x] what nobody measured is absent and not zero
- [x] a reading of this machine says how long it has been up wherever it runs
- [x] **nothing in it is a judgement**

## CU24 — Where the data comes from

```python
from somatize import Graph, Store
from somatize.data import Parquet, settle, to_polars

sms = Parquet(Store("/data"), "sms/train")
g = Graph.somatize(sms.named("sms").frozen() >> Clean().named("clean").cached())
settle(g)

g.forward({"at": 0, "take": 64}, store="/data")
```

The third of the four layers, and the first thing it decided was that there is
no `Source` trait.

### A source is a node, and being one is the whole design

A source takes something and answers with something, which is `Node`. A second
trait with a method that does what `forward` does is two things this project
exists not to build: a hole with one tenant, and the `error[E0034]` the rules
warn about — two traits with a same-named method in scope make that name
unusable.

Being a node is not a saving, it is the point. The DSL, `.on()`, `.at()`,
`.cached()`, `.mapped()`, the record and the figure all reach a dataset without
one line written here for any of them.

### What the graph is handed, and what that saves

Not a batch — a **coordinate**. A cache has to name what it is being asked, and
everything but the input is already a name: the node's class, its settled state,
its salt. The input is the one thing it has to look at, all of it, because two
batches differing in one number have to end up with different names. It is a
library reading the whole book to check whether it already has it.

Measured on 24 August 2026, release build, a graph of one node that returns a
constant so that only the input grows:

| what is handed to `forward` | with a store behind it |
|---|---|
| 1 MB of tensor | 4,9 ms |
| 19 MB of tensor (32×3×224×224) | **121 ms**, on every step, hit or miss |
| a `Span` | **0,027 ms** |

Linear in the batch, and paid on the hits as well — you pay to ask whether you
can avoid the work. And there is nothing to optimize there: `torch.save` is
1 ms/MB and sha256 is 2 ms/MB, so a faster hash saves nothing. **The answer is
not to weigh the batch**, and it is available because the rows can be named by
the span they are and by the version they came from.

### And the version was free

The other half of that name, and a source has to state it **without reading
itself** or it does the very work the cache exists to avoid. Against a `Store`
it costs nothing: a name resolves to a digest and the digest **is** the hash of
the content. One `resolve`, no bytes.

It goes where the digest of settled weights goes — `Memory::freeze(id, digest)`,
the call that is *made twice on purpose*: the declaration says a node is
settled, and whoever knows what is inside says what it is settled at. For
weights that means hashing them; here it means repeating what the store already
knew. `somatize.data.settle(g)` is the second call, and it is the same shape as
`somatize.torch.freeze(g)`.

`_has_state` grew a third duck for it, and that was a silent bug rather than a
nicety: a source declared `.frozen()` looked exactly like a tokenizer — nothing
to settle — so its version stayed out of the key and two datasets shared a name.
That is the one failure a cache must not have, and it is now refused before the
first node runs.

A source also **reads the version it stated**, not the name it was given: it
keeps the digest, so a dataset rebound under the same name mid-run does not
change what that run is reading. Resolving once is not an optimization, it is
what makes the version true.

### Arrow is the type, polars is the tool

`data/` takes `arrow` and `parquet` and stops there. Whoever wants expressions
over the rows brings their own engine: putting polars in the contract would
charge 370 crates to the worker that only tokenizes, and it earns nothing this
crate needs. What crosses an edge is a `Frame` — a `RecordBatch` inside an
`Opaque`, so the core still has no dependencies and never learns what a column
is — and `to_polars` / `to_arrow` turn it into whichever dataframe is installed,
neither of them a dependency of anything.

For a node that only wants the values, `frame.column("sms")` hands over plain
Python ones. That exists so the 193 MB worker with no torch in it does not have
to install a dataframe library to read a column of text.

**No runtime came in with any of it** — zero `tokio`, zero `futures` — and that
is the reason **SQL is not in this slice**. Every Rust driver worth using
carries a runtime inside, `Store` is synchronous on purpose, and
`store/Cargo.toml` already says what that costs: *«an SDK with a runtime inside
would have made the trait async, and that is the objection that has kept a bus
out of this repo twice»*. When SQL arrives it arrives synchronous, and DuckDB is
the candidate: sync, Arrow-native, and it attaches Postgres, MySQL and SQLite
rather than being N drivers.

### A frame crosses a wire

`Ipc` is the **second implementor of `Codec`**, from another crate, which is the
bar that keeps a hole a hole — the first was `python/`'s registry of
`dump`/`load` pairs. Arrow IPC is the buffers as they already are, with no
encoding pass, which is the reason Arrow is the type rather than something
converted to at the edges.

Two duplicates went before they could exist: the shape of a written-down opaque
(`{"__soma_opaque__": kind, "bytes": …}`) and `Packing` — a `Keeper` with a
codec in front of it — both moved to `transport/`, beside the trait they are
about. It is the same rule that keeps a store's record in the store and not in
the directory: two copies of a convention are two chances to disagree, and the
day they did nothing would fail.

The `kind` is `arrow.RecordBatch`, named after the **format** and not the
language: what is on disk is an Arrow IPC stream and whoever reads it back may
be a polars on the other side of the wall.

### Batch and stream are the same graph

Nothing above says the dataset is finite and nothing has to. A span is a
**position**, and a position can be asked for twice: rows 400..500 are the same
rows tomorrow, however much has arrived since. So a source read by span is
settled, and what moves is not its state — it is **which spans exist**. A source
that answers *whatever is newest* is the other thing, and the engine already
refuses to cache under it, because its answer cannot be asked for twice.

Nobody wrote a rule for streams. It falls out of `.frozen()` meaning what it
always meant, which is why the sentence this slice is about is not a slogan:

> **The difference between training and deploying is how many rows the frame
> brings.**

4096 from a folder of parquet while training, one from a topic in production.
The same graph, the same nodes, the same codec, the same figure, and no second
code path anywhere.

Push and pull both have a home already, and only one of them was missing. Pull
with a position — Kafka by offset, CDC by LSN, a file that grows — is this
slice. Push with no history — a websocket, a sensor — is whoever pushes calling
`forward`, which is request/response and has always worked; and whoever receives
writes into something with retention, which is CU20's rule and what keeps a
diagnosis reproducible.

### Where it is read is where it runs

`.at()` decides, exactly as it decides for any other node: the data is read on
the client, or on the machine that has it. What is missing to make that true is
small and named:

- a source has to **travel with its name** and not with a `Store`: `sms/train`
  means something over there and a path from here does not;
- and the worker has to be able to hand it its own store, which is `Ctx` — the
  channel that exists so that *whoever executes hands a node what it knows*, the
  same way a device is handed today. Adding to it is additive and no node
  signature changes.

The piece that is design and not plumbing is the **version**, because the client
computes the keys and cannot resolve a name that only exists over there. The
answer is one this project has already given twice: *whoever knows how to hash
is whoever has the thing in front of them*. `somatize.torch.freeze` settles
weights on the machine that holds them; a remote source is settled by the
**worker**, against its own store, when the slice arrives. The machinery is
already there — `Memory` travels in the cargo, the worker answers with
`outcome.keys`, and `elsewhere` merges them — so the client learns the name
afterwards and can go on naming its own nodes from it.

What is lost is the pre-pass: nothing can be foreseen above a remote source, so
nothing is skipped there. Conservative, and the same side it already gives up on
for a mapped node.

It is **not built**, and the reason is the rule: there is no worker anywhere
today that can see a store with data in it — not even in the containers, where
the volume is shared between them and not with whoever runs the tests.

### What is not in it

- **SQL**, decided rather than delayed. See above.
- **Ranged reads.** `Store::get` answers with every byte of a blob, so a file is
  read once and held. A dataset that does not fit in memory needs the store to
  learn to read a range first.
- **A source on a worker**, for the reason above.

### Questionnaire

**The rows** (`soma-data/tests/unit/`, `soma-python/tests/test_source.py`)
- [x] a span is the rows it names
- [x] the last span of a dataset is short, and one past the end is empty
- [x] a span that crosses a row group still comes back as one frame
- [x] the columns come back with their names and their types
- [x] **declaring a dataset does not open it**
- [x] and the bytes arrive once however many spans are asked for
- [x] a column comes over as plain Python values, with no dataframe library
- [x] a name nobody bound says so before anything runs

**The version** (`soma-data/tests/unit/parquet.rs`, `soma-python/tests/test_source.py`)
- [x] it is what the store already knew, and costs one lookup and no bytes
- [x] the same data under two names is the same version
- [x] and different data under one name is a different version
- [x] **other data under the same name is not the same answer**
- [x] a source nobody settled is refused before anything runs

**The frame** (`soma-data/tests/unit/frame.rs`, `soma-data/tests/unit/ipc.rs`)
- [x] it crosses an edge and is the same frame on the other side
- [x] what the columns are called is there without reading a value
- [x] it does not travel on its own — an opaque is an opaque
- [x] written down and read back it is the same frame
- [x] and what it became can leave the process
- [x] a frame is found however deep it is
- [x] an opaque that is not a frame says so
- [x] somebody else's kind is left exactly as it arrived
- [x] **rows read here are tokenized over there**
- [x] a kept frame means the dataset is not opened again

**The whole thing** (`soma-python/tests/cluster/test_searching.py`)
- [x] the dataset goes into the shared store once and every machine reads spans
- [x] the graph is still cut across three machines and the study across two

## CU25 — What only fed an answer that was kept is not run

Opened by a question about the notebook above: *if `widest` is cached, should
`sms` and `clean` not be skipped?* They should, and they were not.

The cache skipped the node whose own output was there and ran everything else
anyway. On a graph fed by a dataset that is the expensive half — the file is
read, the rows are tokenized, and none of it is looked at.

### A name is knowable before anything runs

`key_for` had already said so: *the name this node's output will have, **before**
it has one*. Only the graph's input is hashed by content; from there down a key
is made of keys. So the engine can name the whole plan with nothing executed,
ask which of those answers it already has, and then work **backwards** from the
leaves: a node whose answer is kept does not need its inputs.

`Keeper::present` is the question and it is new in the hole. The default is
honest and expensive — it reads them — and a store overrides it with
`resolve_many`: one scan, no fetches. Asking early has to be free or it is not
worth asking, which is the same price list CU20 wrote down.

Two places it gives up, both towards keeping a node rather than skipping one: a
`.mapped()` node is named out of the **content of its items**, so its names are
not knowable until it has them; and a node with no key at all is a miss for the
same reason it was never cached.

A slice nobody needs is **not sent**. The saving there is the round trip and not
the work at the far end.

### And it says so

`Fact::Spared` is a fact rather than an absence, because a node missing from a
record cannot be told from one that was never in the graph, and *why is there no
time for `clean`* is a question whoever reads a run will have. In the notebook
it reads:

```text
{'fact': 'spared', 'node': 'sms'}
{'fact': 'spared', 'node': 'clean'}
{'fact': 'recalled', 'node': 'widest', 'key': 'sha256:c521a0…'}
{'fact': 'finished', 'took_us': '155'}
```

`RunError::Vanished` is the rare other end: something that was there when the
store was asked and gone when it was read, after what feeds it had already been
skipped because of that answer. It says what happened instead of leaving a
puzzle.

### What the notebook caught

Re-running `examples/10-a-dataset.ipynb` after writing this, the 19 MB toll had
gone from 121 ms to **245 ms**. The pre-pass named the root and the walk named
it again, so the batch was hashed **twice**: asking early cost exactly what it
saves. The walk reuses the name that was already worked out, and it is back at
121 ms.

No test would have caught it — they all pass either way. It is the third real
thing a notebook has found in this project, and the reason they are executed
rather than written.

### Questionnaire

**The pruning** (`soma-core/tests/unit/execution.rs`)
- [x] what only fed an answer that was kept is not run
- [x] and the answer is still the answer
- [x] and it says so rather than leaving a hole in the record
- [x] **but what somebody else still reads is run**
- [x] a node that maps keeps everything above it
- [x] and a slice nobody needs is not sent at all

**End to end** (`soma-data/tests/unit/parquet.rs`)
- [x] the second run finds the answer under a name it could work out, and never
      opens the dataset

## CU26 — What an edit did, before paying to find out

CU25 built the pre-pass and then used it for one thing. The engine names the
whole plan with nothing executed, asks which of those answers are already there,
and skips what only fed one — and then throws the names away.

They are worth keeping. Two versions of one graph name a node differently
**exactly when** its recipe changed, so comparing two sets of names says what an
edit did. The question is the one everybody with a cache asks out loud on a
Tuesday afternoon — *did I just invalidate the encoder, or only the head?* — and
today it is answered by running and watching which nodes take time.

`Executor::foreseen` is `pub` for that, `foreseen_json` is the bridge, and
`somatize.foreseen` is where it is asked:

```python
from somatize import foreseen

foreseen.names(g)                       # {node: name}, nothing executed
foreseen.unneeded(g, x, store=store)    # what would not have to run at all
foreseen.changes(before, after)         # what the edit did
foreseen.snapshot(g)                    # the same, kept for a later comparison
```

### Findings and not buckets, and the case that decided it

The first shape was a partition: every node into exactly one of `changed`,
`downstream`, `stale`, `added`, `gone`, `unknown`, `same`. It reads well and it
is wrong, and the counterexample took one run to appear — edit the encoder's
code **and** bump the head's salt:

```text
{'embed': ['STALE'], 'head': ['SALTED', 'SUSPECT']}
```

The head's name moved, so it runs again: `SALTED` is true. And it runs on what
the encoder handed back, which is the old code's answer: `SUSPECT` is true too.
A partition can carry one of the two, and the one it carries is *this node will
be recomputed* — the reassuring half of a node about to compute the wrong
thing.

So the shape is `{node: [finding, ...]}`, which is what `somatize.health`
already answers with, and for the same reason: what happens to a node is more
than one fact. A node with nothing said about it is fine, and absence being the
good answer is what keeps the ones that matter readable.

| finding | what it says |
|---|---|
| `CHANGED` | its **shape** moved: another class, or who feeds it |
| `RESETTLED` | it is frozen at another state — other weights, another version |
| `SALTED` | its salt moved |
| `DOWNSTREAM` | none of those moved and its name moved anyway |
| `STALE` | its name did not move and its code did |
| `SUSPECT` | something above it is `STALE` |
| `ADDED` / `GONE` | it is in one graph and not the other |
| `UNVERSIONED` | its answer is kept and nobody can say whether its code moved |
| `UNKNOWN` | it cannot be named on one side or the other |

`CHANGED` against `DOWNSTREAM` is what makes a list of forty nodes readable: one
of them is where the edit is and the rest is what inherited it. Who feeds a node
is part of its shape because rewiring it moves its key without touching anything
the node is made of — and that is an edit, not something somebody else did.

### One question, or two

The first three are the same finding split three ways, and the split came from
the other end of the wire. A sibling slice — a tree of experiments, one snapshot
per commit — asked *what did the code do*, and to that question a node re-frozen
at Tuesday's checkpoint is a false positive: the architecture is the same one,
trained again. To the question this module opened with — *does my cache still
hold* — it is not a false positive at all: the answer under that name really is
a different answer, and the name moving is the cache being right.

Both are true, and neither is a reason to give the other the wrong answer. So a
name that moved says **which part of the recipe moved it**, and the two readers
take the part they came for: all three, or `CHANGED` alone. Weights belong to a
version; they are not a version.

`UNKNOWN` is the absence CU25 already gives up on, said out loud: a `.mapped()`
node is named out of the content of its items and nothing under it can be named
either. It is the one place where *cannot tell* has to survive all the way
to the reader, because the alternative reads as *checked, and fine*.

### `STALE`, which is the finding the key cannot give

The fingerprint of the code is **deliberately not in the key** — CU13 decided
it, and the reason still holds: a cosmetic refactor would invalidate half the
store in silence. It is kept beside the value and compared on a hit, which
turns that into a line on `stderr`.

The cost of that decision is that editing the body of a `forward` renames
nothing. A diff that only compared names would answer *nothing changed* to the
very edit being asked about — which is why the fingerprint is looked at here,
where it is an **opinion and not an invalidation**. `STALE` is the finding that
says *you should have bumped the salt*, and the two are the same edit answered
and not answered:

```text
edited, salt bumped   {'embed': ['CHANGED'], 'head': ['DOWNSTREAM']}
edited, salt not      {'embed': ['STALE'],   'head': ['SUSPECT']}
```

And it **reaches down**, which is what `SUSPECT` is. A stale node hits, so
everything under it goes on being fed the old code's answer, whatever became of
its own name.

### `UNVERSIONED`, which the notebook found

It needs both fingerprints, and a class with no source to read has none — a
notebook cell, an `exec`. The first draft treated that as no opinion and said
nothing, and writing `examples/11-what-an-edit-did.ipynb` is what showed what
that means: **in a
notebook every node is defined in a cell**, so a graph compared with an edited
copy of itself came back `{}` — *nothing to report* about an afternoon of edits,
in the one place where the question is asked most.

So the absence is a finding. Its scope is the scope a version is recorded at —
`_note_the_code` computes one **only for what is kept**, because parsing an AST
for a node nobody remembers anything about would be paid by everyone who
declares a graph. A graph with no cache in it gets no opinion about its code and
is not told so once per node.

### Two graphs, or two snapshots

`changes` takes either. A `Graph` is a live object and two versions of one
module do not coexist in an interpreter, so comparing **two commits** means
comparing what was written down: `snapshot(g)` is plain JSON with the names
already worked out — the shape of each node, its state, its salt, what is kept,
the fingerprints, and the edges, which `SUSPECT` needs to reach down.

The cut is the one this project keeps making between **the fact and whoever
draws it**, and it is what lets the sibling slice delete its own comparison and
consume this one. Two snapshots are comparable when they were taken with the
same input, which the default — none at all — always is.

### What it does not need

**A store's contents.** Naming is the `Keeper`'s and the keeper is the store's —
the core computes no hash — so a store is where the hash function comes from and
nothing else is asked of it. `store=None` opens a temporary directory and gives
the same names. `unneeded` is the only one of the three whose answer is about
what is in there, and it is the only one that demands one.

**The input.** Every key on both sides carries the same hash of it, so which
input it is cancels out of every comparison. `changes(before, after)` with
nothing at all gives the same findings as with the real batch, and does not pay
the 121 ms of weighing it that CU24 measured.

### What is not in it

- **The figure.** `overlaid` puts findings on a graph, and its channel is
  health: the outline turns red and red means ill. A node that changed is not
  ill, so it would need a channel or a colour of its own, and that is a drawing
  decision and not this slice's.
- **A silent edit above a cached node.** A version is recorded only for what
  is kept, so a code change in an **uncached** node moves nothing: its key does
  not carry its code, and the cached node below it goes on hitting. The runtime
  has the same blind spot for the same reason — it compares a fingerprint on a
  hit, and there is no hit to compare on. Closing it means versioning every
  node at `somatize`, which is CU13's trade to reopen and not this slice's.
- **A fan-in whose order does not matter.** `(a | b) >> c` and `(b | a) >> c`
  name `c` differently, though what reaches it is a dict keyed by node id and
  the order is nothing to the node. A conservative miss, nobody's answer is
  wrong, and it is CU13's to decide — not something a diff should hide.
  `_recipe` carries the order for exactly that reason: it says the truth about
  what the key does today.

### Questionnaire

**A name, without a run** (`soma-core/tests/unit/execution.rs`,
`soma-python/tests/test_foreseen.py`)
- [x] the names foreseen are the names things are kept under
- [x] nothing ran to find out — a node that cannot run at all still has one
- [x] and it does not depend on having a store
- [x] what cannot be foreseen is missing rather than wrong
- [x] without a keeper nothing is named, which is the silence a run gets
- [x] what is already kept says what would not have to run
- [x] and asking that needs somewhere to look

**What an edit did** (`soma-python/tests/test_foreseen.py`)
- [x] a graph compared with itself has nothing said about it
- [x] a changed recipe renames what is under it and nothing above it
- [x] and `CHANGED` against `DOWNSTREAM` says which is which
- [x] another class under the same name is a change of shape
- [x] **other weights under the same code are not**
- [x] and two parts of one recipe moving says both
- [x] **rewiring a node is a change of its own and not an inherited one**
- [x] a node that is only in one of them is not a change to a name
- [x] a mapped node cannot be told about either way
- [x] the input cancels out of the comparison

**The code that changed and the name that did not say so**
(`soma-python/tests/test_foreseen.py`)
- [x] an edit the key cannot see is said out loud
- [x] what is under a stale node is not told it is fine
- [x] **and a node that recomputes still recomputes from a stale answer**
- [x] **a class nobody can version says so rather than nothing**
- [x] and a node nothing is kept of is not told off for having none
- [x] a bumped salt is what `STALE` is asking for

**Two graphs, or two snapshots** (`soma-python/tests/test_foreseen.py`)
- [x] a snapshot answers exactly as the graph it was taken of
- [x] **and it survives the graph it came from**, through JSON
- [x] what is under a stale node is found through a snapshot too

## CU27 — What a node was built with

Escalated out of CU26 by the slice consuming its findings, and it is not a
missing axis in a diff. It is a **wrong value served in silence**.

```python
Embed(512).named("embed").frozen().cached()
Embed(64).named("embed").frozen().cached()
```

One class, one identity, one name in the store — and two different answers. The
second run is handed the first one's, with no error and no warning. It is the
same failure CU13 already refuses for a checkpoint nobody hashed, arriving
through the one door that check does not cover: `_check_it_was_obeyed` asks for
a digest only from something with `state_dict`, `parameters` or `version`, and
a node whose behaviour is a **number in its constructor** answers none of them.

So `key(node) = H(identity, declaration, state, keys above)`. The class is half
of *what this node is* and what it was built with is the other half.

### What was passed in, not what is lying around

The first attempt read the declaration off `vars(obj)`, and four tests fell
over in a way worth writing down: **what a node holds is not what it was built
with.** A node that counts its calls, caches a client on first use or moves a
tensor onto a device has attributes that move **while the graph runs**, and a
key made of those renames a node nobody touched. In `test_freeze.py` the
encoder ran three times instead of once, because `calls` was in its own name.

So it is captured at `__init__`, by `Node.__init_subclass__`, and never read
off the object afterwards. Bound against the signature with the defaults filled
in, so `Layer(64, 32)`, `Layer(64, out=32)` and `Layer(in_=64, out=32)` are one
declaration — a key that depended on how the call was typed would miss for a
rename.

### Two ways to be wrong, and they are not symmetric

A key is computed on the client and computed **again** on a worker, so the text
has to mean the same thing in another process:

- **Unstable** — one declaration, two texts. `<mine.Helper object at 0x7f43…>`
  is the usual one. The cost is a cache that misses forever and never says why.
- **Lossy** — two declarations, one text. A truncated tensor, or an address
  scrubbed down to `<Helper>`. The cost is the wrong value served in silence,
  which is the bug being closed.

Neither is accepted, and the second is why **scrubbing the address away is not
the answer**: `<Helper>` is stable and says nothing, so two configurations go
on colliding. Anything that cannot be written faithfully *and* identically
twice raises `CannotDeclare`, and the graph is refused before the first node
with the attribute named — `` `Given.held` is a lambda, and every lambda is
written down under the same name ``.

### A test on the type does not catch an address

The rule that looks right and is not: `type(obj).__repr__ is object.__repr__`.
It is correct about the object in front of it and blind to what it holds — a
**list** of those objects has `list.__repr__`, which is defined, and the
addresses come through from inside. Containers are walked rather than repr'd
for exactly that, and the one place a `__repr__` nobody here wrote is trusted,
the text it produced is checked afterwards.

A `set` is the same trap in other clothes: string hashing is seeded per
process, so its repr's order is stable in one interpreter and different in the
next. It is sorted before anything sees it, and the test that says so runs a
**subprocess** — a fixture pretending to be another process proves nothing
about hash seeds.

### What the digest is believed about, and what it is not

`freeze(id, digest)` stays authoritative about the **state**: whoever knows how
to hash weights says what a node is settled at, and the core believes them. It
is not authoritative about the declaration, and one test changed to say so. Two
nodes built the same way and settled at one digest are one name — that is the
digest being believed. Two nodes built *differently* and settled at one digest
are two names, because they answer `6.0` and `15.0` and no assertion about
their weights makes that untrue.

### What is not in it

- **A node that is not a `Node`.** Anything held as an argument has nothing
  captured at construction, so it is read off its attributes instead. That errs
  the safe way — an over-sensitive miss costs time, being blind costs the
  answer.
- **A `__repr__` that hides a set inside itself.** The angle-bracket rule
  catches most of what cannot be written — CPython's own convention for *this
  has no faithful repr* — and enums, classes and functions are taken out before
  it since they wear the brackets and are perfectly stable. Past that, the
  answer is `salt=`.
- **A hard-coded constant.** `self.dim = 512` in a body is code, and code is
  the fingerprint's question: `STALE`, not `CHANGED`. That line is worth reading
  twice, because moving one number from an argument to a constant moves the same
  edit from *the cache will miss* to *the cache will hit and hand back the old
  code's answer* — two very different afternoons, and the reason the two
  findings are two words.

### Questionnaire

**Faithful: two declarations, two texts** (`soma-python/tests/test_declaration.py`)
- [x] two arguments are two declarations, and the same arguments are one
- [x] **what a node holds is followed and not believed**
- [x] a mapping built in another order is the same mapping, and another is not
- [x] a name and not a source is what a class or a function writes

**Steady: one declaration, one text, in any process**
- [x] **a set is written down in an order of its own**
- [x] and what a node holds is the same text in another process

**And what can be neither is refused, saying which**
- [x] a repr that writes its own address
- [x] something that writes itself in angle brackets
- [x] a lambda, because every lambda has the same name
- [x] data held as an attribute, rather than truncated
- [x] something that holds itself

**What the graph does with it** (`soma-python/tests/test_declaration.py`,
`soma-core/tests/unit/execution.rs`)
- [x] what a node was built with is in its name, and reaches everything under it
- [x] **and the answer is the one that was asked for**
- [x] a graph built by hand is told apart too
- [x] **what a node keeps for itself is not what it was built with**
- [x] a cache that cannot be named is refused before the first node
- [x] and a graph that keeps nothing is not asked
- [x] nobody saying what built it is not a reason to refuse a name

---

## After CU27 — The wire leaves, and soma-fabric opens

```
soma/transport/   →   soma-fabric/wire/     (sixteen commits, and their history)
```

Not a use case: nothing can be written the day after that could not be written
the day before. It is a **cut**, made when four things that shared the name
*remote execution* turned out to want opposite things:

| | what it is | what it wants | where it lives |
|---|---|---|---|
| **placement** | `.at("w1")`: the graph says a node runs elsewhere | to be a *declaration*, part of the graph's meaning | **stays** |
| **transport** | bytes over a wire, framed, with codecs | low latency, a hot connection | leaves |
| **provisioning** | how the code reaches the other side | reproducibility | leaves |
| **coordination** | who does which work | durability, retries, leases | leaves |

There is no single thing that is both *a hot connection* and *durability with
retries*, which is why one name over the four of them read as a tangle rather
than a design. Placement is a declaration and belongs beside the graph that makes
it; the other three are mechanism. What soma keeps is `Transport`, **a hole with
one method**, and the `.at()` that fills it with a name.

The boundary is **not a layer**, and pretending it were would have put the codec,
the store or the `Value` on the wrong side of the cut: `soma-fabric/wire` depends
on `soma-core` and `soma-store`, and `soma-python` depends on
`soma-fabric-wire`. No cycle — they are different crates — but what the
arrangement assumes is that **the two repositories sit side by side and move
together**, which is what the path dependencies say out loud. The day either is
published on its own they become versions.

It is not free, and the place it shows is the one nobody predicts: the cluster
images build from **two contexts** now, because a single root over both is a
directory of unrelated projects and every byte of it would go to the daemon.

The move is verified the only way a move can be: **87 tests green on the other
side of it**, having changed nothing.

---

## CU28 — A client talks to a broker

```python
from somatize import Broker, Graph, Worker

g = Graph.somatize(Encode() >> Classify().at("gpu-box"))
g.forward(x, broker=Broker.embedded({"gpu-box": Worker.at("gpu-box:7000")}))
```

Status: **closed**. A crate in soma-fabric, `soma-fabric-broker`, with 45 tests
— 16 of the thread, 13 of the handle, 11 of the protocol, 5 of the paths —
plus `test_remote.py` (41) and four in `soma-core/tests/unit/placement.rs`.

### The question: what changes for somebody who has no platform?

The answer this use case exists to give is **which broker, and that is a URL**.
Not a degraded mode with its own code, not a branch on whether there is an
account: the same call, the same protocol, the same handle.

| deployment | what it is | what it adds |
|---|---|---|
| **embedded** | in the client's own process | nothing. It is what makes soma work alone |
| local | a process on a head node | reachable by more than one client |
| platform | ours | authentication, pairing, leases, metering |

Only the first exists today, and there is deliberately **no `Broker` trait**: a
trait with one implementor is the shape this project was started to stop
writing. It arrives with the second.

### A `Worker` stops being a connection and becomes a declaration

That is the change a caller can see, and it has a consequence worth stating
rather than discovering: **an unreachable host now fails when it is needed
rather than when it is named.** `Worker.at("bad:7000")` used to fail in the
constructor; now it fails inside the run, from the slice that wanted it. Better
behaviour — a graph names hosts a run may never reach, and a branch not taken is
a worker not needed — and a change, not a side effect.

### Ask eagerly, connect lazily

The two costs pull in opposite directions and both are real, so the split is
between them:

- **Asking** where a host is costs tens of bytes, and has to happen **before**
  the first node runs, because what gets packed for a host depends on which
  hosts turn out to be the same place.
- **Connecting** costs a socket, a process, or both.

So a rendezvous is asked for once and remembered by the session, and the wire it
describes is opened the first time somebody actually sends work down it. The ask
happens once however it was triggered: a client that resolved every host up front
to decide what to pack finds the answers already there when the run reaches them.

### Two names for one place are one wire, and it is not a nicety

A worker has **one** catalog, and half of one is a different catalog.
Provisioning the same process twice, once per host name, replaces what it had
live and takes every activation over there with it — a run that quietly loses
its state, not an extra socket.

Only the session sees more than one host at a time, so only the session can
know. But **what to pack is Python's**, because it is the half that knows what
a `cloudpickle` is. The two are reconciled without either learning the other's
business: `wire_token(host)` answers with **opaque bytes that are equal exactly
when two hosts are one wire**. Python groups by equality and never finds out
what a path is, nor when two of them count as one.

What decides it is `Path::shared`, and the asymmetry in it is the point: an
**address** is an identity — the same host and port is the same process — while
a **command** is a thing to run, and running it twice gives two of them. Today's
suite stands up two hosts from an identical `argv` and requires two processes.

### The ladder of four, of which two can be answered

This is the one place the crate builds ahead of its consumer, on purpose:

1. **same process** — nothing is transferred
2. **shared mount** — a path is written and read; free, and a cluster has one
3. **direct socket** — one crossing, lowest latency. The broker steps out
4. **relayed** — streamed through the broker, no disk and no durability

Two of them can be answered today. All four are in the message, because the
alternative is that adding a rung later changes `Reply::Met` — and a message
that changes is a version that changes for everybody. **The ladder is the
design; what arrives later is the probing that chooses, not the vocabulary.**

An object store is not one of the rungs. It is where durable things live, and
renting eleven nines for an activation that lives forty milliseconds is the
wrong shape before it is a bill.

The rung that transfers nothing answers with a `SlotId` and **never a handle**,
which looks like a needless indirection and is not: every message here has to
survive a round trip through bytes, including the ones an embedded broker
answers without leaving the process. It costs nothing and buys a conformance
suite that can round-trip every message there is.

### The embedded broker is a thread, and it really serializes

Both of those look like waste and neither is, because **control and cargo go by
different routes**:

| | what crosses | how much | how often |
|---|---|---|---|
| the rendezvous | a path | tens of bytes | once per host per session |
| the wire next door | an activation | megabytes | once per forward, fifty thousand forwards |

A run across four workers is nine messages — one greeting, four rendezvous, four
goodbyes — some tens of microseconds, once, outside the loop. The broker is in
the first row and steps out of the second, so being honest here is not
measurable there. And being honest buys the thing that matters: **the protocol
is exercised for real from the first day**, by a round trip that actually
happens, before any broker exists outside a process. A protocol whose only
implementation never serialized anything would be a protocol nobody had tested.

The failure that type exists not to have is a client blocked forever on an
answer that is never coming. Every channel operation maps to `Unanswered::Gone`
and never to an `unwrap`, and that is pinned by a test rather than by care —
which is why `Embedded::served_by` is public: without a way to stand up a desk
that fails, the one failure mode worth testing is the one that cannot happen.

### Which hosts does this graph name?

`Placement::hosts()`, the half of `host_of` that reads the other way, and it
exists because of **who asks**. A client handed a dictionary of workers already
knew the names — they were its keys. A client that talks to a broker does not.

Once each, because a host named by ten nodes is one rendezvous and not ten. And
**sorted**, which is not tidiness: they come out of a `HashMap`, and iterating
one gives a different order every run. That would make the order rendezvous are
asked for — and so the order failures happen in — irreproducible, and this
project has already paid for a nondeterministic order once, when an artifact's
id changed because the caller reordered a dictionary.

### What is not honoured yet, said out loud

A `Reply::Met` can carry a `good_for`, and **nothing enforces it**. No broker
issues one today — the embedded one has no policy — so enforcing it would be a
mechanism with no tenant. The day one does, the enforcement belongs to the
handle and not to the engine: it is the only thing that knows when the
rendezvous was granted.

### Questionnaire

**The protocol survives meeting a binary that disagrees with it**
(`broker/tests/unit/protocol.rs`)
- [x] every question and every answer goes and comes back equal
- [x] a greeting from a version we do not speak is **still readable**
- [x] and is refused naming both numbers
- [x] **a greeting that grew a field could not be read, which is why it must not grow**
- [x] leftovers are as suspicious as missing bytes, and a truncated message is refused
- [x] an answer is not a question

**The four paths, of which two can be answered** (`broker/tests/unit/path.rs`)
- [x] all four cross, **including the one that transfers nothing**
- [x] a pipe and a socket are the same path
- [x] a command keeps its arguments in order
- [x] how long it is good for crosses as a duration

**A broker on a thread, which must never become a hang**
(`broker/tests/unit/embedded.rs`)
- [x] the session stays open across rendezvous, and is greeted once
- [x] **a desk that panics is reported and not waited on**
- [x] and one that has fallen over stays fallen over instead of hanging
- [x] one thread for the broker, and not one per ask
- [x] dropping it ends its thread rather than leaving it behind
- [x] two threads reach one broker at once
- [x] bytes that are not a message are refused by the desk and are not fatal
- [x] a host it does not know is told what it does know

**One host, standing in the engine's hole** (`broker/tests/unit/reaching.rs`)
- [x] building a handle asks the broker nothing
- [x] **an unreachable host fails when it is needed and not when it is named**
- [x] four hosts share one greeting, and a host is asked about once however many
      times it is wanted
- [x] a rendezvous nobody took is not let go of, and one that was taken is
- [x] a slice reaches a real worker through the broker, and the second reuses
      the wire without asking again
- [x] the paths the negotiation has not arrived for are refused **by name**

**Two names for one place** (`broker/tests/unit/reaching.rs`)
- [x] two hosts at one address share one wire
- [x] **two hosts with the same command are two processes**
- [x] and what is packed for them is packed **once**
- [x] two names for one place declared with different packing are refused, **by
      name**, saying what each of them asked for

**Which hosts a placement names** (`soma-core/tests/unit/placement.rs`)
- [x] they come back once each
- [x] a placement that sends nothing away names no hosts
- [x] **the order does not depend on the order they were placed in**
- [x] moving a node elsewhere leaves no ghost behind

**What a client writes** (`soma-python/tests/test_remote.py`)
- [x] a worker is declared with an address or a command, and declaring it starts
      nothing
- [x] a broker takes a dict from host to `Worker`, and says which host was wrong
- [x] two workers are two processes, and the artifact is sent only once
- [x] a worker that gets nothing is told nothing
- [x] a graph run in pieces keeps the worker it had
- [x] `provision` says out loud what `forward` says on its own

**Two names for one place, which is one catalog** (`soma-python/tests/test_remote.py`)
- [x] two names for one place are told once each, about **one** artifact holding
      both halves
- [x] while two addresses are two catalogs with half each
- [x] two names for one place packed differently is refused, naming both
- [x] a host the broker never heard of is left out and not raised over

> Proved against an **artifact** and not against a wire, which is the half this
> side owns. The grouping was mutated — keyed by host name instead of by wire —
> and two of the four fail; the two that survive are the contrast rows, which is
> what they are for.

### What is pending

- **`session.rs` has no test module**, though it holds the rules that matter
  most — the ask remembered, the wire shared, the token. They are reached
  through `reaching.rs`, which is real coverage and not a file of its own.
- **The local and platform brokers**, the path negotiation, the agent and the
  queue. They arrive with a consumer and not before.

## CU29 — A kept value says where it came from

```python
g.forward(x, store=where, stamping={"run": "an-investigation/3847d0c1"})
```

```
$ cat where/names/*/* | jq -r .meta
[["node","embed"],["fingerprint","920aac16"],
 ["input","sha256:1064…"],["env","2ffccc9569ca"],["run","an-investigation/3847d0c1"]]
```

Status: **closed**. `Executor::stamping` and the `INPUT` constant in
`soma-core/src/execution.rs` with 4 tests, `somatize/_environment.py`, and
`test_provenance.py` (11).

### The question: what is a hash six months later?

A store outlives every process that ever wrote to it. What survives in it is a
pile of names that are hashes of recipes, and a recipe **does not run
backwards**: from a key there is no path back to what made it. Inside one
afternoon of trying five things, that becomes five sets of intermediates nobody
can tell apart. They are not wrong. They are mute, which with time is worse.

So what cannot be recovered is written down at the moment it is known.

| written by | what | recoverable later? |
|---|---|---|
| the engine | the node, the fingerprint of its code | it already was |
| the engine | the input, by the name its content has | **never** — only a keeper can hash a value |
| `somatize` | the environment | **never** — it is in no key |
| the caller | a run, a commit, an investigation | not by anybody else |

### Why the environment, when there is already a fingerprint

Because a fingerprint stops at what is **installed**: a distribution goes in by
name and version, and the standard library by its bare name, since the
interpreter is compared at the greeting rather than hashed into every class.
That is right for naming and wrong for provenance — two interpreters name the
same node identically, and only one of them produced what is on disk.

It is filed once as `env/<digest>` and carried on each value as twelve
characters. Both, because whoever reads this store back in a year needs the
short name to group by and the long one to understand.

### Why the core is told nothing about any of it

A commit, an investigation, an environment are facts about the world outside a
graph. A core with a field for one of them is a core that has learnt a word
belonging to whoever stands above it — so `stamping` is opaque text it passes
through untouched, the same division of labour as `Meta` itself and as the name
a study is filed under. The Python layer fills in the environment because that
is where soma meets an interpreter; everything else arrives from outside.

**Nobody has to remember.** Four of the five are written with no argument
passed, and that is the point: provenance that has to be asked for is missing
from exactly the runs nobody thought were going to matter.

### What the caller may not say

`node`, `fingerprint` and `input` are refused where somebody is typing, and
dropped in the core if they arrive anyway. Not ordering — dropped: whether the
first or the last of two pairs wins is the reader's convention, and the obvious
way to read a list of pairs takes the last. A value that came back naming
another node would be the one mistake the whole mechanism exists to prevent.

### What is pending

- **A `.mapped()` node** is named by the content of its items, so each item's
  key is its own. They are stamped like everything else and attributable one by
  one; what nothing can do is *foresee* them, so a reader working from a probe
  alone will not find them.
- **A slice on a worker says nothing about the graph's input**, deliberately:
  what arrives at a slice is not what arrived at the graph. Whoever coordinates
  knows the real one and can send it in a stamp; nothing does yet.
- **Nothing deletes.** The `Store` trait has no `forget`, and a
  content-addressed store where two versions legitimately share a blob needs
  unbinding and a sweep over *every* name, not an `rm`. It arrives with a
  consumer and a decision, not before.

## CU30 — The reasoning of an investigation, versioned beside the code

Twenty-nine slices answer *what does this graph do*. None of them answers *what
was I trying to find out* — and that is the half an investigation actually
loses. A repository keeps every edit and no motive. A store keeps every number
and no question. Six months later the code is readable, the results are
readable, and why anybody ran them is gone.

What this has to do, and everything below is judged against it: **pin the code,
the approach and the result so the three are versioned together**, and **give
the reasoning enough structure that what worked, what did not, and why can be
seen** rather than reconstructed from memory.

The oracle is `/mnt/cluster/projects/soma-tree`, which built this once and
proved it against a finished paper. It is read as a questionnaire: below is
what has to be true, and the call shapes are decided here.

### Two layers, and the rule that says which

**The record** is what can be recalculated: commits, the graph at each of them,
what an edit did node by node, the trials that ran with each version. Nobody
types it — it is worked out from git, from a probe and from a store.

**The reasoning** is what somebody thought: the questions, the hypotheses, what
was tried, what the evidence said, what was decided. No amount of reading the
repository recovers it.

> **If it can be recalculated it is record. If somebody thought it it is
> reasoning.**

So a pruned line is derived and never stored, and a verdict is written down and
never guessed at. The two do not share a unit either: a commit is nobody's
decision, a question nobody has tried yet has no commit, and one move can
produce three branches.

### Five kinds, and there are no more

| kind | what it is | what only it can do |
|---|---|---|
| `question` | what is not known | the only one that can stand with nothing under it — an untried question is pending work |
| `hypothesis` | a proposed, falsifiable answer | gets **validated** or **refuted**, verbs a question does not have |
| `attempt` | what was tried | the only one that touches the record |
| `finding` | what the evidence says | where the verb edges come from |
| `decision` | what is done about it | separate from the finding, because two people can agree on one and disagree on the other |

### It is a DAG, and the case that forces it

Two live questions — does more capacity improve interpretability? does it
improve performance? — one variant that bears on each, and then the question
neither contained: what if I combine them? That attempt hangs under **both**.
With a single parent you either choose, or duplicate the node — and a duplicated
node is two nodes that drift apart.

So `under` is multivalued and a cycle is refused at write time: a walk over one
does not terminate, including the walk that would draw it.

Which attempts it is *made of* is a different edge: `combines`. That is what
makes *each one worked alone, together they cancel* readable as what it is,
rather than as two results that happen to sit near each other.

### Scope, and why standing cannot be a field

An answer holds **where it holds**. Without that, `validated` and `refuted` on
one hypothesis look like a contradiction, when the ordinary case is two facts
about two situations. A scope is a set of **roots** — "the whole encoder
branch" is a root, the whole investigation is none at all — which is what makes
*do these overlap?* a walk rather than something to materialise.

Standing is **derived and never stored**:

```
open · answered · partly · validated · partly-validated
refuted · partly-refuted · disputed · depends
```

A field somebody overwrites loses the previous fact, and the previous fact is
what lets a hypothesis go back to open on its own when what refuted it is
marked invalid. Two of the nine are why it is not a field at all: `disputed` is
edges of opposite sign whose scopes **touch**; `depends` is validated in some
situations and refuted in others **without** touching — not half an answer and
not a conflict, but the answer depending on the case, which is the most
informative outcome an investigation gives.

That is not hypothetical. Seeded from a finished paper, the central
hypothesis — *a pure per-symptom decoder is interpretable by construction and
can match an aggregator* — comes out `depends`. An earlier seeding put both
halves at the same scope and got `disputed`, which reads as two people
disagreeing rather than as an answer with a domain.

### What pins the three together

An `attempt` **cites** the record, and a commit is only half of what ran:
`--decorr-weight 0.1` and `0.5` are the same commit and different experiments.
So the resolved invocation is kept content-addressed beside it, and a citation
carries both.

The other direction had to exist too. A commit you cannot ask *what was this
for* is a change without a motive. So a commit carries the moves that cite it,
**derived from the citations and kept in no index** — the reasoning is already
in memory to be drawn, and an index would be a second place saying the same
thing and the one of the two that goes stale.

It pays for itself on real data: one commit came back cited by **five** attempts,
told apart only by their configurations. From the code side that was invisible.

And the link to results needs no mechanism at all, because it is already a
name: `exp/<tree>/<commit>` is both where a commit's journal lives and a study
name, so trials land under it with nothing written to make them.

### A move carries a name somebody chose

The original identifies a move by the integer the store hands out. That works
while you hold it in a variable and stops working the moment you do not — a
second process, a tool call, or the same person a week later, none of whom
remember that the capacity question was `7`.

So a move carries a **name its author chooses**, unique within the tree, beside
the id. This is the one addition to the model, and it is not for a machine:
picking a move up again after a week is the normal case.

### What is drawn, and where the line is

Seeing what worked is half of this, and it splits cleanly. **Deriving is the
framework's**: the layout of the DAG, the standing of every question, whether
two scopes touch, what cites what, which lines are folded because somebody
abandoned them. **Interacting is an app's**: folding what you have read,
clicking through, editing.

The rules the drawing has to get right are knowledge and not taste:

- **Nothing moves, and the walk goes away from where it started.** A position
  is derived from the shape; a position somebody dragged would have to be
  stored, and it is not a fact about the investigation. This is not `git log`:
  an exploration is read from where it began, because what you want to see is
  what came *out* of it.
- **Depth grows to the right and siblings stack downward.** A move carries
  prose, so its card is wide and short, and stacking siblings across would give
  columns two words wide. The record's rail is the other way up — a commit's
  subject is one line — and that is the one difference between the two.
- **A lane per line, never handed out twice.** Freeing a lane when a branch ends
  looks thrifty and stacks three variants into one lane pretending to be one
  history.
- **A parent is centred over its children's span**, not their average — an
  uneven fan drawn on the average leans and looks like it is falling over.
- **Siblings in the order they were made**, never the order a walk arrives in.
- **What could not be reached is still drawn.** A move nobody has hung anywhere
  is work waiting for a place, not a move that hides.
- **A colour is never the only place a finding lives.** `STALE` is written in
  words as well: it is the finding the whole thing exists for, and a palette is
  not something to bet it on.

### Folding is the reader's, and it is not pruning

Two ways a line disappears and they are not the same control. A **pruned** line
comes folded because somebody decided to abandon it, and it says how many it
hides and **why**, in words. Anything else folds because the reader has read it
— no reason, nothing stored, because closing what you have read is not a claim
about the investigation. Without the second, an outline of fifty-seven moves is
fifty-seven rows, always.

And pruning never deletes. A line that did not work is the most reusable thing
an investigation produces.

### What seeding a real investigation caught

Three decisions were written with their scope pointing at the line that
**produced** them rather than the line they abandon. All three read fine in
prose and all three were wrong, and the model said so the only way it could:
pruning folded the branch carrying the paper's headline.

The fix is a fact about the model. **A decision's scope names what is
abandoned, so what is abandoned has to be a move** — and an attempt nobody ever
ran is still a move, and precisely the one needed to be able to say it was
never run.

### What does not earn its place

The original served seventeen routes. Some of them were an in-browser editor
rather than this, and chasing them would be building for a requirement that
does not fit:

- **Formatting source** (`ruff format` behind a route) pins nothing and
  structures nothing. An editor's convenience.
- **A single `check` verb** — parses, linter quiet, graph builds, node runs —
  decomposes into things that already exist: what an edit did is
  `foreseen.changes`, and whether it runs is running it. One verb that bundles
  a linter into the framework is the framework learning what a linter is.
- **A knowledge lake** to export findings to. Named in the original as not
  built, with nothing waiting for it. A hole with no tenant.
- **`/api/health`**, exit codes, `--json`: a server's and a terminal's.

- **Editing is forking**, which looked like the one exception and is not: it
  dissolved into `go`. What it was defending is right — a commit is a version
  that has already been measured, so changing one is wanting another variant
  from here, never touching an existing branch and never rewriting anything —
  and `go` enforces exactly that. The original spliced a file, committed and
  branched because it was driven **from a browser**, where you cannot edit
  files, and called the result a fork. In a terminal the person typing holds the
  checkout: `go`, then edit, then commit. What is left over is the in-browser
  editor, which is the first thing on this list.

And a **fork** in the word's own sense — taking a state and starting a *new
investigation* from it, a second `tree` seeded from a move — is a different
thing that this does not have. It is worth naming and it has no tenant today,
which is the bar.

### Going back to try another idea

The original navigated by clicking, so it needed no verb for this and none was
extracted from it. In a terminal it needs one, and it is the first thing that
pays back the cost of having written the reasoning down at all.

`git checkout` asks for a hash. What anybody actually remembers is the idea:

```
$ somatize-tree go decorr-0.1
```

Go to the commit that attempt ran, on a branch of its own, ready to try
something else from there. Git cannot do this, because git does not know which
attempt that was — the move's name is what makes it reachable, which is why a
move carries one.

**A branch of its own and never an existing one.** A commit is a version that
has already been measured, so arriving at one is arriving to make the *next*
variant, not to rewrite that one. The original said the same thing about
editing and enforced it by working in a worktree nobody could see; here the
person typing is the one holding the checkout, so the honest primitive is a
branch — and unstaged work is a refusal rather than something to carry along.

And the way back matters as much: standing on a commit, ask what it was for.
That is derived from the citations and kept in no index, so it is true the
moment somebody cites it and cannot go stale.

### Where the two layers meet, and it only runs one way

A standing being derived is what lets one **come back**, and the case is the one
the model exists for: a refutation read off a measurement that lied. So a move
citing a commit somebody judged `invalid` is **withdrawn** — what it said stops
counting — and the hypothesis goes back to `open` with nobody saying anything
again. A later `sound` puts it back, because the journal keeps the last word.

`invalid` is deliberately not a `Course`: judging the code wrong is not deciding
where to go next, and the word for it stays in the journal. Which is why this is
the one place the reasoning reads the record, and it reads the **direct** verdict
only — a commit under an invalid one inherits doubt and not a judgement, and
inheriting it would need an ancestry nothing here asks git for.

It reaches **up the DAG**, which running it on a real investigation is what
showed: a finding cites the trial it was seen in and hangs under the attempt,
and it is the attempt that names the commit. Looking only at the finding's own
citation, the rule fired on almost nothing. Walking up can pick up nothing but
an attempt or a finding, since those are the only kinds that cite at all.

Nothing is deleted and nothing is a field: the edge is still written, still
drawn, and says it was withdrawn — a standing that moved on its own has to say
what moved it.

### Read back in names, and laid out from the rows

The store hands out a slot and the slot stops identifying a move the moment
nobody is holding it in a variable — which is exactly what reading one back is.
So the whole read-back answers in **names**: what a move hangs under, where a
scope holds, who says what to whom. The id stays as a field, because it is what
says which of three variants was tried first and no walk recovers that.

It is derived once, in `soma-tree`, and read two ways — `somatize-tree moves`
prints an outline and `somatize.reasoning` hands the same rows to Python. A
second copy of the derivation behind the drawing would be a view that quietly
disagreed with the terminal about what an investigation contains.

Two things came out of using it on a seeded investigation rather than designing
it. A decision hung under nothing and scoped at what it abandons was drawn
**floating**, seven rows from the line it ended — so a decision belongs beside
what it abandons, which is the same rule `decided` already runs the other way.
And a move nobody ran leaves no trace in anything derived from commits, since it
cites none: what folds is worked out over the **moves** and reaches it.

The layout is a pure function of those rows, so **folding is what you hand it**:
the lines in `folds` come folded with how many and why, and handing it none
opens everything. An app that folds what its reader has already read hands its
own list and needs nothing added here — which is where the line between deriving
and interacting actually falls.

### Which half is a command and which is a library

Both, and the line is not taste: **the terminal is for what happens between
runs, the library for what happens inside one, and for looking at it.**

Asking a question, hanging an attempt under it, deciding to abandon a line —
those happen while somebody is thinking, one at a time, with nothing else
running. `note` and `verdict` were already commands; their siblings belong
beside them. The original listed their absence as its own outstanding gap:
*"asking a question or hanging a move under another has no CLI."*

| | where |
|---|---|
| `ask`, `suppose`, `tried`, `found`, `decide` — writing the reasoning | command |
| `hang`, `combines`, rewording | command |
| `go`, and what a commit was for | command |
| `diff`, `log`, `show`, `trials`, `data` — reading the record | command, and already there |
| `moves` — the reasoning as an outline | command, because nine verbs that write with no way to read them back is a tool you cannot check your own typing against |
| `keep` — the invocation half of a version | command, because it is what `tried --ran` cites |
| reporting a result from the code that produced it | library, and already there |
| citing a trial from the finding that reads it | library |
| the reasoning read back: moves, standings, scopes, what cites what | library |
| **drawing** it — the tree, what worked, what did not | library, and a notebook |

A command is also the shape a tool call takes later, so an MCP wraps this with
less between it and the work than an API would.

### Questionnaire

- [x] a question can stand with nothing under it, and reads as pending work
- [x] a hypothesis can be validated and refuted, and a question cannot
- [x] only an attempt and a finding cite the record
- [x] a decision carries a course, and nothing else does
- [x] one move hangs under two parents, and neither is the parent
- [x] `combines` is not `under`, and says an attempt **is** the composition
- [x] a cycle is refused when it is written, not when it is walked
- [x] a move is reached by a name its author chose, from a process that never saw it created
- [x] standing is derived from what was said, and never read from a field
- [x] two edges of opposite sign whose scopes touch are `disputed`
- [x] the same two whose scopes do not touch are `depends`, and not `partly`
- [x] a hypothesis goes back to open on its own when what refuted it is invalidated
- [x] a decision's scope names what is abandoned, and what is abandoned is a move
- [x] an attempt nobody ran can be written down, and be what a decision abandons
- [x] a commit says which moves cite it, without an index saying so
- [x] a citation carries the commit **and** the resolved invocation
- [x] trials land under a version with nothing written to link them
- [x] nothing is ever updated: saying something claims the next slot
- [x] a line that was abandoned is still readable
- [x] what is suspect below an invalid commit is worked out, never stored
- [x] a walk sees every branch, not the ancestry of one tip
- [x] the reasoning can be drawn from what is stored, with nothing run again
- [x] a move nobody hung anywhere is drawn
- [x] pruning folds and never deletes, and says how many and why
- [x] folding what you have read writes nothing down

Going back:

- [x] going to a move by name lands on the commit its attempt cited
- [x] on a branch of its own, and never on one that already exists
- [x] unstaged work is a refusal, not something carried along
- [x] a move that cites no commit cannot be visited, and says so rather than guessing
- [x] an attempt citing a commit **and** a resolved invocation restores both halves
- [x] standing on a commit says which moves cite it, and says so when none do
