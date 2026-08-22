# soma-next

A re-derivation of [Soma](/mnt/cluster/projects/soma) written by hand, one use
case at a time. The goal is not a better design: it is **authorship**. A system
you designed yourself you can hold in your head even with three hundred types in
it; one you did not, you cannot — and that is why the original, which works and
is published, stopped being maintainable by its author.

Secondary goal, non-negotiable: learn Rust by writing it.

## Rules

**Vertical slices, not crate layers.** Nothing is written without a real
consumer *today*. Building a whole crate before anything uses it is how the
original ended up with 14 traits with a single implementor and 2 with none.

**Type taxonomy** — the rule that avoids the `dyn Trait` soup:

- **enum** when the set is closed and you know it. It is the domain's language,
  and the compiler keeps track when a variant is added.
- **trait** only when the implementation is supplied by *someone else* (the
  user). If you cannot name two real implementors today, it is a struct. And two
  traits with a same-named method in scope make that name unusable
  (`error[E0034]`), even when the signatures differ.
- **struct with typestate** for the invariants: so an impossible state cannot be
  written, rather than validated.

**One file per type.** The type, its inherent `impl`s and the errors its
operations produce, together. An inherent `impl` is **never** split across
files: if you feel like splitting it, the operation probably was not a method of
that type (that was the case of `run`, which needed graph *and* catalog *and*
input, and ended up a function in `execution.rs`).

A family of types gets a **folder with a `mod.rs`** inside it — never a
`family.rs` sitting next to a `family/`. The folder is created when the family
already has members, never in anticipation: nested folders per concept are how
`soma-runtime/src/optimizer/sampler/mod.rs` happened. The tests mirror the same
shape.

What cannot be sustained in Rust, and is worth knowing if you come from Java:
behaviour does not belong to a type, it belongs to the **(type, trait)** pair.
`impl Filter for PyFilter` lives in another crate. Trait `impl`s scatter out of
necessity and are looked up by the trait, not by the type.

**The tests live outside `src/`.** They are another crate: they only see the
public API, so they cannot pass by leaning on anything private. One test binary
per type (`tests/unit/main.rs` with one `mod` per module), not one per file.

**The core knows nothing about Python.** `#[pyclass]` does not go into `core/`.
The moment a core type carries one, it can no longer be used without an
interpreter loaded. `python/` translates; if a domain rule ends up written
there, it is in the wrong place.

## The original as an oracle, not a template

`/mnt/cluster/projects/soma` is **frozen**: consultation and bugfixes, zero
features. Its 31,994 lines of tests are the executable specification of what has
to be true.

They are read as a **questionnaire, not as code**. An old test mixes two things:
what has to be true (knowledge, irreplaceable) and how it is invoked (design,
which is decided here). Copying them as they are would recreate the old API,
which is exactly what we do not want. From each file the list of guarantees is
extracted and each one answered with whatever call shape we decide.

**The DSL is the normal way of writing a graph**, in both languages:
`Graph.somatize(a >> (b | c) >> d)` in Python, `(a >> (b | c) >> d).somatize()`
in Rust. `node()`/`edge()` remain for when the topology is built in a loop or
comes from outside.

**A node is one single thing**, in both languages: `forward(input, ctx)` takes
what arrived along the edges and returns what it produced. There is no "filter"
type and "step" type, and after CU18 there is no return type that could tell them
apart either. Whatever a node takes to answer — a retry, a model, three rounds of
something — happens **inside it**, holding whatever client that takes.

## Commands

```bash
# The Rust side needs one thing from Python: an interpreter PyO3 supports, and
# the system one is ahead of it. `.python-version` says 3.13 and `uv run` puts it
# on the PATH — there is no environment to activate and none to keep.
uv run cargo test --workspace
uv run cargo clippy --workspace -- -D warnings && uv run cargo fmt --all -- --check

# The Python side still needs `mos`, because that is where torch is.
conda activate mos
cd python && maturin develop && python -m pytest tests/ -q

# A real cluster: containers, one per host. Opt-in; the images build themselves
# the first time. `SOMA_CLUSTER=build` forces a rebuild, which is what you want
# after touching `python/src` or the Dockerfile. Both live in `tests/cluster/`.
SOMA_CLUSTER=1 python -m pytest tests/cluster -q
docker compose -f python/tests/cluster/docker/compose.yaml --profile gpu build worker-gpu worker-gpu-b
```

`maturin develop` is not optional before `pytest`: the Python tests run against
the **installed** extension, so a change in `python/src/` that is not rebuilt
means the suite is green about code that is not the code.

## Status

Nineteen use cases closed: the graph, the engine, the plan, the fans, the DSL, a
single node contract, `Opaque`, the waves, the device, training, the distributed
worker, the cache, training the half that is not here, federated rounds, the
grain of an item, the study, handing it out of a folder, and a graph that draws
itself. A graph is declared with `>>`, `|`, `.on("cuda:0")` and `.cached()`,
executed in Rust, spread across processes with `.at("worker1")`, trained from
outside with `soma_next.torch.Trainer` — including the part of it that runs on
another machine, where a **trainer travels to stand beside the node** and the
node is never asked to know it — and **printed in a notebook**, where the figure
shows what runs at once and what leaves the machine. See `docs/use-cases.md`.

**Five orthogonal facts**, and confusing them is the easy mistake: `Graph` says
**what** exists, `Catalog` **who** executes it, `Placement` **where**, `Plan`
**when**, and `Memory` **what is remembered** of each node. The device
deliberately does not live in the plan.

**Three holes, and the core provides them without filling any**: `Node` is the
user's, `Transport` carries a slice elsewhere, `Keeper` hashes a recipe and keeps
what it names. There were four: `Driver` served what a suspended node asked for,
and it went — after eighteen use cases it had **no consumer outside the tests**,
its own docstring said it was there to keep the agentic layer out of the core,
and a hole with no tenant is what this project exists not to build. What is kept
is the channel: `Ctx` is where whoever executes hands a node what it knows, so an
agentic layer that wants something injected puts it there and **no node signature
changes**. The core still has
no dependencies. `transport` has two of its own, filled from `python/`:
`Provision` turns an artifact into a catalog, and `Codec` writes down what only
exists in one process — so an `Opaque` crosses a wire, and what does not is the
one nobody registered a codec for.

**Where a graph gets cut is the pair `(host, trained)`**: `.at()` already said
the first half and the graph owns it; the second is a fact of the training run,
so `stages` is **told** and no node is ever asked. A trainer lets go of the
activation exactly as a cable does, and a backward pass is a `forward` of the
transposed stage — no new variant in `Plan`, nothing new on the wire.

**Three levels, and none knows the one above exists**: the graph is a network —
the scale of one `forward` —, the `Trainer` is a training run — the scale of an
afternoon —, and N training runs are a Python list, not a type and certainly not
a graph. A graph earns its keep when there are dependencies to declare.

The two pendings CU14 left are closed: an `Opaque` **crosses a wire** with a
codec in front of it, so an activation travels as bytes and the same node is
handed the same shape wherever it runs; and `Trainer(every=N)` makes **a group of
steps into one update**, with whatever trains itself elsewhere making the same
group out of the same steps.

**CU15 closed**: a training run exports its weights node by node, `fedavg` is a
**function**, and a federated round is a `for` — level 3 has no type and that is
on purpose. Across machines it is a folder they all mounted and nothing else:
`Store` opened by hand, `claim` to hand work out, and `gather` for the round,
where **whoever finds it complete claims the averaging** so there is no
coordinator to keep alive. No port, no protocol, no `Plan::Remote`; Slurm
distributes.

**CU16 closed**, and it split what the plan had joined: micro-batches are level 2
—the batch is the caller's, so `torch.chunk` reaches it— while `.mapped()` is the
engine's, because caching item by item has to **name** each item and a name comes
from its **content and not its place**.

**CU17 closed**: `study/`, the first crate with **no dependencies at all, not
even the core's** — level 3, the one that has no type. Three families of the same
shape, each an enum of structs: `Partition` (where to cut), `Sampler` (where to
look) and `Pruner` (when to give up). The line between the languages is drawn by
**shape and not by language**: Rust keeps what is pure, deterministic and
hashable; the loop stays in Python where torch is, and **no callback crosses** —
the original's `TrialExecutor` has one implementor and it is a closure wrapper.
So nothing is asked of a `Trainer`: a pruner answers and the loop stops calling.
And `ask` is a function of the **index**, not of what was asked before, so a
machine that claimed trial 7 from a shared folder derives it without replaying
six.

**CU18, first half**: those pieces now search **together**. `take`, `report`,
`finished` and `curves` are functions over a `Store` — Python, like `gather`,
because touching the folder is not pure — and handing out work costs **no
message**: a trial is a number, `ask` is a function of it, and `claim` settles
who gets it, so the state *is* the queue. Two samplers were added that cover the
space **for every prefix** rather than in expectation, which is what stops two
machines proposing neighbours: `Halton` (arithmetic, no ceiling) and `Sobol`
(Joe & Kuo's table, 32 knobs). The record keeps the configuration as text beside
the score, so a sampler's whole history is **one scan and zero fetches**; only a
pruner's curves cost a fetch each. And a guided sampler spread out now knows what
the others are holding: `ask` takes a score that **may be absent**, and absent
means running — those points are kept away from without voting on how big the
good pile is, which is the part that measurement showed a made-up bad score gets
backwards.

And the first level-3
test with a real pipeline under it: `tests/cluster/test_searching.py` searches
hyper-parameters over real SMS messages with the graph cut across containers —
tokenising where there is no torch at all — **and** the study cut across
machines. Two distributions at once, and they are not the same one.

**CU19 opens observability, and the first thing it did was split it in three**:
the declaration **drawn** — a graph can be drawn having never run —, the record
of what **happened**, and the **diagnosis**, which is an opinion about the record
and not a fact in it. The original keeps all three in one enum of 37 variants,
which is how `NodeStarted` ends up beside `HealthFlag`. The invariant that makes
the split real is a test: **a diagnosis has to be reproducible from the stored
record, without training again.** CU19 is the first of the three only, so it
touches no event, no bus and no store — and a bus is not refused, only deferred
to where it earns its place.

What is drawn is the **plan**: a `Wave` is what runs at once and a `Remote` what
leaves the machine. The layout needs no heuristic because `Plan` is a tree. But
`decompose` falls back to a flat `Sequence` when a graph is not series-parallel,
and there the nesting stops saying who feeds whom — so **the boxes say *when* and
the arrows say *what feeds what***, and the `N` (`a→c`, `a→d`, `b→d`) is the test
that keeps the figure honest.

See the distribution report for the full order.
