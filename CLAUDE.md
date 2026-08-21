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

**A node is one single thing**, in both languages: `forward(input, ctx)` returns
`Done(value)` or `Await([requests])`. There is no "filter" type and "step" type
— a filter is a node that always answers `Done`, and that is said by its
transition, not by its type.

## Commands

```bash
conda activate mos                  # PyO3 does not compile outside this env
cargo test --workspace
cargo clippy --workspace -- -D warnings && cargo fmt --all -- --check
cd python && maturin develop && python -m pytest tests/ -q

# A real cluster: containers, one per host. Opt-in; the images build themselves
# the first time. `SOMA_CLUSTER=build` forces a rebuild, which is what you want
# after touching `python/src` or the Dockerfile.
SOMA_CLUSTER=1 python -m pytest tests/cluster -q
docker compose -f docker/compose.yaml --profile gpu build worker-gpu   # once, 11 GB
```

`maturin develop` is not optional before `pytest`: the Python tests run against
the **installed** extension, so a change in `python/src/` that is not rebuilt
means the suite is green about code that is not the code.

## Status

Fourteen use cases closed: the graph, the engine, the plan, the fans, the DSL, a
single node contract, `Opaque`, the waves, the device, training, the distributed
worker, the cache and training the half that is not here. A graph is declared
with `>>`, `|`, `.on("cuda:0")` and `.cached()`, executed in Rust, spread across
processes with `.at("worker1")`, and trained from outside with
`soma_next.torch.Trainer` — including the part of it that runs on another
machine, where a **trainer travels to stand beside the node** and the node is
never asked to know it. See `docs/use-cases.md`.

**Five orthogonal facts**, and confusing them is the easy mistake: `Graph` says
**what** exists, `Catalog` **who** executes it, `Placement` **where**, `Plan`
**when**, and `Memory` **what is remembered** of each node. The device
deliberately does not live in the plan.

**Four holes, and the core provides them without filling any**: `Node` is the
user's, `Driver` serves what a step asks for, `Transport` carries a slice
elsewhere, `Keeper` hashes a recipe and keeps what it names. The core still has
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

Next up: **CU15, federated**, which is what a training run exports rather than
what it does; and micro-batches, which is where the grain per item (`.mapped()`)
joins them. See the distribution report for the full order.
