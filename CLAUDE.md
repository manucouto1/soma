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

# A bucket, for the half of the store's contract that needs one. Opt-in the same
# way and with the same handshake on both sides: `SOMA_S3` set means there is one.
docker compose -f store/tests/docker/compose.yaml up -d
SOMA_S3=http://127.0.0.1:9000 uv run cargo test -p soma-next-store --features s3
SOMA_S3=http://127.0.0.1:9000 python -m pytest tests/test_bucket.py -q
```

`maturin develop` is not optional before `pytest`: the Python tests run against
the **installed** extension, so a change in `python/src/` that is not rebuilt
means the suite is green about code that is not the code.

`examples/` holds ten notebooks — declaring a graph, watching a run, training,
a study, the health of a network, one problem end to end, a real architecture
diagnosed in **problem → symptoms → solution → healthy** cycles, what can be
said before a step is taken, a fleet of machines, and where the data comes from
— **with their outputs stored**, so opening one shows what it does. Every
figure is kept twice: the plotly JSON for a live viewer and a PNG for a static
one, which is what `PLOTLY_RENDERER="plotly_mimetype+png"` decides at execution
time. Re-run them with `nbclient` when the Python API moves — `nbconvert` is not
installed and is not needed, it is only the CLI around it. Writing them is how
two real bugs were found.

## Status

Twenty-seven use cases closed: the graph, the engine, the plan, the fans, the DSL, a
single node contract, `Opaque`, the waves, the device, training, the distributed
worker, the cache, training the half that is not here, federated rounds, the
grain of an item, the study, handing it out of a folder, a graph that draws
itself, the record of what happened, the health of a network, what can be
said before a step is taken, the machines it ran on, where the data comes from,
not running what nobody needs, what an edit did before paying to find out, and
what a node was built with.
A graph is declared with `>>`, `|`, `.on("cuda:0")` and `.cached()`,
executed in Rust, spread across processes with `.at("worker1")`, trained from
outside with `soma_next.torch.Trainer` — including the part of it that runs on
another machine, where a **trainer travels to stand beside the node** and the
node is never asked to know it — and **printed in a notebook**, where the figure
shows what runs at once and what leaves the machine. See `docs/use-cases.md`.

**Five orthogonal facts**, and confusing them is the easy mistake: `Graph` says
**what** exists, `Catalog` **who** executes it, `Placement` **where**, `Plan`
**when**, and `Memory` **what is remembered** of each node. The device
deliberately does not live in the plan.

**Four holes, and the core provides them without filling any**: `Node` is the
user's, `Transport` carries a slice elsewhere, `Keeper` hashes a recipe and keeps
what it names, and `Watcher` is told what happened.

There have been four before, and the one that went is the lesson: `Driver` served
what a suspended node asked for, and after eighteen use cases it had **no
consumer outside the tests** — its own docstring said it was there to keep the
agentic layer out of the core, and a hole with no tenant is what this project
exists not to build. `Watcher` arrived with two implementors in two crates on the
first day, which is the bar. What `Driver` left behind is the channel: `Ctx` is
where whoever executes hands a node what it knows, so an agentic layer that wants
something injected puts it there and **no node signature changes**. The core
still has no dependencies. `transport` has two of its own, filled from `python/`:
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

**A store is a directory or a bucket**, and that is the second implementor of
the trait — `Store.on_bucket(...)`, behind a `s3` feature that is off by default.
It went in front of CU20 because only one of the three uses of a store demands a
shared disk: a cache and an artifact degrade to a miss, but handing out work does
not. A bucket and a directory are **the same store** — the same layout and the
same JSON — and what is different is that `claim` is a conditional PUT. An
endpoint that takes `If-None-Match: *` and writes anyway would give every trial
to every machine and say nothing, so `Bucket::at` spends **two round trips
proving it does not** before handing the store over. Nothing above learned a new
word: `take`, `report` and `gather` never asked what kind of store they had.

**CU20 is the record of what happened**, and the requirement that shaped it was
not "a log": that somebody training in a notebook, with half the graph on other
machines, **keeps being told** what is going on. So `run()` cannot hand the facts
back at the end — there is a `Watcher`, injected like the `Keeper`. **Emitting is
synchronous and delivering is the implementor's**, which is how live costs no
runtime: an `async` there would be `async` in every caller and would drag `Store`
with it.

An enum of facts is fine and the original's 37 variants are not the mistake — the
mistake is that they are **three vocabularies in one**. Each level keeps its own:
the engine's in `core/src/fact.rs`, a training run's (`loss`, `updated`) where
the loss is, a trial's on disk since CU18. **They do not meet in Rust, they meet
in the record** — a fact is emitted as an enum and written as `(kind, pairs)`,
which is the shape `Meta` already had, so what you print is what you would find
in the store.

And what happens on another machine **comes back down the connection that was
already open**: `dispatch` was blocked in `recv` anyway, so `Answer` gained one
non-terminal variant and reading one answer became reading until one is terminal.
No port, no second connection, no bus. The rule: **where a connection is open,
facts come back down it; where there is none, they go to the store and whoever
wants them scans** — which is what CU18 was already doing. A relay attributes
nothing: the client wraps what arrives, because the host's *name* is the graph's.

One record per `forward`, at `run/<id>/<n>`. A loss is computed after the forward
that made it, so it goes into the record that closed last, rewritten — and there
is no guessing, because the two vocabularies come through different doors.

**And `soma_next.record` reads it back**, functions over a `Store` like `gather`
and `take`. It is a **price list**: `runs`, `forwards` and `curve` are one scan,
`facts` is one fetch and `nodes` a fetch per forward — so everything a progress
view asks for is free and the per-node breakdown is asked once, not once a step.
`Recorder(..., summarising=["loss"])` is what puts a curve on the free side, and
which kinds those are is the caller's: the store still does not learn what a loss
is. `curve_costs` says whether it scanned or fetched, because a reader that is
quietly a thousand times slower is worse than one that says so.

**And a run is drawn**, which was asked for in the same breath as the reader:
`progress` and `spent` read a store, `Live` is handed facts as they happen, and
the two fill **one drawing function** — they can, because a fact read back is the
very dict a watcher was given. A live view and a report written twice are two
things that slowly stop agreeing.

The colours are **one table** in `soma_next._theme` and the graph of CU19 moved
onto it: a library whose graph is light and whose curves are dark is two
libraries. One fact per channel still holds — hue says *where*, never
good-or-bad, and the only red marks a `forward` that broke, which is a fact.
The smooth line is a **centred rolling mean** and not a spline: a spline through
measured points invents the values between them, and a loss dipping below a
minimum that never happened is a figure that lies. And an edge that would cross a
box it does not belong to is **routed around it**, one lane each — an arrow drawn
over a node reads as an arrow into it.

A study is drawn from the library and not from a notebook: `table`, `influence`
and `coordinates` in `soma_next.study`, with `importance` — **Spearman's ρ**,
which the original names as fANOVA-deferred and never wrote — beside the other
readers. `coordinates` is hand-drawn out of splines because plotly's `Parcoords`
only draws straight segments; it trades brushing for a trial reading as one
curve. In all three, pruned and finished are never ranked together.

**CU21 is the third row: the diagnosis, and it says it is an opinion.** A crate
`health/` with **no dependencies at all, not even the core's** — numbers in,
flags out, no measuring and no clock — which is what makes CU19's invariant a
test rather than an aspiration: *a diagnosis has to be reproducible from the
stored record, without training again*. Change a bound and ask again; the record
has not moved, and an argument about a threshold costs a scan.

It inherits the original's taxonomy with its two findings, which are knowledge
and not design: **`DEAD` and `SATURATED` read the maximum** over a window and
never the mean — a layer that dies one step in four is dead — and **dormant is
not dead**. It adds three from the literature: `STALLED`/`OVERSTEPPING` from the
update-to-weight ratio, which the original measured and never said anything
about and which lands a healthy layer at ~1e-3; and `LOSING_PLASTICITY` as a
**conjunction**, because weights growing or units going quiet alone is a network
that is training.

`NARROWING` is in the vocabulary and **off by default, because it was measured
and the measurement did not support it**: the published monitor's certificate is
the deviation from a healthy baseline and one run has none. See
`health/tests/narrowing.py`. The metric is recorded and drawn; the alarm was not
invented.

Measuring is `soma_next.torch`'s: `Trainer(..., auditing=True)` hooks the nodes
and emits `health` facts through the same `watching=`. Thresholds never go near
it — baked into the measurement, they would make an argument cost an afternoon
of GPU. `Audit(inside=True)` looks **inside** a node, because a node is often a
whole architecture and *this node is unhealthy* is not an answer when it is
twenty layers; findings are keyed `node.path.to.submodule`.

**A node is opened up, and an architecture is a graph**: `architecture(g, x)`
traces what a node is made of — `fx` where it can, because it sees the
operations that are **not** modules and a residual connection is exactly one;
a real forward where it cannot, saying so, because a residual that is missing
looks like a residual that is not there. The unit is the **node**, since a node
holding two modules composes them in its own `forward`.

`g.figure(inside=...)` draws its box as a **frame** — the shape a `Wave` and a
`Remote` already are — and lays the inside out **by what feeds what**, so a skip
runs down a gutter and enters from the side. The rules that make it readable: a
kind, not a class name, decides the **silhouette** — a convolution is a
parallelogram, a recurrent cell has a tab, an attention block has its corners
cut and says what is in it, a normalisation is a capsule, a non-linearity is
pointed, and anything that changes the width is tapered the way it goes; a
composite everybody recognises is one box and `depth=` opens it; blocks that are
the same block collapse to `×N`, and when the block is **more than one layer**
that `×N` goes on a frame **around** them rather than on each of them — four
encoder layers opened up are eight boxes each saying `×4`, which is the count
said eight times and the block said none; something that runs identical lanes
at once — the heads of an attention block, the groups of a convolution — is
drawn with plates behind it and **never** as separate boxes, because torch packs
the heads into one projection and four of them wired together would be a graph
nobody built; and the **shape is written on the layer**, because that is the
only thing that makes a bottleneck a picture.

Findings are coloured by **family** — numeric, signal, activation, step,
capacity, data — with a legend of the ones on the figure, because six alarms
that all look the same are one alarm.

Every number on a layer **says what it is**: `4 batch · 16 steps · 24 dim`, not
`4×16×24`. The batch is checked rather than assumed — the caller knows how many
rows went in — and a layer that did not change the shape keeps the names of the
one that did, so a `BatchNorm1d` in a convolutional trunk says `ch` and `len`. And `overlaid(..., inside=...)` puts each
one on the layer it is about: the audit's scope and the drawing's are now the
same scope, which is what makes *what is measured has a box* true rather than
hopeful.

**Where is a question the graph answers**: `overlaid` marks the ill nodes — and
the ill **layers inside them** — on the figure, and health gets a **channel of
its own** — the fill goes on saying where
a node runs and the outline turns red, because on a graph over three machines
*where does this run* is the answer somebody came for. `alerts` is the loud one,
cards a cell shows on its own. And `gantt` is the timeline: every fact carries
how far into the `forward` it began, so a `Wave` draws as overlapping bars and a
remote slice sits inside the round trip it arrived under — an offset into a
slice is a fact about the slice, and two wall clocks would not have composed.

**And a third question, which is not about the network at all**:
`soma_next.data.contribution` shuffles one input and scores again, and the drop
is what that input was worth. `health` asks whether a network is **learning**;
this asks whether it is learning **what you meant**, which no amount of looking
at a gradient will ever say. It exists because of a real project — symptom
channels for a mental-health condition, months on the architecture, and the
signal was in the self-disclosure. `IGNORED_INPUT` is the finding that would
have said so in an afternoon; `SOLE_RELIANCE` is the other end of the same
worry. **Shuffled and not zeroed**: a zero is a value, and what is being asked
about is the correspondence with the answer.

**CU22 is the static half, and it costs seconds rather than an afternoon.** CU21
asked whether a network **is** learning, which needs it to have been learning;
this asks whether it **can**. `soma_next.torch.probe(g, x)` is **one `forward`
that was recorded and never trained** — literally `run/<id>/0`, through the same
`Watcher`, under the same keys — so `diagnose`, `seen`, `profile`, `flags`,
`where`, `overlaid` and `alerts` all read it with **no new code at all**. That is
CU20's decision to write a fact as `(kind, pairs)` being paid back: a probe is a
new producer and not a new vocabulary.

It measures three things and **none of them is a gradient norm**: at
initialisation there is no loss, so a parameter gradient would be taken against a
target somebody made up and would land in the very field the audit fills from a
real one. The backward direction is `jacobian_gain`, a ratio that needs no
target. The cost is two forwards and `k` backwards **over the whole network** —
every layer reads its own `Jᵀv` off the same pass — and the second forward is
`architecture`'s, because the probe takes its scope from **what the figure will
draw** rather than walking the modules itself. Those two stop being the same set
once `fx` has had its say, and a finding on a layer with no box lands nowhere.

Of the three, **only one earned an alarm**. `MISSING_NORMALISATION` fires and it
is a conjunction whose structural half lives in the *measurement*: a
normalisation resets the reference and reports no gain of its own, because
changing the scale is its job. It is also **one-sided**, which the measurement
decided and the design did not — a stack whose signal arrives five
ten-thousandths of the size it went in trains as well as a healthy one, because
Adam is scale-invariant per parameter. The two Jacobian numbers raise nothing:
they **rank** and do not **separate**, and the network with the tighter spectrum
was the one that failed. Which leaves the rule this slice is really about:

> **What separates is a runaway. What ranks is a proxy.**

The forward scale is geometric — it stays put or leaves by decades, with nothing
in between to be wrong about. Everything continuous is a ranking, and a ranking
belongs at level 3 where a number only means something next to another
candidate's. Which is exactly where the five zero-cost proxies went:
`soma_next.torch.proxies` is a **cheap objective** a study's loop scores with,
never a `Flag`, and `health/tests/proxies.py` asks the only question worth asking
of one — *does it beat counting parameters?*

**CU23 is workers and jobs, and the first thing it decided was what not to
build.** A machine does work here in exactly two ways — a worker serving slices,
which the client is talking to right now, and a machine claiming trials from a
folder, which CU18 already watches by *is it still writing* — and there is no
third case, so there is **no registry and no heartbeat**. The original needs a
`WorkerStatus` with a `last_heartbeat` because it has a coordinator; CU15
removed that.

What was missing was a **view**: `fleet(store, run=...)` turns the record the
other way up, because it is written run → `forward` → node and *where* is an
attribute. The column that earns it is `waiting_us` — the round trip **minus**
what ran over there, which is the wire and the queue — and neither half of that
subtraction belongs to a node.

Plus the half no record can derive: how loaded a machine is. The worker says it,
in **a vocabulary of its own** in `transport/`, crossing as
`Fact::Said { kind, pairs }` — a carrier and not a vocabulary, so the core never
learns what a load average is. That cost **nothing on the wire**: `Answer::Saw`
already carries a `Fact` and the engine already wraps what comes back in
`Elsewhere`, so a reading arrives saying which host without a line attributing
it. Read and never judged — a bound on it would belong in `health/`.

The **idle** machine is in it too, and its pipe follows the same rule rather
than an exception to it: an idle worker's connection is one nobody is reading,
so `--reporting SECONDS` writes to the store on a clock — **one name per machine,
rewritten**, which is CU18's shape and makes `quiet_s` a scan with no fetches,
measured writer against writer. It files under what the machine calls *itself*,
because `w1` is the graph's word and a worker does not know it; the two names
meet only on a reading that came down a wire, and `fleet` joins them there.

**CU24 is where the data comes from, and it built no layer.** A source is a
**node** — there is no `Source` trait, because a second one with a method that
does what `forward` does is a hole with one tenant and the `E0034` the rules
warn about — so `.at()`, `.cached()`, the record and the figure reach a dataset
with nothing written for them. What changed is what the graph is **handed**: not
a batch but a **coordinate**. Measured, release build: naming a 19 MB batch
costs 121 ms on every step, hit or miss, because a cache has to look at all of a
value to name it — and a span costs 0,027 ms. There is no faster hash to reach
for; the answer is not to weigh the batch.

The other half of the name was free: a name in a `Store` resolves to a digest
and the digest **is** the hash of the content, so a source states its version
with one lookup and no bytes, through `Memory::freeze` — the call made twice on
purpose. That closed a silent bug: a source declared `.frozen()` looked exactly
like a tokenizer, so its version stayed out of the key and two datasets shared a
name.

**Arrow is the type and polars is the tool**: `arrow` + `parquet` in `data/`,
and whoever wants expressions brings their own engine rather than charging 370
crates to the worker that only tokenizes. **No runtime came in** — zero `tokio`
— which is exactly why SQL is not in it: every driver worth using carries one,
and `Store` is synchronous on purpose. `Ipc` is the **second implementor of
`Codec`**, from another crate, so a frame is kept and a frame crosses a wire.
Which leaves the sentence the slice is about: **the difference between training
and deploying is how many rows the frame brings** — 4096 from a folder of
parquet, one from a topic, the same graph and no second code path. A span is a
position and a position can be asked for twice, so a stream source is *settled*
and what moves is which spans exist, not its state.

**CU25 came out of a question about the notebook**: if the head is cached,
should what feeds it be skipped? It should. `key_for` already promised *the name
this node's output will have, before it has one*, so the engine names the whole
plan with nothing executed, asks `Keeper::present` — a scan, no fetches — and
works **backwards** from the leaves: a node whose answer is kept does not need
its inputs. It gives up towards keeping a node in the two places it cannot
foresee, a `.mapped()` node and one with no key; a slice nobody needs is **not
sent**; and `Fact::Spared` says it out loud, because a node missing from a
record cannot be told from one that was never in the graph. The notebook caught
the regression no test could: the pre-pass named the root and the walk named it
again, so the batch was hashed twice and asking early cost exactly what it
saves.

**CU26 keeps the names CU25 threw away.** The pre-pass already names the whole
plan with nothing executed, and that answer is worth having **without** the run:
two versions of one graph name a node differently exactly when its recipe
changed, so `soma_next.foreseen.changes(before, after)` says what an edit did
— *did I invalidate the encoder, or only the head?* — for the price of two
hashes.
The first shape was a partition of the nodes and one run killed it: edit the
encoder's code **and** salt the head, and the head both recomputes and
recomputes **from a stale answer**. A partition carries one of those two, and it
carries the reassuring one. So it answers `{node: [finding, ...]}`, which is
what `health` already answers with and for the same reason.

**And a name that moved says which part of it moved**: `CHANGED` is the shape —
the class, and who feeds it —, `RESETTLED` is another checkpoint and `SALTED` is
a bumped salt. The split came from the other end of the wire: a sibling slice
asking *what did the code do* reads a re-freeze as a false positive, while *does
my cache still hold* reads it as the cache being right. Both are true, so the
finding says which and each reader takes the part it came for — **weights belong
to a version, they are not a version**. And `changes` takes a `Graph` or a
`snapshot` of one, because two versions of a module do not coexist in an
interpreter: comparing two commits is comparing what was written down.

The finding this exists for is `STALE`, and it is CU13's decision paid for
rather than undone: the fingerprint of the code is deliberately not in the key —
a cosmetic refactor must not invalidate half the store — which means editing a
`forward` renames nothing, and a diff that only compared names would answer
*nothing changed* to the very edit being asked about. So the fingerprint is
looked at here, where it is an **opinion and not an invalidation**: `STALE` says
*you should have bumped the salt*, and `SUSPECT` is everything under it, which
goes on being fed the old code's answer whatever became of its own name. And
`UNVERSIONED` is what the notebook found: a class defined in a **cell** has no
source to read and so no version, so the first draft answered `{}` — *nothing to
report* — to an afternoon of edits, in the one place the question gets asked
most. Its scope is the scope a version is recorded at, which is what is kept.
Neither `names` nor `changes` reads or writes anything — a store is only where
the hash comes from — and the input cancels out of every comparison, so asking
costs nothing and skips the 121 ms of weighing a batch.

**CU27 came out of CU26 and is not a diff at all**: `Embed(512)` and
`Embed(64)` are one class, one identity and **one name in the store**, so the
second run was handed the first one's answer with no error and no warning. The
same failure CU13 refuses for an unhashed checkpoint, through the door that
check does not cover — `_check_it_was_obeyed` asks for a digest only from
something with `state_dict`, `parameters` or `version`, and a node whose
behaviour is a number in its constructor answers none of them. So the key
gained a part: `H(identity, declaration, state, keys above)`.

The lesson the first attempt paid for: **what a node holds is not what it was
built with**. Reading `vars(obj)` put `calls` in a key and the encoder ran
three times instead of once — a node that counts, caches a client or moves a
tensor has attributes that move *while the graph runs*. So it is captured at
`__init__` by `Node.__init_subclass__`, bound against the signature with
defaults filled in, so `Layer(64, 32)` and `Layer(in_=64, out=32)` are one
declaration.

And two ways to be wrong that are **not symmetric**, because a key is computed
on the client and again on a worker: *unstable* (one declaration, two texts —
an address) costs a cache that misses forever; *lossy* (two declarations, one
text — a truncated tensor, or an address scrubbed to `<Helper>`) costs the
wrong value in silence. Neither is accepted, which is why scrubbing is not the
answer. What can be neither raises `CannotDeclare` and the graph is refused
before the first node with the attribute named. The rule that looks right and
is not: a test on the **type** — a `list` of address-bearing objects has
`list.__repr__`, which is defined, and the addresses come through from inside.

See the distribution report for the full order.
