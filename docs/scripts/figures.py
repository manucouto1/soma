#!/usr/bin/env python3
"""Draw the figures the hand-written pages show, and commit them.

Twenty-one pages described a framework whose argument is that it draws itself,
and not one of them showed a drawing. The tutorials have figures because they
come from executed notebooks; everything else said *this returns a plotly
figure* and left the reader to imagine it. On `Start here`, which is where
somebody decides whether any of this is for them, that is the whole page
failing at its job.

**Committed, and for the same reason `python-surface.json` is.** Drawing these
needs torch, plotly and kaleido; the site builds on a runner with a stdlib
`python3` and nothing else. So the PNGs live in git and this script is how they
are made again:

    python docs/scripts/figures.py            # all of them
    python docs/scripts/figures.py study      # just one group

There is no `--check`. A figure is not a hash: plotly and torch move, fonts
move, and a byte comparison would go red for reasons that are not the drawing.
What guards these instead is astro, which fails the build when a page points at
an image that is not there — so a figure can go stale, but it cannot go
missing.

Every seed is fixed, so running this twice gives the same pictures.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _output import rewrite  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "src/assets/figures"

# Wide enough for the prose column at two-times scale, and short enough that a
# figure does not push the paragraph explaining it off the screen.
WIDE, TALL, SCALE = 1000, 460, 2


def drawn(
    figure, name: str, into: dict[str, bytes], height: int = TALL, width: int = WIDE
) -> None:
    """One figure, as the PNG a static page can show.

    Sized per figure rather than once: a graph of six nodes in a 1000-wide
    canvas is half a picture and half empty ground, and a thirty-row table cut
    off mid-row reads as a page that broke.
    """
    into[f"{name}.png"] = figure.to_image(
        format="png", width=width, height=height, scale=SCALE
    )


# ── The graph, drawn ─────────────────────────────────────────────────────────


def a_graph(into: dict[str, bytes]) -> None:
    """The quickstart's own `spread`, and the `N` that is not series-parallel.

    The very graph that page declares, class names and all — a drawing of a
    different one would be decoration, and this one is the thing being talked
    about. It is also the graph that carries every suffix at once: `.at()`,
    `.on()`, `.cached()`, `.frozen()` and `.mapped()`, which is what makes one
    picture worth the four paragraphs above it.
    """
    from somatize import Graph, Node

    class Tokenize(Node):
        def forward(self, text, ctx):
            return [float(len(word)) for word in text.split()]

    class Embed(Node):
        def __init__(self, scale):
            self.scale = scale

        def forward(self, counts, ctx):
            return [n * self.scale for n in counts]

    class Score(Node):
        def forward(self, values, ctx):
            return sum(values) / len(values)

    spread = Graph.somatize(
        Tokenize().named("tokenize").at("worker1").mapped()
        >> Embed(0.5).named("embed").on("cuda:0").cached().frozen()
        >> (Score().named("strict") | Score().named("loose").at("worker2"))
    )
    drawn(spread.figure(), "graph-declared", into, width=760, height=430)

    # `a→c`, `a→d`, `b→d`: the minimal shape with no series-parallel tree, so
    # the plan falls back to a flat `Sequence` and the nesting stops saying who
    # feeds whom. It is the test that keeps the drawing rules honest — the
    # boxes say *when* and the arrows say *what feeds what*.
    n = Graph()
    for who in ("a", "b", "c", "d"):
        n.node(who, Tokenize())
    n.edge("a", "c")
    n.edge("a", "d")
    n.edge("b", "d")
    drawn(n.figure(), "graph-the-n", into, width=620, height=340)


# ── A study ──────────────────────────────────────────────────────────────────


def a_study(into: dict[str, bytes]) -> None:
    """A real search, with a real `Trainer` under it.

    The page used to show a stand-in `train()` returning a made-up curve, which
    is exactly the friction this documentation exists to avoid: the reader asks
    *don't you have a Trainer?* and they are right.
    """
    import torch

    import somatize.torch  # noqa: F401  (registers the tensor codec)
    from somatize import Graph, Node, Opaque, Store
    from somatize.study import (
        DONE,
        PRUNED,
        Pruner,
        Sampler,
        Space,
        coordinates,
        curves,
        finished,
        importance,
        influence,
        report,
        table,
        take,
        trials,
    )
    from somatize.torch import Trainer, parameters

    torch.manual_seed(0)
    truth = torch.randn(8, 1)

    def batch(how_many=64):
        x = torch.randn(how_many, 8)
        return x, x @ truth + 0.1 * torch.randn(how_many, 1)

    class Body(Node):
        def __init__(self, width):
            self.net = torch.nn.Sequential(torch.nn.Linear(8, width), torch.nn.ReLU())

        def forward(self, x, ctx):
            return Opaque(self.net(x))

        def parameters(self):
            return list(self.net.parameters())

    class Head(Node):
        def __init__(self, width):
            self.out = torch.nn.Linear(width, 1)

        def forward(self, x, ctx):
            return Opaque(self.out(x))

        def parameters(self):
            return list(self.out.parameters())

    space = (
        Space()
        .real("lr", 1e-4, 1e-1, log=True)
        .int("width", 8, 64)
        .choice("opt", ["adam", "sgd"])
    )
    sampler = Sampler.sobol(seed=0)
    pruner = Pruner.median(goal="min", warmup=4, startup=6)
    store = Store(tempfile.mkdtemp())
    STUDY, me = "widths", "the page that drew this"

    for trial in range(30):
        point = sampler.ask(space, trial, finished(store, space, study=STUDY))
        if not take(store, point, study=STUDY, trial=trial, me=me, goal="min"):
            continue

        g = Graph.somatize(
            Body(point["width"]).named("body") >> Head(point["width"]).named("head")
        )
        make = torch.optim.Adam if point["opt"] == "adam" else torch.optim.SGD
        t = Trainer(
            g,
            objective=torch.nn.functional.mse_loss,
            optimizer=make(parameters(g), lr=point["lr"]),
        )

        said, why, so_far = [], None, curves(store, study=STUDY)
        for _ in range(8):
            said.append(sum(t.step(batch()) for _ in range(10)) / 10)
            # A pruner stops nothing: it answers, and the loop stops calling.
            if why := pruner.verdict(said, so_far):
                break
        report(store, point, said, study=STUDY, trial=trial, me=me,
               state=PRUNED if why else DONE, because=why,
               score=said[-1], goal="min")

    drawn(table(store, space, study=STUDY), "study-table", into, height=940)
    drawn(influence(store, space, study=STUDY), "study-influence", into)
    drawn(coordinates(store, space, study=STUDY), "study-coordinates", into)

    seen = trials(store, space, study=STUDY)
    pruned = sum(one["state"] == PRUNED for one in seen)
    best = sorted(finished(store, space, study=STUDY), key=lambda p: p[1])[:3]
    print(f"  study: {len(seen)} trials, {pruned} pruned")
    print(f"  importance: {[(k, round(v, 2)) for k, v in importance(store, space, study=STUDY)]}")
    for point, score in best:
        print(f"    {score:.4f}  {point}")


# ── A run, watched ───────────────────────────────────────────────────────────


def a_run(into: dict[str, bytes]) -> None:
    """A short training run, written down and read back.

    A fan on purpose: `spent` and `gantt` are about *when*, and a graph that is
    a straight line has nothing to say about it. The two branches run in one
    `Wave`, which is what makes the timeline draw as overlapping bars rather
    than a staircase.
    """
    import torch

    import somatize.torch  # noqa: F401
    from somatize import Graph, Node, Opaque, Recorder, Store
    from somatize.record import gantt, progress, spent
    from somatize.torch import Trainer, parameters

    torch.manual_seed(0)
    WIDTH = 16
    # Something to learn. Training against `randn` targets draws a loss that
    # hovers, which shows the figure works and the framework does not — and a
    # front page should not have to explain why its own curve is flat.
    truth = torch.randn(WIDTH, WIDTH)

    def batch(how_many=32):
        x = torch.randn(how_many, WIDTH)
        return x, torch.tanh(x @ truth)

    class Block(Node):
        def __init__(self, width=WIDTH):
            self.net = torch.nn.Sequential(
                torch.nn.Linear(width, width), torch.nn.ReLU()
            )

        def forward(self, x, ctx):
            return Opaque(self.net(x))

        def parameters(self):
            return list(self.net.parameters())

    class Mean(Node):
        # A node with two predecessors is handed a map keyed by who sent what.
        # What is in it is the value itself: an `Opaque` is only visible from
        # OUTSIDE the graph — the engine opens it on the way in, whatever the
        # node's arity — so there is nothing here to unwrap.
        def forward(self, said, ctx):
            values = list(said.values())
            return Opaque(sum(values) / len(values))

    g = Graph.somatize(
        Block().named("encode")
        >> (Block().named("strict") | Block().named("loose"))
        >> Mean().named("vote")
    )
    store = Store(tempfile.mkdtemp())
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.Adam(parameters(g), lr=0.01),
        watching=Recorder(store, run="tuesday", summarising=["loss"]),
    )
    for _ in range(120):
        t.step(batch())

    drawn(progress(store, run="tuesday", smooth=5), "record-progress", into)
    drawn(spent(store, run="tuesday"), "record-spent", into, height=380)
    drawn(gantt(store, run="tuesday", forward=0), "record-gantt", into, height=380)


# ── The health of a network ──────────────────────────────────────────────────


def health(into: dict[str, bytes]) -> None:
    """A stack that cannot train, diagnosed from its record.

    Five sigmoid layers in a row, and what it earns is `STALLED` on the first
    three: the update is tiny next to the weights it moves. Built to be ill on
    purpose, because a figure of a healthy network shows that the drawing works
    and not that the diagnosis does.

    Which flag it earns was **read off the run and not decided here** — the
    guess when this was written was `VANISHING`, and the audit said otherwise.
    """
    import torch

    import somatize.torch  # noqa: F401
    from somatize import Graph, Node, Opaque, Recorder, Store
    from somatize.health import diagnose, flags, overlaid, profile
    from somatize.torch import Trainer, parameters

    torch.manual_seed(0)
    WIDTH = 16

    class Block(Node):
        def __init__(self, width=WIDTH, activation="sigmoid"):
            self.net = torch.nn.Linear(width, width)
            self.after = {
                "relu": torch.nn.ReLU(),
                "sigmoid": torch.nn.Sigmoid(),
            }[activation]

        def forward(self, x, ctx):
            return Opaque(self.after(self.net(x)))

        def parameters(self):
            return list(self.net.parameters())

    wired = Block().named("b0")
    for i in range(1, 5):
        wired = wired >> Block().named(f"b{i}")
    g = Graph.somatize(wired)

    store = Store(tempfile.mkdtemp())
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.SGD(parameters(g), lr=0.05),
        auditing=True,
        watching=Recorder(store, run="sigmoids", summarising=["loss"]),
    )
    for _ in range(20):
        t.step((torch.randn(32, WIDTH), torch.randn(32, WIDTH)))

    drawn(profile(store, run="sigmoids"), "health-profile", into)
    drawn(flags(store, run="sigmoids"), "health-flags", into, height=250)
    drawn(overlaid(g, store, run="sigmoids"), "health-overlaid", into, width=760, height=560)
    print(f"  diagnosed: {diagnose(store, run='sigmoids')}")


# ── A node opened up ─────────────────────────────────────────────────────────


def an_architecture(into: dict[str, bytes]) -> None:
    """Three nodes, each a real architecture, drawn inside their own boxes.

    Chosen to put every drawing rule on one picture rather than to be a good
    model: a convolutional trunk for the silhouettes and the `ch`/`len` shape
    names, a transformer stack for the `xN` frame and the plates behind
    attention heads, and a head that narrows so the taper has something to say.
    """
    import torch
    from torch import nn

    import somatize.torch  # noqa: F401
    from somatize import Graph, Node, Opaque
    from somatize.torch import architecture

    torch.manual_seed(0)

    class Trunk(Node):
        def __init__(self):
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            )

        def forward(self, x, ctx):
            return Opaque(self.net(x))

        def parameters(self):
            return list(self.net.parameters())

    class Encode(Node):
        def __init__(self):
            one = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, batch_first=True
            )
            self.net = nn.TransformerEncoder(one, num_layers=4)

        def forward(self, x, ctx):
            return Opaque(self.net(x.transpose(1, 2)))

        def parameters(self):
            return list(self.net.parameters())

    class Head(Node):
        def __init__(self):
            self.net = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x, ctx):
            return Opaque(self.net(x.mean(1)))

        def parameters(self):
            return list(self.net.parameters())

    g = Graph.somatize(
        Trunk().named("trunk") >> Encode().named("encode") >> Head().named("head")
    )
    # Wrapped, because a bare tensor does not cross an edge — the same rule as
    # `Opaque` on the way out, seen from the caller's side.
    example = Opaque(torch.randn(4, 1, 32))
    inside = architecture(g, example)
    print(f"  architecture traced: {sorted(inside)}")
    drawn(g.figure(inside=inside), "architecture", into, width=980, height=900)

    # The same graph with the composite opened. `depth=` is the whole of the
    # difference: a `TransformerEncoderLayer` everybody recognises is one box
    # until somebody asks, and then it is the `xN` frame and the plates.
    deeper = architecture(g, example, depth=2)
    drawn(g.figure(inside=deeper), "architecture-opened", into, width=980, height=1200)

    # An attention block written out rather than torch's, which is where the
    # **residual** shows: `y + x` is not a module, so a hand-written block draws
    # the skip that the encoder layer keeps inside a `forward` nobody can see
    # into. The plates behind the heads are on both figures.
    class Block(Node):
        def __init__(self):
            self.att = nn.MultiheadAttention(64, 4, batch_first=True)
            self.norm = nn.LayerNorm(64)
            self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))

        def forward(self, x, ctx):
            y, _ = self.att(x, x, x)
            y = self.norm(y + x)
            return Opaque(self.norm(y + self.ff(y)))

        def parameters(self):
            return [
                *self.att.parameters(), *self.norm.parameters(), *self.ff.parameters()
            ]

    b = Graph.somatize(Block().named("block"))
    drawn(
        b.figure(inside=architecture(b, Opaque(torch.randn(4, 32, 64)))),
        "architecture-attention", into, width=880, height=700,
    )


# ── The reasoning of an investigation ────────────────────────────────────────


def reasoning(into: dict[str, bytes]) -> None:
    """The DAG behind an investigation, drawn from what was written down.

    The only group that needs something built rather than installed: the moves
    are written by the `somatize-tree` binary, because asking, supposing and
    deciding happen between runs and the library only reads them back.

    It runs the very session `start/an-investigation` shows, against the very
    fixture the end-to-end tests use, so the page's figure and the page's
    commands cannot drift apart.
    """
    import os
    import subprocess

    from somatize import Store
    from somatize.reasoning import figure, moves, standing

    root = Path(__file__).resolve().parent.parent.parent
    binary = os.environ.get("SOMA_TREE_BIN", str(root / "target/release/somatize-tree"))
    if not Path(binary).exists():
        raise SystemExit(
            f"{binary} is not there. This group writes moves with the CLI:\n"
            "  cargo build --release -p somatize-tree"
        )

    where = Path(tempfile.mkdtemp())
    repo, cache = where / "repo", where / "cache"
    repo.mkdir()
    subprocess.run(
        ["bash", str(root / "soma-tree/tests/an-investigation.sh"), "--only-build", str(repo)],
        check=True, capture_output=True,
    )

    # Its own cache, because the default store is shared by every repository
    # answering to the same `tree` in `soma-tree.toml` — two runs of this would
    # otherwise collide on the move names.
    env = {**os.environ, "XDG_CACHE_HOME": str(cache)}

    def tree(*args: str) -> None:
        subprocess.run([binary, *args], cwd=repo, env=env, check=True, capture_output=True)

    strict = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD~2"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    ran = subprocess.run(
        [binary, "keep"], cwd=repo, env=env, check=True, capture_output=True, text=True,
        input="python -m experiments.run --threshold 2.0 --seed 7",
    ).stdout.strip()

    tree("ask", "why-recall", "-m",
         "Recall sits at 0.61 and nothing we change moves it. What is holding it down?")
    tree("suppose", "threshold-too-strict", "--under", "why-recall", "-m",
         "The strict classifier at 2.0 throws away the short documents.")
    tree("suppose", "embedding-too-flat", "--under", "why-recall", "-m",
         "A linear embedding cannot separate them at all, whatever the threshold.")
    tree("tried", "at-2.0", "--under", "threshold-too-strict", "--cites", strict,
         "--ran", ran, "-m", "Ran the strict classifier at 2.0 on the full split.")
    tree("found", "short-docs-lost", "--under", "at-2.0", "-m",
         "Recall on documents under 20 tokens is 0.31; on the rest it is 0.79.")
    tree("says", "short-docs-lost", "validates", "threshold-too-strict", "--about", "at-2.0")
    tree("says", "short-docs-lost", "refutes", "embedding-too-flat", "--partly")
    tree("decide", "abandon", "drop-flat-embedding", "--about", "embedding-too-flat", "-m",
         "The split by length explains it; the embedding is not the problem.")

    store = Store(str(cache / "somatize-tree"))
    print(f"  moves: {len(moves(store, tree='an-investigation'))}, "
          f"standing: {standing(store, tree='an-investigation')}")
    drawn(figure(store, tree="an-investigation"), "reasoning", into, width=980, height=300)


GROUPS = {"graph": a_graph, "reasoning": reasoning, "architecture": an_architecture, "study": a_study, "run": a_run, "health": health}


def main() -> None:
    wanted = sys.argv[1:] or list(GROUPS)
    unknown = [name for name in wanted if name not in GROUPS]
    if unknown:
        raise SystemExit(f"no such group: {unknown}. Have: {sorted(GROUPS)}")

    into: dict[str, bytes] = {}
    for name in wanted:
        print(f"drawing {name}…")
        GROUPS[name](into)

    # Only a full run may prune: asking for one group and having it delete the
    # others is the kind of helpfulness that costs an afternoon.
    if set(wanted) == set(GROUPS):
        rewrite(OUT, into)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, body in into.items():
            (OUT / name).write_bytes(body)
    print(f"figures → {OUT.name}/: {len(into)} drawn")


if __name__ == "__main__":
    main()
