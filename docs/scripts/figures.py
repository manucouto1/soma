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


GROUPS = {"graph": a_graph, "study": a_study}


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
