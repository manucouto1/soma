"""The nodes the cluster runs, and **the containers cannot see this file**.

That is the whole reason it is a module of its own: it is mounted nowhere, it is
in no image, and no `PYTHONPATH` over there names it. Whatever a worker executes
out of here got there by travelling, and if it did not travel the worker has to
say so instead of guessing.

They are plain Python on purpose, torch aside: what is being tested is the wire,
and a 2.5 GB image per worker would be paying for a tensor nobody looks at. The
one that needs a GPU says so in its name.
"""

from __future__ import annotations

import os
import time

from soma_next import Await, Done, Node


def whereabouts():
    """Which machine and which process. The container's hostname is its id, so
    two of them cannot pretend to be one."""
    return {"host": os.uname().nodename, "pid": float(os.getpid())}


class Shout(Node):
    """Text in, text out. Nothing to install."""

    def forward(self, text, ctx):
        return Done({"text": text.upper(), **whereabouts()})


class Wrap(Node):
    """Reads what the one before it produced, so something really crosses."""

    def forward(self, got, ctx):
        return Done({"text": f"[{got['text']}]", "before": got["host"], **whereabouts()})


class Slow(Node):
    """Takes its time, so that two of them overlapping is visible from outside.

    A rendezvous through a shared file is what the same-machine tests use; across
    containers the honest way is the clock — two of these in a wave take about as
    long as one, or they did not overlap.
    """

    SECONDS = 0.6

    def forward(self, x, ctx):
        time.sleep(self.SECONDS)
        return Done(whereabouts())


class Asks(Node):
    """Answers only if there is a driver **where it runs**."""

    def forward(self, x, ctx):
        if ctx.turn == 0:
            return Await(["are you there?"])
        return Done({"heard": ctx.results[0], **whereabouts()})


class Answers:
    """A driver, which travels in the same artifact as the nodes."""

    def __init__(self, with_what="yes"):
        self.with_what = with_what

    def perform(self, requests):
        return [f"{self.with_what}: {r}" for r in requests]


class Sized(Node):
    """Produces something worth keeping: a list as long as you say.

    Not a tensor, so that a worker with no torch can keep it — what is kept is a
    value, and a value does not have to be a tensor.
    """

    def __init__(self, how_many=1000):
        self.how_many = how_many

    def forward(self, seed, ctx):
        base = int(seed)
        return Done({"data": [float(base + i) for i in range(self.how_many)], **whereabouts()})


class OnTheDevice(Node):
    """Says where torch put its tensor, which is the only thing worth asking of
    a placement that crossed a wire.

    It obeys `ctx.device` the way any placed node does — the core says where and
    the node moves itself — and reports what it ended up with.
    """

    def forward(self, x, ctx):
        import torch

        tensor = torch.ones(4)
        if ctx.device:
            tensor = tensor.to(ctx.device)
        return Done(
            {
                "said": ctx.device or "",
                "landed": str(tensor.device),
                "cuda": float(torch.cuda.is_available()),
                **whereabouts(),
            }
        )


class Stamp(Node):
    """Answers something that cannot be answered twice the same.

    A clock reading is the honest way to see a cache across machines: if the
    second run — on **another worker** — comes back with the same reading and the
    first one's hostname, it did not compute anything, it read what the other one
    left in the store they share.
    """

    def forward(self, x, ctx):
        return Done({"when": time.monotonic(), **whereabouts()})
