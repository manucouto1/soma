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


class Trainable(Node):
    """Weights on the far side, and nothing to receive their gradient here.

    From the client this is the trap: it runs, it produces, the loss comes down
    because whatever is downstream of it **is** learning — and these weights
    never move. What crosses a wire is the value, not the graph that made it.
    """

    def __init__(self, wide=8, tall=6):
        import torch

        self.lin = torch.nn.Linear(wide, tall)

    def forward(self, x, ctx):
        import torch

        return Done(self.lin(torch.tensor(x, dtype=torch.float32)).tolist())

    def parameters(self):
        return list(self.lin.parameters())


class SplitPart(Node):
    """The far half of a **split learning** cut, which is the answer to the trap
    above rather than a way around it.

    Two messages and one node, because a node is one contract: forward keeps its
    activation alive **here**, where its autograd graph is; backward takes the
    gradient of the seam — a tensor like any other — and carries on with the
    chain rule from there, with an optimizer of its own.

    Nothing about the weights ever travels. What goes out is activations, what
    comes back is `dL/da`, and each side updates what it holds.
    """

    def __init__(self, wide=8, tall=6, lr=0.1):
        import torch

        self.lin = torch.nn.Linear(wide, tall)
        self.opt = torch.optim.SGD(self.lin.parameters(), lr=lr)
        self.held = None

    def forward(self, msg, ctx):
        import torch

        value = torch.tensor(msg["value"], dtype=torch.float32)
        if msg["kind"] == "forward":
            self.held = self.lin(value).relu()
            return Done({"value": self.held.detach().tolist(), **whereabouts()})
        self.opt.zero_grad()
        self.held.backward(value)
        self.opt.step()
        return Done({"weights": float(self.lin.weight.abs().sum()), **whereabouts()})

    def parameters(self):
        return list(self.lin.parameters())


class Head(Node):
    """The near half: plain torch, and it is the one that gets a gradient."""

    def __init__(self, wide=6, tall=3):
        import torch

        self.lin = torch.nn.Linear(wide, tall)

    def forward(self, x, ctx):
        import torch
        from soma_next import Opaque

        return Done(Opaque(self.lin(torch.tensor(x, dtype=torch.float32))))

    def parameters(self):
        return list(self.lin.parameters())
