"""How a tensor is written down, so that it can be kept.

`Opaque` exists because a live Python object cannot leave the process. A tensor
is one, and this is what stops it being one: with a codec registered, what
crosses into a store is bytes, and `soma_next_store` never finds out any of it
was ever a tensor.

The frontier does not vanish, it moves — from "an opaque does not travel" to "an
opaque nobody registered a codec for does not travel", which is the more precise
of the two.

What it does **not** restore is the graph the tensor came from: what `load`
gives back is a **leaf**, and a backward pass stops there. That is exactly why a
cached prefix has to be settled, and why the core refuses the other case rather
than letting a net stop training in silence.
"""

from __future__ import annotations

import io

import torch

from soma_next._soma_next import codec

KIND = "torch.Tensor"
"""What gets written beside the bytes, and what a store keeps forever. Named
after the type and not after the run, because it has to still mean something the
day somebody reads that record by hand."""


def register():
    """Says how a tensor is written down. Called on importing `soma_next.torch`,
    once: registering it again with the same name replaces it with itself."""
    codec(KIND, torch.Tensor, dump=dump, load=load)


def dump(tensor):
    """A tensor in bytes, `torch.save`'s own, and **detached**.

    What `load` gives back is a leaf whatever is done here, so carrying
    `requires_grad` across preserves nothing — and it costs: a tensor that
    arrives on a worker still asking for a gradient builds a graph there that
    nobody reads, in a `forward` that is not training anything. Whoever does
    train across a cut says so itself, once, on the input of the stage.
    """
    buffer = io.BytesIO()
    torch.save(tensor.detach(), buffer)
    return buffer.getvalue()


def load(raw):
    """And back, **on the cpu**.

    Not where it was saved: a store is shared between machines, and one that
    only reads back on the box that wrote it is not shared at all. Whoever
    receives it moves it, which is what a placed node already does with its
    input.

    `weights_only=True` because what is read here comes from a store that may be
    shared: unpickling arbitrary objects out of it would make a cache into a way
    in.
    """
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
