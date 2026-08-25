"""Making `.frozen()` true, which is the only part of it torch has to do.

The core **declares** — a settled node's state does not change while the graph
runs — and reasons over that: it is what lets it refuse a cache that cannot be
honoured. It does not know what a gradient is, and it is not going to. Obeying
is here, exactly as moving a tensor to a GPU is the node's job and not the
core's.

Two things happen here, and they are the same thing seen twice:

- `requires_grad_(False)`, so the gradient really does stop. Without it, the
  value restored from a cache is a **leaf** and the net above it stops training
  in silence.
- the digest of the weights, **hashed once, here**. Settling is what makes both
  the gradient rule and the stability of the key true at the same time, so this
  is the one moment worth paying for it — and without it, two checkpoints of the
  same class would share a key, which is the one kind of hit that is a bug.

The digest is not `torch.save`'s bytes: it is the names, dtypes, shapes and raw
bytes of the tensors, in order. That way the same weights hash the same whether
they sit on a GPU or on a CPU, which they have to for a store two machines
share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from soma_next._graph import Graph

import hashlib

import torch


def freeze(graph: "Graph", *node_ids: str) -> None:
    """Settles these nodes — or whatever was already declared `.frozen()`.

    With ids it declares and obeys; with none it only obeys, which is what
    `Trainer` calls so that a `.frozen()` in the expression is true before the
    first step rather than after somebody notices.
    """
    for node_id in node_ids or list(graph.frozen()):
        implementation = graph.implementation(node_id)
        _stop_the_gradient(implementation)
        graph.freeze(node_id, state_digest(implementation))


def state_digest(implementation: object) -> str | None:
    """The digest of the state it is settled at, or `None` if it has none.

    Three ducks and not one, because the project's own nodes use all three:
    whoever already knows what it is settled at — a source, whose version the
    store worked out — is asked first and believed; whoever has a `state_dict`
    is asked for it by name; whoever only has `parameters()` is asked for those,
    in order. Whoever has none has no state, and a tokenizer does not stop being
    a node for it.

    It has to be the same three `Graph._check_it_was_obeyed` looks at, or a node
    would be told to settle and then have nothing to settle with.
    """
    # A node that already knows what it is settled at is asked first and
    # believed: a source hashes nothing here, because the store hashed its
    # content when the bytes went in.
    said = getattr(implementation, "version", None)
    if said is not None:
        return said
    named = getattr(implementation, "state_dict", None)
    if named is not None:
        return _digest_of(sorted(named().items()))
    in_order = getattr(implementation, "parameters", None)
    if in_order is not None:
        return _digest_of(enumerate(in_order()))
    return None


def _digest_of(state: Iterable[tuple[Any, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in state:
        digest.update(str(name).encode())
        if not torch.is_tensor(value):
            digest.update(repr(value).encode())
            continue
        # The dtype and the shape too: the same bits read as two dtypes are two
        # different states.
        digest.update(f"{value.dtype}{tuple(value.shape)}".encode())
        digest.update(_raw(value))
    return f"sha256:{digest.hexdigest()}"


def _raw(tensor: "torch.Tensor") -> bytes:
    """The bytes of a tensor, wherever it lives and whatever its dtype.

    Through `uint8` and not `numpy()`: half precision has no numpy dtype, and
    what is wanted here is the bits and not the numbers.
    """
    flat = tensor.detach().cpu().contiguous().flatten()
    return flat.view(torch.uint8).numpy().tobytes()


def _stop_the_gradient(implementation: object) -> None:
    """No parameter of this node asks for a gradient any more.

    Freezing a node is not freezing its prefix: the gradient still crosses it
    towards whatever is above. That is why the rule the core checks is about the
    **whole prefix** and not about one node.
    """
    parameters = getattr(implementation, "parameters", None)
    if parameters is None:
        return
    for parameter in parameters():
        parameter.requires_grad_(False)
