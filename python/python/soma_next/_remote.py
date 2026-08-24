"""`Worker` — the process that gets sent slices, from this side.

Almost all of it is in Rust. What cannot be is packing the nodes: a `pickle`
artifact is made by `cloudpickle`, which is Python's, and the transport
deliberately does not look at what it carries.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

from soma_next._soma_next import Worker as _RustWorker


@contextmanager
def _by_value(modules: Iterable[str]) -> Iterator[None]:
    """Makes these modules travel **inside** the artifact.

    cloudpickle serializes by reference whatever comes from an importable
    module, which leaves out exactly the case a generic worker is for: your
    nodes in `my_package/net.py` and a worker that cannot import it. It
    unregisters afterwards, because that registry is global to the process.
    """
    import cloudpickle

    registered = []
    try:
        for name in modules:
            # The whole package, not just the named module: a node can use
            # something from a sibling of its own.
            for loaded in [name, *[m for m in list(sys.modules) if m.startswith(f"{name}.")]]:
                module = sys.modules.get(loaded) or importlib.import_module(loaded)
                cloudpickle.register_pickle_by_value(module)
                registered.append(module)
        yield
    finally:
        for module in registered:
            cloudpickle.unregister_pickle_by_value(module)


class Worker(_RustWorker):
    """A process that executes the slices you send it.

    A worker is **an address and a way of packing**, nothing else: which nodes
    go to it is decided by the graph at run time::

        # on the other machine, in the background
        python -m soma_next.worker --listen 0.0.0.0:7000

        # here
        g.place_at("tokenize", "w1")
        g.forward(x, workers={"w1": Worker.at("node3:7000")})

    `mode` says what gets sent: `"project"` *(default)* sends names, versions
    and state, and the worker supplies the code from its clone; `"network"`
    sends the code too, with `cloudpickle`, and `send=["my_package"]` makes your
    own modules travel inside it.
    """

    _mode: str
    """How this worker is packed for: `"project"` or `"network"`. Declared here
    and set by `_remember`, because an attribute hung on an instance from outside
    the class is one a reader — and a checker — has no way to find."""

    _send: tuple[str, ...]
    """Which of your modules travel inside the artifact rather than by name."""

    @classmethod
    def at(cls, addr: str, mode: str = "project", send: Sequence[str] = ()) -> Worker:
        """Connects to a worker that **was already standing**."""
        return _remember(cls(addr), mode, send)

    @classmethod
    def spawn(cls, argv: list[str], mode: str = "project", send: Sequence[str] = ()) -> Worker:
        """Starts a worker as a child process. For testing: while the client
        starts the process, there is no independent worker worth the name."""
        return _remember(cls(argv), mode, send)

    @classmethod
    def generic(
        cls,
        mode: str = "project",
        send: Sequence[str] = (),
        python: str | None = None,
    ) -> Worker:
        """A child running `python -m soma_next.worker`."""
        return cls.spawn(
            [python or sys.executable, "-m", "soma_next.worker"], mode=mode, send=send
        )

    def carry(self, nodes: dict[str, Any]) -> None:
        """Packs these nodes and tells the worker it is going to need them.
        `Graph.forward` calls it — an artifact is how anything gets there."""
        kind, blob = _pack(nodes, self._mode, self._send)
        self.provision(
            kind, "sha256:" + hashlib.sha256(blob).hexdigest(), blob, _runtime()
        )


def _remember(worker: Worker, mode: str, send: Sequence[str]) -> Worker:
    """Hangs on the worker how it packs, which is all it is missing."""
    if mode not in ("project", "network"):
        raise ValueError(f"`mode` is 'project' or 'network', not {mode!r}")
    worker._mode, worker._send = mode, tuple(send)
    return worker


def _pack(nodes: dict[str, Any], mode: str, send: Sequence[str]) -> tuple[str, bytes]:
    """The nodes as an artifact, whichever way applies.

    **By id, always.** The artifact's id is the digest of these bytes, and a dict
    pickles in insertion order — so the same nodes handed over in another order
    were another artifact. Two things fell out of that, and neither was
    intentional: a `Worker` serving two hosts changed id when the caller
    reordered `workers={...}`, which defeats the `have`/`want` and the store's
    artifact cache; and a second graph over the same nodes — the transpose of the
    first, which is how a backward pass crosses a wire — provisioned the worker
    again, **swapping the catalog it had live** and losing with it every
    activation and every optimizer state over there.

    Sorting makes the id depend on what is in the artifact and on nothing else.
    """
    nodes = dict(sorted(nodes.items()))
    if mode == "project":
        from soma_next import _manifest

        return _manifest.KIND, _manifest.pack(nodes)

    import cloudpickle

    with _by_value(send):
        return "pickle", cloudpickle.dumps(nodes)


def _runtime() -> str:
    from soma_next.worker import runtime

    return runtime()
