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

from soma_next._soma_next import Worker as _RustWorker


@contextmanager
def _by_value(modules):
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

    @classmethod
    def at(cls, addr, mode="project", send=()):
        """Connects to a worker that **was already standing**."""
        return _remember(cls(addr), mode, send)

    @classmethod
    def spawn(cls, argv, mode="project", send=()):
        """Starts a worker as a child process. For testing: while the client
        starts the process, there is no independent worker worth the name."""
        return _remember(cls(argv), mode, send)

    @classmethod
    def generic(cls, mode="project", send=(), python=None):
        """A child running `python -m soma_next.worker`."""
        return cls.spawn(
            [python or sys.executable, "-m", "soma_next.worker"], mode=mode, send=send
        )

    def carry(self, nodes, driver=None):
        """Packs these nodes and this driver and tells the worker it is going to
        need them. `Graph.forward` calls it.

        The driver goes in the same artifact, by the same strategy: a node that
        returns `Await` has to be served **where it runs**, and the artifact is
        how anything gets there.
        """
        kind, blob = _pack(nodes, driver, self._mode, self._send)
        self.provision(
            kind, "sha256:" + hashlib.sha256(blob).hexdigest(), blob, _runtime()
        )


def _remember(worker, mode, send):
    """Hangs on the worker how it packs, which is all it is missing."""
    if mode not in ("project", "network"):
        raise ValueError(f"`mode` is 'project' or 'network', not {mode!r}")
    worker._mode, worker._send = mode, tuple(send)
    return worker


def _pack(nodes, driver, mode, send):
    """The nodes and the driver as an artifact, whichever way applies."""
    if mode == "project":
        from soma_next import _manifest

        return _manifest.KIND, _manifest.pack(nodes, driver)

    import cloudpickle

    with _by_value(send):
        return "pickle", cloudpickle.dumps((nodes, driver))


def _runtime():
    from soma_next.worker import runtime

    return runtime()
