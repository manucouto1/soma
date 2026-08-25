"""`Worker` and `Broker` — where a slice goes, and who knows where that is.

A `Worker` is a **declaration**: an address or a command, and how to pack for
it. It opens nothing. What resolves it into a wire is a `Broker`, and the wire
is opened the first time somebody actually sends work — so a graph that names a
host a run never reaches costs nothing for naming it.

Almost all of the rest is in Rust. What cannot be is packing the nodes: a
`pickle` artifact is made by `cloudpickle`, which is Python's, and the wire
deliberately does not look at what it carries.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

from soma_next._soma_next import Broker as _RustBroker


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


class Worker:
    """Where a slice goes, and how to pack for it. It opens nothing.

    A declaration and not a connection, which is the change a broker brings::

        # on the other machine, in the background
        python -m soma_next.worker --listen 0.0.0.0:7000

        # here
        g.place_at("tokenize", "w1")
        g.forward(x, broker=Broker.embedded({"w1": Worker.at("node3:7000")}))

    `mode` says what gets sent: `"project"` *(default)* sends names, versions
    and state, and the worker supplies the code from its clone; `"network"`
    sends the code too, with `cloudpickle`, and `send=["my_package"]` makes your
    own modules travel inside it.

    Because it declares rather than connects, **a host that is not there fails
    when it is needed rather than when it is named** — inside the run, and not
    in this constructor.
    """

    target: str | list[str]
    """A `"host:port"` for a worker already standing, or an `argv` for one to be
    started as a child."""

    _mode: str
    """How this worker is packed for: `"project"` or `"network"`."""

    _send: tuple[str, ...]
    """Which of your modules travel inside the artifact rather than by name."""

    def __init__(
        self,
        target: str | list[str],
        mode: str = "project",
        send: Sequence[str] = (),
    ) -> None:
        if mode not in ("project", "network"):
            raise ValueError(f"`mode` is 'project' or 'network', not {mode!r}")
        if not isinstance(target, str):
            if not isinstance(target, (list, tuple)) or not all(
                isinstance(one, str) for one in target
            ):
                raise ValueError(
                    'a worker is declared with a `"host:port"` address or with an '
                    "`argv` list"
                )
            if not target:
                raise ValueError("a worker needs at least a program")
            target = list(target)
        self.target, self._mode, self._send = target, mode, tuple(send)

    @classmethod
    def at(cls, addr: str, mode: str = "project", send: Sequence[str] = ()) -> Worker:
        """A worker that **is already standing** somewhere."""
        return cls(addr, mode, send)

    @classmethod
    def spawn(cls, argv: list[str], mode: str = "project", send: Sequence[str] = ()) -> Worker:
        """A worker to be started as a child process. For testing: while the
        client starts the process, there is no independent worker worth the
        name."""
        return cls(argv, mode, send)

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

    def packed(self, nodes: dict[str, Any]) -> tuple[str, str, bytes]:
        """These nodes as an artifact the way this worker wants them: its kind,
        its id, and its bytes."""
        kind, blob = _pack(nodes, self._mode, self._send)
        return kind, "sha256:" + hashlib.sha256(blob).hexdigest(), blob

    def __repr__(self) -> str:
        where = self.target if isinstance(self.target, str) else " ".join(self.target)
        return f"Worker({where})"


class Broker(_RustBroker):
    """Where the hosts of a graph are, and who resolves them.

    One deployment today — the one inside this process, which is what makes soma
    work with no platform, no head node and no internet::

        g.forward(x, broker=Broker.embedded({"w1": Worker.at("node3:7000")}))

    The others — a local one on a head node, and the platform's — speak the same
    protocol, so what changes for a client is which broker, and that is a URL.
    """

    _packing: dict[str, Worker]
    """How to pack for each host. It stays on this side because what can be
    packed depends on what is installed **on that machine**, and because the
    Rust half deliberately does not know what a `cloudpickle` is."""

    @classmethod
    def embedded(cls, workers: dict[str, Worker]) -> Broker:
        """A broker inside this process, knowing where these hosts are."""
        for host, worker in workers.items():
            if not isinstance(worker, Worker):
                raise ValueError(
                    f"a broker is given a dict from host to Worker; for `{host}` a "
                    f"`{type(worker).__name__}` arrived"
                )
        broker = cls({host: worker.target for host, worker in workers.items()})
        broker._packing = dict(workers)
        return broker

    def packing_for(self, host: str) -> Worker:
        """The worker declared for this host."""
        return self._packing[host]

    def token_for(self, host: str) -> bytes | None:
        """What this host shares a wire with, or `None` if the broker does not
        know it.

        `None` rather than an exception because a graph may name a host nobody
        listed, and that is not this step's failure to report: either the run
        reaches it or it does not, and whichever happens says so with the slice
        in front of it.
        """
        try:
            return self.wire_token(host)
        except RuntimeError:
            return None


def _pack(nodes: dict[str, Any], mode: str, send: Sequence[str]) -> tuple[str, bytes]:
    """The nodes as an artifact, whichever way applies.

    **By id, always.** The artifact's id is the digest of these bytes, and a dict
    pickles in insertion order — so the same nodes handed over in another order
    were another artifact. Two things fell out of that, and neither was
    intentional: a `Worker` serving two hosts changed id when the caller
    reordered the dict it was given, which defeats the `have`/`want` and the
    store's artifact cache; and a second graph over the same nodes — the transpose of the
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
