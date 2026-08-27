"""`Graph` — the Rust object plus what can only live in Python.

Almost all of it is in Rust. What cannot be is `somatize`, which walks an
expression made of Python objects, and it is written in the class body rather
than assigned at import time: a `#[pyclass]` is immutable, and what is assigned
is invisible to `help()`, an IDE and a type checker.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from somatize import _dsl
from somatize._somatize import Graph as _RustGraph
from somatize._remote import Broker, Worker, _runtime
from somatize._somatize import Store
from somatize._typing import Fact, Figure, Inside, Overlay

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class Carrier(Protocol):
    """A worker this side can hand an artifact to. A `Protocol` and not
    `somatize.Worker`, because what `provision` needs is the one method and the
    Rust `Worker` the engine downcasts to does not have it. `_share_out` asks for
    it by name and says so when it is missing.
    """

    def carry(self, nodes: dict[str, Any]) -> None: ...



class Graph(_RustGraph):
    """A computation graph: nodes, edges and what each one executes.

    Everything else — `node`, `edge`, `plan` and the topology queries — is
    inherited from the Rust class.
    """

    _slice_of: Graph | None = None
    """The graph this one is a piece of, for a graph run in pieces."""

    def node(self, *args: Any) -> str:
        """Adds a node and returns its id, noting **what it was built with**.

        Here and not in `_dsl`, because a graph built by hand in a loop has the
        same collision: `Embed(512)` and `Embed(64)` are one class and two
        answers. What cannot be written down the same way in two processes is
        passed over here and refused in `_check_it_was_obeyed`.
        """
        from somatize import _declaration

        node_id = super().node(*args)
        try:
            self.declared_as(node_id, _declaration.digest(self.implementation(node_id)))
        except _declaration.CannotDeclare:
            pass
        return node_id

    def figure(self, overlay: Overlay | None = None, inside: Inside | None = None) -> Figure:
        """The graph drawn, as a `plotly.graph_objects.Figure`. Nothing is
        executed to draw it. Needs the `viz` extra.

        `inside` opens a node up — `{node: [(path, what), ...]}`, which
        `somatize.torch.architecture` reads off the modules it holds. `overlay`
        lays what **happened** over what was declared, `{node: [flag, ...]}`.
        """
        from somatize import _figure

        return _figure.figure(self, overlay, inside)

    def _repr_mimebundle_(
        self,
        include: object = None,
        exclude: object = None,
    ) -> dict[str, Any] | None:
        """What a notebook shows for `g` on its own: the figure.

        The *mimebundle* and not `_repr_html_`, because that is how a plotly
        figure reaches a cell. `None` without plotly, for a graph too big to
        read, and outside a notebook — in all three the cell falls back to
        `__repr__`, and `figure()` by hand still draws.
        """
        from somatize import _figure

        if len(self) > _figure.TOO_MANY:
            return None
        try:
            drawn = self.figure()
        except RuntimeError:
            return None
        return drawn._repr_mimebundle_(include=include, exclude=exclude) or None

    # The narrowing is deliberate, and it is the design: the Rust `Broker` this
    # overrides knows where a host is and nothing else, while **this** one
    # carries the packing table — what can be packed depends on what is
    # installed on that machine, which the Rust half does not know a
    # `cloudpickle` about. A bare one would reach `provision` with no
    # `packing_for`.
    def forward(  # type: ignore[override]
        self,
        input: Any | None = None,
        *,
        broker: Broker | None = None,
        store: Store | str | None = None,
        watching: Callable[[Fact], None] | list[Callable[[Fact], None]] | None = None,
        stamping: dict[str, str] | None = None,
    ) -> Any:
        """Executes the whole graph and returns what it produced.

        `broker=Broker.embedded({"w1": Worker.at(...)})` says who knows where
        each host is. **This method sends the nodes**, not you.

        `store` is a directory or a `Store`: with one, whatever was declared
        `.cached()` is looked up before being computed and kept after.

        `watching` is told what happened, as it happens::

            g.forward(x, watching=print)                    # in a notebook
            g.forward(x, watching=Recorder(store))          # kept
            g.forward(x, watching=[Recorder(store), draw])  # both

        A fact arrives as a `dict` with a `fact` key naming it — the same shape
        it is written down as. What a worker saw comes back down the connection
        that was already open.
        """
        self._check_it_was_obeyed()
        self.provision(broker)
        return super().forward(
            input,
            broker=broker,
            store=store,
            watching=watching,
            stamping=self._provenance(store, stamping),
        )

    def _provenance(
        self, store: "Store | str | None", stamping: dict[str, str] | None
    ) -> dict[str, str] | None:
        """What gets written beside everything this run keeps.

        The environment goes in **without anybody asking**: provenance that has
        to be remembered is missing from exactly the runs nobody thought were
        going to matter, and no key will ever carry it since a fingerprint stops
        at what is installed. Bound once under `env/<digest>`. Only with a store.
        """
        if store is None:
            return stamping

        from somatize import _environment

        said = _environment.environment()
        name = _environment.named(said)

        # Bound and not claimed: two runs in the same environment write the same
        # reading, so a claim would be asking who won a race with no loser. And
        # a failure here does not stop a run — refusing to execute over it would
        # be losing the work to save the label.
        try:
            kept = store if isinstance(store, Store) else Store(str(store))
            kept.keep(f"{_environment.WHERE}/{name}", said)
        except Exception as why:  # noqa: BLE001
            print(f"the environment could not be written down: {why}", file=sys.stderr)

        # The caller's on top: `stamping` is how soma-tree says which commit and
        # which investigation this was, and neither is anything this can guess.
        return {"env": name, **(stamping or {})}

    def provision(self, broker: Broker | dict[str, Any] | None) -> None:
        """Tells each host what it is going to need, before the first node runs.

        `forward` calls it, and whoever runs a graph **in pieces** does not have
        to either: a piece provisions the graph it is a piece of, entire. That is
        why the method exists — a worker has **one** catalog, and half of one is
        a different catalog, refused mid-session and swallowed in silence by a
        worker that has not greeted yet.

        A host that gets nothing is told nothing, and **two hosts that turn out
        to be one place are told once**, with the union of what they hold.
        """
        if broker is None:
            return
        whole = self._slice_of or self
        if not isinstance(broker, Broker):
            # A dict of things that know how to `carry`. It is what a `Broker`
            # replaces, and it survives here for whoever stands one in.
            for carrier, nodes in whole._share_out_by_carrier(broker).items():
                if nodes:
                    carrier.carry(nodes)
            return

        for hosts, nodes in whole._share_out(broker):
            if not nodes:
                continue
            declared = broker.packing_for(hosts[0])
            for host in hosts[1:]:
                other = broker.packing_for(host)
                if (other._mode, other._send) != (declared._mode, declared._send):
                    raise ValueError(
                        f"`{hosts[0]}` and `{host}` are the same place, so they get one "
                        f"catalog, and they are declared with different packing "
                        f"({declared._mode!r} and {other._mode!r}). Say the same thing "
                        f"for both, or give them different addresses"
                    )
            kind, ident, blob = declared.packed(nodes)
            for host in hosts:
                broker.provision(host, kind, ident, blob, _runtime())

    def _check_it_was_obeyed(self) -> None:
        """That whoever was declared settled really was settled.

        The core cannot ask this: it cannot tell a node with **no state to
        settle** — a tokenizer — from one whose weights **nobody has hashed yet**,
        since both arrive as `None`. Without the digest the key does not depend
        on the weights, so two checkpoints of one class share a name and the
        wrong tensor comes back with no error and no warning.
        """
        from somatize import _declaration

        if not self.cached():
            return
        declared = self.declarations()
        for node_id in self.frozen():
            # Everything upstream of a cache is frozen — `cacheable` refuses the
            # graph otherwise — so this is exactly the set whose keys have to
            # mean something. Read and not recomputed: this runs on **every**
            # forward, and writing an architecture down again every step of a
            # training run is a toll for an answer that cannot have changed.
            if node_id in declared:
                continue
            try:
                _declaration.written(self.implementation(node_id))
                continue
            except _declaration.CannotDeclare as why:
                raise ValueError(
                    f"`{node_id}` is kept, or feeds something that is, and what "
                    f"it was built with cannot be written down: {why}. That "
                    f"half of a key is what tells one "
                    f"`{type(self.implementation(node_id)).__name__}` from "
                    f"another, so without it two of them would be kept under "
                    f"one name and you would get the other one back. Fix what "
                    f"it holds, or say `.cached(salt=...)` and take the naming "
                    f"on yourself"
                ) from why
        for node_id, state in self.frozen().items():
            if state is not None:
                continue
            implementation = self.implementation(node_id)
            if not _has_state(implementation):
                continue
            raise ValueError(
                f"`{node_id}` is declared frozen and has state, and nobody has "
                f"settled it: the digest of what it is settled at — weights, or "
                f"the dataset it reads — is what puts it in its key, so without "
                f"it two different states of `{type(implementation).__name__}` "
                f"would be kept under one name and you would get the other one "
                f"back. Call "
                f"`somatize.torch.freeze(g)` before running — or "
                f"`somatize.data.settle(g)` if what it holds is a dataset. "
                f"Declaring it is this graph's half, making it true belongs to "
                f"whoever knows how to hash what is inside"
            )

    def _share_out(self, broker: Broker) -> list[tuple[list[str], dict[str, Any]]]:
        """Which nodes fall to each **wire**, and which host names share it.

        Grouped by wire and not by host: two names for one process get one
        catalog, and provisioning it twice with half each time leaves it without
        the other half. A host the broker does not know is left out rather than
        raised over — whichever way the run goes, it says so with the slice in
        front of it.
        """
        hosts = self.hosts()
        share: dict[bytes, tuple[list[str], dict[str, Any]]] = {}
        for host in sorted(set(hosts.values())):
            token = broker.token_for(host)
            if token is None:
                continue
            names, nodes = share.setdefault(token, ([], {}))
            names.append(host)
            for node_id, where in hosts.items():
                if where == host:
                    nodes[node_id] = self.implementation(node_id)
        return list(share.values())

    def _share_out_by_carrier(self, workers: dict[str, Any]) -> dict[Carrier, dict[str, Any]]:
        """The same, for a dict of things that carry rather than a broker.

        Grouped by the object, because that is what identity means when the
        caller is the one holding it.
        """
        hosts = self.hosts()
        share: dict[Carrier, dict[str, Any]] = {}
        for host, worker in workers.items():
            if not hasattr(worker, "carry"):
                raise ValueError(
                    f"provisioning takes a Broker, or a dict from host to something "
                    f"that carries; for `{host}` a `{type(worker).__name__}` arrived"
                )
            theirs = share.setdefault(cast(Carrier, worker), {})
            for node_id, where in hosts.items():
                if where == host:
                    theirs[node_id] = self.implementation(node_id)
        return share

    @classmethod
    def somatize(cls, topology: _dsl.Topology) -> Graph:
        """Materializes an expression into an executable graph.

        You think it, Soma somatizes it::

            Graph.somatize(Source() >> (Left() | Right()) >> Mean())
        """
        return _dsl.somatize(cls, topology)


def _has_state(implementation: object) -> bool:
    """Whether this node has anything worth hashing, asked as a duck. Three and
    not two since there are sources: weights are `state_dict` or `parameters`,
    and a dataset is `version`. One question — *what is this node settled at* —
    and one failure if nobody answers it.
    """
    return any(
        getattr(implementation, what, None) is not None
        for what in ("state_dict", "parameters", "version")
    )
