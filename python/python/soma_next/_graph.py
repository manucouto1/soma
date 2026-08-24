"""`Graph` — the Rust object plus what can only live in Python.

Almost all of `Graph` is in Rust. What cannot be is `somatize`: it receives an
expression made of Python objects and walks it, so it lives here.

It is written as a method in the class body, and not assigned onto the Rust
class at import time, because a `#[pyclass]` is an immutable type — attributes
cannot be hung on it — and because even if they could, what is assigned is
invisible to `help()`, to an IDE, and to a type checker.
"""

from __future__ import annotations

from soma_next import _dsl
from soma_next._soma_next import Graph as _RustGraph


class Graph(_RustGraph):
    """A computation graph: nodes, edges and what each one executes.

    Everything else — `node`, `edge`, `plan` and the topology queries — is
    inherited from the Rust class.
    """

    _slice_of = None
    """The graph this one is a piece of, for a graph run in pieces."""

    def node(self, *args):
        """Adds a node and returns its id, noting **what it was built with**.

        Here and not in `_dsl`, because a graph built by hand in a loop reaches
        the same door and has the same collision without it: `Embed(512)` and
        `Embed(64)` are one class and two answers, and what tells them apart is
        the only half of a key that lives in the object.

        Something that cannot be written down the same way in two processes is
        passed over in silence here, and refused in `_check_it_was_obeyed` if a
        cache turns out to depend on it. Declaring a graph is not the moment to
        fail: running one with a cache it cannot honour is.
        """
        from soma_next import _declaration

        node_id = super().node(*args)
        try:
            self.declared_as(node_id, _declaration.digest(self.implementation(node_id)))
        except _declaration.CannotDeclare:
            pass
        return node_id

    def figure(self, overlay=None, inside=None):
        """The graph drawn, as a `plotly.graph_objects.Figure`.

        Nothing is executed to draw it: everything on the figure was declared.
        Needs the `viz` extra; without it the error says so.

        `inside` opens a node up — `{node: [(path, what), ...]}`, which
        `soma_next.torch.architecture` reads off the modules it holds. A node is
        often a whole architecture and a cube is not a picture of one.

        `overlay` lays what **happened** over what was declared —
        `{node: [flag, ...]}`, which `soma_next.health.overlaid` builds out of a
        diagnosis. An empty one draws exactly what no overlay draws.
        """
        from soma_next import _figure

        return _figure.figure(self, overlay, inside)

    def _repr_mimebundle_(self, include=None, exclude=None):
        """What a notebook shows for `g` on its own: the figure.

        The *mimebundle* and not `_repr_html_`, because that is how a plotly
        figure actually reaches a cell — a notebook sanitises the `<script>` a
        hand-written HTML repr would need, which is the same wall the original
        soma hit and answered by writing its own SVG renderer.

        `None` when there is no plotly, when the graph is too big to read, and
        when plotly itself answers with nothing — which is what it does outside a
        notebook, where no renderer is configured. In all three the cell falls
        back to `__repr__`. Asking for `figure()` by hand still draws whatever
        you asked for: the guard is against a surprise, not against you.
        """
        from soma_next import _figure

        if len(self) > _figure.TOO_MANY:
            return None
        try:
            drawn = self.figure()
        except RuntimeError:
            return None
        return drawn._repr_mimebundle_(include=include, exclude=exclude) or None

    def forward(self, input=None, *, workers=None, store=None, watching=None):
        """Executes the whole graph and returns what it produced.

        With `workers={"w1": Worker.at(...)}` you say what each host resolves
        to. **This method sends the nodes**, not you: the graph is the one that
        knows which goes where.

        `store` is a directory or a `Store`: with one, whatever was declared
        `.cached()` is looked up before being computed and kept afterwards. Both
        and not one, because a `Store` is the only one of the two that can say
        "a bucket" — and a path is what whoever has a directory already has.

        `watching` is told what happened, as it happens::

            g.forward(x, watching=print)                    # in a notebook
            g.forward(x, watching=Recorder(store))          # kept
            g.forward(x, watching=[Recorder(store), draw])  # both

        A fact arrives as a `dict` with a `fact` key naming it and text beside
        it — the same shape it is written down as, so what you print is what you
        would find in the store. **A node on another machine is no different**:
        what its worker saw comes back down the connection that was already
        open, and says which host it was.
        """
        self._check_it_was_obeyed()
        self.provision(workers)
        return super().forward(input, workers=workers, store=store, watching=watching)

    def provision(self, workers):
        """Tells each worker what it is going to need, before the first node runs.

        `forward` calls it, so whoever runs a graph in one go never says it out
        loud. Whoever runs one **in pieces** — stage by stage, or its transpose —
        does not have to either: a piece provisions the graph it is a piece of,
        entire.

        That is the whole reason for the method to exist, and it is not a
        courtesy: a worker has **one** catalog, and half of one is a different
        catalog. Handing it over is refused mid-session and swallowed in silence
        by a worker that has not greeted yet, taking with it every activation and
        every optimizer state that lived over there.

        A worker that gets nothing is told nothing, for the same reason: an
        artifact with no nodes in it is a catalog too, and it would take the
        place of the one the next graph is about to send.
        """
        whole = self._slice_of or self
        for worker, nodes in whole._share_out(workers or {}).items():
            if nodes:
                worker.carry(nodes)

    def _check_it_was_obeyed(self):
        """That whoever was declared settled really was settled.

        The core cannot ask this. It says `.frozen()` means "this node's state
        does not change", and it cannot tell a node with **no state to settle** —
        a tokenizer — from one whose weights **nobody has hashed yet**: both
        arrive as a state of `None`. Only whoever knows what a weight is can
        tell, and here that is a duck: something with `state_dict` or
        `parameters` has state.

        And it is not a detail. Without the digest of the weights the key does
        not depend on them, so two different checkpoints of the same class share
        a name — and what comes back is the wrong tensor, with no error and no
        warning. It is the one failure a cache must not have, and it is checked
        wherever a cache is declared, store or no store.
        """
        from soma_next import _declaration

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
                f"`soma_next.torch.freeze(g)` before running — or "
                f"`soma_next.data.settle(g)` if what it holds is a dataset. "
                f"Declaring it is this graph's half, making it true belongs to "
                f"whoever knows how to hash what is inside"
            )

    def _share_out(self, workers):
        """Which nodes fall to each worker, grouped by worker and not by host.

        Two hosts can point at the same one, and provisioning it twice with half
        each time would leave it without the other half: the session opens with
        the first artifact that reaches it.
        """
        hosts = self.hosts()
        share = {}
        for host, worker in workers.items():
            if not hasattr(worker, "carry"):
                raise ValueError(
                    f"`workers` takes a dict from host to Worker; for `{host}` a "
                    f"`{type(worker).__name__}` arrived"
                )
            theirs = share.setdefault(worker, {})
            for node_id, where in hosts.items():
                if where == host:
                    theirs[node_id] = self.implementation(node_id)
        return share

    @classmethod
    def somatize(cls, topology):
        """Materializes an expression into an executable graph.

        You think it, Soma somatizes it::

            Graph.somatize(Source() >> (Left() | Right()) >> Mean())
        """
        return _dsl.somatize(cls, topology)


def _has_state(implementation):
    """Whether this node has anything worth hashing, asked as a duck: whoever
    answers none of them has no state, and does not stop being a node for it.

    Three and not two since there are sources: weights are `state_dict` or
    `parameters`, and a dataset is `version`. They are the same question — *what
    is this node settled at* — and the same failure if nobody answers it, which
    is two different things kept under one name.
    """
    return any(
        getattr(implementation, what, None) is not None
        for what in ("state_dict", "parameters", "version")
    )
