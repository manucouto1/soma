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

    def forward(self, input=None, *, driver=None, workers=None, store=None):
        """Executes the whole graph and returns what it produced.

        With `workers={"w1": Worker.at(...)}` you say what each host resolves
        to. **This method sends the nodes**, not you: the graph is the one that
        knows which goes where.

        The `driver` goes with them. It serves what the steps ask for here and
        over there alike, so a node that returns `Await` does not stop being
        executable for having been sent away.

        `store` is a directory: with one, whatever was declared `.cached()` is
        looked up before being computed and kept afterwards.
        """
        self._check_it_was_obeyed()
        for worker, nodes in self._share_out(workers or {}).items():
            worker.carry(nodes, driver)
        return super().forward(input, driver=driver, workers=workers, store=store)

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
        if not self.cached():
            return
        for node_id, state in self.frozen().items():
            if state is not None:
                continue
            implementation = self.implementation(node_id)
            if not _has_state(implementation):
                continue
            raise ValueError(
                f"`{node_id}` is declared frozen and has state, and nobody has "
                f"settled it: the digest of its weights is what puts them in its "
                f"key, so without it two different checkpoints of "
                f"`{type(implementation).__name__}` would be kept under one name "
                f"and you would get the other one back. Call "
                f"`soma_next.torch.freeze(g)` before running — declaring it is "
                f"this graph's half, making it true is torch's"
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
    """Whether this node has anything worth hashing, asked with the same duck as
    `parameters()`: whoever answers neither has no state, and does not stop
    being a node for it."""
    return any(
        getattr(implementation, what, None) is not None
        for what in ("state_dict", "parameters")
    )
