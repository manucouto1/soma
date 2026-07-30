"""Tracked runs for the native training loop.

Installs ``graph.track_run(...)`` — a context manager around
``Graph.begin_run`` (which snapshots the graph topology into the run
directory) that exposes the run to the audit/orchestrator via
``py_state['active_run']`` and finalizes the run status even on
exceptions.

``params`` are the hyperparameters that live outside the graph — they
are what lets a later variant record a ``ParamChanged`` derivation.
``parent`` names the run this one descends from; omit it and soma
resolves one from ``$SOMA_PARENT_RUN`` or ``.soma/HEAD`` (see
:func:`soma.checkout`). ``hypothesis`` records what you expected before
you saw the result — written afterwards it would be a conclusion, and
:func:`soma.record_conclusion` is where those go::

    with g.track_run("mos-baseline", tags=["mos"], params={"lr": 0.01}) as run:
        with g.gradient_audit(channels=True) as audit:
            for epoch in range(30):
                run.log_epoch(epoch, total=30)
                ...  # native loop: context/forward/backward/step
                run.log("val_f1", evaluate(g), step=epoch)
    # .soma/runs/<run_id>/ now holds manifest, status, graph.json/.mmd,
    # fingerprint.json, events.jsonl, metrics.jsonl and diagnostics/.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from soma._soma import Graph as _RustGraph


@contextlib.contextmanager
def _track_run(
    self: _RustGraph,
    name: str,
    *,
    root: str = ".soma",
    kind: str = "train",
    tags: tuple[str, ...] | list[str] = (),
    params: dict | None = None,
    parent: str | None = None,
    hypothesis: str | None = None,
) -> Iterator:
    run = self.begin_run(
        name,
        root=root,
        kind=kind,
        tags=list(tags),
        params=params,
        parent=parent,
        hypothesis=hypothesis,
    )
    self.py_state["active_run"] = run
    self.py_state["train_step"] = 0
    try:
        yield run
    except BaseException:
        run.finish("failed")
        raise
    else:
        run.finish("completed")
    finally:
        self.py_state.pop("active_run", None)
        self.py_state.pop("train_step", None)


_RustGraph.track_run = _track_run
