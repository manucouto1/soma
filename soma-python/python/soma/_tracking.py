"""Tracked runs for the native training loop.

Installs ``graph.track_run(...)`` — a context manager around
``Graph.begin_run`` that snapshots the graph topology into the run
directory, exposes the run to the audit/orchestrator via
``py_state['active_run']``, and finalizes the run status even on
exceptions::

    with g.track_run("mos-baseline", tags=["mos"]) as run:
        with g.gradient_audit(channels=True) as audit:
            for epoch in range(30):
                run.log_epoch(epoch, total=30)
                ...  # native loop: context/forward/backward/step
                run.log("val_f1", evaluate(g), step=epoch)
    # .soma/runs/<run_id>/ now holds manifest, status, graph.json/.mmd,
    # events.jsonl, metrics.jsonl and diagnostics/.
"""

from __future__ import annotations

import contextlib
import pathlib
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
) -> Iterator:
    run = self.begin_run(name, root=root, kind=kind, tags=list(tags))
    run_dir = pathlib.Path(run.dir)
    (run_dir / "graph.json").write_text(self.graph_json())
    (run_dir / "graph.mmd").write_text(self.to_mermaid())
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
