"""Lineage helpers: which run the next one descends from.

Soma resolves a run's parent in four steps, most explicit first: the
``parent=`` argument, ``$SOMA_PARENT_RUN``, ``.soma/HEAD``, then nothing.
HEAD advances automatically after every *successful* run, so a linear
session builds a linear lineage with no bookkeeping::

    with g.track_run("baseline", params={"lr": 0.01}):
        ...                       # HEAD -> baseline

    with g.track_run("wider", params={"lr": 0.05}):
        ...                       # parent = baseline, HEAD -> wider

To go back and try something else, rewind::

    soma.checkout(baseline_id)    # HEAD -> baseline
    with g.track_run("deeper"):   # parent = baseline, a sibling of "wider"
        ...

The parent is never guessed from timestamps: "the run before this one"
is a different claim from "the run this one came from", and one false
edge poisons every metric delta computed downstream of it.
"""

from __future__ import annotations

from soma import _soma


def checkout(run_id: str, *, root: str = ".soma") -> None:
    """Point HEAD at ``run_id`` so the next run branches from it.

    Raises if the run does not exist under ``root`` — attaching an
    experiment to a parent that is not there is worse than not
    branching at all.
    """
    _soma.checkout_run(run_id, root=root)


def head(*, root: str = ".soma") -> str | None:
    """The run id the next run will descend from, or None if detached."""
    return _soma.read_head_run(root=root)


def detach(*, root: str = ".soma") -> None:
    """Clear HEAD: the next run starts its own research line."""
    _soma.clear_head_run(root=root)


def reindex(*, root: str = ".soma") -> int:
    """Rebuild ``<root>/experiments.jsonl`` from the run directories.

    Migration, backfill and disaster recovery in one call: the run
    directories are the source of truth, the journal is an index.
    Returns the number of records written.
    """
    return _soma.kb_reindex(root=root)
