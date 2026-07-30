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

import json

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


def find_similar(
    query: str = "",
    *,
    like_run: str | None = None,
    limit: int = 5,
    research_line: str | None = None,
    tags: "list[str] | tuple[str, ...] | None" = None,
    half_life_days: float | None = None,
    root: str = ".soma",
) -> list[dict]:
    """Rank past experiments against a query — "what have I already
    tried that bears on this?".

    The score adds four terms rather than multiplying them::

        0.40·text + 0.25·architecture + 0.15·recency + 0.20·importance

    A product would let recency alone veto a year-old dead end, which is
    exactly the thing worth surfacing. Failures that carry a conclusion
    have their importance floored, for the same reason.

    Pass ``like_run`` to match an experiment's *architecture* as well as
    (or instead of) its text.

    Each hit is a dict with ``score``, ``why`` (the score broken down),
    ``components`` and ``record``.
    """
    return json.loads(
        _soma.kb_find_similar_json(
            query,
            like_run=like_run,
            limit=limit,
            research_line=research_line,
            tags=list(tags) if tags else None,
            half_life_days=half_life_days,
            root=root,
        )
    )


def record_conclusion(
    run_id: str,
    notes: str,
    *,
    hypothesis: str | None = None,
    tags: "list[str] | tuple[str, ...] | None" = None,
    root: str = ".soma",
) -> str:
    """Retain what you learned about a run. Returns the amendment id.

    Appended as its own journal line, so the original record is never
    rewritten — a note added today cannot corrupt what was recorded when
    the run happened. The text is indexed like any other, so a later
    :func:`find_similar` surfaces it.

    Worth doing for failures especially: that is what stops the next
    person, or the next agent, rediscovering the same dead end.
    """
    return _soma.kb_record_conclusion(
        run_id,
        notes,
        hypothesis=hypothesis,
        tags=list(tags) if tags else None,
        root=root,
    )


def lineage(run_id: str, *, root: str = ".soma") -> dict | None:
    """An experiment with its ancestors and descendants, or None.

    ``{"focus": record, "ancestors": [...], "descendants": [{"record":
    ..., "depth": n}, ...]}`` — ancestors oldest first, descendants in
    pre-order so the list reads as an indented tree.
    """
    raw = _soma.kb_lineage_json(run_id, root=root)
    return json.loads(raw) if raw is not None else None


def diff(a: str, b: str, *, root: str = ".soma") -> dict:
    """The move between any two experiments, related or not.

    A recorded ``derivation`` only exists on a parent→child edge. This
    computes the same diff for two records that never met, which is the
    comparison you want between sibling branches.
    """
    return json.loads(_soma.kb_diff_json(a, b, root=root))
