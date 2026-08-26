"""The reasoning of an investigation, read out of a store.

Functions over a `Store`, like `gather` and `take`: what is being read is a
folder, and a class around one would be the store with a longer name. `tree=`
on every call and never read from `soma-tree.toml` — a second reader of that
file is how a `--tree` reaches the journal and not the walk, with nothing
saying so.

There is no price list here, unlike `somatize.record`: an investigation is
fifty-odd moves, so it is read whole, once, and everything below is a walk over
what came back.

Everything cross-references by **name**. The id is in each row because it says
what order the moves were made in, and nothing else.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from somatize import _somatize

if TYPE_CHECKING:
    from somatize._somatize import Store

#: One line of an answer here. `Any` because the columns differ by question and
#: are named in each docstring.
Row = dict[str, Any]

__all__ = ["covered", "cites", "folds", "moves", "says", "standing"]

KINDS = ("question", "hypothesis", "attempt", "finding", "decision")
"""The five kinds, and there are no more."""

STANDINGS = (
    "open",
    "answered",
    "partly",
    "validated",
    "partly-validated",
    "refuted",
    "partly-refuted",
    "disputed",
    "depends",
)
"""How a question or a hypothesis can stand. Derived from what was said and
never read from a field, which is what lets one go back to `open` on its own."""


def moves(store: "Store", *, tree: str) -> list[Row]:
    """Every move, in the order they were made — which is the order siblings are
    read and drawn in.

    | column | what it is |
    |---|---|
    | `name` | what its author called it, and how it is reached everywhere else |
    | `id` | the store's slot; what says which of two variants was tried first |
    | `kind` | one of `KINDS` |
    | `prose` | what somebody wrote |
    | `under` | what it hangs under, multivalued: one move can answer two questions |
    | `about` | where it belongs when it hangs nowhere — a decision's scope names what it abandons |
    | `scope` | where it holds, as roots; empty is everywhere |
    | `cites` | `[{"what": …, "id": …}]`; only an attempt and a finding carry one |
    | `course` | `pursue`, `abandon`, `superseded`; only a decision |
    | `standing` | one of `STANDINGS`; only a question or a hypothesis, `None` otherwise |
    | `pruned` | whether something abandoned the line it is on — derived, never stored |
    | `who`, `when` | who wrote it down and at what second |

    `standing` is `None` and not `"open"` on an attempt: an attempt is not a
    question nobody has answered.
    """
    return _read(store, tree)["moves"]


def says(store: "Store", *, tree: str) -> list[Row]:
    """Everything anybody said from one move towards another: `from`, `says`,
    `to`, `scope` and `partly`, all in names.

    `answers`, `validates` and `refutes` are what make a standing; `combines`
    says an attempt **is** the composition of those, which is what lets *each
    worked alone, together they cancel* read as what it is.
    """
    return _read(store, tree)["says"]


def folds(store: "Store", *, tree: str) -> list[Row]:
    """The lines somebody abandoned: `root`, `by`, `course`, `why` and `hides`.

    One row per move a decision named, with `hides` naming what folds with it —
    so how many is `len` and why is in words. **Pruning never deletes**: a line
    that did not work is the most reusable thing an investigation produces.

    Folding what you have *read* is not here and never will be. It writes
    nothing down, because closing what you have read is not a claim about the
    investigation — it is the reader's, and so it is an app's.
    """
    return _read(store, tree)["folded"]


def standing(store: "Store", *, tree: str) -> dict[str, str]:
    """How each question and hypothesis stands, by name.

    A word, and the reason is in `says`: two edges of opposite sign whose scopes
    **touch** are `disputed`, and the same two that do not touch are `depends` —
    the answer depending on the case, which is the most informative outcome an
    investigation gives. Use `covered` to tell one from the other.
    """
    return {
        one["name"]: one["standing"]
        for one in _read(store, tree)["moves"]
        if one["standing"] is not None
    }


def covered(store: "Store", *, tree: str, by: list[str]) -> list[str]:
    """What a scope with those roots reaches: the roots and everything under
    them, in the order they were made.

    The one walk a reader cannot redo by hand and get right, because `under` is
    multivalued and a scope is a DAG and not a subtree. With it, *do these two
    scopes touch* is `set(a) & set(b)`. Fails if a name is nobody's here.
    """
    return _somatize.reasoning_covers(store, tree, list(by))


def cites(store: "Store", *, tree: str) -> dict[str, dict[str, list[str]]]:
    """The way back: `{what: {id: [move, …]}}` — which moves cite each commit,
    trial or configuration.

    A commit you cannot ask *what was this for* is a change without a motive.
    Derived from the citations here and **kept in no index**, so it is true the
    moment somebody cites one and cannot go stale. On real data one commit came
    back cited by five attempts, told apart only by their configurations.
    """
    said: dict[str, dict[str, list[str]]] = {}
    for one in _read(store, tree)["moves"]:
        for cited in one["cites"]:
            said.setdefault(cited["what"], {}).setdefault(cited["id"], []).append(one["name"])
    return said


def _read(store: "Store", tree: str) -> dict[str, Any]:
    """The whole reasoning, derived in Rust and handed over as JSON.

    Read whole and read again per call: an investigation is small, and a handle
    that cached it would answer with what the store said before the last thing
    somebody wrote from the terminal.
    """
    return json.loads(_somatize.reasoning(store, tree))
