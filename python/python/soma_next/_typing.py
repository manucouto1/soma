"""Names for the shapes this package passes around but does not own.

Two kinds of thing live here, and they are not the same kind:

- **`Figure`** is a type from a library that ships no `py.typed`. Imported under
  `TYPE_CHECKING` so it reads as what it is; a checker resolves it to `Any`
  today, and the day plotly ships types the annotation gets better on its own
  without a line changing here.

- **`Fact`, `Overlay`, `Inside`** are shapes this package really does define and
  hands over as plain dicts and lists. They are aliases and not classes on
  purpose: a fact read back out of a store *is* the dict a watcher was given, and
  wrapping it in a type would make the two stop being interchangeable — which is
  the property the record's reader and its live view are built on.

It is a module of names and no code, so importing it costs nothing and anything
may.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from plotly.graph_objects import Figure as Figure

    from soma_next.torch._inside import Inside as _Inside
else:  # pragma: no cover - the alias only has to exist at runtime
    Figure = Any

#: One thing that happened: a `fact` key naming it and text beside it. The same
#: shape whether it arrived through a `Watcher` or was read back off a store,
#: which is what lets a live view and a report share one drawing function.
Fact: TypeAlias = "dict[str, Any]"

#: What a diagnosis says about a graph: the flags on each node, by id. Empty for
#: a node with nothing wrong, absent for a node nobody looked at — and those two
#: are not the same, which is why this is a mapping and not a list.
Overlay: TypeAlias = "dict[str, list[str]]"

#: What each node is made of, for the nodes somebody asked about — exactly what
#: `soma_next.torch.architecture` answers. Behind `TYPE_CHECKING` because the
#: class lives in the `torch` half and this module is imported by the half that
#: has no torch in it.
Inside: TypeAlias = "dict[str, _Inside]"
