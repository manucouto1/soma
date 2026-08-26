"""The reasoning drawn: where an investigation opened into variants, and where
it converged.

Depth grows **to the right** and siblings stack downward, because a move carries
prose and columns two words across cannot be read. Nothing is draggable and no
axis is a preference: this draws something that already happened, so a position
is derived from the shape — a position somebody dragged would have to be stored,
and it is not a fact about the investigation.

The layout is `cards`, which is pure and has no plotly near it; `figure` draws
what it placed. That split is what makes the rules below testable rather than
looked at:

- a lane per line, never handed out twice — a leaf takes a row and no later
  branch reuses it, or three variants stack into one row pretending to be one
  history
- a parent centred over its children's **span** and not their average, or an
  uneven fan leans and looks like it is falling over
- siblings in the order they were made, never the order a walk arrives in
- what could not be reached is still drawn: a move nobody hung anywhere is work
  waiting for a place, not a move that hides
- a colour is never the only place something lives — how a question stands is
  written on it in words, every time

**Deriving is here, interacting is an app's.** Folding what you have read,
clicking through and editing are not in this file and will not be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from somatize import _theme
from somatize.reasoning._read import Row, _read

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from somatize._somatize import Store

__all__ = ["Card", "Edge", "cards", "figure"]

#: One arrow: `(from, to, how)`. `how` is `"under"` for the line a move hangs
#: on, `"again"` for a second parent — a move can answer two questions and
#: neither is the parent — and `"combines"` for an attempt that **is** the
#: composition of others.
Edge = tuple[str, str, str]

WIDE, TALL = 190.0, 46.0
"""How big a card is. Wide and short, which is the shape prose comes in."""

ACROSS, DOWN = 74.0, 24.0
"""The gap between columns, and between rows."""

ENOUGH = 34
"""How much of a name fits on a card before it is cut. The prose is in the
hover: a card is read for the shape and hovered for what it says."""


@dataclass(frozen=True)
class Card:
    """One move, placed. `hides` is how many a folded line takes with it, and is
    `None` on a card that is not folded — pruning says how many and why."""

    name: str
    kind: str
    x: float
    y: float
    w: float
    h: float
    said: str
    """What it stands at, in words: a standing, or a course, or nothing."""
    prose: str
    pruned: bool
    hides: int | None = None
    why: str | None = None


def cards(
    moves: "Sequence[Row]",
    says: "Sequence[Row]" = (),
    folds: "Sequence[Row]" = (),
    *,
    under: str | None = None,
) -> tuple[list[Card], list[Edge]]:
    """Where every card goes, and the edges between them.

    Pure, over the rows `somatize.reasoning` reads — nothing here touches a
    store, which is what makes the layout rules testable on their own.

    **Folding is what you hand it**: the lines in `folds` come folded, saying how
    many they hide and why, and passing none opens everything. An app that folds
    what its reader has already read hands its own list and needs nothing from
    here.
    """
    return _laid_out(list(moves), list(says), list(folds), under)


def figure(
    store: "Store",
    *,
    tree: str,
    folded: bool = True,
    under: str | None = None,
) -> "Figure":
    """The reasoning as a `plotly.graph_objects.Figure`, drawn from what is
    stored with nothing run again.

    Hue says which of the five kinds a move is and never whether it went well;
    how a question stands is written on it. The one outline that changes is
    `disputed` — edges of opposite sign whose scopes touch — because that is the
    one a reader has to go and settle.
    """
    go = _theme.plotly()
    read = _read(store, tree)
    placed, edges = cards(
        read["moves"],
        read["says"],
        read["folded"] if folded else (),
        under=under,
    )

    drawing = go.Figure()
    if not placed:
        return _nothing(drawing, "Nothing has been written down here yet.")

    where = {card.name: card for card in placed}
    shapes: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for card in placed:
        fill, line, ink = _theme.MOVES[card.kind]
        width = 1.4
        if card.pruned:
            # Abandoned and still there: a line that did not work is the most
            # reusable thing an investigation produces.
            fill, line, ink = _theme.RAISED, _theme.EDGE, _theme.MUTED
        elif card.said == "disputed":
            line, width = _theme.SERIES["alarm"], 2.4
        shapes.append(_rect(card, fill, line, width, "dot" if card.pruned else None))
        notes.append(_text(card.x + 10, card.y + 15, _shortened(card.name), ink, size=11))
        notes.append(_text(card.x + 10, card.y + 32, _under(card), _theme.MUTED, size=10))

    for source, target, how in edges:
        notes.append(_edge(where[source], where[target], how))

    drawing.add_trace(
        go.Scatter(
            x=[card.x + card.w / 2 for card in placed],
            y=[card.y + card.h / 2 for card in placed],
            mode="markers",
            marker={"size": 1, "opacity": 0},
            hoverinfo="text",
            hovertext=[_hover(card) for card in placed],
            showlegend=False,
        )
    )
    wide = max(card.x + card.w for card in placed)
    tall = max(card.y + card.h for card in placed)
    drawing.update_layout(
        shapes=shapes,
        annotations=notes,
        xaxis={"visible": False, "range": [-20, wide + 20]},
        # Reversed, so the first thing made is at the top and the walk goes
        # down and away from it. This is not `git log`: an exploration is read
        # from where it started, because what you want to see is what came
        # **out** of it.
        yaxis={"visible": False, "range": [tall + 20, -20], "scaleanchor": "x"},
        **_theme.layout(
            margin={"l": 16, "r": 16, "t": 16, "b": 16},
            width=min(1400.0, wide + 80),
            height=min(1600.0, tall + 80),
        ),
    )
    return drawing


def _laid_out(
    moves: list[Row],
    says: list[Row],
    folds: list[Row],
    under: str | None,
) -> tuple[list[Card], list[Edge]]:
    """The whole layout, and it is the only place a position is decided."""
    if under is not None and not any(one["name"] == under for one in moves):
        raise ValueError(f"nothing here is called `{under}`")
    by_name = {one["name"]: one for one in moves}
    hiding = {one["root"]: one for one in folds}

    below: dict[str, list[str]] = {}
    for one in moves:
        # `about` and not only `under`: a decision's scope names what it
        # abandons, and a decision drawn floating beside the line it ended is
        # the one thing a reader cannot join up.
        for parent in list(one["under"]) + list(one["about"]):
            below.setdefault(parent, []).append(one["name"])

    roots = (
        [under]
        if under is not None
        else [one["name"] for one in moves if not one["under"] and not one["about"]]
    )
    column = _columns(moves, below, roots)

    placed: dict[str, Card] = {}
    edges: list[Edge] = []
    rows: list[float] = [0.0]

    def lay(name: str, from_: str | None) -> float | None:
        one = by_name[name]
        if name in placed:
            # Drawn once and pointed at twice: the second parent arrives from
            # the side rather than the branch being repeated.
            if from_ is not None:
                edges.append((from_, name, "again"))
            return None
        if from_ is not None:
            edges.append((from_, name, "under"))
        shut = hiding.get(name)
        kids = [] if shut else below.get(name, [])
        spans = [row for kid in kids for row in [lay(kid, name)] if row is not None]
        # Centred over the **span** and not the average: with an uneven fan an
        # average leans, and the fan looks like it is falling over.
        row = (spans[0] + spans[-1]) / 2 if spans else _next(rows)
        placed[name] = Card(
            name=name,
            kind=one["kind"],
            x=column[name] * (WIDE + ACROSS),
            y=row * (TALL + DOWN),
            w=WIDE,
            h=TALL,
            said=one["standing"] or one["course"] or "",
            prose=one["prose"],
            pruned=one["pruned"],
            hides=len(shut["hides"]) if shut else None,
            why=shut["why"] if shut else None,
        )
        return row

    for root in roots:
        lay(root, None)
    for said in says:
        if said["says"] == "combines" and said["from"] in placed and said["to"] in placed:
            edges.append((said["to"], said["from"], "combines"))
    return [placed[one["name"]] for one in moves if one["name"] in placed], edges


def _next(rows: list[float]) -> float:
    """The next lane, and it is never handed back. Freeing one when a branch
    ends looks thrifty and stacks three variants into a row pretending to be one
    history."""
    rows[0] += 1.0
    return rows[0] - 1.0


def _columns(
    moves: list[Row],
    below: dict[str, list[str]],
    roots: list[str],
) -> dict[str, int]:
    """How deep each move is: the **longest** way down to it, so a move under two
    parents sits to the right of both rather than beside the nearer one.

    A walk and not a recursion over the DAG, because `under` is multivalued; a
    cycle cannot arrive here, since one is refused when it is written.
    """
    depth = {name: 0 for name in roots}
    reach = list(roots)
    while reach:
        name = reach.pop()
        for kid in below.get(name, ()):
            deeper = depth[name] + 1
            if depth.get(kid, -1) < deeper:
                depth[kid] = deeper
                reach.append(kid)
    return {one["name"]: depth.get(one["name"], 0) for one in moves}


def _under(card: Card) -> str:
    """The second line of a card: what it is, and what it stands at."""
    said = f"{card.kind} · {card.said}" if card.said else card.kind
    return f"{said} · ⋯{card.hides} folded" if card.hides else said


def _hover(card: Card) -> str:
    """What it says, and why it folded if it did. **Pruning says why or it is
    deletion with a nicer name.**"""
    lines = [f"<b>{_safe(card.name)}</b>", _safe(_under(card)), "", _safe(card.prose)]
    if card.why:
        lines += ["", f"<i>{_safe(card.why)}</i>"]
    return "<br>".join(lines)


def _edge(source: Card, target: Card, how: str) -> dict[str, Any]:
    """One arrow, from the right of a card into the left of another. The line a
    move hangs on is solid; a second parent and a composition are fainter,
    because they are said **about** the tree and are not the tree."""
    arrow = {
        "x": target.x,
        "y": target.y + target.h / 2,
        "ax": source.x + source.w,
        "ay": source.y + source.h / 2,
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "text": "",
        "showarrow": True,
        "arrowhead": 2,
        "arrowsize": 1,
        "arrowwidth": 1.2,
        "arrowcolor": _theme.MUTED,
    }
    if how == "under":
        return arrow
    return {**arrow, "arrowwidth": 1.0, "arrowcolor": _theme.EDGE, "opacity": 0.8}


def _rect(
    card: Card,
    fill: str,
    line: str,
    width: float,
    dash: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "rect",
        "x0": card.x,
        "y0": card.y,
        "x1": card.x + card.w,
        "y1": card.y + card.h,
        "fillcolor": fill,
        "line": {"color": line, "width": width, **({"dash": dash} if dash else {})},
        "layer": "below",
    }


def _text(x: float, y: float, said: str, ink: str, size: int = 11) -> dict[str, Any]:
    return {
        "x": x,
        "y": y,
        "text": said,
        "showarrow": False,
        "font": {"size": size, "color": ink, "family": _theme.FONT},
        "align": "left",
        "xanchor": "left",
        "yanchor": "middle",
    }


def _shortened(name: str) -> str:
    return name if len(name) <= ENOUGH else f"{name[: ENOUGH - 1]}…"


def _safe(text: object) -> str:
    """Prose goes into a hover, and plotly reads that as HTML."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _nothing(drawing: "Figure", what: str) -> "Figure":
    """An empty investigation is a statement, not an exception."""
    drawing.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": what,
                "showarrow": False,
                "font": {"size": 13, "color": _theme.MUTED, "family": _theme.FONT},
            }
        ],
        **_theme.layout(height=180),
    )
    return drawing
