"""A diagnosis, drawn.

    from somatize.health import flags, profile

    profile(store, run="tuesday")        # the depth profile, which is the picture
    flags(store, run="tuesday")          # what is wrong, and what to do about it

## Why the profile is the picture

Vanishing is not a property of a layer, it is a **shape over depth**: with a
saturating non-linearity the backpropagated signal shrinks geometrically, so the
early layers go quiet while the last one still learns. One number per node says
nothing about that and the profile says all of it — which is why `about()` for
`VANISHING` tells you to look here rather than at the node it fired on.

Drawn in log, because the interesting range is six orders of magnitude and a
linear axis would show one bar and seven zeros.

## The one colour that means a judgement

Everywhere else in this library hue says **where** something ran and never
good-or-bad. Here it is allowed to mean bad, because here that is the subject:
this module draws opinions, and it is the only one that does. What it may not
do is put that colour on the figures that draw facts — CU20's curves stay as
they are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from somatize._typing import Figure, Inside, Overlay

if TYPE_CHECKING:
    from somatize._graph import Graph
    from somatize._somatize import Store, Thresholds

from somatize import _theme
from somatize.health._read import about, diagnose, seen

__all__ = ["Alerts", "alerts", "flags", "overlaid", "profile", "where"]

#: What a healthy update-to-weight ratio is, for the line drawn across it.
HEALTHY_RATIO = 1e-3


def profile(
    store: "Store",
    *,
    run: str,
    of: str = "grad_norm",
    thresholds: "Thresholds | None" = None,
    last: int | None = None,
) -> Figure:
    """One measurement per node, across the graph — the shape over depth.

    `of` is any number the audit wrote down: `grad_norm`, `update_ratio`,
    `param_norm`, `act_abs_mean`, `eff_rank`, `update_rank`. A node that raised
    a flag is drawn in the alarm colour and says which flag on its hover.

    Nodes are in the order they were declared, which for a chain is the order
    the gradient travelled — backwards. That is the axis the pathology lives on.
    """
    go = _theme.plotly()
    measured = seen(store, run=run, last=last)
    said = diagnose(store, run=run, thresholds=thresholds, last=last)
    nodes = [node for node in measured if of in measured[node]]
    if not nodes:
        return _nothing(go, f"{run} — nothing measured {of} here")

    values = [measured[node][of] for node in nodes]
    ill = [bool(said.get(node)) for node in nodes]
    figure = go.Figure(
        go.Bar(
            x=nodes,
            y=values,
            marker={
                "color": [
                    _theme.SERIES["alarm"] if one else _theme.SERIES["took"] for one in ill
                ],
                "line": {"color": _theme.EDGE, "width": 1},
            },
            customdata=[", ".join(said.get(node, [])) or "nothing tripped" for node in nodes],
            hovertemplate="<b>%{x}</b><br>%{y:.3e}<br>%{customdata}<extra></extra>",
        )
    )
    if of == "update_ratio":
        # The number practice puts a healthy layer at. A line and not a band,
        # because it is a rule of thumb and drawing it as a range would dress
        # it up as a measurement.
        figure.add_hline(
            y=HEALTHY_RATIO,
            line={"color": _theme.MUTED, "width": 1, "dash": "dot"},
            annotation={"text": "~1e-3", "font": {"color": _theme.MUTED, "size": 10}},
            # On the line and not in the corner: a label for a line that is not
            # touching it is a label for something else.
            annotation_position="top left",
        )
    return (
        figure.update_layout(
            **_theme.layout(
                title=_theme.titled(f"{run} — {of} across the graph"),
                height=320,
                showlegend=False,
                bargap=0.35,
            )
        )
        .update_yaxes(**_theme.axis(type="log", title_text=of))
        .update_xaxes(**_theme.axis(title_text="declared in this order"))
    )


def flags(
    store: "Store",
    *,
    run: str,
    thresholds: "Thresholds | None" = None,
    last: int | None = None,
) -> Figure:
    """What is wrong with each node, and what to do about it.

    A table and not a list, because a diagnosis without its advice is a word
    somebody has to go and look up.

    A node with nothing wrong is not a row. Empty would read as *checked and
    fine*, and no flags does not mean that: a metric nobody measured cannot
    raise one.
    """
    go = _theme.plotly()
    said = diagnose(store, run=run, thresholds=thresholds, last=last)
    rows = [(node, flag, about(flag)) for node, raised in said.items() for flag in raised]
    if not rows:
        measured = seen(store, run=run, last=last)
        return _nothing(
            go,
            f"{run} — nothing tripped in {len(measured)} node(s) measured"
            if measured
            else f"{run} — nothing was measured, so nothing can be said",
        )
    return go.Figure(
        go.Table(
            columnwidth=[0.8, 1.0, 4.0],
            header={
                "values": ["<b>node</b>", "<b>flag</b>", "<b>what it means</b>"],
                "fill_color": _theme.RAISED,
                "line_color": _theme.EDGE,
                "font": {"color": _theme.INK, "size": 12},
                "align": "left",
                "height": 30,
            },
            cells={
                "values": [[one[i] for one in rows] for i in range(3)],
                "fill_color": _theme.GROUND,
                "line_color": _theme.EDGE,
                "font": {"color": _theme.INK, "size": 11},
                "align": "left",
                "height": 40,
            },
        )
    ).update_layout(
        **_theme.layout(
            title=_theme.titled(f"{run} — {len(rows)} finding(s)"),
            height=100 + 40 * (len(rows) + 1),
            margin={"l": 24, "r": 24, "t": 52, "b": 16},
        )
    )


def _nothing(go: Any, what: str) -> Figure:
    """Nothing to draw is a statement and not an exception."""
    return go.Figure().update_layout(
        annotations=[
            {
                "x": 0.5,
                "y": 0.5,
                "xref": "paper",
                "yref": "paper",
                "text": what,
                "showarrow": False,
                "font": {"size": 12, "color": _theme.MUTED, "family": _theme.FONT},
            }
        ],
        **_theme.layout(xaxis={"visible": False}, yaxis={"visible": False}, height=140),
    )


def overlaid(
    graph: "Graph",
    store: "Store",
    *,
    run: str,
    thresholds: "Thresholds | None" = None,
    last: int | None = None,
    inside: Inside | None = None,
) -> Figure:
    """The graph, with what is wrong marked on the nodes it is wrong in.

    The answer to *where* — which is the question a diagnosis of a distributed
    graph actually raises. A list of flags says a node is ill; the graph says
    which node, on which machine, and what feeds it.

    Health gets a **channel of its own**: the fill goes on saying where a node
    runs, the outline turns to the alarm colour, and the flags are a badge under
    the name. Recolouring the fill would have let *is this unhealthy* eat *where
    does this run*, and on a graph spread over three machines that is the
    answer somebody came for.

    `inside` is what `somatize.torch.architecture` gives back, and passing it
    is what makes a finding land **on the layer it is about** rather than piling
    every one of them into the node's label. Without it a graph of four nodes
    with ten findings comes out ten times wider than it is tall with nothing
    readable in it, which is what it did.

    Findings from inside a node whose architecture was not drawn land on the
    node, because that is then the only box there is to mark.
    """
    return graph.figure(
        overlay=where(store, run=run, thresholds=thresholds, last=last, inside=inside),
        inside=inside,
    )


def where(
    store: "Store",
    *,
    run: str,
    thresholds: "Thresholds | None" = None,
    last: int | None = None,
    inside: Inside | None = None,
) -> Overlay:
    """A diagnosis folded onto the nodes of a graph: `{node: [flag, ...]}`.

    What `overlaid` hands to the figure, and what to hand to `graph.figure()`
    yourself if you are composing something else. A finding inside a node is
    named on its flag — `LEAKAGE in net.2` — because the box it lands on is the
    node and the layer would otherwise be lost.
    """
    drawn = {
        f"{node}.{one.path}"
        for node, made in (inside or {}).items()
        for one in made.layers
    }
    # And where a box was folded away into a `×N`, the box that stands for it.
    stands_for = {
        f"{node}.{was}": f"{node}.{now}"
        for node, made in (inside or {}).items()
        for was, now in made.folded.items()
    }
    folded: dict[str, list[str]] = {}
    for at, raised in diagnose(store, run=run, thresholds=thresholds, last=last).items():
        one = stands_for.get(at, at)
        if one in drawn:
            # It has a box of its own, so it goes on it and the node stays
            # readable.
            folded.setdefault(one, []).extend(raised)
            continue
        node, _, within = one.partition(".")
        said = folded.setdefault(node, [])
        said.extend(f"{flag} in {within}" if within else flag for flag in raised)
    # Six identical blocks folded into one box mean the same finding arrives six
    # times. Said once, in the order it first arrived.
    return {where: list(dict.fromkeys(said)) for where, said in folded.items()}


class Alerts:
    """What is wrong, as cards a notebook cell shows on its own.

        alerts(store, run="tuesday")

    The loud one. A table is for reading and this is for **noticing**: it is
    what the original framework put on the screen as toasts, and the reason it
    exists is that a finding nobody saw is a finding nobody had.

    HTML and not a figure, because that is what a card is. It carries the node,
    the flag, and what to do about it — the advice comes from the same place the
    thresholds do, so a card cannot say something the verdict did not.

    Outside a notebook it prints, so a script says the same thing without
    needing a browser.
    """

    def __init__(
        self,
        found: Overlay,
        run: str,
        measured: dict[str, dict[str, float | bool]],
    ) -> None:
        self.found = found
        self.run = run
        self.measured = measured

    def __bool__(self) -> bool:
        return bool(self.found)

    def __len__(self) -> int:
        return sum(len(raised) for raised in self.found.values())

    def __repr__(self) -> str:
        if not self.found:
            return self._quiet()
        lines = [f"{self.run} — {len(self)} finding(s)"]
        for where, raised in self.found.items():
            for flag in raised:
                lines.append(f"  ⚠ {where}: {flag}\n      {about(flag)}")
        return "\n".join(lines)

    def _quiet(self) -> str:
        """What to say when nothing tripped — which is not *healthy*."""
        if not self.measured:
            return f"{self.run} — nothing was measured, so nothing can be said"
        return f"{self.run} — nothing tripped in {len(self.measured)} place(s) measured"

    def _repr_html_(self) -> str:
        if not self.found:
            return (
                f'<div style="{_CARD};border-left:3px solid {_theme.SERIES["took"]};'
                f'color:{_theme.MUTED}">{_escaped(self._quiet())}</div>'
            )
        cards = [
            f'<div style="{_CARD};border-left:3px solid {_theme.SERIES["alarm"]}">'
            f'<div style="font-size:12px;color:{_theme.SERIES["alarm"]};'
            f'letter-spacing:.04em">⚠ {_escaped(flag)}</div>'
            f'<div style="font-size:14px;color:{_theme.INK};margin:2px 0 4px">'
            f"{_escaped(where)}</div>"
            f'<div style="font-size:12px;color:{_theme.MUTED};line-height:1.45">'
            f"{_escaped(about(flag))}</div></div>"
            for where, raised in self.found.items()
            for flag in raised
        ]
        return (
            f'<div style="font-family:{_theme.FONT};background:{_theme.GROUND};'
            f'padding:10px;border-radius:8px">'
            f'<div style="color:{_theme.MUTED};font-size:12px;margin:2px 0 8px">'
            f"{_escaped(self.run)} — {len(self)} finding(s)</div>"
            + "".join(cards)
            + "</div>"
        )


#: One card. Kept out of the f-strings because repeating it four times is how
#: two of them end up different.
_CARD = (
    "background:#1a1e28;border-radius:6px;padding:8px 12px;margin:0 0 6px;"
    "font-family:inherit"
)


def alerts(
    store: "Store",
    *,
    run: str,
    thresholds: "Thresholds | None" = None,
    last: int | None = None,
) -> Alerts:
    """What is wrong, loudly. See [`Alerts`]."""
    return Alerts(
        diagnose(store, run=run, thresholds=thresholds, last=last),
        run,
        seen(store, run=run, last=last),
    )


def _escaped(text: object) -> str:
    """What came from a node's name is not markup.

    The same guard the graph figure has, for the same reason: an id is the
    user's and a card is a page.
    """
    import html

    return html.escape(str(text))
