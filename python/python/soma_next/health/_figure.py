"""A diagnosis, drawn.

    from soma_next.health import flags, profile

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

from soma_next import _theme
from soma_next.health._read import about, diagnose, seen

__all__ = ["flags", "profile"]

#: What a healthy update-to-weight ratio is, for the line drawn across it.
HEALTHY_RATIO = 1e-3


def profile(store, *, run, of="grad_norm", thresholds=None, last=None):
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


def flags(store, *, run, thresholds=None, last=None):
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


def _nothing(go, what):
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
