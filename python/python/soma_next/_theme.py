"""One table of colours, for everything this library draws.

CU19 said it about the graph — *"looked up with `[]` and never with
`.get(…, default)`; in the original soma the same colours lived in four tables
keyed by the same strings, so a typo came out as the alarm colour instead of
failing"* — and the moment there was a second figure the same rule applied one
level up. A product whose graph is light and whose curves are dark is two
products.

Dark, and not as a preference dressed up as a decision: what these figures are
for is being looked at for a long time beside a training run, and a bright
rectangle in a notebook at two in the morning is a bright rectangle. The ink is
off-white rather than white and the ground is off-black rather than black,
because full-contrast edges buzz.

# What a colour is allowed to mean

The same discipline as the graph's fill: **one fact per channel**. Hue says
*where* something ran or *which* series it is; it never doubles as good-or-bad.
The only colour that means a judgement is `alarm`, and it is used for exactly one
thing — a `forward` that broke — which is a fact and not an opinion. Whatever
CU21 decides is *unhealthy* will need its own channel and it is not this one.
"""

from __future__ import annotations

INK = "#e7e9ef"
"""Text. Off-white: full white on a dark ground buzzes at small sizes."""

MUTED = "#868ea3"
"""Text that is there to be read only if you look for it: units, counts, axes."""

GROUND = "#101319"
"""The paper and the plot both. Off-black, and slightly blue so that the greys
above it do not read as brown."""

RAISED = "#1a1e28"
"""A surface that sits on the ground: the fill of a box, a bar's track."""

EDGE = "#2c3242"
"""A line that separates without being read as content: grids, outlines."""

PALETTE = {
    # Where a node runs. The graph has used these since CU19 and they keep
    # their meaning here: green is a device, orange is another machine.
    "cpu": (RAISED, "#3d4459", INK),
    "cuda": ("#12291f", "#3f9d6d", "#a9e7c5"),
    "meta": ("#171921", "#343a49", MUTED),
    "wave": ("rgba(0,0,0,0)", EDGE, MUTED),
    "remote": ("rgba(0,0,0,0)", "#eb6834", "#f2a681"),
}
"""Fill, outline and ink, by what the thing is. The only table."""

SERIES = {
    "loss": "#eb6834",
    "took": "#4fc3b0",
    "recalled": "#6f7bd1",
    "alarm": "#ef5f6b",
}
"""One colour per series. `alarm` is the only one that means a judgement, and it
is used for one thing: a `forward` that broke."""

FONT = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


def titled(what):
    """A title where a title goes: left, and against the **plot** and not the
    window — `xref="container"`, which is plotly's default, puts `x=0` hard
    against the edge and the first letter is clipped off."""
    return {
        "text": what,
        "x": 0,
        "xref": "paper",
        "xanchor": "left",
        "y": 0.97,
        "yanchor": "top",
        "font": {"size": 14},
    }


def layout(**over):
    """The layout every figure here starts from, with `over` on top.

    Kept in one place for the same reason the colours are: a figure that gets
    its margins from somewhere else is a figure that will drift.
    """
    return {
        "paper_bgcolor": GROUND,
        "plot_bgcolor": GROUND,
        "font": {"family": FONT, "size": 12, "color": INK},
        "margin": {"l": 56, "r": 24, "t": 44, "b": 44},
        "hoverlabel": {
            "align": "left",
            "bgcolor": RAISED,
            "bordercolor": EDGE,
            "font": {"family": FONT, "color": INK},
        },
        # To the right, because the title is on the left and a legend that
        # lands on top of it is the first thing anybody notices about a figure.
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": MUTED},
        },
        **over,
    }


def axis(**over):
    """An axis that is there to be read and not to be looked at."""
    return {
        "gridcolor": EDGE,
        "zerolinecolor": EDGE,
        "linecolor": EDGE,
        "tickfont": {"color": MUTED, "size": 11},
        "title": {"font": {"color": MUTED, "size": 11}},
        **over,
    }


def plotly():
    """`plotly.graph_objects`, or an error that says how to get it."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise RuntimeError(
            "drawing needs plotly — install it with: pip install 'soma-next[viz]'"
        ) from e
    return go
