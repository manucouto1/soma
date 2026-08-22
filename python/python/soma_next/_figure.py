"""The graph, drawn. What was declared, before anything has run.

    g.figure()          # a plotly Figure, to show or to compose
    g                   # in a notebook: the same figure, straight in the cell

## What is drawn, and why it is the plan and not the graph

The **plan** — `compile` then `distribute` — because that is where the decisions
show: a `Wave` is what runs at once, a `Remote` is what crosses to another
machine. A bare list of edges says neither.

The plan is a **tree**, so placing it needs no layout engine and no crossing
heuristic: one pass upwards asking each subtree its size, one pass downwards
handing out positions. That is the whole algorithm.

| in the plan | on the figure |
|---|---|
| `Execute` | a box, filled by the device it runs on |
| `Sequence` | its children stacked, top to bottom |
| `Wave` | its children side by side, inside a frame |
| `Remote` | a frame labelled with the host |
| `Empty` | an empty figure that says so |

`Sequence` gets no frame of its own: top-to-bottom is already how the figure is
read, and the root is always one — a box around everything is a border.

## The arrows are not decoration

`decompose` in the core is a real series-parallel decomposition, and it has a way
out at the bottom (`plan.rs`): a graph that is **not** series-parallel — only
reachable through `node()`/`edge()`, never through the DSL — falls back to a flat
`Sequence`. There the nesting no longer says who feeds whom; the truth lives
entirely in each step's `from`.

So the boxes say **when**, and the arrows say **what feeds what**. For a graph
built with `>>` and `|` the two agree. For the other one the arrows are all there
is, and a figure without them would be a lie. The `N` — `a→c`, `a→d`, `b→d` — is
the case, and it is in the tests.

## One table of colours

The fill says **where a node runs**, and nothing else. Whether it is cached,
frozen or mapped is a badge in the label: three facts cannot share one fill, and
inventing a precedence between them would only hide two of the three.

The table is looked up with `[]` and never with `.get(…, default)`. In the
original soma the same colours lived in four tables keyed by the same strings,
two of which ended in a catch-all arm — so a typo came out as the alarm colour
instead of failing. Here a typo raises.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

__all__ = ["Box", "TOO_MANY", "boxes", "figure", "steps"]

#: How wide a character is at the label's size. An estimate and not a
#: measurement — there is no font metric available here — and it is what the
#: original sized its SVG with too.
CHAR = 7.4

NODE_H = 44.0
"""How tall a node's box is: two lines of text and the room around them."""

MIN_W = 96.0
"""How narrow a node's box may get, however short its name."""

PAD_X = 14.0
"""The room between a node's longest line and the side of its box."""

GAP_X = 26.0
"""Between two branches of a wave."""

GAP_Y = 38.0
"""Between two steps of a sequence: an arrow has to fit in here."""

FRAME_PAD = 14.0
"""How far a frame stands off what it contains."""

FRAME_HEAD = 24.0
"""The strip at the top of a frame where its label goes."""

TOO_MANY = 80
"""Past this many nodes a diagram stops being readable, so the notebook does not
draw one on its own. `figure()` still obeys if you ask it by hand — the guard is
against a surprise, not against you."""

PALETTE = {
    "cpu": ("#fcfcfb", "#c3c2b7", "#0b0b0b"),
    "cuda": ("#e8f5e9", "#2e7d32", "#1b5e20"),
    "meta": ("#f4f2ef", "#898781", "#52514e"),
    "wave": ("rgba(0,0,0,0)", "#c3c2b7", "#52514e"),
    "remote": ("rgba(0,0,0,0)", "#eb6834", "#a5401a"),
}
"""Fill, outline and ink, by what the thing is. The only table."""


@dataclass(frozen=True)
class Box:
    """One rectangle, placed. `kind` is `"node"`, `"wave"` or `"remote"`."""

    kind: str
    x: float
    y: float
    w: float
    h: float
    node: str | None = None
    label: str | None = None

    @property
    def cx(self) -> float:
        """The middle of it, across."""
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        """The middle of it, down."""
        return self.y + self.h / 2


def steps(plan):
    """Every `Execute` in the plan, as `(node, from)`, in declaration order.

    A `Remote` is entered: what a plan does does not depend on where it runs, and
    an arrow into a slice that travelled is still an arrow.
    """
    if plan == "Empty":
        return
    (kind, body), = plan.items()
    if kind == "Execute":
        yield body["node"], body["from"]
    elif kind == "Remote":
        yield from steps(body["inner"])
    else:
        for child in body:
            yield from steps(child)


def boxes(plan, labels=None):
    """Where every box goes, for a plan as `Graph.plan_json()` gives it.

    `labels` maps a node id to the lines that will be written in it, and is only
    read to work out how wide the box has to be; without it a node is as wide as
    its id. Pure, and with no plotly anywhere near it — which is what makes the
    layout testable on its own.
    """
    out: list[Box] = []
    _place(plan, 0.0, 0.0, labels or {}, out)
    return out


def figure(graph):
    """The graph as a `plotly.graph_objects.Figure`.

    Everything drawn is read back from what was declared — `nodes()`, `edges()`,
    `devices()`, `hosts()`, `cached()`, `frozen()`, `mapped_nodes()`,
    `identities()`, `fingerprints()` — so this never runs anything and never
    needs a store.
    """
    go = _plotly()
    import json

    plan = json.loads(graph.plan_json())
    devices, hosts = graph.devices(), graph.hosts()
    cached, frozen = graph.cached(), graph.frozen()
    mapped, identities = set(graph.mapped_nodes()), graph.identities()
    fingerprints = graph.fingerprints()

    labels = {
        node: _lines(node, identities.get(node), devices.get(node), badges)
        for node in graph.nodes()
        for badges in [_badges(node, cached, frozen, mapped)]
    }
    placed = boxes(plan, labels)
    where = {box.node: box for box in placed if box.kind == "node"}

    figure = go.Figure()
    if not where:
        return _nothing(figure, go)

    shapes, notes = [], []
    for box in placed:
        if box.kind == "node":
            fill, line, ink = PALETTE[_family(devices.get(box.node))]
            shapes.append(_rect(box, fill, line, 1.4))
            notes.append(_text(box.cx, box.cy, "<br>".join(labels[box.node]), ink))
        else:
            fill, line, ink = PALETTE[box.kind]
            shapes.append(_rect(box, fill, line, 1.6, dash="dot"))
            notes.append(
                _text(box.x + FRAME_PAD, box.y + FRAME_HEAD / 2, box.label, ink, left=True)
            )

    for node, comes_from in steps(plan):
        for source in comes_from:
            if source in where:
                notes.append(_arrow(where[source], where[node]))

    figure.add_trace(
        go.Scatter(
            x=[box.cx for box in where.values()],
            y=[box.cy for box in where.values()],
            mode="markers",
            marker={"size": 1, "opacity": 0},
            hoverinfo="text",
            hovertext=[
                _hover(node, identities, devices, hosts, cached, frozen, mapped, fingerprints)
                for node in where
            ],
            showlegend=False,
        )
    )

    span_x = max(box.x + box.w for box in placed)
    span_y = max(box.y + box.h for box in placed)
    figure.update_layout(
        shapes=shapes,
        annotations=notes,
        xaxis={"visible": False, "range": [-20, span_x + 20]},
        # Reversed, because the layout counts downwards the way a plan reads.
        yaxis={"visible": False, "range": [span_y + 20, -20], "scaleanchor": "x"},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        margin={"l": 16, "r": 16, "t": 16, "b": 16},
        width=min(1100, max(360, span_x + 80)),
        height=min(1400, max(240, span_y + 80)),
        hoverlabel={"align": "left"},
    )
    return figure


def _plotly():
    """`plotly.graph_objects`, or an error that says how to get it."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise RuntimeError(
            "drawing a graph needs plotly — install it with: pip install 'soma-next[viz]'"
        ) from e
    return go


def _measure(plan, labels):
    """How much room a plan takes, before anybody decides where it goes."""
    if plan == "Empty":
        return 0.0, 0.0
    (kind, body), = plan.items()
    if kind == "Execute":
        return _width(body["node"], labels), NODE_H
    if kind == "Remote":
        w, h = _measure(body["inner"], labels)
        return w + 2 * FRAME_PAD, h + FRAME_PAD + FRAME_HEAD
    sizes = [_measure(child, labels) for child in body]
    if kind == "Sequence":
        return (
            max(w for w, _ in sizes),
            sum(h for _, h in sizes) + GAP_Y * (len(sizes) - 1),
        )
    return (
        sum(w for w, _ in sizes) + GAP_X * (len(sizes) - 1) + 2 * FRAME_PAD,
        max(h for _, h in sizes) + FRAME_PAD + FRAME_HEAD,
    )


def _place(plan, x, y, labels, out):
    """Hands out positions, top down, from a size already known."""
    if plan == "Empty":
        return
    (kind, body), = plan.items()
    width, height = _measure(plan, labels)

    if kind == "Execute":
        out.append(Box("node", x, y, width, height, node=body["node"]))
    elif kind == "Remote":
        out.append(Box("remote", x, y, width, height, label=body["host"]))
        _place(body["inner"], x + FRAME_PAD, y + FRAME_HEAD, labels, out)
    elif kind == "Sequence":
        top = y
        for child in body:
            w, h = _measure(child, labels)
            _place(child, x + (width - w) / 2, top, labels, out)
            top += h + GAP_Y
    else:
        out.append(Box("wave", x, y, width, height, label="wave"))
        left, inside = x + FRAME_PAD, y + FRAME_HEAD
        room = height - FRAME_HEAD - FRAME_PAD
        for child in body:
            w, h = _measure(child, labels)
            _place(child, left, inside + (room - h) / 2, labels, out)
            left += w + GAP_X


def _width(node, labels):
    """How wide a node's box has to be to hold its longest line."""
    lines = labels.get(node) or (node,)
    return max(MIN_W, CHAR * max(len(line) for line in lines) + 2 * PAD_X)


def _family(device):
    """`cuda:0` and `cuda:1` are painted the same; nothing said means `cpu`."""
    return (device or "cpu").split(":")[0]


def _badges(node, cached, frozen, mapped):
    """The marks a node carries beside its name, in a fixed order."""
    marks = []
    if node in cached:
        marks.append("⟳ cached")
    if node in frozen:
        marks.append("❄ frozen")
    if node in mapped:
        marks.append("⋯ mapped")
    return marks


def _lines(node, identity, device, badges):
    """What is written inside a node's box: at most three lines."""
    lines = [_safe(node)]
    if identity and not _named_after(node, identity):
        lines.append(_safe(identity))
    tail = ([device] if device and device != "cpu" else []) + badges
    if tail:
        lines.append(_safe(" · ".join(tail)))
    return tuple(lines)


def _named_after(node, identity):
    """Whether the id says nothing the class name does not.

    A node with no id of its own gets the class lowercased — `Tokenize` becomes
    `tokenize` — and a second one of the same class gets `_2` after it. Writing
    both lines then says the same word twice. The class is still on the hover,
    where it costs nothing.
    """
    lowered = identity.lower()
    return node == lowered or node.startswith(f"{lowered}_")


def _hover(node, identities, devices, hosts, cached, frozen, mapped, fingerprints):
    """Everything that was said about a node, for the pointer."""
    said = [f"<b>{_safe(node)}</b>"]
    if identity := identities.get(node):
        said.append(f"class: {_safe(identity)}")
    said.append(f"device: {_safe(devices.get(node) or 'cpu')}")
    if host := hosts.get(node):
        said.append(f"host: {_safe(host)}")
    if node in cached:
        salt = cached[node]
        said.append("cached" + (f" (salt {_safe(salt)})" if salt else ""))
    if node in frozen:
        state = frozen[node]
        said.append("frozen" + (f" ({_safe(state)})" if state else ""))
    if node in mapped:
        said.append("mapped over its items")
    if written := fingerprints.get(node):
        said.append(f"written as {_safe(written)}")
    return "<br>".join(said)


def _safe(text):
    """Escapes what came from whoever declared the graph.

    Plotly reads a subset of HTML in labels and in hover text, so a node called
    `<script>` is not a curiosity: it is the same hole the original soma had a
    test for, and this is where it is closed.
    """
    return html.escape(str(text), quote=False)


def _rect(box, fill, line, width, dash=None):
    """One rectangle, in plotly's shape form."""
    return {
        "type": "rect",
        "x0": box.x,
        "y0": box.y,
        "x1": box.x + box.w,
        "y1": box.y + box.h,
        "fillcolor": fill,
        "line": {"color": line, "width": width, **({"dash": dash} if dash else {})},
        "layer": "below",
    }


def _text(x, y, said, ink, left=False):
    """One label, without an arrow attached to it."""
    return {
        "x": x,
        "y": y,
        "text": said,
        "showarrow": False,
        "font": {"size": 11, "color": ink, "family": "system-ui, sans-serif"},
        "align": "left" if left else "center",
        "xanchor": "left" if left else "center",
        "yanchor": "middle",
    }


def _arrow(source, target):
    """One edge, from the bottom of what produced to the top of what reads."""
    return {
        "x": target.cx,
        "y": target.y,
        "ax": source.cx,
        "ay": source.y + source.h,
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "text": "",
        "showarrow": True,
        "arrowhead": 2,
        "arrowsize": 1,
        "arrowwidth": 1.2,
        "arrowcolor": "#898781",
    }


def _nothing(figure, go):
    """What a graph with no nodes looks like: a statement, not an exception."""
    del go
    figure.update_layout(
        annotations=[_text(0.5, 0.5, "empty graph — add nodes with g.node(...)", "#52514e")],
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        width=420,
        height=160,
        margin={"l": 16, "r": 16, "t": 16, "b": 16},
    )
    return figure
