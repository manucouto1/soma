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

## One table of colours, and it is not in this file any more

The fill says **where a node runs**, and nothing else. Whether it is cached,
frozen or mapped is a badge in the label: three facts cannot share one fill, and
inventing a precedence between them would only hide two of the three.

The table is looked up with `[]` and never with `.get(…, default)`. In the
original soma the same colours lived in four tables keyed by the same strings,
two of which ended in a catch-all arm — so a typo came out as the alarm colour
instead of failing. Here a typo raises.

When a second figure arrived — a run, drawn — the same rule applied one level up,
so the table moved to `soma_next._theme` and both read it from there.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from soma_next import _theme

__all__ = ["Box", "TOO_MANY", "boxes", "figure", "steps"]

#: How wide a character is at the label's size. An estimate and not a
#: measurement — there is no font metric available here — and it is what the
#: original sized its SVG with too.
CHAR = 7.4

NODE_H = 44.0
"""How tall a node's box is with two lines in it, and the floor for any."""

LINE_H = 15.0
"""One line of label. A box grows with what is written in it — a node with a
device, three badges and a flag is four lines, and a fixed height would have
written them over its own outline."""

LAYER_H = 24.0
"""One layer of an expanded node."""

LAYER_GAP = 5.0
"""Between two of them. Small: they are a stack and not a sequence, and an
arrow does not go between them."""

MIN_W = 96.0
"""How narrow a node's box may get, however short its name."""

PAD_X = 14.0
"""The room between a node's longest line and the side of its box."""

PAD_Y = 10.0
"""And above and below what is written in it."""

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

PALETTE = _theme.PALETTE
"""Fill, outline and ink, by what the thing is. **The only table**, and it now
lives in `_theme` because there is a second figure — a product whose graph is
light and whose curves are dark is two products."""


@dataclass(frozen=True)
class Box:
    """One rectangle, placed.

    `kind` is `"node"`, `"wave"`, `"remote"` or `"layer"`; a layer's `mark` says
    what **sort** of thing it is, which is what decides how it is drawn. A
    `Linear` and a `Sigmoid` are not the same kind of thing and drawing them the
    same says they are.
    """

    kind: str
    x: float
    y: float
    w: float
    h: float
    node: str | None = None
    label: str | None = None
    mark: str | None = None
    shape: str | None = None

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


def boxes(plan, labels=None, inside=None):
    """Where every box goes, for a plan as `Graph.plan_json()` gives it.

    `labels` maps a node id to the lines that will be written in it, and is only
    read to work out how wide the box has to be; without it a node is as wide as
    its id.

    `inside` maps a node to `[(path, what), ...]` — what it is **made of** — and
    turns its box into a frame with those stacked in it. It is data and this
    module does not know where it came from: `soma_next.torch.architecture`
    reads it off the modules, and something that is not torch could answer the
    same question about itself.

    Pure, and with no plotly anywhere near it — which is what makes the layout
    testable on its own.
    """
    out: list[Box] = []
    _place(plan, 0.0, 0.0, labels or {}, out, inside)
    return out


def figure(graph, overlay=None, inside=None):
    """The graph as a `plotly.graph_objects.Figure`.

    Everything drawn is read back from what was declared — `nodes()`, `edges()`,
    `devices()`, `hosts()`, `cached()`, `frozen()`, `mapped_nodes()`,
    `identities()`, `fingerprints()` — so this never runs anything and never
    needs a store.

    `inside` opens a node up: `{node: [(path, what), ...]}`, which
    `soma_next.torch.architecture` reads off the modules a node holds. A node is
    often a whole architecture and a cube is not a picture of one — so its box
    becomes a **frame**, which is the shape a `Wave` and a `Remote` already are.

    `overlay` is what **happened**, laid over what was declared:
    `{node: [flag, ...]}`, which is what `soma_next.health.overlaid` builds out
    of a diagnosis. An empty one has to give a byte-identical drawing, and that
    is a test — it is what lets the declaration keep being drawable by somebody
    who has never run anything.

    It gets a **channel of its own**. The fill goes on saying where a node runs
    and nothing else; health is the outline and a badge. Recolouring the fill
    would be two facts in one channel, and the answer to *is this unhealthy*
    would have eaten the answer to *where does it run*.
    """
    go = _theme.plotly()
    import json

    plan = json.loads(graph.plan_json())
    devices, hosts = graph.devices(), graph.hosts()
    cached, frozen = graph.cached(), graph.frozen()
    mapped, identities = set(graph.mapped_nodes()), graph.identities()
    fingerprints = graph.fingerprints()

    ill = dict(overlay or {})
    labels = {
        node: _lines(node, identities.get(node), devices.get(node), badges, ill.get(node))
        for node in graph.nodes()
        for badges in [_badges(node, cached, frozen, mapped)]
    }
    placed = boxes(plan, labels, inside)
    where = {box.node: box for box in placed if box.kind == "node"}

    figure = go.Figure()
    if not where:
        return _nothing(figure, go)

    shapes, notes = [], []
    for box in placed:
        if box.kind == "layer":
            # What a node is made of, drawn by **what it is**: something that
            # holds weights gets a box, and a non-linearity gets a mark, because
            # a box says *there is something living here* and an activation has
            # nothing to live.
            fill, line, ink, _ = _theme.MARKS.get(box.mark or "other", _theme.MARKS["other"])
            if box.node in ill:
                line, ink = _theme.SERIES["alarm"], _theme.SERIES["alarm"]
            shapes.append(_rect(box, fill, line, 1.0))
            notes.append(
                _text(box.cx, box.cy, _safe(_labelled(box)), ink, size=10)
            )
        elif box.kind == "node":
            fill, line, ink = PALETTE[_family(devices.get(box.node))]
            width = 1.4
            if box.node in ill:
                # The one place in this library where a colour means bad. It is
                # the outline and never the fill: two facts, two channels.
                line, width = _theme.SERIES["alarm"], 2.6
            shapes.append(_rect(box, fill, line, width))
            lines = labels[box.node]
            # A node that was opened writes its name at the **top**, where a
            # frame's label goes; a plain one keeps it in the middle.
            opened = (inside or {}).get(box.node)
            at = (box.y + PAD_Y + LINE_H * len(lines) / 2) if opened else box.cy
            notes.append(_text(box.cx, at, "<br>".join(lines), ink))
        else:
            fill, line, ink = PALETTE[box.kind]
            shapes.append(_rect(box, fill, line, 1.6, dash="dot"))
            notes.append(
                _text(box.x + FRAME_PAD, box.y + FRAME_HEAD / 2, box.label, ink, left=True)
            )

    # Outside every box, so a routed edge never has to guess which way is clear.
    span = (min(box.x for box in placed), max(box.x + box.w for box in placed))
    lanes, boxed = {}, list(where.values())
    for node, comes_from in steps(plan):
        for source in comes_from:
            if source not in where:
                continue
            from_, to = where[source], where[node]
            if not _crosses(from_, to, boxed):
                notes.append(_arrow(from_.cx, from_.y + from_.h, to.cx, to.y))
                continue
            # One lane per edge on that side, handed out in declaration order so
            # the same graph is drawn the same way twice.
            side = -1 if abs(from_.cx - span[0]) <= abs(span[1] - from_.cx) else 1
            apart = lanes[side] = lanes.get(side, -1) + 1
            around, head = _routed(from_, to, span, apart)
            shapes.extend(around)
            notes.append(head)

    figure.add_trace(
        go.Scatter(
            x=[box.cx for box in where.values()],
            y=[box.cy for box in where.values()],
            mode="markers",
            marker={"size": 1, "opacity": 0},
            hoverinfo="text",
            hovertext=[
                _hover(
                    node, identities, devices, hosts, cached, frozen, mapped,
                    fingerprints, ill.get(node),
                )
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
        **_theme.layout(
            margin={"l": 16, "r": 16, "t": 16, "b": 16},
            width=min(1100, max(360, span_x + 80)),
            height=min(1400, max(240, span_y + 80)),
        ),
    )
    return figure


def _measure(plan, labels, inside=None):
    """How much room a plan takes, before anybody decides where it goes."""
    if plan == "Empty":
        return 0.0, 0.0
    (kind, body), = plan.items()
    if kind == "Execute":
        return _node_size(body["node"], labels, inside)
    if kind == "Remote":
        w, h = _measure(body["inner"], labels, inside)
        return w + 2 * FRAME_PAD, h + FRAME_PAD + FRAME_HEAD
    sizes = [_measure(child, labels, inside) for child in body]
    if kind == "Sequence":
        return (
            max(w for w, _ in sizes),
            sum(h for _, h in sizes) + GAP_Y * (len(sizes) - 1),
        )
    return (
        sum(w for w, _ in sizes) + GAP_X * (len(sizes) - 1) + 2 * FRAME_PAD,
        max(h for _, h in sizes) + FRAME_PAD + FRAME_HEAD,
    )


def _place(plan, x, y, labels, out, inside=None):
    """Hands out positions, top down, from a size already known."""
    if plan == "Empty":
        return
    (kind, body), = plan.items()
    width, height = _measure(plan, labels, inside)

    if kind == "Execute":
        node = body["node"]
        out.append(Box("node", x, y, width, height, node=node))
        _stack(node, (inside or {}).get(node), x, y, width, labels, out)
    elif kind == "Remote":
        out.append(Box("remote", x, y, width, height, label=body["host"]))
        _place(body["inner"], x + FRAME_PAD, y + FRAME_HEAD, labels, out, inside)
    elif kind == "Sequence":
        top = y
        for child in body:
            w, h = _measure(child, labels, inside)
            _place(child, x + (width - w) / 2, top, labels, out, inside)
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
    return max(MIN_W, CHAR * max(len(_bare_markup(line)) for line in lines) + 2 * PAD_X)


def _node_size(node, labels, inside=None):
    """A node's box: its own lines, plus room for what it is made of.

    A node with an `inside` becomes a **frame** — the same shape a `Wave` or a
    `Remote` already is — because that is what it turns out to be: a thing that
    contains things. Nothing new had to be invented for it, which is usually the
    sign that the layout was right.
    """
    lines = labels.get(node) or (node,)
    tall = max(NODE_H, 2 * PAD_Y + LINE_H * len(lines))
    held = (inside or {}).get(node)
    if not held:
        return _width(node, labels), tall
    rows = _rows_of(held)
    return (
        max(_width(node, labels), _inner_width(held) + 2 * FRAME_PAD),
        tall + rows * LAYER_H + max(rows - 1, 0) * LAYER_GAP + FRAME_PAD,
    )


def _stack(node, inside, x, y, width, labels, out):
    """The layers of an expanded node, placed **by what feeds what**.

    A stack cannot show a skip connection. What is placed here is a small DAG,
    by rank — the longest way down from an input — so what runs at the same
    depth sits on the same row and an edge that jumps a row is a skip and looks
    like one.

    No crossing minimisation and no Sugiyama: an architecture is mostly a line
    with a few jumps in it, and a heuristic that reorders rows would move boxes
    around between two runs of the same figure.
    """
    if not inside:
        return
    lines = labels.get(node) or (node,)
    top = y + max(NODE_H, 2 * PAD_Y + LINE_H * len(lines))
    for one, place in _ranked(inside):
        row, across, wide = place
        # The height is the kind's, and it is decided **here** and not when it
        # is drawn: a figure that paints something other than the box it laid
        # out has two truths in it, and the tests can only see one of them.
        tall = LAYER_H * _theme.MARKS.get(one.kind, _theme.MARKS["other"])[3]
        out.append(
            Box(
                "layer",
                x + FRAME_PAD + across,
                top + row * (LAYER_H + LAYER_GAP) + (LAYER_H - tall) / 2,
                wide,
                tall,
                node=f"{node}.{one.path}",
                label=one.label,
                mark=one.kind,
                shape=one.shape,
            )
        )


def _ranked(inside):
    """Where each layer goes: `(layer, (row, x, width))`.

    The rank is the longest way down, which is what puts a skip's two ends on
    rows that are not adjacent — and that gap is the whole reason the picture is
    worth drawing.
    """
    layers = {one.path: one for one in inside.layers}
    feeds = {path: [] for path in layers}
    for a, b in inside.edges:
        if a in layers and b in layers:
            feeds[b].append(a)
    rank, order = {}, list(layers)
    for path in order:
        rank[path] = 1 + max((rank.get(one, 0) for one in feeds[path]), default=-1)
    rows = {}
    for path in order:
        rows.setdefault(rank[path], []).append(layers[path])

    placed, widest = [], _inner_width(inside)
    for row, beside in sorted(rows.items()):
        each = (widest - LAYER_GAP * (len(beside) - 1)) / len(beside)
        for at, one in enumerate(beside):
            placed.append((one, (row, at * (each + LAYER_GAP), each)))
    return placed


def _inner_width(inside):
    """How wide the widest row of an architecture has to be."""
    return max(
        (CHAR * len(_layer_text(one)) + 2 * PAD_X for one in inside.layers),
        default=MIN_W,
    )


def _rows_of(inside):
    """How many rows deep it is."""
    return len({place[0] for _, place in _ranked(inside)}) if inside else 0


def _layer_text(one):
    """What is written on a layer: what it is, and what it produces."""
    return one.label + (f"   {one.shape}" if one.shape else "")


def _bare_markup(line):
    """A label's length as it is read, not as it is written: the flag badge
    carries a `<span>` that takes no room on the page."""
    import re

    return re.sub(r"<[^>]+>", "", line)


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


def _lines(node, identity, device, badges, flags=None):
    """What is written inside a node's box: at most four lines."""
    lines = [_safe(node)]
    if identity and not _named_after(node, identity):
        lines.append(_safe(identity))
    tail = ([device] if device and device != "cpu" else []) + badges
    if tail:
        lines.append(_safe(" · ".join(tail)))
    if flags:
        # The names and not a count: `2 findings` is a number somebody has to go
        # and look up, and the whole point of putting it on the box is not
        # having to.
        said = " · ".join(sorted({_bare(one) for one in flags}))
        lines.append(f"<span style='color:{_theme.SERIES['alarm']}'>⚠ {_safe(said)}</span>")
    return tuple(lines)


def _labelled(box):
    """What is written on a layer: what it is, and what it produces.

    The shape is on it and not on the hover, because it is the one thing that
    makes a **bottleneck** a picture: `512 → 8 → 512` is visible and
    `Linear · Linear · Linear` is not.
    """
    return box.label + (f"   {box.shape}" if box.shape else "")


def _bare(flag):
    """A flag without what it counts: `DEAD_CHANNELS(7)` is `DEAD_CHANNELS` on a
    box, and the seven is on the hover where there is room for it."""
    return flag.split("(", 1)[0]


def _named_after(node, identity):
    """Whether the id says nothing the class name does not.

    A node with no id of its own gets the class lowercased — `Tokenize` becomes
    `tokenize` — and a second one of the same class gets `_2` after it. Writing
    both lines then says the same word twice. The class is still on the hover,
    where it costs nothing.
    """
    lowered = identity.lower()
    return node == lowered or node.startswith(f"{lowered}_")


def _hover(
    node, identities, devices, hosts, cached, frozen, mapped, fingerprints, flags=None
):
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
    if flags:
        said.append("")
        for one in flags:
            said.append(f"<b>{_safe(one)}</b>")
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


def _text(x, y, said, ink, left=False, size=11):
    """One label, without an arrow attached to it."""
    return {
        "x": x,
        "y": y,
        "text": said,
        "showarrow": False,
        "font": {"size": size, "color": ink, "family": _theme.FONT},
        "align": "left" if left else "center",
        "xanchor": "left" if left else "center",
        "yanchor": "middle",
    }


LANE = 22.0
"""How far outside everything the first routed edge runs."""

LANE_APART = 14.0
"""And how far apart the next one runs from it. Without this, three edges that
all have to go around share one line and the figure stops saying there are
three."""

BEND = 16.0
"""How round its corners are."""

HEAD = 12.0
"""The last straight bit, which is the part that carries the arrowhead."""


def _crosses(source, target, obstacles):
    """Whether the straight edge would pass through a box that is not its ends.

    An edge drawn over a node reads as an edge **into** that node, which is the
    figure saying something that is not true. Cheap to ask and worth asking: it
    only happens where the nesting already stopped saying who feeds whom.
    """
    x0, y0 = source.cx, source.y + source.h
    x1, y1 = target.cx, target.y
    return any(
        _hits(x0, y0, x1, y1, box)
        for box in obstacles
        if box is not source and box is not target
    )


def _hits(x0, y0, x1, y1, box):
    """Segment against rectangle, by the slab test — exact, and about ten lines.

    Sampling the segment would miss a thin box, and a figure that is *usually*
    honest is the kind of thing nobody ever finds.
    """
    dx, dy = x1 - x0, y1 - y0
    near, far = 0.0, 1.0
    for delta, start, low, high in (
        (dx, x0, box.x, box.x + box.w),
        (dy, y0, box.y, box.y + box.h),
    ):
        if abs(delta) < 1e-9:
            if start < low or start > high:
                return False
            continue
        one, other = (low - start) / delta, (high - start) / delta
        near, far = max(near, min(one, other)), min(far, max(one, other))
        if near > far:
            return False
    return True


def _routed(source, target, span, apart):
    """One edge that cannot go straight, as `(shapes, annotation)`.

    Around means **outside everything**, down, and in through the side of what
    reads it. The lane is outside the whole drawing rather than outside the boxes
    in the way, because a lane threaded between two of them is a lane that will
    cross a third the next time the layout changes.

    `apart` is which lane on that side this one gets, so that edges which all
    have to go around stay countable.
    """
    left, right = span
    # The near side, so an edge that skips one box does not cross the figure.
    outward = -1 if abs(source.cx - left) <= abs(right - source.cx) else 1
    out = LANE + apart * LANE_APART
    lane = (left - out) if outward < 0 else (right + out)
    x0, y0 = source.cx, source.y + source.h
    into = target.x if outward < 0 else target.x + target.w
    y1 = target.cy
    path = (
        f"M {x0},{y0} "
        f"C {x0},{y0 + BEND} {lane},{y0} {lane},{y0 + BEND} "
        f"L {lane},{y1 - BEND} "
        f"C {lane},{y1} {lane},{y1} {into - outward * HEAD},{y1}"
    )
    return (
        [{"type": "path", "path": path, "xref": "x", "yref": "y",
          "line": {"color": _theme.MUTED, "width": 1.2}, "layer": "below"}],
        _arrow(into - outward * HEAD, y1, into, y1),
    )


def _arrow(ax, ay, x, y):
    """The head, and whatever straight run carries it."""
    return {
        "x": x,
        "y": y,
        "ax": ax,
        "ay": ay,
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


def _nothing(figure, go):
    """What a graph with no nodes looks like: a statement, not an exception."""
    del go
    figure.update_layout(
        annotations=[
            _text(0.5, 0.5, "empty graph — add nodes with g.node(...)", _theme.MUTED)
        ],
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
        **_theme.layout(
            width=420, height=160, margin={"l": 16, "r": 16, "t": 16, "b": 16}
        ),
    )
    return figure
