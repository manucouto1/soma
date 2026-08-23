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
from dataclasses import dataclass, replace

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

LAYER_H = 36.0
"""One layer of an expanded node: two lines, because what it is and what it
produces are two different things and putting them on one line makes a reader
parse a sentence to find a number."""

LAYER_GAP = 16.0
"""Between two rows of an architecture. An arrow has to fit in here."""

GUTTER = 22.0
"""The lane down the side of an opened node where a **skip** runs.

Inside the frame and not outside it, because a skip belongs to the architecture
and drawing it out in the graph's own margin would say it belonged to the
graph."""

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

GROUP_HEAD = 20.0
"""And the strip at the top of a repeated block, where its `×N` goes.

Shallower than a frame's, because a block sits **inside** one and two headers
of the same height read as two frames of the same kind."""

GROUP_PAD = 9.0
"""How far a block's frame stands off the layers in it."""

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

    `kind` is `"node"`, `"wave"`, `"remote"`, `"layer"` or `"group"`; a layer's
    `mark` says what **sort** of thing it is, which is what decides how it is
    drawn. A `Linear` and a `Sigmoid` are not the same kind of thing and drawing
    them the same says they are.

    A `"group"` is a repeated block, drawn as a frame around the layers in it
    with its `×N` on the frame — because four encoder layers opened up are eight
    boxes each saying `×4`, which is the count said eight times and the block
    said none.
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
    row: int | None = None
    narrows: int | None = None
    made_of: str | None = None
    dims: tuple | None = None
    parallel: int | None = None
    """How many identical lanes this one layer is — the heads of an attention
    block, the groups of a convolution. Drawn as plates behind it and **never**
    as separate boxes: torch packs the heads into one projection, so there is no
    second module and edges between four of them would be a graph nobody
    built."""

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
            width = 1.0
            if box.node in ill:
                line = ink = _worst(ill[box.node])
                width = 1.8
            how = _theme.SHAPES.get(box.mark or "other", "box")
            if how == "box" and box.narrows and _tapers(box.mark):
                how = "trapezoid"
            shapes.extend(_plates(box, line, how))
            shapes.append(_silhouette(box, fill, line, width, how))
            notes.append(_text(box.cx, box.cy, _labelled(box), ink, size=10))
        elif box.kind == "node":
            fill, line, ink = PALETTE[_family(devices.get(box.node))]
            width = 1.4
            if box.node in ill:
                # The one place in this library where a colour means bad. It is
                # the outline and never the fill: two facts, two channels. And
                # the colour is the **family** of the trouble, because six
                # alarms that all look the same are one alarm.
                line, width = _worst(ill[box.node]), 2.6
            shapes.append(_rect(box, fill, line, width))
            lines = labels[box.node]
            # A node that was opened writes its name at the **top**, where a
            # frame's label goes; a plain one keeps it in the middle.
            opened = (inside or {}).get(box.node)
            at = (box.y + PAD_Y + LINE_H * len(lines) / 2) if opened else box.cy
            notes.append(_text(box.cx, at, "<br>".join(lines), ink))
        elif box.kind == "group":
            # A block that repeats, drawn once with its count on it. Dotted like
            # a wave, because it is the same statement — *what is in here goes
            # together* — and a reader who has learnt one has learnt both.
            _, line, ink = PALETTE["wave"]
            shapes.append(_rect(box, None, line, 1.0, dash="dot"))
            notes.append(
                _text(box.x + GROUP_PAD, box.y + GROUP_HEAD / 2, box.label, ink, left=True,
                      size=10)
            )
        else:
            fill, line, ink = PALETTE[box.kind]
            shapes.append(_rect(box, fill, line, 1.6, dash="dot"))
            notes.append(
                _text(box.x + FRAME_PAD, box.y + FRAME_HEAD / 2, box.label, ink, left=True)
            )

    # What a node is made of feeds each other too, and that is the only thing
    # that can tell a residual from a stack.
    for node, held in (inside or {}).items():
        where_in = {box.node: box for box in placed if box.kind == "layer"}
        blocks = {box.node: box for box in placed if box.kind == "group"}
        frame = next((box for box in placed if box.node == node), None)
        if frame is None:
            continue
        mine = {one.path: one.block for one in held.layers}
        for a, b in held.edges:
            from_, to = where_in.get(f"{node}.{a}"), where_in.get(f"{node}.{b}")
            if from_ is None or to is None:
                continue
            # An edge that comes **down** into a block ends at the block, not
            # at the layer inside it: the frame's header is where the `×N` is
            # written, and an arrow through a label reads as neither. A skip
            # comes in through the side and never touches the header, so it
            # keeps going to the layer it really feeds — which is the `+`, and
            # saying *into the block* there would lose the one thing the skip
            # is about.
            around, head = _inner_edge(
                from_,
                _entered(from_, to, blocks.get(f"{node}.{mine.get(b)}"))
                if mine.get(a) != mine.get(b)
                else to,
                frame,
            )
            shapes.extend(around)
            notes.append(head)

    # Outside every box, so a routed edge never has to guess which way is clear.
    span = (min(box.x for box in placed), max(box.x + box.w for box in placed))
    # What the drawing has to hold, which starts as the boxes and grows to take
    # in every lane a routed edge asks for.
    reach = [span[0], span[1]]
    lanes, boxed = {}, list(where.values())
    for node, comes_from in steps(plan):
        for source in comes_from:
            if source not in where:
                continue
            from_, to = where[source], where[node]
            if not _crosses(from_, to, boxed):
                around, head = _bent(from_, to)
                shapes.extend(around)
                notes.append(head)
                continue
            # One lane per edge on that side, handed out in declaration order so
            # the same graph is drawn the same way twice.
            side = -1 if abs(from_.cx - span[0]) <= abs(span[1] - from_.cx) else 1
            apart = lanes[side] = lanes.get(side, -1) + 1
            around, head, lane = _routed(from_, to, span, apart)
            shapes.extend(around)
            notes.append(head)
            reach = [min(reach[0], lane), max(reach[1], lane)]

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

    # A legend, and only of the families that are actually on the figure. Six
    # colours nobody can read are one colour, and a legend of families that are
    # not here is a reader looking for something that is not there.
    span_y = max(box.y + box.h for box in placed)
    if ill:
        notes.extend(_legend(ill, span_y + 26))
        span_y += 26
    figure.update_layout(
        shapes=shapes,
        annotations=notes,
        xaxis={"visible": False, "range": [reach[0] - 20, reach[1] + 20]},
        # Reversed, because the layout counts downwards the way a plan reads.
        yaxis={"visible": False, "range": [span_y + 20, -20], "scaleanchor": "x"},
        **_theme.layout(
            margin={"l": 16, "r": 16, "t": 16, "b": 16},
            **_sized(reach[1] - reach[0] + 80, span_y + 80),
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
        # `under` and not `inside`, which is the argument: a wave's branches are
        # nodes and every one of them may be opened up too.
        left, under = x + FRAME_PAD, y + FRAME_HEAD
        for child in body:
            w, _ = _measure(child, labels, inside)
            # Top-aligned, not centred: everything in a wave starts at the same
            # moment, and hanging a short branch halfway down the frame says it
            # starts later.
            _place(child, left, under, labels, out, inside)
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
        max(_width(node, labels), _inner_width(held) + 2 * FRAME_PAD + GUTTER),
        tall
        + rows * LAYER_H
        + max(rows - 1, 0) * LAYER_GAP
        + len(held.groups) * (GROUP_HEAD + GROUP_PAD)
        + FRAME_PAD,
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
    placed = _ranked(inside)
    lifts = _lifted(placed, inside)
    held = {}
    for one, place in placed:
        row, across, wide, narrows = place
        # A layer in a repeated block is indented, so the frame around it has
        # somewhere to be. It is the only reason the block is narrower.
        inset = GROUP_PAD if one.block in inside.groups else 0.0
        # The height is the kind's, and it is decided **here** and not when it
        # is drawn: a figure that paints something other than the box it laid
        # out has two truths in it, and the tests can only see one of them.
        tall = LAYER_H * _theme.MARKS.get(one.kind, _theme.MARKS["other"])[3]
        box = Box(
            "layer",
            x + FRAME_PAD + GUTTER + across + inset,
            top + lifts[row] + row * (LAYER_H + LAYER_GAP) + (LAYER_H - tall) / 2,
            wide - 2 * inset,
            tall,
            node=f"{node}.{one.path}",
            label=one.label,
            mark=one.kind,
            shape=one.shape,
            row=row,
            narrows=narrows,
            made_of=one.made_of if one.kind in ("attention", "recurrent") else None,
            dims=one.dims,
            parallel=one.parallel,
        )
        out.append(box)
        if one.block in inside.groups:
            held.setdefault(one.block, []).append(box)
    # The frame last, so it is behind nothing and its label is not covered.
    for block, boxed in held.items():
        name, count = inside.groups[block]
        out.append(
            Box(
                "group",
                min(one.x for one in boxed) - GROUP_PAD,
                min(one.y for one in boxed) - GROUP_HEAD,
                max(one.x + one.w for one in boxed) - min(one.x for one in boxed)
                + 2 * GROUP_PAD,
                max(one.y + one.h for one in boxed) - min(one.y for one in boxed)
                + GROUP_HEAD + GROUP_PAD,
                node=f"{node}.{block}",
                label=f"{name}  ×{count}" if name else f"×{count}",
            )
        )


def _lifted(placed, inside):
    """How far each row drops to make room for the headers above it.

    A repeated block gets a strip at the top for its `×N`, and every row from
    there down moves by that much. Accumulated rather than per block, because
    two blocks in a row each want their own strip and the second one has to
    clear the first.
    """
    rows = {}
    for one, (row, *_) in placed:
        if one.block in inside.groups:
            rows.setdefault(one.block, []).append(row)
    opens = {min(where) for where in rows.values()}
    lifts, so_far = {}, 0.0
    for row in sorted({place[0] for _, place in placed}):
        if row in opens:
            so_far += GROUP_HEAD
        lifts[row] = so_far
    return lifts


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

    # Which way each layer changes the width, so a taper can be drawn the way
    # it really goes. The last dimension of what it produced against the last
    # dimension of what fed it: `+1` narrower, `-1` wider, `0` neither.
    narrows = {}
    for path in layers:
        mine = _last_dim(layers[path].shape)
        # What really went in, when the trace knows it. Only when it does not —
        # a functional operation `fx` recovered, which no hook ever saw — does
        # this fall back to whatever is above it on the figure.
        came_in = getattr(layers[path], "came_in", None)
        # Only when the two are the same **rank**: an `Embedding` is handed
        # token indices and returns vectors, and comparing the last number of
        # one with the last number of the other is comparing a length with a
        # width. A lookup does not narrow or widen, it replaces.
        theirs = (
            _last_dim(came_in)
            if came_in and layers[path].shape and _rank(came_in) == _rank(layers[path].shape)
            else None
        )
        if theirs is None and came_in is None:
            theirs = _width_before(path, layers, feeds, set())
        narrows[path] = (
            None if mine is None or theirs is None else (mine < theirs) - (mine > theirs)
        )

    placed, widest = [], _inner_width(inside)
    for row, beside in sorted(rows.items()):
        each = (widest - LAYER_GAP * (len(beside) - 1)) / len(beside)
        for at, one in enumerate(beside):
            placed.append((one, (row, at * (each + LAYER_GAP), each, narrows[one.path])))
    return placed


def _width_before(path, layers, feeds, seen):
    """The width of the nearest thing above this that has one.

    Walking back past what has no shape is the whole of it: an `Add` has no
    shape of its own, and stopping at one is how a pooling layer that really
    does go from thirty-two to one comes out drawn as a plain box.
    """
    for one in feeds.get(path, []):
        if one in seen or one not in layers:
            continue
        seen.add(one)
        found = _last_dim(layers[one].shape)
        if found is not None:
            return found
        deeper = _width_before(one, layers, feeds, seen)
        if deeper is not None:
            return deeper
    return None


def _rank(shape):
    """How many numbers a shape has, which is what says what they mean."""
    return len(shape.split("×"))


def _last_dim(shape):
    """The last number of a shape as it is written — the width of what came out."""
    try:
        return int(shape.split("×")[-1])
    except (AttributeError, ValueError):
        return None


def _inner_width(inside):
    """How wide the widest row of an architecture has to be."""
    return max(
        (CHAR * len(_layer_text(one)) + 2 * PAD_X for one in inside.layers),
        default=MIN_W,
    )


def _rows_of(inside):
    """How many rows deep it is."""
    return len({place[0] for _, place in _ranked(inside)}) if inside else 0


def _tapers(kind):
    """Whether a kind is drawn changing the width when it changes it.

    Only what really carries the shape: a non-linearity that happens to sit
    where the width changed did not change it.
    """
    return kind in ("learned", "shaping")


def _layer_text(one):
    """The longest line a layer needs, for working out how wide its box is."""
    return max(_two_lines(one), key=len)


def _two_lines(one):
    """What is written on a layer, over two lines.

    What it **is** on the first and what it **produces** on the second. One line
    makes a reader parse a sentence to find a number, and the number is what
    they came for.

    A non-linearity and a dropout get **one** line: they cannot change a shape,
    so writing the one they were handed says nothing and takes the room their
    silhouette needs to stay thin.
    """
    top = one.label + (f"  ·  {one.made_of}" if getattr(one, "made_of", None) else "")
    if (one.mark if hasattr(one, "mark") else one.kind) in ("activation", "regular"):
        return (top, "")
    return (top, _measured(one.shape, getattr(one, "dims", None)))


def _measured(shape, dims):
    """A shape with each number said out loud: `4 batch · 24 ch · 32 len`.

    Three numbers and no way to tell which is the batch, which is time and which
    is the width is the thing that makes a shape useless at a glance.
    """
    if not shape:
        return ""
    sizes = shape.split("×")
    if not dims or len(dims) != len(sizes):
        return "×".join(sizes)
    return " · ".join(
        size if name == "?" else f"{size} {name}" for size, name in zip(sizes, dims)
    )


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
        # At most three names, and then a count. Writing every finding of every
        # layer into the node's label is what made a graph of four nodes come
        # out ten times wider than it was tall, with nothing readable in it.
        names = sorted({_bare(one) for one in flags})
        said = " · ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
        lines.append(f"<span style='color:{_worst(flags)}'>⚠ {_safe(said)}</span>")
    return tuple(lines)


def _labelled(box):
    """What is written on a layer: what it is, and what it produces.

    The shape is on it and not on the hover, because it is the one thing that
    makes a **bottleneck** a picture: `512 → 8 → 512` is visible and
    `Linear · Linear · Linear` is not.
    """
    top, below = _two_lines(box)
    # `<br>` and `<i>` an annotation does take; a `<span style=…>` it does not,
    # and an unknown tag comes out as the letters of the tag.
    if not below:
        return _safe(top)
    return f"{_safe(top)}<br><i>{_safe(below)}</i>"


def _legend(ill, at):
    """One line saying what each colour on this figure means."""
    families = []
    for flags in ill.values():
        for flag in flags:
            try:
                one = _flag_family(_bare(flag))
            except (TypeError, ValueError):
                continue
            if one not in families:
                families.append(one)
    across = 0.0
    said = []
    for one in families:
        colour = _theme.ALARM.get(one, _theme.SERIES["alarm"])
        said.append(_text(across, at, f"■ {one}", colour, left=True, size=10))
        across += CHAR * (len(one) + 4) + 18
    return said


def _worst(flags):
    """What colour a set of findings is drawn in: the family of the first one.

    First and not blended: `verdict` already puts what stops a run soonest at
    the front, so the colour is the family of the thing to look at first. A
    blend of six families is a seventh colour that means nothing.
    """
    for flag in flags:
        try:
            return _theme.ALARM[_flag_family(_bare(flag))]
        except (KeyError, TypeError, ValueError):
            continue
    return _theme.SERIES["alarm"]


def _flag_family(name):
    """Which family a flag belongs to, from the crate that decides it.

    Not `_family`, which this file already had and which answers a different
    question — which device family a node runs on. Two names for two questions,
    and the day they were one name the graph was drawn with a flag's colour.
    """
    from soma_next._soma_next import family

    return family(name)


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


def _silhouette(box, fill, line, width, how):
    """One layer, drawn as the **kind of thing** it is.

    A `Linear`, a convolution, a recurrent cell and a non-linearity are four
    different kinds of thing, and four identical rectangles with different words
    in them make the reader do the sorting a picture was supposed to have done.

    An SVG path in data coordinates, so the silhouette scales with the box and
    there is nothing to keep in step by hand.
    """
    x, y, w, h = box.x, box.y, box.w, box.h
    r, cut, skew = h / 2, min(h / 2, 10.0), min(w / 8, 12.0)
    said = {"type": "path", "xref": "x", "yref": "y", "layer": "below",
            "fillcolor": fill,
            "line": {"color": line, "width": width}}
    if how in ("capsule", "dashed"):
        # Polygons and never arcs: this figure's y axis is **reversed** — a plan
        # is read downwards — and an SVG arc's sweep flag is about a direction,
        # so every one of them comes out inside out. A cut corner says *rounded*
        # well enough and cannot be wrong.
        said["path"] = (
            f"M {x + r},{y} L {x + w - r},{y} L {x + w},{y + h / 2} "
            f"L {x + w - r},{y + h} L {x + r},{y + h} L {x},{y + h / 2} Z"
        )
        if how == "dashed":
            said["line"]["dash"] = "dot"
    elif how == "lens":
        # Pointed at both ends: nothing lives in it, it is passed through.
        said["path"] = (
            f"M {x},{y + h / 2} L {x + skew},{y} L {x + w - skew},{y} "
            f"L {x + w},{y + h / 2} L {x + w - skew},{y + h} L {x + skew},{y + h} Z"
        )
    elif how == "skewed":
        # A window sliding along, which is what a convolution is.
        said["path"] = (
            f"M {x + skew},{y} L {x + w},{y} L {x + w - skew},{y + h} L {x},{y + h} Z"
        )
    elif how == "cut":
        said["path"] = (
            f"M {x + cut},{y} L {x + w - cut},{y} L {x + w},{y + cut} "
            f"L {x + w},{y + h - cut} L {x + w - cut},{y + h} L {x + cut},{y + h} "
            f"L {x},{y + h - cut} L {x},{y + cut} Z"
        )
    elif how == "looped":
        # It feeds itself, and the tab on the right side is that and nothing
        # else — drawn out of straight lines for the same reason as the capsule.
        tab = min(w / 6, 16.0)
        said["path"] = (
            f"M {x},{y} L {x + w - tab},{y} L {x + w - tab},{y - h / 4} "
            f"L {x + w},{y - h / 4} L {x + w},{y + h + h / 4} "
            f"L {x + w - tab},{y + h + h / 4} L {x + w - tab},{y + h} L {x},{y + h} Z"
        )
    elif how == "trapezoid":
        said["path"] = _tapered(box, skew)
    else:
        said["path"] = f"M {x},{y} L {x + w},{y} L {x + w},{y + h} L {x},{y + h} Z"
    return said


PLATE = 4.0
"""How far behind a layer each of its lanes is drawn."""

PLATES = 2
"""And how many of them, at most.

A count and not the count: eight heads drawn as eight plates is a smudge, and
what says *eight* is the word `8 heads` written on the front one. The plates
say **there are several of these**, which is the part a shape can carry.
"""


def _plates(box, line, how):
    """The lanes behind a layer that runs several of itself at once.

    Offset copies of its own silhouette, and **no edges between them**. Torch
    packs the heads of a `MultiheadAttention` into one projection, so there is
    no second module anywhere and four boxes wired together would be a graph
    nobody built. What is true is that this one operation happens several times
    over, and that is what a stack of plates says.
    """
    if not box.parallel or box.parallel < 2:
        return []
    said = []
    for at in range(min(box.parallel - 1, PLATES), 0, -1):
        # Downwards and to the right, never up: the first layer of a repeated
        # block has its frame's `×N` immediately above it, and a plate drawn
        # into a label is two things in one place.
        behind = replace(box, x=box.x + at * PLATE, y=box.y + at * PLATE)
        said.append(_silhouette(behind, None, line, 0.8, how))
    return said


def _tapered(box, skew):
    """A shape that changes the shape, drawn changing: narrowing when what comes
    out is smaller than what went in, widening when it is bigger.

    This is what makes a **bottleneck** look like one instead of like three
    identical boxes with different numbers written on them.
    """
    x, y, w, h = box.x, box.y, box.w, box.h
    if box.narrows is None or box.narrows == 0:
        return f"M {x},{y} L {x + w},{y} L {x + w},{y + h} L {x},{y + h} Z"
    if box.narrows > 0:
        # Narrower coming out than going in: **wide at the top**, which is the
        # way the data goes. It was the other way round, and a funnel drawn
        # upside down says the opposite of what it means.
        return f"M {x},{y} L {x + w},{y} L {x + w - skew},{y + h} L {x + skew},{y + h} Z"
    return f"M {x + skew},{y} L {x + w - skew},{y} L {x + w},{y + h} L {x},{y + h} Z"


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
    """One edge that cannot go straight, as `(shapes, annotation, lane)`.

    Around means **outside everything**, down, and in through the side of what
    reads it. The lane is outside the whole drawing rather than outside the boxes
    in the way, because a lane threaded between two of them is a lane that will
    cross a third the next time the layout changes.

    `apart` is which lane on that side this one gets, so that edges which all
    have to go around stay countable.

    The lane comes back because **the caller has to make room for it**. Outside
    every box is outside the extent the boxes gave, and a canvas measured from
    the boxes alone cuts the lane off — which is a figure whose arrows leave it
    on one side and come back on the other.
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
        lane,
    )


BIGGEST = 1600.0
"""How large a figure is allowed to get, in pixels, on its longer side."""


def _sized(wide, tall):
    """A figure big enough to hold what is in it, **in proportion**.

    The y axis is anchored to the x so a box is not stretched into a different
    box. That makes width and height one decision and not two: capping the width
    of a figure whose contents are wider than the cap does not shrink it, it
    **cuts the right-hand side off** — and the arrows that reached the node over
    there went with it.

    So both are scaled by the same factor when either is too big, which is the
    only way to make a thing smaller without making it shorter.
    """
    scale = min(1.0, BIGGEST / max(wide, tall, 1.0))
    return {"width": max(360, wide * scale), "height": max(240, tall * scale)}


def _bent(source, target):
    """One edge with nothing in the way, as `(shapes, annotation)`.

    Straight down when it really is straight down; a curve when it has to move
    across. A long diagonal cutting over a figure is the thing that makes a
    graph of four nodes look like a cat's cradle, and a bend that leaves and
    arrives vertically reads as *this feeds that* rather than as a line.
    """
    x0, y0 = source.cx, source.y + source.h
    x1, y1 = target.cx, target.y
    if abs(x1 - x0) < 2.0:
        return [], _arrow(x0, y0, x1, y1)
    # Enough to leave and arrive vertically, and no more: a deep bend on a long
    # drop swings out across the figure, and what was wanted was *this feeds
    # that* and not a detour.
    lean = min(max((y1 - y0) * 0.35, 12.0), 70.0)
    path = f"M {x0},{y0} C {x0},{y0 + lean} {x1},{y1 - lean} {x1},{y1 - HEAD}"
    return (
        [{"type": "path", "path": path, "xref": "x", "yref": "y",
          "line": {"color": _theme.MUTED, "width": 1.2}, "layer": "below"}],
        _arrow(x1, y1 - HEAD, x1, y1),
    )


def _entered(source, target, block):
    """The block an edge lands in, when it lands on it from directly above.

    The row is kept from the layer, because a skip is *how many rows it jumps*
    and a block has no row of its own — measuring the picture instead of the
    graph is the mistake `_inner_edge` already has a comment about. And the
    height is trimmed to the header, so the arrow stops on the frame's top edge
    rather than at the middle of everything inside it.
    """
    if block is None or source.row is None or target.row is None:
        return target
    if target.row - source.row > 1:
        return target
    return replace(block, row=target.row, h=GROUP_HEAD)


def _inner_edge(source, target, frame):
    """One edge between two layers of the same node.

    Straight down when they are next to each other; **out into the gutter** when
    the edge jumps a row, which is what a skip connection is. Drawing a skip as
    a long straight arrow through everything between its ends is drawing an
    arrow into each of them.
    """
    # By **row** and not by how far apart they look: a non-linearity is drawn
    # shorter than a Linear, so two neighbours can be further apart in pixels
    # than a skip that jumps one — and measuring the picture instead of the
    # graph got exactly that wrong.
    if target.row is not None and source.row is not None and target.row - source.row <= 1:
        return [], _arrow(source.cx, source.y + source.h, target.cx, target.y)

    x0, y0 = source.cx, source.y + source.h
    # In through the **side**, because a skip does not come from above: coming
    # down onto the top of a box is what a step does.
    into, y1 = target.x, target.cy
    lane = frame.x + FRAME_PAD + GUTTER / 2
    path = (
        f"M {x0},{y0} "
        f"C {x0},{y0 + BEND} {lane},{y0} {lane},{y0 + BEND} "
        f"L {lane},{y1 - BEND} "
        f"C {lane},{y1} {lane},{y1} {into - HEAD},{y1}"
    )
    return (
        [{"type": "path", "path": path, "xref": "x", "yref": "y",
          "line": {"color": _theme.SERIES["took"], "width": 1.2}, "layer": "below"}],
        _arrow(into - HEAD, y1, into, y1, colour=_theme.SERIES["took"]),
    )


def _arrow(ax, ay, x, y, colour=None):
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
        "arrowcolor": colour or _theme.MUTED,
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
