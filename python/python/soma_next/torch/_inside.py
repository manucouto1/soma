"""What a node is made of, as a **graph** and not as a list.

    from soma_next.torch import architecture
    g.figure(inside=architecture(g))

A list of children cannot show a skip connection. It cannot show a recurrent
cell looping back on itself, or a bottleneck, or which of a transformer block's
fourteen leaves feed which. What it takes is the **dataflow**, and that is a
graph.

## The graph is declared; the inside of a node is observed

The outer figure is drawn from what was declared and needs nothing to have run —
that is CU19, and it holds. This is the other half of the same sentence: what is
inside a node belongs to the node, this library did not write it, and the only
honest way to know its shape is to look.

Two ways to look, and the figure says which one it used:

| | how | what it misses |
|---|---|---|
| `torch.fx` | traces symbolically, no data | anything with control flow that depends on the values |
| hooks | runs it once for real | operations that are not modules — a `+`, a `cat`, a slice |

`fx` is tried first because it sees the functional operations, which is exactly
where a residual connection lives. When it cannot — a hand-written recurrent
cell, an `if` on a length — the hook path runs and **says so on the figure**,
because a residual that is missing looks precisely like a residual that is not
there.

Both answer in the **same shape**, and there is a test that they agree on a
module both can handle. Two paths that produce two shapes are two things that
stop coinciding slowly, which this project has already been bitten by once.

## What a kind is

A `Linear` and a `Sigmoid` are not the same kind of thing and drawing them the
same says they are. The set is closed and small, which is what lets a figure
have a rule per kind instead of a colour per class name.
"""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

__all__ = ["KINDS", "Inside", "Layer", "architecture", "kind_of", "traced"]

#: What a thing in an architecture **is**, and what a figure draws it as.
#:
#: Closed, and by role rather than by class: whoever writes a new activation
#: gets the activation treatment without this table learning its name.
KINDS = (
    "learned",  # holds weights: Linear, Embedding
    "conv",  # holds weights and slides them along
    "recurrent",  # holds weights and loops back on itself
    "attention",  # a composite everybody recognises, drawn as one thing
    "norm",  # LayerNorm, BatchNorm — no capacity, changes the scale
    "activation",  # elementwise, no weights: not a box at all
    "regular",  # Dropout — nothing at inference
    "shaping",  # changes the shape and nothing else: reshape, pool, cat
    "block",  # a container of the above
    "other",  # honestly unknown, and drawn as such
)

_BY_NAME = {
    "Linear": "learned",
    "Bilinear": "learned",
    "LazyLinear": "learned",
    "Embedding": "learned",
    "EmbeddingBag": "learned",
    "Conv1d": "conv",
    "Conv2d": "conv",
    "Conv3d": "conv",
    "ConvTranspose1d": "conv",
    "ConvTranspose2d": "conv",
    "ConvTranspose3d": "conv",
    "RNN": "recurrent",
    "RNNCell": "recurrent",
    "LSTM": "recurrent",
    "LSTMCell": "recurrent",
    "GRU": "recurrent",
    "GRUCell": "recurrent",
    "MultiheadAttention": "attention",
    "TransformerEncoderLayer": "attention",
    "TransformerDecoderLayer": "attention",
    "LayerNorm": "norm",
    "BatchNorm1d": "norm",
    "BatchNorm2d": "norm",
    "BatchNorm3d": "norm",
    "GroupNorm": "norm",
    "InstanceNorm1d": "norm",
    "InstanceNorm2d": "norm",
    "RMSNorm": "norm",
    "Dropout": "regular",
    "Dropout1d": "regular",
    "Dropout2d": "regular",
    "AlphaDropout": "regular",
    "Flatten": "shaping",
    "Unflatten": "shaping",
    "Identity": "shaping",
}

_ACTIVATIONS = (
    "ReLU", "LeakyReLU", "PReLU", "RReLU", "ELU", "SELU", "CELU", "GELU", "SiLU",
    "Mish", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "Softplus", "Softsign",
    "Hardtanh", "Hardsigmoid", "Hardswish", "GLU", "Threshold",
)

_SHAPING = ("Pool", "Pad", "Upsample", "Interpolate", "Fold", "Unfold", "PixelShuffle")

#: Functional operations `fx` sees and modules do not. A residual connection
#: lives in exactly one of these, which is why the symbolic path is tried first.
_FUNCTIONS = {
    "add": ("shaping", "+"),
    "iadd": ("shaping", "+"),
    "mul": ("shaping", "×"),
    "cat": ("shaping", "concat"),
    "stack": ("shaping", "stack"),
    "getitem": ("shaping", "slice"),
    "reshape": ("shaping", "reshape"),
    "view": ("shaping", "reshape"),
    "permute": ("shaping", "permute"),
    "transpose": ("shaping", "transpose"),
    "flatten": ("shaping", "flatten"),
    "matmul": ("shaping", "@"),
    "mean": ("shaping", "mean"),
    "sum": ("shaping", "sum"),
}


def kind_of(what):
    """What kind of thing this is, by role and never by exact class name.

    A `Sigmoid` and a `GELU` are the same kind; a `Linear` and a `Conv2d` are
    the same kind; a class this table has never heard of that ends in `Norm` is
    a normalisation. Guessing by suffix is a guess and it is a good one — the
    alternative is calling half of everybody's models `other`.
    """
    if isinstance(what, str):
        name = what
    else:
        name = type(what).__name__
    if name in _BY_NAME:
        return _BY_NAME[name]
    if name in _ACTIVATIONS:
        return "activation"
    if "Conv" in name:
        return "conv"
    if name.endswith("Norm"):
        return "norm"
    if name.endswith("Dropout"):
        return "regular"
    if any(one in name for one in _SHAPING):
        return "shaping"
    if "Attention" in name:
        return "attention"
    if torch is not None and isinstance(what, torch.nn.Module) and any(what.children()):
        return "block"
    return "other"


class Layer:
    """One thing in an architecture: where it is, what it is, what it produces."""

    __slots__ = ("path", "kind", "label", "shape", "made_of", "dims", "came_in")

    def __init__(self, path, kind, label, shape=None, made_of=None, dims=None):
        self.path = path
        self.kind = kind
        self.label = label
        #: What is inside it, for a composite that is drawn whole. A
        #: `TransformerEncoderLayer` kept as one box still owes the reader an
        #: answer to *one box of what* — and the answer is short enough to write
        #: on it.
        self.made_of = made_of
        #: What each number of `shape` is — `("batch", "steps", "dim")`. A shape
        #: nobody can read is three numbers.
        self.dims = dims
        #: And what went in, so that *does this narrow* is a fact about the
        #: layer rather than about what happens to sit above it on a figure.
        self.came_in = None
        #: The shape of what it produces, as text — `(32, 8)`. The one thing
        #: that makes a **bottleneck** visible: `512 → 8 → 512` is a picture and
        #: `Linear · Linear · Linear` is not. `None` when nobody ran it.
        self.shape = shape

    def __eq__(self, other):
        return isinstance(other, Layer) and self._as_tuple() == other._as_tuple()

    def __hash__(self):
        return hash(self._as_tuple())

    def _as_tuple(self):
        return (self.path, self.kind, self.label, self.shape, self.made_of, self.dims)

    def __repr__(self):
        said = f"Layer({self.path!r}, {self.kind!r}, {self.label!r}"
        return said + (f", {self.shape!r})" if self.shape else ")")


class Inside:
    """What one node is made of: layers, how they feed each other, and how it
    was found out.

    `how` is `"symbolic"` or `"traced"`, and it is on the figure rather than in
    a log: a residual connection the hook path could not see looks exactly like
    a residual connection that is not there, and the reader has to be able to
    tell those apart.
    """

    __slots__ = ("layers", "edges", "how", "why", "folded")

    def __init__(self, layers, edges, how, why=None, folded=None):
        self.layers = list(layers)
        self.edges = list(edges)
        self.how = how
        #: What was collapsed into what: `{"body.2.norm": "body.0.norm"}`. Six
        #: identical blocks drawn once leaves five sets of paths with no box,
        #: and a finding on one of them has to land somewhere — on the box that
        #: stands for it, which is this.
        self.folded = dict(folded or {})
        #: Why the symbolic path was not used, when it was not. Kept because
        #: "it did not work" is not something anybody can act on.
        self.why = why

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        return f"Inside({len(self.layers)} layers, {len(self.edges)} edges, {self.how})"


def traced(module, example=None):
    """What this module is made of. `fx` if it can, a real forward if it cannot.

    `example` is an input to run it on. Without one only the symbolic path is
    available, and a module that needs a forward answers `None` rather than a
    guess.
    """
    said, why = _symbolic(module, example)
    if said is not None:
        return said
    if example is None:
        return None
    return _by_running(module, example, why)


def _symbolic(module, example=None):
    """`torch.fx`, which sees the functional operations — and a residual
    connection is one."""
    try:
        from torch.fx import symbolic_trace
    except ImportError as e:  # pragma: no cover
        return None, str(e)
    try:
        graphed = symbolic_trace(module)
    except Exception as e:
        # Control flow that depends on the values, most often. It is not a
        # failure of the model and the message is what says so.
        return None, f"{type(e).__name__}: {e}"

    shapes = _shapes_of(graphed, example)
    named = dict(module.named_modules())
    layers, edges = [], []
    # What each `fx` node **reaches back to**: itself if it is worth a box, and
    # otherwise whatever fed it. Bridging like this is not a nicety — without
    # it, dropping one uninteresting node silently cuts the path through it and
    # a bottleneck comes out as two boxes and no edges.
    reaches = {}
    for one in graphed.graph.nodes:
        made = _from_fx(one, named)
        before = [where for feeds in one.all_input_nodes for where in reaches.get(feeds.name, ())]
        if made is None:
            reaches[one.name] = before
            continue
        made.shape = shapes.get(one.name)
        reaches[one.name] = [made.path]
        layers.append(made)
        edges.extend((where, made.path) for where in dict.fromkeys(before))
    return Inside(layers, edges, "symbolic"), None


def _shapes_of(graphed, example):
    """What each node produces, when there is something to run through it.

    `fx` alone knows the shape of nothing — it never saw a number. One pass with
    a real input fills them in, and they are what makes a **bottleneck** a
    picture instead of three identical boxes.
    """
    if example is None:
        return {}
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        ShapeProp(graphed).propagate(example)
    except Exception:
        # Not worth a word: the graph is still right, it just has no numbers on
        # it, and every box says so by having no shape rather than a wrong one.
        return {}
    said = {}
    for one in graphed.graph.nodes:
        meta = one.meta.get("tensor_meta")
        if meta is not None and getattr(meta, "shape", None) is not None:
            said[one.name] = "×".join(str(n) for n in tuple(meta.shape))
    return said


def _from_fx(one, named):
    """One `fx` node as a `Layer`, or `None` for what is not worth a box."""
    if one.op == "call_module":
        held = named.get(one.target)
        return Layer(one.target, kind_of(held), type(held).__name__)
    if one.op == "placeholder":
        # Kept, and it is not bookkeeping: `x + f(x)` **forks** here, and
        # without a box to fork from the skip has nowhere to start. Called
        # `fork` and not `input` because that is what it is wherever it lands —
        # a module spliced into the middle of a node has its placeholder in the
        # middle too, and "input" there reads as the node's.
        return Layer(one.name, "shaping", "fork")
    if one.op in ("call_function", "call_method"):
        name = getattr(one.target, "__name__", str(one.target)).strip("_")
        if name in _NOT_WORTH_A_BOX:
            return None
        kind, label = _FUNCTIONS.get(name, (kind_of(_titled(name)), name))
        return Layer(one.name, kind, label)
    # The output is where the picture ends, not a box in it.
    return None


def _titled(name):
    """`relu` as `ReLU`-ish, so a functional activation is recognised as one.

    Functional and module forms are the same operation written twice, and a
    table that knows `Sigmoid` and not `sigmoid` would draw them differently.
    """
    return {"relu": "ReLU", "gelu": "GELU", "silu": "SiLU", "elu": "ELU"}.get(
        name, name[:1].upper() + name[1:]
    )


#: `fx` sees the plumbing too. These are not operations anybody draws.
_NOT_WORTH_A_BOX = frozenset(
    {"size", "shape", "dim", "len", "getattr", "to", "contiguous", "detach", "clone"}
)


def _by_running(module, example, why):
    """One real forward, watching which module produced which tensor.

    Edges come from **tensor identity**: what a module was handed was produced
    by whoever last returned that same object. It is exact where it applies and
    blind where it does not — a `+` is not a module, so the sum it returns has
    no producer and the edge falls back to whatever ran before it.

    That blindness is the reason `how` is on the figure.
    """
    layers, edges, made_by, order = [], [], {}, []

    def watch(path, one):
        def saw(_module, args, output):
            layers.append(Layer(path, kind_of(one), type(one).__name__, _shape(output)))
            for before in _tensors(args):
                producer = made_by.get(id(before))
                if producer is not None and producer != path:
                    edges.append((producer, path))
            if not any(to == path for _, to in edges) and order:
                # Nothing known fed it, so the thing that ran before it did —
                # which is right for a stack and is a guess anywhere else.
                edges.append((order[-1], path))
            for after in _tensors((output,)):
                made_by[id(after)] = path
            order.append(path)

        return saw

    hooks = [
        one.register_forward_hook(watch(path, one))
        for path, one in module.named_modules()
        if path
    ]
    try:
        with torch.no_grad():
            module(example)
    finally:
        for hook in hooks:
            hook.remove()
    return Inside(layers, edges, "traced", why)


def _made_of(module):
    """What a composite is made of, in one line: `attention · norm ×2 · linear ×2`.

    Only for something kept whole — a leaf is made of itself. It is the answer
    to *one box of what*, which a reader is owed the moment a fourteen-part
    block is drawn as one thing.
    """
    if torch is None or not any(True for _ in module.children()):
        return None
    counted = {}
    for _, one in module.named_modules():
        if one is module or any(True for _ in one.children()):
            continue
        kind = kind_of(one)
        counted[kind] = counted.get(kind, 0) + 1
    if not counted:
        return None
    return " · ".join(
        one + (f" ×{how_many}" if how_many > 1 else "")
        for one, how_many in sorted(counted.items(), key=lambda kv: -kv[1])
    )


def _tensors(what):
    """Every tensor in whatever a module was handed or returned, **in order**.

    In order and not in whatever a stack pops: a recurrent cell returns
    `(output, h_n)`, and reversing that shows the hidden state where the output
    belongs — `1×4×24` for a GRU whose output is `4×16×24`, which is a wrong
    number written confidently on a figure.
    """
    found = []
    for one in what:
        if torch is not None and isinstance(one, torch.Tensor):
            found.append(one)
        elif isinstance(one, (tuple, list)):
            found.extend(_tensors(one))
        elif isinstance(one, dict):
            found.extend(_tensors(list(one.values())))
    return found


def _shape(output):
    """What a layer produces, as text. The one thing that makes a bottleneck
    visible at all."""
    found = _tensors(output if isinstance(output, list) else (output,))
    if not found:
        return None
    return "×".join(str(one) for one in tuple(found[0].shape))


def _dims(kind, shape, batch=None):
    """What each number in a shape **is**: `64×16×24` as batch, steps, dim.

    Unlabelled numbers are the thing that makes a shape useless to read at a
    glance — three numbers and no way to tell which is the batch, which is time
    and which is the width. There is no protocol that says so, so this is a
    reading of the conventions torch itself uses, and it says `?` rather than
    guessing when it does not know.

    The batch is the one thing that can be **checked** rather than assumed: the
    caller knows how many rows went in.
    """
    if not shape:
        return None
    sizes = [int(one) for one in shape.split("×")]
    named = ["?"] * len(sizes)
    at = 0
    if batch is not None and sizes and sizes[0] == batch:
        named[0] = "batch"
        at = 1
    rest = len(sizes) - at
    if kind == "conv":
        # `(batch, channels, …)`, and whatever is left is what it slides along.
        for which in range(at, len(sizes)):
            named[which] = "ch" if which == at else "len"
    elif kind == "recurrent":
        # `batch_first` puts time second; without a batch to anchor on there is
        # nothing to tell the two apart, and it says so.
        if rest >= 2:
            named[at] = "steps" if named[0] == "batch" else "?"
            named[-1] = "dim"
    elif rest >= 1:
        named[-1] = "dim"
        for which in range(at, len(sizes) - 1):
            named[which] = "steps"
    return tuple(named)


def architecture(graph, example=None, *, most=48, depth=0, workers=None):
    """What each node is made of, as `{node: Inside}` — ready for a figure.

        g.figure(inside=architecture(g, x))

    **The graph is run once**, with hooks on everything every node holds. That
    is not an optimisation: a node in the middle of a graph is handed what the
    nodes above it produced, and tracing it on the graph's own input feeds a
    fan-in the wrong thing entirely — which is what it did, until a picture with
    an empty box in it said so.

    Then, module by module, `torch.fx` is asked for the same thing. Where it can
    answer it wins, because it sees the operations that are **not** modules —
    and a residual connection is exactly one of those. Where it cannot, the run
    stands and the figure says so.

    That is the seam, and it is a module boundary rather than a judgement call:
    two paths, one shape, spliced where both of them agree the world divides.

    `example` is one input to run **the graph** on, so it obeys the same rules
    every input does — a tensor arrives wrapped in an `Opaque`, and a graph with
    branches is fed a map. Without one nothing can be traced at all, and every
    node answers nothing rather than a guess.

    A composite everybody recognises — an attention block, an LSTM — is **one**
    box and is not opened: read as its fourteen leaves it is fourteen things and
    a diagram nobody looks at twice. `depth=` opens them, for when the inside of
    one is exactly what is being asked about. And blocks that are the same block
    collapse to one and a `×N`, because twelve identical transformer layers
    drawn twelve times is a figure nobody reads.
    """
    if example is None or torch is None:
        return {}
    watched = {}
    for node in graph.nodes():
        for name, module in _held(graph.implementation(node)):
            for path, one in _worth_drawing(module, depth):
                watched[(node, f"{name}.{path}" if path else name)] = one

    batch = _rows_in(example)
    layers, edges, made_by, order, hooks = {}, {}, {}, {}, []
    for (node, where), one in watched.items():
        layers.setdefault(node, [])
        edges.setdefault(node, [])
        # One running order **per node**: the fallback edge is *whatever ran
        # before this*, and across nodes that is a different node's last layer,
        # which is not an edge inside anything.
        order.setdefault(node, [])
        hooks.append(
            one.register_forward_hook(
                _watch(where, one, layers[node], edges[node], made_by, order[node], batch)
            )
        )
    try:
        with torch.no_grad():
            graph.forward(example, workers=workers)
    finally:
        for hook in hooks:
            hook.remove()

    said = {}
    for node in graph.nodes():
        if not layers.get(node):
            if any(one == node for one, _ in watched):
                import warnings

                warnings.warn(
                    f"`{node}` holds modules and none of them ran, so its architecture "
                    f"is not drawn",
                    stacklevel=2,
                )
            continue
        mine, its = layers[node], edges[node]
        for name, module in _held(graph.implementation(node)):
            if not any(True for _ in module.children()):
                continue
            finer, _ = _symbolic(module, None)
            if finer is not None:
                mine, its = _spliced(mine, its, name, finer)
        said[node] = _at_most(
            node,
            _repeated(_inherited(_without_a_lone_input(Inside(mine, its, "traced", None)))),
            most,
        )
    return said


def _rows_in(example):
    """How many rows went in, which is the one dimension that can be checked
    rather than assumed."""
    found = _tensors([_as_the_engine_would(example)])
    return int(found[0].shape[0]) if found and found[0].dim() else None


def _as_the_engine_would(example):
    """The input as a node really receives it: with the wrappers off.

    An `Opaque` is what makes a tensor cross an edge, and the engine unwraps it
    on the way **in** — a node is handed a tensor. Tracing with the wrapper
    still on hands a `Conv1d` an `Opaque`, every node raises, and the whole
    architecture comes back empty without a word about why.
    """
    from soma_next import Opaque

    if isinstance(example, Opaque):
        return example.value
    if isinstance(example, dict):
        return {name: _as_the_engine_would(one) for name, one in example.items()}
    if isinstance(example, (list, tuple)):
        return type(example)(_as_the_engine_would(one) for one in example)
    return example


def _watch(where, one, layers, edges, made_by, order, batch=None):
    """A hook that writes down what ran and what it was handed."""

    def saw(_module, args, output):
        kind = kind_of(one)
        shape = _shape(output)
        made = Layer(
            where, kind, type(one).__name__, shape, _made_of(one), _dims(kind, shape, batch)
        )
        # What went **in**, which is the only honest way to know whether a layer
        # narrows: the first layer of a node has nothing above it to compare
        # with, and it is exactly the one a bottleneck usually starts at.
        made.came_in = _shape(_tensors(list(args)) and _tensors(list(args))[0:1])
        layers.append(made)
        known = False
        for before in _tensors(args):
            producer = made_by.get(id(before))
            if producer is not None and producer[0] != where:
                edges.append((producer[0], where))
                known = True
        if not known and order:
            # Nothing known fed it, so whatever ran before it did — right for a
            # stack, a guess anywhere else, and the reason `fx` is preferred.
            edges.append((order[-1], where))
        for after in _tensors((output,)):
            # The tensor is kept alive beside its id **on purpose**: CPython
            # reuses an id the moment the object behind it is freed, and an
            # intermediate that nobody holds is freed at once. Without this, a
            # later tensor lands on a dead one's id and the figure draws an edge
            # that never existed — which is worse than a missing one, because a
            # missing edge looks like a missing edge.
            made_by[id(after)] = (where, after)
        order.append(where)

    return saw


#: Composites everybody recognises by name, and nobody wants to read as their
#: parts. A `TransformerEncoderLayer` is **one** thing on a figure; drawn as its
#: fourteen leaves it is fourteen things and a diagram nobody looks at twice.
WHOLE = ("attention", "recurrent")


def _worth_drawing(module, depth=0):
    """Which of a module's parts get a box, as `[(path, module)]`.

    Two rules and they are the same rule: **draw the smallest thing that is
    still a thing**. A leaf is one. A composite everybody recognises — an
    attention block, an LSTM — is one too, and is not opened, because reading
    it as its parts is reading it as something nobody named.

    `depth` opens the composites that many levels further, for when the inside
    of one is exactly what is being asked about.
    """
    said, closed, opened = [], [], []
    for path, one in module.named_modules():
        # `""` is the module itself, and it can be the composite: closing it
        # has to mean *everything under it*, which an empty prefix does only
        # if it is written as one.
        if any(path.startswith(shut) for shut in closed):
            continue
        composite = kind_of(one) in WHOLE
        # `depth` counts **composites opened**, not path components: a
        # `TransformerEncoderLayer` sits three names deep inside a
        # `TransformerEncoder`, and asking for one level of detail should not
        # have to know that.
        under = sum(1 for above in opened if path.startswith(above))
        whole = composite and under >= depth
        if composite:
            opened.append(f"{path}." if path else "")
        if whole:
            closed.append(f"{path}." if path else "")
        if whole or not any(True for _ in one.children()):
            said.append((path, one))
    return said


def _inherited(inside):
    """A layer that did not change the shape keeps the names of the one that did.

    A `BatchNorm1d` in a convolutional trunk produces `(batch, channels, length)`
    because that is what it was handed — naming it by its own kind gets `steps`
    and `dim`, which are the right words for the wrong tensor. What a shape's
    numbers **are** is a fact about where the shape came from.
    """
    by_path = {one.path: one for one in inside.layers}
    feeds = {}
    for a, b in inside.edges:
        feeds.setdefault(b, []).append(a)
    for one in inside.layers:
        if one.kind in ("conv", "recurrent") or not one.shape:
            continue
        older = _same_shape_above(one.path, one.shape, by_path, feeds, set())
        if older is not None:
            one.dims = older
    return inside


def _rank(shape):
    """How many numbers a shape has, which is what says what they mean."""
    return len(shape.split("×"))


def _same_shape_above(path, shape, by_path, feeds, seen):
    """The names of the nearest thing above with the same shape.

    Walking back past what has no shape is the whole of it: a residual's `+` has
    none, so stopping at the first predecessor leaves everything after a skip
    named by its own kind — which is how a `BatchNorm1d` in a convolutional
    trunk came out with `steps` and `dim` on it.
    """
    for before in feeds.get(path, []):
        if before in seen:
            continue
        seen.add(before)
        older = by_path.get(before)
        if older is None:
            continue
        # Against the **original** shape all the way down: comparing against
        # whatever we are standing on means comparing against a `+`, which has
        # no shape, and nothing ever matches.
        #
        # By **rank** and not by the exact sizes: a pooling layer changes what
        # the numbers are, not what they mean — `(batch, channels, length)` with
        # a shorter length is still a length.
        if older.dims and older.shape and _rank(older.shape) == _rank(shape):
            return older.dims
        if not older.shape:
            deeper = _same_shape_above(older.path, shape, by_path, feeds, seen)
            if deeper is not None:
                return deeper
    return None


def _without_a_lone_input(inside):
    """Drops a `fork` box that nothing actually forks from.

    It earns its place when something **other** than the next layer reads it —
    that fork is where a residual starts, and without a box to fork from the
    skip has nowhere to begin. When one thing reads it, it is a box that says
    *and then it began*, which every figure already says by having a top.
    """
    lone = {
        one.path
        for one in inside.layers
        if one.label == "fork" and sum(a == one.path for a, _ in inside.edges) <= 1
    }
    if not lone:
        return inside
    feeds = {a: b for a, b in inside.edges if a in lone}
    return Inside(
        [one for one in inside.layers if one.path not in lone],
        [
            (a, b)
            for a, b in inside.edges
            if a not in lone and b not in lone
        ]
        + [(a, feeds[b]) for a, b in inside.edges if b in lone and b in feeds],
        inside.how,
        inside.why,
        inside.folded,
    )


def _repeated(inside):
    """Blocks that are the same block, collapsed to one and a count.

    Twelve identical transformer layers drawn twelve times is a figure nobody
    reads.

    A **block** is whatever a numbered path component names: `body.layers.3` and
    `body.3.norm` both belong to a third something, and that number is how every
    container in torch says *these are the same thing repeated*. Sameness is by
    shape and not by name — the ordered kinds, labels and relative paths of what
    is in it — and only **consecutive** blocks collapse: two identical ones with
    something else between them are two blocks, and saying `×2` would move one.
    """
    belongs = _blocks_of(inside)
    blocks, order = {}, []
    for one in inside.layers:
        which = belongs[one.path]
        if which not in blocks:
            order.append(which)
        blocks.setdefault(which, []).append(one)
    # By shape and not by name: the same kinds in the same order, and where
    # each sits relative to the block. `body.0.norm` and `body.5.norm` are the
    # same position of two blocks and have to compare equal.
    signature = {
        which: tuple((one.kind, one.label, at) for at, one in enumerate(held))
        for which, held in blocks.items()
    }

    kept, counts, folded = [], {}, {}
    at = 0
    while at < len(order):
        # A repeating unit is not always one block: `Linear, ReLU, Linear, ReLU`
        # repeats with a **period of two**, and comparing neighbours one at a
        # time never sees it. Longest period first, so `A B A B` is two of `A B`
        # and not four of nothing.
        period, times = _period(order, signature, at)
        for step in range(period):
            which = order[at + step]
            kept.append(which)
            counts[which] = times
            for gone in range(1, times):
                for one, mine in zip(blocks[order[at + gone * period + step]], blocks[which]):
                    folded[one.path] = mine.path
        at += period * times

    if all(count == 1 for count in counts.values()):
        return inside
    del signature
    layers = [
        _carried(
            Layer(
                one.path,
                one.kind,
                _times(one.label, counts[which]),
                one.shape,
                one.made_of,
                one.dims,
            ),
            one,
        )
        for which in kept
        for one in blocks[which]
    ]
    at = {one.path: which for which, one in enumerate(layers)}
    edges, seen = [], set()
    for a, b in inside.edges:
        one = (folded.get(a, a), folded.get(b, b))
        if one[0] not in at or one[1] not in at or one in seen:
            continue
        # Forwards only. Folding six blocks into one turns the edge from the
        # sixth back into the first into a loop, and the `×6` already says the
        # thing repeats — drawing it as an arrow going up says something else.
        if at[one[0]] >= at[one[1]]:
            continue
        seen.add(one)
        edges.append(one)
    return Inside(layers, edges, inside.how, inside.why, {**inside.folded, **folded})


def _blocks_of(inside):
    """Which block each layer belongs to.

    A numbered path says it itself. An operation that `fx` recovered does not —
    `symbolic_trace` flattens a container, so a residual's `+` comes back named
    at the parent — and it belongs to **the block it consumes from**. Without
    that, the `+`s sit between the blocks and break the run, and six identical
    residuals never collapse because no two of them are ever adjacent.
    """
    said = {one.path: _block(one.path) for one in inside.layers}
    feeds = {}
    for a, b in inside.edges:
        feeds.setdefault(b, []).append(a)
    for one in inside.layers:
        if _numbered(said[one.path]):
            continue
        from_ = [said.get(a) for a in feeds.get(one.path, [])]
        # Only into a block it could belong to: something under the same
        # container. Without this guard the `Linear` after a stack of encoder
        # layers gets adopted by the last of them, and four identical blocks
        # come out as three and an odd one — which is exactly what happened.
        numbered = [
            which
            for which in from_
            if which and _numbered(which) and one.path.startswith(_container(which))
        ]
        if numbered:
            said[one.path] = max(numbered, key=_ordinal)
    return said


def _container(which):
    """What a numbered block hangs off: `body.layers.3` hangs off `body.layers.`."""
    return which.rsplit(".", 1)[0] + "."


def _ordinal(which):
    """The number a block is, for picking the later of two."""
    return int(which.rsplit(".", 1)[-1])


def _period(order, signature, at):
    """How long the repeating unit starting here is, and how many times it runs.

    **Shortest** first: `A A A A` is four of `A` and not two of `A A`, and a
    longer period that also fits says the same thing less usefully. `A B A B`
    then falls out at period two, which is the case a neighbour-at-a-time scan
    never sees at all. `(1, 1)` when nothing repeats, which is the ordinary case
    and costs one comparison.
    """
    left = len(order) - at
    for period in range(1, left // 2 + 1):
        if not all(_numbered(order[at + step]) for step in range(period)):
            continue
        times = 1
        while (at + (times + 1) * period <= len(order)) and all(
            _numbered(order[at + times * period + step])
            # And out of the same container: a `Linear` after a stack of them
            # has the same shape as one of the stack and is not one of them.
            and _container(order[at + times * period + step]) == _container(order[at + step])
            and signature[order[at + times * period + step]] == signature[order[at + step]]
            for step in range(period)
        ):
            times += 1
        if times > 1:
            return period, times
    return 1, 1


def _block(path):
    """The numbered thing a path belongs to, or the path itself.

    `body.layers.3.self_attn` belongs to `body.layers.3`; `emb` belongs to
    itself. The first numeric component is where a container stopped naming and
    started counting.
    """
    parts = path.split(".")
    for at, one in enumerate(parts):
        if one.isdigit():
            return ".".join(parts[: at + 1])
    return path


def _numbered(which):
    """Whether that block is one of a counted family."""
    return which.rsplit(".", 1)[-1].isdigit()


def _carried(made, from_):
    """The same layer with what went into it kept. Rebuilding a `Layer` and
    losing that is how a funnel goes back to being a rectangle."""
    made.came_in = getattr(from_, "came_in", None)
    return made


def _times(label, count):
    return label if count == 1 else f"{label}  ×{count}"


def _spliced(layers, edges, name, finer):
    """One module's boxes replaced by what `fx` saw inside it.

    The shapes are kept: `fx` traced without data and the run had it, so the
    numbers on the boxes are the ones that really flowed. What is gained is the
    **edges** — a `+` that no hook could have seen.
    """
    mine = f"{name}."
    was = {one.path: one for one in layers if one.path.startswith(mine) or one.path == name}
    outside = [one for one in layers if one.path not in was]
    around = [(a, b) for a, b in edges if a not in was and b not in was]
    into = [(a, b) for a, b in edges if a not in was and b in was]
    out_of = [(a, b) for a, b in edges if a in was and b not in was]

    renamed = [
        _carried(
            Layer(
                f"{mine}{one.path}",
                one.kind,
                one.label,
                (was.get(f"{mine}{one.path}") or one).shape,
                (was.get(f"{mine}{one.path}") or one).made_of,
                (was.get(f"{mine}{one.path}") or one).dims,
            ),
            was.get(f"{mine}{one.path}") or one,
        )
        for one in finer.layers
    ]
    inner = [(f"{mine}{a}", f"{mine}{b}") for a, b in finer.edges]
    known = {one.path for one in renamed}
    # What fed the module now feeds whatever `fx` says came first, and what the
    # module fed is fed by whatever came last.
    first = [one.path for one in renamed if not any(b == one.path for _, b in inner)]
    last = [one.path for one in renamed if not any(a == one.path for a, _ in inner)]
    joined = [(a, b) for a, b in into for b in first] + [(a, b) for _, b in out_of for a in last]
    at = min((which for which, one in enumerate(layers) if one.path in was), default=len(outside))
    return (
        outside[:at] + renamed + outside[at:],
        around + inner + [(a, b) for a, b in joined if a in known or b in known] + [
            (a, b) for a, b in into for b in first
        ],
    )


def _at_most(node, inside, most):
    """A figure holds what a figure holds, and says what it dropped."""
    if len(inside.layers) <= most:
        return inside
    import warnings

    warnings.warn(
        f"`{node}` is {len(inside.layers)} things and the figure holds {most}: drawing "
        f"the first {most}. Raise `architecture(most=...)`, or draw it on its own",
        stacklevel=3,
    )
    kept = {one.path for one in inside.layers[:most]}
    return Inside(
        inside.layers[:most],
        [(a, b) for a, b in inside.edges if a in kept and b in kept],
        inside.how,
        inside.why,
        inside.folded,
    )


def _held(node):
    """The torch modules a node holds, as `[(attribute, module)]`."""
    if torch is None or isinstance(node, type):
        return []
    return [(name, one) for name, one in vars(node).items() if isinstance(one, torch.nn.Module)]
