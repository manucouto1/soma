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
    "learned",  # holds weights: Linear, Conv, Embedding, the cell of an LSTM
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
    "Conv1d": "learned",
    "Conv2d": "learned",
    "Conv3d": "learned",
    "ConvTranspose1d": "learned",
    "ConvTranspose2d": "learned",
    "ConvTranspose3d": "learned",
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

    __slots__ = ("path", "kind", "label", "shape")

    def __init__(self, path, kind, label, shape=None):
        self.path = path
        self.kind = kind
        self.label = label
        #: The shape of what it produces, as text — `(32, 8)`. The one thing
        #: that makes a **bottleneck** visible: `512 → 8 → 512` is a picture and
        #: `Linear · Linear · Linear` is not. `None` when nobody ran it.
        self.shape = shape

    def __eq__(self, other):
        return isinstance(other, Layer) and self._as_tuple() == other._as_tuple()

    def __hash__(self):
        return hash(self._as_tuple())

    def _as_tuple(self):
        return (self.path, self.kind, self.label, self.shape)

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

    __slots__ = ("layers", "edges", "how", "why")

    def __init__(self, layers, edges, how, why=None):
        self.layers = list(layers)
        self.edges = list(edges)
        self.how = how
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
        # Kept, and it is not bookkeeping: `x + f(x)` **forks** at the input,
        # and without a box to fork from the skip has nowhere to start.
        return Layer(one.name, "shaping", "input")
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


def _tensors(what):
    """Every tensor in whatever a module was handed or returned."""
    found = []
    stack = list(what)
    while stack:
        one = stack.pop()
        if torch is not None and isinstance(one, torch.Tensor):
            found.append(one)
        elif isinstance(one, (tuple, list)):
            stack.extend(one)
        elif isinstance(one, dict):
            stack.extend(one.values())
    return found


def _shape(output):
    """What a layer produces, as text. The one thing that makes a bottleneck
    visible at all."""
    found = _tensors((output,))
    if not found:
        return None
    return "×".join(str(one) for one in tuple(found[0].shape))


def architecture(graph, example=None, *, most=48):
    """What each node is made of, as `{node: Inside}` — ready for a figure.

        g.figure(inside=architecture(g, x))

    **The unit is the node**, and it has to be: a node holding two modules
    composes them in its own `forward`, and tracing each of them on the same
    input would not only miss that edge, it would feed the second one the wrong
    tensor. So the node's `forward` is run once, with hooks on everything it
    holds, and the edges come out of which tensor each module was handed.

    Then, module by module, `torch.fx` is asked for the same thing. Where it can
    answer it wins, because it sees the operations that are **not** modules —
    and a residual connection is exactly one of those. Where it cannot, the run
    stands and the figure says so.

    That is the seam, and it is a module boundary rather than a judgement call:
    two paths, one shape, spliced where both of them agree the world divides.

    `example` is one input to run it on. Without one nothing can be traced at
    all, and every node answers nothing rather than a guess.
    """
    if example is None or torch is None:
        return {}
    said = {}
    for node in graph.nodes():
        inside = _of_node(graph.implementation(node), example)
        if inside is None or not inside.layers:
            continue
        said[node] = _at_most(node, inside, most)
    return said


def _of_node(held, example):
    """One node's own `forward`, run once and watched."""
    from soma_next import Ctx

    modules = _held(held)
    if not modules:
        return None
    layers, edges, made_by, order = [], [], {}, []
    hooks = []
    for name, module in modules:
        for path, one in module.named_modules():
            where = f"{name}.{path}" if path else name
            hooks.append(one.register_forward_hook(_watch(where, one, layers, edges, made_by, order)))
    try:
        with torch.no_grad():
            held.forward(example, Ctx())
    except Exception as e:
        for hook in hooks:
            hook.remove()
        return Inside([], [], "traced", f"{type(e).__name__}: {e}")
    for hook in hooks:
        hook.remove()

    # And now the half `fx` does better, one module at a time — but only for a
    # module that **contains** something. Asking it about a bare `Linear` gets
    # back "an input, then a linear", which is one box of noise and the loss of
    # the name and the shape the run already had.
    for name, module in modules:
        if not any(True for _ in module.children()):
            continue
        finer, _ = _symbolic(module, None)
        if finer is not None:
            layers, edges = _spliced(layers, edges, name, finer)
    return Inside(layers, edges, "traced", None)


def _watch(where, one, layers, edges, made_by, order):
    """A hook that writes down what ran and what it was handed."""

    def saw(_module, args, output):
        # Only the leaves: a container's box is the boxes of what is in it.
        if any(True for _ in _module.children()):
            return
        layers.append(Layer(where, kind_of(one), type(one).__name__, _shape(output)))
        known = False
        for before in _tensors(args):
            producer = made_by.get(id(before))
            if producer is not None and producer != where:
                edges.append((producer, where))
                known = True
        if not known and order:
            # Nothing known fed it, so whatever ran before it did — right for a
            # stack, a guess anywhere else, and the reason `fx` is preferred.
            edges.append((order[-1], where))
        for after in _tensors((output,)):
            made_by[id(after)] = where
        order.append(where)

    return saw


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
        Layer(
            f"{mine}{one.path}",
            one.kind,
            one.label,
            (was.get(f"{mine}{one.path}") or one).shape,
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
    return (
        outside + renamed,
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
    )


def _held(node):
    """The torch modules a node holds, as `[(attribute, module)]`."""
    if torch is None or isinstance(node, type):
        return []
    return [(name, one) for name, one in vars(node).items() if isinstance(one, torch.nn.Module)]
