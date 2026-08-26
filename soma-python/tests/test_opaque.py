"""Values that cross the graph without being converted."""

import pytest

from somatize import Graph, Node, Opaque


class Wraps(Node):
    """Returns whatever arrives, opaque."""

    def forward(self, x, ctx):
        return Opaque(x)


class Remembers(Node):
    """Notes the object it received, so identity can be checked."""

    def __init__(self):
        self.seen = None

    def forward(self, x, ctx):
        self.seen = x
        return Opaque(x)


def test_any_object_crosses_without_being_converted(g):
    class Odd:
        pass

    original = Odd()
    remembers = Remembers()
    g.node("remembers", remembers)

    output = g.forward(Opaque(original))
    assert output is original, "the same object has to come out, not a copy"
    assert remembers.seen is original


def test_the_node_receives_it_unwrapped(g):
    remembers = Remembers()
    g.node("remembers", remembers)
    g.forward(Opaque({1, 2, 3}))  # a set, which unwrapped would not cross
    assert remembers.seen == {1, 2, 3}


def test_it_crosses_several_nodes_staying_the_same_one(g):
    class Odd:
        pass

    original = Odd()
    for name in ("a", "b", "c"):
        g.node(name, Wraps())
    g.edge("a", "b")
    g.edge("b", "c")
    assert g.forward(Opaque(original)) is original


def test_unwrapped_it_still_raises(g):
    g.node("x", Wraps())
    with pytest.raises(TypeError, match="Opaque"):
        g.forward({1, 2, 3})


def test_it_fits_in_a_list_and_in_a_map(g):
    class Odd:
        pass

    one, other = Odd(), Odd()

    class AsList(Node):
        def forward(self, x, ctx):
            return [Opaque(one), Opaque(other)]

    g.node("list", AsList())
    output = g.forward()
    assert output[0] is one and output[1] is other


def test_the_repr_says_what_type_it_is():
    assert repr(Opaque({1: 2})) == "Opaque(dict)"


def test_torchs_autograd_survives_the_graph(g):
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Layer(Node, nn.Module):
        def __init__(self, m):
            nn.Module.__init__(self)
            self.m = m

        def forward(self, x, ctx):
            return Opaque(self.m(x))

    l1, l2 = nn.Linear(4, 3), nn.Linear(3, 2)
    g.node("l1", Layer(l1))
    g.node("relu", Layer(nn.ReLU()))
    g.node("l2", Layer(l2))
    g.edge("l1", "relu")
    g.edge("relu", "l2")

    x = torch.randn(5, 4, requires_grad=True)
    y = g.forward(Opaque(x))

    assert y.requires_grad, "the output is still attached to the graph"
    assert y.grad_fn is not None

    y.pow(2).sum().backward()
    assert x.grad is not None, "the backward pass crosses all three nodes"
    assert l1.weight.grad is not None
    assert l2.weight.grad is not None


def test_converting_to_numbers_breaks_autograd_which_is_why_opaque_exists(g):
    torch = pytest.importorskip("torch")

    class Copy(Node):
        def forward(self, x, ctx):
            return x.tolist()  # unwrapped: it gets converted

    g.node("copy", Copy())
    x = torch.randn(3, requires_grad=True)
    output = g.forward(Opaque(x))

    assert output == pytest.approx(x.tolist())
    assert not torch.tensor(output).requires_grad
