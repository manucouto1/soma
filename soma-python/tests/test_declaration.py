"""What a node was built with, which is the other half of what a node is.

`Embed(512)` and `Embed(64)` are one class, one identity and two different
answers. Before this they shared a name in a store and the second one was handed
the first one's — no error, no warning, the wrong tensor. Same failure as a
checkpoint nobody hashed, and it is in a key for the same reason.

Everything here defends one of two things, and they pull in opposite directions:

**Faithful** — two different declarations must never be written down the same
way. That is the collision above, and it is the dangerous one: a wrong value,
served in silence.

**Steady** — one declaration must be written down the same way in another
process, because a key is computed on the client and computed again on a worker.
Getting this wrong costs a cache that misses forever and never says why.

Anything that cannot be both is refused, with the attribute named, before the
first node runs.
"""

import subprocess
import sys
import textwrap

import pytest

from somatize import Graph, Node
from somatize._declaration import CannotDeclare, digest, written


class Embed(Node):
    """A node whose answer depends on what it was built with, which is the whole
    case."""

    def __init__(self, dim, dropout=0.0):
        self.dim = dim
        self.dropout = dropout

    def forward(self, x, ctx):
        return x * self.dim


class Helper:
    """Somebody else's object, with no `__repr__` of its own — which is what
    makes it the trap."""

    def __init__(self, k=1):
        self.k = k


class Holds:
    """Whatever you give it, as attributes."""

    def __init__(self, **held):
        self.__dict__.update(held)


def in_another_process(building):
    """The digest of that object, worked out by an interpreter of its own.

    A subprocess and not a fixture pretending to be one: hash randomisation is
    per-process, so a set written down in *this* one proves nothing at all.
    """
    said = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(building)],
        capture_output=True,
        text=True,
        check=True,
    )
    return said.stdout.strip()


def test_two_arguments_are_two_declarations():
    assert digest(Embed(512)) != digest(Embed(64))


def test_the_same_arguments_are_one_declaration():
    assert digest(Embed(512, dropout=0.1)) == digest(Embed(512, dropout=0.1))


def test_what_a_node_holds_is_followed_and_not_believed():
    # The trap the type of a thing cannot catch: a **list** of objects with no
    # `__repr__` has `list.__repr__`, which is defined, so anyone trusting the
    # repr of what they were handed lets two addresses through from inside. They
    # are walked instead, and what comes out tells the two lists apart.
    assert (
        written(Holds(h=[Helper(1), Helper(2)]))
        == "Holds(h=[Helper(k=1), Helper(k=2)])"
    )
    assert digest(Holds(h=[Helper(1)])) != digest(Holds(h=[Helper(2)]))


def test_a_mapping_built_in_another_order_is_the_same_mapping():
    assert digest(Holds(c={"a": 1, "b": 2})) == digest(Holds(c={"b": 2, "a": 1}))


def test_and_another_mapping_is_not():
    assert digest(Holds(c={"a": 1})) != digest(Holds(c={"a": 2}))


def test_a_set_is_written_down_in_an_order_of_its_own():
    # A set's repr follows the hash table, and string hashing is seeded per
    # process, so the naive answer is stable in one interpreter and different in
    # the next — a cache that misses forever without ever saying why.
    here = digest(Holds(tags={"zebra", "apple", "mango"}))
    there = in_another_process(
        """
        from somatize._declaration import digest
        class Holds:
            def __init__(self, **held): self.__dict__.update(held)
        print(digest(Holds(tags={"zebra", "apple", "mango"})))
        """
    )

    assert here == there


def test_what_a_node_holds_is_the_same_text_in_another_process():
    here = digest(Holds(h=[Helper(1), Helper(2)], conf={"b": 2, "a": 1}, dim=512))
    there = in_another_process(
        """
        from somatize._declaration import digest
        class Helper:
            def __init__(self, k=1): self.k = k
        class Holds:
            def __init__(self, **held): self.__dict__.update(held)
        print(digest(Holds(h=[Helper(1), Helper(2)], conf={"b": 2, "a": 1}, dim=512)))
        """
    )

    assert here == there


def test_a_repr_that_writes_its_own_address_is_refused():
    class Wraps:
        def __init__(self, inner):
            self.inner = inner

        def __repr__(self):
            return f"Wraps({self.inner!r})"

    with pytest.raises(CannotDeclare) as why:
        written(Holds(w=Wraps(Helper())))

    assert "Holds.w" in str(why.value) and "process" in str(why.value)


def test_something_that_writes_itself_in_angle_brackets_is_refused():
    # CPython's own convention for *this has no faithful repr*: a socket, a
    # file, a generator. The stable ones that wear them — an enum, a class, a
    # function — are taken out before this rule is reached.
    import socket

    with pytest.raises(CannotDeclare) as why:
        written(Holds(s=socket.socket()))

    assert "Holds.s" in str(why.value)


def test_a_lambda_is_refused_because_every_lambda_has_the_same_name():
    with pytest.raises(CannotDeclare) as why:
        written(Holds(key=lambda x: x))

    assert "lambda" in str(why.value) and "salt" in str(why.value)


def test_data_held_as_an_attribute_is_refused_rather_than_truncated():
    class Arrayish:
        shape = (1000, 1000)
        dtype = "float32"

        def __repr__(self):
            return "Arrayish([[0.1, 0.2, ..., 0.9]])"

    with pytest.raises(CannotDeclare) as why:
        written(Holds(w=Arrayish()))

    assert "freeze" in str(why.value)


def test_something_that_holds_itself_is_refused_rather_than_followed():
    it = Holds()
    it.me = it

    with pytest.raises(CannotDeclare) as why:
        written(it)

    assert "holds itself" in str(why.value)


def test_a_class_with_no_arguments_says_so_and_nothing_else():
    # It binds against `object.__init__(*args, **kwargs)`, so the honest but
    # useless answer is `Plain(args=(), kwargs={})`. Empty is nothing, and this
    # text is what somebody reads to find out why a key moved.
    class Plain(Node):
        def forward(self, x, ctx):
            return x

    assert written(Plain()) == "Plain()"


def test_and_what_was_spread_is_written_only_when_there_is_some():
    class Spread(Node):
        def __init__(self, dim, *rest, **said):
            self.dim = dim

        def forward(self, x, ctx):
            return x

    assert written(Spread(4)) == "Spread(dim=4)"
    assert (
        written(Spread(4, 5, mode="x"))
        == "Spread(dim=4, rest=(5,), said={'mode': 'x'})"
    )


def test_a_tuple_is_written_the_way_python_writes_one():
    # `(,)` is not something anybody can paste back into an interpreter, and
    # this text exists to be read.
    assert written(Holds(t=())) == "Holds(t=())"
    assert written(Holds(t=(1,))) == "Holds(t=(1,))"
    assert written(Holds(t=(1, 2))) == "Holds(t=(1, 2))"


def test_a_name_and_not_a_source_is_what_a_class_or_a_function_writes():
    # What they *do* is the fingerprint's question, and noting them by name is
    # the same choice made there: moving a helper to another file must not
    # invalidate a store.
    assert written(Holds(cls=Helper, fn=len)) == "Holds(cls=Helper, fn=len)"


def test_two_nodes_built_differently_are_not_kept_under_one_name(tmp_path):
    from somatize import foreseen

    def built(dim):
        return Graph.somatize(Embed(dim).named("embed").frozen().cached())

    assert foreseen.names(built(512)) != foreseen.names(built(64))


def test_and_the_answer_is_the_one_that_was_asked_for(tmp_path):
    store = str(tmp_path)
    wide = Graph.somatize(Embed(10).named("embed").frozen().cached())
    narrow = Graph.somatize(Embed(2).named("embed").frozen().cached())

    assert wide.forward(1.0, store=store) == 10.0
    assert narrow.forward(1.0, store=store) == 2.0, "it was handed the other one"


def test_a_graph_built_by_hand_is_told_apart_too():
    # The same door, reached without the DSL. A topology built in a loop has the
    # same collision and had no answer for it.
    def built(dim):
        g = Graph()
        g.node("embed", Embed(dim))
        g.freeze("embed")
        g.cache("embed")
        return g

    assert built(512).declarations() != built(64).declarations()


class Given(Node):
    """Whatever you hand it. What it does with it is not the question."""

    def __init__(self, held):
        self.held = held

    def forward(self, x, ctx):
        return x


def test_a_cache_that_cannot_be_named_is_refused_before_the_first_node():
    g = Graph.somatize(Given(lambda x: x).named("given").frozen().cached())

    with pytest.raises(ValueError) as why:
        g.forward(1.0)

    assert "given" in str(why.value) and "lambda" in str(why.value)


def test_and_a_graph_that_keeps_nothing_is_not_asked():
    # The check is about a key that has to mean something. Nothing is kept here,
    # so there is no name to collide and nothing to refuse.
    assert Graph.somatize(Given(lambda x: x).named("given")).forward(1.0) == 1.0


def test_what_a_node_keeps_for_itself_is_not_what_it_was_built_with():
    # The distinction the whole thing rests on. A counter, a client built on
    # first use, a tensor moved onto a device: all of them move **while the
    # graph runs**, and a key made of them would rename a node nobody touched.
    class Counts(Node):
        def __init__(self, dim):
            self.dim = dim
            self.calls = 0
            self.helper = lambda x: x

        def forward(self, x, ctx):
            self.calls += 1
            return x

    it = Counts(512)
    before = digest(it)
    it.calls = 99

    assert digest(it) == before
    assert written(it) == "Counts(dim=512)"

