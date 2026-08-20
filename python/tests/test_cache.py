"""Keeping what a node produced, so it is not produced again.

The case this exists for is labchain's: an expensive, settled node — an encoder,
an embedding — under a head that changes twenty times an afternoon. What is
defended here:

**A name is known before anything runs.** It is a hash of the *recipe*, not of
the data: only the graph's input is hashed by content, and from there down they
are hashes of hashes. So changing the head does not touch the name of what is
underneath it.

**What cannot be honoured is refused before the first node**, not discovered as
a net that quietly stopped training.
"""

import pytest

from soma_next import Done, Graph, Node, Opaque

from conftest import Add


class Counts(Node):
    """Says how many times it was really asked. A cache hit is invisible from
    the outside unless somebody counts."""

    def __init__(self):
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return Done(x)


class Encoder(Node):
    """The same thing under another name, for when two classes have to differ."""

    def __init__(self):
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return Done(x)


@pytest.fixture
def store(tmp_path):
    """A store of its own, which is a directory."""
    return str(tmp_path)


# ── That it keeps, and that it reads back ──


def test_what_is_kept_is_not_computed_again(store):
    encoder = Counts()
    g = Graph.somatize(encoder.named("encoder").frozen().cached())

    assert g.forward(7.0, store=store) == 7.0
    assert g.forward(7.0, store=store) == 7.0
    assert encoder.calls == 1, "the second run had the answer already"


def test_a_different_input_is_a_different_name(store):
    # The other half, and the reason the input is the one thing hashed by its
    # content: a cache that answers the same for two inputs is a bug.
    encoder = Counts()
    g = Graph.somatize(encoder.named("encoder").frozen().cached())

    assert g.forward(1.0, store=store) == 1.0
    assert g.forward(2.0, store=store) == 2.0
    assert encoder.calls == 2


def test_without_a_store_nothing_is_kept(store):
    # All of it hangs off `store=`: the same graph, run twice, with nowhere to
    # keep anything.
    encoder = Counts()
    g = Graph.somatize(encoder.named("encoder").frozen().cached())

    g.forward(7.0)
    g.forward(7.0)
    assert encoder.calls == 2


def test_changing_the_head_does_not_invalidate_the_embedding(store):
    # **The labchain case**, which is what all of this is for. The encoder is
    # the same object and the same declaration; what changes is what reads it.
    encoder = Counts()
    first = Graph.somatize(encoder.named("encoder").frozen().cached() >> Add(1).named("head"))
    second = Graph.somatize(encoder.named("encoder").frozen().cached() >> Add(100).named("head"))

    assert first.forward(1.0, store=store) == 2.0
    assert second.forward(1.0, store=store) == 101.0
    assert encoder.calls == 1, "the head changed, and it renamed what was below it"


def test_a_node_that_keeps_nothing_does_not_break_the_chain(store):
    # `.cached()` is opt-in because keeping costs. A node without it still gets
    # a name and still passes it on: otherwise declaring it node by node would
    # be declaring it for the whole graph.
    encoder, head = Counts(), Counts()
    g = Graph.somatize(
        encoder.named("encoder").frozen() >> head.named("head").frozen().cached()
    )

    g.forward(7.0, store=store)
    g.forward(7.0, store=store)

    assert encoder.calls == 2, "nothing was kept of it"
    assert head.calls == 1, "and it was still named out of a name nobody kept"


# ── The salt, which is the knob for what the key cannot see ──


def test_the_same_recipe_twice_is_one_name(store):
    # What the salt exists to break: two runs the key cannot tell apart.
    first, second = Counts(), Counts()
    Graph.somatize(first.named("n").frozen().cached()).forward(1.0, store=store)
    Graph.somatize(second.named("n").frozen().cached()).forward(1.0, store=store)

    assert second.calls == 0, "it read what the first one kept"


def test_a_salt_is_another_name(store):
    first, second = Counts(), Counts()
    Graph.somatize(first.named("n").frozen().cached()).forward(1.0, store=store)
    Graph.somatize(second.named("n").frozen().cached(salt="a100-fp16")).forward(
        1.0, store=store
    )

    assert second.calls == 1, "the salt has to reach the key, or it does nothing"


# ── The prefix rule ──


def test_a_cache_over_something_that_can_still_change_is_refused(store):
    # Freezing the node is not enough: what matters is that nothing **above** it
    # can change. Two independent reasons give the same line — the restored
    # value is a leaf, and a node that still trains never hits anyway.
    g = Graph.somatize(Add(1).named("encoder") >> Counts().named("head").frozen().cached())

    with pytest.raises(ValueError, match="encoder"):
        g.forward(1.0, store=store)


def test_it_says_which_one_it_is_and_which_one_moves(store):
    g = Graph.somatize(Add(1).named("encoder") >> Counts().named("head").frozen().cached())

    with pytest.raises(ValueError) as raised:
        g.forward(1.0, store=store)

    said = str(raised.value)
    assert "head" in said and "encoder" in said
    assert "frozen" in said


def test_the_whole_prefix_settled_is_what_passes(store):
    g = Graph.somatize(
        Add(1).named("encoder").frozen() >> Counts().named("head").frozen().cached()
    )
    assert g.forward(1.0, store=store) == 2.0


def test_nothing_is_checked_where_nothing_is_kept():
    # Deliberate: the question is asked by whoever is about to keep something.
    # Without a store there is nothing to keep and nothing to be wrong about.
    g = Graph.somatize(Add(1).named("encoder") >> Counts().named("head").cached())
    assert g.forward(1.0) == 2.0


# ── The fingerprint, which is beside the value and not in the name ──


def test_the_code_changing_says_so_and_uses_what_is_kept(store, capfd):
    # A cosmetic refactor must not invalidate half a store in silence, so the
    # fingerprint is **not** in the key. What it is, is compared on a hit.
    kept = _first_version()
    Graph.somatize(kept.named("encoder").frozen().cached()).forward(7.0, store=store)
    capfd.readouterr()

    changed = _second_version()
    assert (
        Graph.somatize(changed.named("encoder").frozen().cached()).forward(
            7.0, store=store
        )
        == 7.0
    )

    said = capfd.readouterr().err
    assert "encoder" in said, said
    assert "fingerprint" in said, said
    assert changed.calls == 0, "it warned, and it used what was kept"


def _first_version():
    class Twin(Node):
        def __init__(self):
            self.calls = 0

        def forward(self, x, ctx):
            self.calls += 1
            return Done(x)

    return Twin()


def _second_version():
    class Twin(Node):
        def __init__(self):
            self.calls = 0

        def forward(self, x, ctx):
            self.calls += 1
            # The same answer out of other code, which is exactly the case the
            # warning is for.
            return Done(x + 0.0)

    return Twin()


# ── What cannot be written down ──


def test_an_opaque_nobody_can_write_down_is_said_and_not_fatal(store, capfd):
    # A store that cannot take something is not a reason to lose an afternoon's
    # run: it is said on `stderr` and the value is simply not kept. The name of
    # the type is in the message, because that is what you have to register a
    # codec for.
    class Wraps(Node):
        def forward(self, x, ctx):
            return Done(Opaque(object()))

    g = Graph.somatize(Wraps().named("wraps").frozen().cached())
    g.forward(1.0, store=store)

    said = capfd.readouterr().err
    assert "object" in said, said
    assert "codec" in said, said


def test_two_classes_of_the_same_name_are_the_same_identity(store):
    # The known and narrow window of a false hit, written down so nobody
    # discovers it by accident: the identity is the **class's name**. Two
    # different classes called the same, with the same input, share a name — and
    # the fingerprint is what says so out loud.
    assert Counts().__class__.__name__ != Encoder().__class__.__name__
    g = Graph.somatize(Counts().named("n").frozen().cached())
    assert g.forward(1.0, store=store) == 1.0


# ── How it is declared ──


def test_a_whole_piece_is_declared_at_once():
    g = Graph.somatize(
        (Counts().named("a") >> Counts().named("b")).frozen().cached()
        >> Counts().named("c")
    )

    assert list(g.frozen()) == ["a", "b"]
    assert list(g.cached()) == ["a", "b"]


def test_settled_and_kept_are_two_questions():
    g = Graph.somatize(Counts().named("a").frozen() >> Counts().named("b").cached())

    assert list(g.frozen()) == ["a"]
    assert list(g.cached()) == ["b"]


def test_the_innermost_salt_wins():
    # The same rule as `.on()`, and here it is finally observable: what the
    # outer `.cached()` hands out only reaches whoever was not told already.
    g = Graph.somatize(
        (Counts().named("a").cached(salt="a100") >> Counts().named("b")).cached()
    )

    assert g.cached() == {"a": "a100", "b": None}


def test_what_implements_each_node_is_noted_on_declaring_it():
    # Half of what a key is built out of, and it is filled where the object is.
    g = Graph.somatize(Counts().named("a") >> Encoder().named("b"))

    assert g.identities() == {"a": "Counts", "b": "Encoder"}


def test_the_version_of_the_code_is_noted_only_for_what_is_kept():
    # Computing it means parsing an AST, so it is paid where it is used.
    g = Graph.somatize(Counts().named("a") >> Counts().named("b").frozen().cached())

    assert list(g.fingerprints()) == ["b"]
