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

from soma_next import Graph, Node, Opaque

from conftest import Add


class Counts(Node):
    """Says how many times it was really asked. A cache hit is invisible from
    the outside unless somebody counts."""

    def __init__(self):
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return x


class Encoder(Node):
    """The same thing under another name, for when two classes have to differ."""

    def __init__(self):
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return x


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


def test_a_graph_that_declares_nothing_is_never_asked_anything():
    # The other side of it: what the check costs is a walk of the graph, and it
    # is only walked by whoever declared a cache. Everything that came before
    # this slice pays nothing.
    g = Graph.somatize(Add(1).named("encoder") >> Counts().named("head"))
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
            return x

    return Twin()


def _second_version():
    class Twin(Node):
        def __init__(self):
            self.calls = 0

        def forward(self, x, ctx):
            self.calls += 1
            # The same answer out of other code, which is exactly the case the
            # warning is for.
            return x + 0.0

    return Twin()


# ── What cannot be written down ──


def test_an_opaque_nobody_can_write_down_is_said_and_not_fatal(store, capfd):
    # A store that cannot take something is not a reason to lose an afternoon's
    # run: it is said on `stderr` and the value is simply not kept. The name of
    # the type is in the message, because that is what you have to register a
    # codec for.
    class Wraps(Node):
        def forward(self, x, ctx):
            return Opaque(object())

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


# ── That what was declared was really obeyed ──


class Stateful(Node):
    """Something with weights, without needing torch to say so: what makes a
    node's state part of its key is that somebody hashed it."""

    def __init__(self, weights):
        self.weights = weights
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return x * self.weights

    def state_dict(self):
        return {"weights": self.weights}


def test_declared_settled_and_never_settled_is_refused(store):
    # **The one failure a cache must not have.** Without the digest of the
    # weights the key does not depend on them, so two checkpoints of the same
    # class share a name and the second run gets the first one's tensor back —
    # no error, no warning, wrong numbers. The core cannot see it: `.frozen()`
    # with no digest is what a tokenizer looks like too.
    g = Graph.somatize(Stateful(2.0).named("encoder").frozen().cached())

    with pytest.raises(ValueError) as raised:
        g.forward(3.0, store=store)

    said = str(raised.value)
    assert "encoder" in said and "checkpoints" in said
    assert "freeze" in said, said


def test_it_is_asked_wherever_a_cache_is_declared(store):
    # Store or no store: it is wrong today, not the day somebody adds a
    # directory to the call.
    g = Graph.somatize(Stateful(2.0).named("encoder").frozen().cached())

    with pytest.raises(ValueError, match="encoder"):
        g.forward(3.0)


def test_settling_it_by_hand_is_enough(store):
    # There is no torch in this file: what `soma_next.torch.freeze` does is call
    # this, and hashing weights is the only part of it that needs torch.
    first, second = Stateful(2.0), Stateful(5.0)
    a = Graph.somatize(first.named("encoder").frozen().cached())
    a.freeze("encoder", "sha256:the-weights-of-monday")
    b = Graph.somatize(second.named("encoder").frozen().cached())
    b.freeze("encoder", "sha256:the-weights-of-tuesday")

    assert a.forward(3.0, store=store) == 6.0
    assert b.forward(3.0, store=store) == 15.0, "two checkpoints, two names"
    assert second.calls == 1


def test_two_checkpoints_settled_the_same_are_one_name(store):
    # The other half: what the digest says is what the key believes.
    first, second = Stateful(2.0), Stateful(5.0)
    for graph, node in ((Graph.somatize(first.named("e").frozen().cached()), first),
                        (Graph.somatize(second.named("e").frozen().cached()), second)):
        graph.freeze("e", "sha256:the-same-weights")
        graph.forward(3.0, store=store)

    assert second.calls == 0, "they were said to be the same state, so they are"


def test_a_node_with_no_state_needs_nobody_to_settle_it(store):
    # A tokenizer does not stop being settled for having nothing to hash, and
    # asking it to be settled by hand would be asking for a digest of nothing.
    g = Graph.somatize(Counts().named("n").frozen().cached())
    assert g.forward(1.0, store=store) == 1.0


# ── With the grain of an item ──


class Embeds(Node):
    """Maps over the items it is handed, and remembers which ones it was made to
    look at — which is the only thing worth observing here."""

    def __init__(self):
        self.seen = []

    def forward(self, items, ctx):
        self.seen += list(items)
        return [x * 10 for x in items]


class Miscounts(Node):
    def forward(self, items, ctx):
        return [1.0]


def mapping():
    embeds = Embeds()
    return embeds, Graph.somatize(
        embeds.named("embed").frozen().mapped().cached()
    )


def test_a_node_that_maps_answers_one_for_each_item(store):
    _, g = mapping()

    assert g.forward([1.0, 2.0, 3.0], store=store) == [10.0, 20.0, 30.0]


def test_and_it_maps_with_no_store_at_all():
    # `.mapped()` is a contract before it is an optimization: a list in, a list
    # as long out. That stays true with nowhere to keep anything.
    _, g = mapping()

    assert g.forward([4.0]) == [40.0]


def test_a_new_item_among_old_ones_is_the_only_one_looked_at(store):
    # **The whole reason this exists.** With one name per node, adding a document
    # changes the name of the list and all of them miss; with one per item, the
    # old ones are read back and the new one runs. And the answer comes out in
    # the order it was asked for, not the order things were computed.
    embeds, g = mapping()

    assert g.forward([1.0, 2.0, 3.0], store=store) == [10.0, 20.0, 30.0]
    assert g.forward([9.0, 1.0, 2.0, 3.0], store=store) == [90.0, 10.0, 20.0, 30.0]

    assert embeds.seen == [1.0, 2.0, 3.0, 9.0], "it looked at some of them twice"


def test_an_item_is_named_after_itself_and_not_after_where_it_sits(store):
    # The same document in another list is the same item. Were a name built out
    # of a position, this would miss on all four — which is the design this was
    # chosen over.
    embeds, g = mapping()

    g.forward([1.0, 2.0, 3.0, 4.0], store=store)
    g.forward([4.0, 3.0, 2.0, 1.0], store=store)

    assert embeds.seen == [1.0, 2.0, 3.0, 4.0], "the same four in another order"


def test_the_second_run_of_the_same_list_looks_at_nothing(store):
    embeds, g = mapping()

    g.forward([1.0, 2.0], store=store)
    g.forward([1.0, 2.0], store=store)

    assert embeds.seen == [1.0, 2.0]


def test_what_reads_a_mapped_node_is_named_after_the_whole_list(store):
    # A node downstream is not mapped: it reads the list, all of it, so its name
    # depends on all of it. Change one item and it has to run again.
    embeds, counts = Embeds(), Counts()
    g = Graph.somatize(
        embeds.named("embed").frozen().mapped().cached()
        >> counts.named("head").frozen().cached()
    )

    g.forward([1.0, 2.0], store=store)
    g.forward([1.0, 2.0], store=store)
    assert counts.calls == 1, "the same list: the head had its answer already"

    g.forward([1.0, 3.0], store=store)
    assert counts.calls == 2, "one item changed and the head has to run again"


def test_something_that_is_not_a_list_is_refused_with_the_node_and_what_arrived():
    _, g = mapping()

    with pytest.raises(ValueError) as e:
        g.forward(1.0)

    assert "embed" in str(e.value)
    assert "number" in str(e.value)


def test_and_so_is_an_answer_with_the_wrong_number_of_items():
    g = Graph.somatize(Miscounts().named("wrong").mapped())

    with pytest.raises(ValueError, match="3 items and answered with 1"):
        g.forward([1.0, 2.0, 3.0])


def test_a_mapped_node_with_two_producers_is_refused_by_what_reaches_it():
    # Two producers make its input a **map**, and "item i" stops meaning one
    # thing. It is refused where it happens, with the node named.
    g = Graph.somatize(
        (Add(1).named("left") | Add(2).named("right"))
        >> Embeds().named("embed").mapped()
    )

    with pytest.raises(ValueError) as e:
        g.forward(1.0)

    assert "embed" in str(e.value) and "map" in str(e.value)
