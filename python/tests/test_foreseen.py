"""The names a graph will use, asked for without running it, and what an edit did.

What this defends is a promise the cache already made: *the name this node's
output will have, **before** it has one*. Only the input is hashed by content, so
every name below it is knowable with nothing executed — and two versions of one
graph name a node differently **exactly when** its recipe changed.

The three answers that would make it useless, and which each have a finding here
so they cannot be given by accident:

- *nothing to say* about a node nobody could name. That is a `.mapped()` node,
  and it has to read as `UNKNOWN`, which is **cannot tell**.
- *nothing to say* about a node whose code was edited. The fingerprint is
  deliberately out of the key, so the name is right and the answer is not: that
  is `STALE`, the finding that says you should have bumped the salt.
- *nothing to say* about what is under one. Everything below a stale node goes on
  being fed the answer the old code gave — including what recomputes, which
  recomputes from it — and that is `SUSPECT`.
- *nothing to say* about a kept node nobody can version. A class defined in a
  cell has no source to read, so in a notebook the code half of the question
  cannot be answered at all, and silence there is the same lie: `UNVERSIONED`.
"""

import json

import pytest

from soma_next import Graph, Node, foreseen

from conftest import Add


class Counts(Node):
    """Says how many times it was really asked, so that *nothing ran* is a
    number and not a hope."""

    def __init__(self):
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return x


class Explodes(Node):
    """Cannot run at all. Its name does not depend on that."""

    def forward(self, x, ctx):
        raise AssertionError("nothing should have asked this to run")


class Encoder(Node):
    """The same thing under another name, for when two classes have to differ."""

    def forward(self, x, ctx):
        return x


@pytest.fixture
def store(tmp_path):
    """A store of its own, which is a directory."""
    return str(tmp_path)


def chain(salt=None, head=Counts):
    """`a >> b`, both settled and both kept, with the salt on the head."""
    return Graph.somatize(
        Counts().named("a").frozen().cached()
        >> head().named("b").frozen().cached(salt=salt)
    )


def three(salt=None):
    """`a >> b >> c`, all settled and all kept, with the salt in the middle."""
    return Graph.somatize(
        Counts().named("a").frozen().cached()
        >> Counts().named("b").frozen().cached(salt=salt)
        >> Counts().named("c").frozen().cached()
    )


# ── A name, without a run ──


def test_a_name_is_had_without_running_the_node_that_earns_it(store):
    g = Graph.somatize(Counts().named("a") >> Explodes().named("b").cached())

    assert sorted(foreseen.names(g, "anything", store=store)) == ["a", "b"]


def test_the_names_do_not_depend_on_having_a_store():
    # A store is where the hash comes from and nothing else is asked of it, so a
    # temporary one gives the names a real one would have given. If that ever
    # stopped being true, a diff would depend on which directory you ran it in.
    g = chain()

    assert foreseen.names(g, "x") == foreseen.names(g, "x", store=None)


def test_what_cannot_be_foreseen_is_missing_rather_than_wrong(store):
    # A mapped node is named by the content of its items, which nobody has yet,
    # and nothing under it can be named either.
    g = Graph.somatize(Add(1).named("a").mapped() >> Counts().named("b"))

    assert foreseen.names(g, [1, 2], store=store) == {}


# ── What would not have to run ──


def test_what_is_already_kept_says_what_would_not_have_to_run(store):
    g = chain()

    assert foreseen.unneeded(g, "x", store=store) == []
    g.forward("x", store=store)

    assert foreseen.unneeded(g, "x", store=store) == ["a"]


def test_asking_what_would_not_run_needs_somewhere_to_look():
    # The one question here whose answer is about what is in the store, so a
    # temporary directory would answer "everything runs" and be believed.
    with pytest.raises(TypeError):
        foreseen.unneeded(chain(), "x")


# ── What an edit did ──


def test_a_graph_compared_with_itself_has_nothing_said_about_it():
    # Absence is the answer for a node that is fine, so it has to be reachable:
    # a finding on every node would make the ones that matter unreadable.
    assert foreseen.changes(three(), three()) == {}


def test_a_salt_changes_the_node_it_is_on_and_nothing_above_it():
    # The asymmetry the whole thing rests on. A salt is the smallest edit to a
    # recipe there is: it has to reach the node it is on and no name above it,
    # which is what tells an edit that invalidated an encoder from one that only
    # touched the head. And the node it is on says which part of it moved.
    assert foreseen.changes(three(), three(salt="a100-fp16")) == {
        "b": ["SALTED"],
        "c": ["DOWNSTREAM"],
    }


def test_another_class_under_the_same_name_is_a_change_of_shape():
    assert foreseen.changes(chain(), chain(head=Encoder)) == {"b": ["CHANGED"]}


def test_other_weights_under_the_same_code_are_not_a_change_of_shape():
    # The split the two questions need. Freezing at another checkpoint really
    # does move the answer, so the cache is right to miss — and nothing about
    # the network was edited, so whoever is asking *what did the code do* has to
    # be able to tell this from a rewrite. Weights belong to a version; they are
    # not one.
    before, after = chain(), chain()
    before.freeze("b", "sha256:monday")
    after.freeze("b", "sha256:tuesday")

    assert foreseen.changes(before, after) == {"b": ["RESETTLED"]}


def test_two_parts_of_one_recipe_moving_says_both():
    # They are not a partition of the node: a rework that also bumps the salt is
    # two true things, and picking one of them would be picking which.
    before, after = chain(), chain(head=Encoder, salt="v2")

    assert foreseen.changes(before, after) == {"b": ["CHANGED", "SALTED"]}


def test_rewiring_a_node_is_a_change_of_its_own_and_not_an_inherited_one():
    # Nothing the node is made of moved — another node feeds it. Its key moves
    # because a key is made of the keys above it, and if `CHANGED` did not
    # include who feeds a node this would read as somebody else's fault.
    def built(feeding):
        g = Graph()
        for node in ("a", "b", "c"):
            g.node(node, Counts())
            g.freeze(node)
            g.written_as(node, "aaaaaaaa")
        for source in feeding:
            g.edge(source, "c")
        g.cache("c")
        return g

    assert foreseen.changes(built(["a", "b"]), built(["a"])) == {"c": ["CHANGED"]}


def test_a_node_that_is_only_in_one_of_them_is_not_a_change_to_a_name():
    before = Graph.somatize(Counts().named("a") >> Counts().named("b"))
    after = Graph.somatize(Counts().named("a") >> Counts().named("c"))

    assert foreseen.changes(before, after) == {"b": ["GONE"], "c": ["ADDED"]}


def test_a_mapped_node_cannot_be_told_about_either_way():
    # And it must not fall through to nothing-said: that is the answer that
    # costs a week, because it is indistinguishable from having checked.
    def built(salt):
        return Graph.somatize(
            Add(1).named("a").mapped() >> Counts().named("b").cached(salt=salt)
        )

    assert foreseen.changes(built(None), built("v2")) == {
        "a": ["UNKNOWN"],
        "b": ["UNKNOWN"],
    }


def test_the_input_cancels_out_of_the_comparison():
    # Every key on both sides carries the same hash of it, so which input it is
    # cannot change a single finding — which is what makes asking with nothing
    # the cheap and correct default rather than a shortcut.
    before, after = three(), three(salt="v2")

    assert foreseen.changes(before, after) == foreseen.changes(
        before, after, "a whole batch"
    )


# ── The code that changed and the name that did not say so ──


def test_an_edit_the_key_cannot_see_is_said_out_loud():
    # The fingerprint is deliberately not in the key: a cosmetic refactor would
    # invalidate half the store in silence. The cost is that a real edit renames
    # nothing, so it is looked at here — as an opinion, next to a name that is
    # correct and an answer that is not.
    before, after = chain(), chain()
    before.written_as("b", "aaaaaaaa")
    after.written_as("b", "bbbbbbbb")

    assert foreseen.changes(before, after) == {"b": ["STALE"]}


def test_what_is_under_a_stale_node_is_not_told_it_is_fine():
    # The half of the graph that quietly goes on running last week's encoder.
    # Nothing about these nodes moved — that is exactly why saying nothing about
    # them would be saying *checked, and fine* about the very thing that is
    # wrong.
    before, after = three(), three()
    before.written_as("a", "aaaaaaaa")
    after.written_as("a", "bbbbbbbb")

    assert foreseen.changes(before, after) == {
        "a": ["STALE"],
        "b": ["SUSPECT"],
        "c": ["SUSPECT"],
    }


def test_a_node_that_recomputes_still_recomputes_from_a_stale_answer():
    # The case that decided the shape of all this. `b` was salted, so it runs
    # again and its own name is honest — and it runs on what `a` handed back,
    # which is the old code's. Two facts about one node, and a design that could
    # only carry one of them would have carried the reassuring one.
    before, after = three(), three(salt="v2")
    before.written_as("a", "aaaaaaaa")
    after.written_as("a", "bbbbbbbb")

    assert foreseen.changes(before, after) == {
        "a": ["STALE"],
        "b": ["SALTED", "SUSPECT"],
        "c": ["DOWNSTREAM", "SUSPECT"],
    }


def test_a_class_nobody_can_version_says_so_rather_than_nothing():
    # A class defined in a notebook cell has no source to read and so no
    # fingerprint — which is what a graph built by hand has too. Silence there
    # would be the lie the whole module is against: in a notebook, where every
    # node is a cell, an afternoon of edits would come back as nothing to
    # report.
    def built():
        g = Graph()
        g.node("a", Counts())
        g.node("b", Counts())
        g.edge("a", "b")
        g.freeze("a")
        g.freeze("b")
        g.cache("b")
        return g

    before, after = built(), built()
    assert not before.fingerprints()

    assert foreseen.changes(before, after) == {"b": ["UNVERSIONED"]}


def test_a_node_nothing_is_kept_of_is_not_told_off_for_having_no_version():
    # A version is only recorded for what is kept, because parsing an AST for a
    # node nobody remembers would be paid by everyone who declares a graph. So
    # the finding has that scope too, or a graph with no cache in it would come
    # back with a line per node saying nothing.
    before = Graph.somatize(Counts().named("a") >> Counts().named("b"))
    after = Graph.somatize(Counts().named("a") >> Counts().named("b"))

    assert not before.fingerprints()
    assert foreseen.changes(before, after) == {}


def test_a_bumped_salt_is_what_stale_is_asking_for():
    # The two findings are the same edit, answered and not answered: with the
    # salt moved the store misses and the node runs its new code, without it the
    # store hits and nobody notices.
    before, after = chain(), chain(salt="v2")
    before.written_as("b", "aaaaaaaa")
    after.written_as("b", "bbbbbbbb")

    assert foreseen.changes(before, after) == {"b": ["SALTED"]}


# ── Two graphs, or two snapshots ──


def test_a_snapshot_answers_exactly_as_the_graph_it_was_taken_of():
    # The whole contract of the second door: whichever side arrives as which,
    # the findings are the same ones. A snapshot that drifted from a graph would
    # be a diff whose answer depended on when you asked it.
    before, after = three(), three(salt="v2")
    kept = (foreseen.snapshot(before), foreseen.snapshot(after))
    live = foreseen.changes(before, after)

    assert foreseen.changes(*kept) == live
    assert foreseen.changes(kept[0], after) == live
    assert foreseen.changes(before, kept[1]) == live


def test_a_snapshot_survives_the_graph_it_came_from():
    # Which is the point: two versions of one module do not coexist in an
    # interpreter, so comparing two commits means comparing what was written
    # down. A round trip through JSON is what "written down" means.
    before = three()
    kept = json.loads(json.dumps(foreseen.snapshot(before)))
    del before

    assert foreseen.changes(kept, three(salt="v2")) == {
        "b": ["SALTED"],
        "c": ["DOWNSTREAM"],
    }


def test_what_is_under_a_stale_node_is_found_through_a_snapshot_too():
    # `SUSPECT` is the one finding that needs the topology and not just the
    # names, so a snapshot has to carry the edges or it would quietly stop
    # reaching down.
    monday, tuesday = three(), three()
    monday.written_as("a", "aaaaaaaa")
    tuesday.written_as("a", "bbbbbbbb")

    assert foreseen.changes(
        foreseen.snapshot(monday), foreseen.snapshot(tuesday)
    ) == {"a": ["STALE"], "b": ["SUSPECT"], "c": ["SUSPECT"]}


# ── The bridge underneath it ──


def test_the_names_foreseen_are_json_and_ordered_by_id(store):
    # It crosses a process boundary, like `resume`'s, so it is ordered rather
    # than however a hash map came out.
    g = Graph.somatize(
        Counts().named("c").frozen().cached() >> Counts().named("a").frozen().cached()
    )

    said = json.loads(g.foreseen_json("x", store=store))

    assert list(said["keys"]) == ["a", "c"]
