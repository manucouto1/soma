"""Where the rows come from: a dataset in a store, read by spans.

What is defended here is the half that only shows up from Python: that a source
is **a node like any other** — the DSL reaches it, the cache reaches it, the next
node is handed a frame — and that its version is in the key of everything
computed from it, which is the one failure a cache must not have.
"""

import sys

import pyarrow
import pyarrow.parquet
import pytest

from soma_next import Broker, Graph, Node, Store, Worker
from soma_next.data import Parquet, settle, to_arrow


def parquet(rows):
    """`rows` numbered rows, as the bytes of a parquet file."""
    table = pyarrow.table({"n": list(range(rows))})
    sink = pyarrow.BufferOutputStream()
    pyarrow.parquet.write_table(table, sink)
    return sink.getvalue().to_pybytes()


@pytest.fixture
def where(tmp_path):
    """The directory, for `forward(store=...)` — which takes a path."""
    return str(tmp_path)


@pytest.fixture
def store(where):
    return Store(where)


def holding(store, name, bytes_):
    """That file, bound under that name."""
    store.bind(name, store.put(bytes_))


class Counts(Node):
    """Says how many rows arrived, and how many times it was really asked."""

    def __init__(self):
        self.calls = 0
        self.columns = None

    def forward(self, frame, ctx):
        self.calls += 1
        self.columns = frame.columns
        return float(frame.rows)


# ── A source is a node ──


def test_a_graph_is_handed_a_coordinate_and_the_node_gets_the_rows(store):
    holding(store, "data/numbers", parquet(1000))
    counts = Counts()
    g = Graph.somatize(
        Parquet(store, "data/numbers").named("sms") >> counts.named("counts")
    )

    assert g.forward({"at": 0, "take": 64}) == 64.0
    assert counts.columns == ["n"]


def test_the_last_span_is_short_and_one_past_the_end_is_empty(store):
    holding(store, "data/numbers", parquet(10))
    g = Graph.somatize(Parquet(store, "data/numbers").named("sms") >> Counts().named("counts"))

    assert g.forward({"at": 8, "take": 64}) == 2.0
    assert g.forward({"at": 10, "take": 64}) == 0.0


def test_a_frame_is_whichever_dataframe_you_have_installed(store):
    holding(store, "data/numbers", parquet(20))

    class Keeps(Node):
        def forward(self, frame, ctx):
            self.frame = frame
            return 0.0

    keeps = Keeps()
    Graph.somatize(Parquet(store, "data/numbers").named("sms") >> keeps.named("k")).forward(
        {"at": 4, "take": 3}
    )

    assert to_arrow(keeps.frame).column("n").to_pylist() == [4, 5, 6]


def test_a_name_nobody_bound_says_so_when_it_is_declared(store):
    with pytest.raises(ValueError, match="data/nothing"):
        Parquet(store, "data/nothing")


# ── And its version is in the key ──


def test_a_dataset_is_not_read_twice_for_the_same_span(store, where):
    holding(store, "data/numbers", parquet(1000))
    counts = Counts()
    g = Graph.somatize(
        Parquet(store, "data/numbers").named("sms").frozen()
        >> counts.named("counts").frozen().cached()
    )
    settle(g)

    # The `Store` itself and not the path: a source needs one open anyway, and
    # writing the same directory twice in two shapes is how a cache ends up on a
    # disk while the data is in a bucket.
    assert g.forward({"at": 0, "take": 64}, store=store) == 64.0
    assert g.forward({"at": 0, "take": 64}, store=store) == 64.0
    assert counts.calls == 1


def test_and_another_span_of_it_is_another_question(store, where):
    holding(store, "data/numbers", parquet(1000))
    counts = Counts()
    g = Graph.somatize(
        Parquet(store, "data/numbers").named("sms").frozen()
        >> counts.named("counts").frozen().cached()
    )
    settle(g)

    g.forward({"at": 0, "take": 64}, store=where)
    g.forward({"at": 64, "take": 64}, store=where)

    assert counts.calls == 2


def test_other_data_under_the_same_name_is_not_the_same_answer(store, where):
    # The whole reason a source has to state a version: without it in the key,
    # this second graph reads the first one's rows back out of the store and
    # nothing anywhere says a word.
    holding(store, "data/numbers", parquet(1000))
    first = Counts()
    g = Graph.somatize(
        Parquet(store, "data/numbers").named("sms").frozen()
        >> first.named("counts").frozen().cached()
    )
    settle(g)
    g.forward({"at": 0, "take": 64}, store=where)

    holding(store, "data/numbers", parquet(10))
    second = Counts()
    other = Graph.somatize(
        Parquet(store, "data/numbers").named("sms").frozen()
        >> second.named("counts").frozen().cached()
    )
    settle(other)

    assert other.forward({"at": 0, "take": 64}, store=where) == 10.0
    assert second.calls == 1, "a different dataset is a different question"


def test_a_source_nobody_settled_is_refused_before_anything_runs(store, where):
    holding(store, "data/numbers", parquet(10))
    g = Graph.somatize(
        Parquet(store, "data/numbers").named("sms").frozen()
        >> Counts().named("counts").frozen().cached()
    )

    with pytest.raises(ValueError, match="settle"):
        g.forward({"at": 0, "take": 8}, store=where)


def test_the_version_is_what_the_store_already_knew(store):
    digest = store.put(parquet(10))
    store.bind("data/numbers", digest)

    assert Parquet(store, "data/numbers").version == digest


# ── And it crosses a wire ──


def test_rows_read_here_are_tokenized_over_there(store, where):
    # The shape a cluster has: the dataset is where the client is, the work is
    # somewhere else. A frame is an opaque like a tensor, so it crosses the same
    # way — Arrow IPC in front of it, and the worker is handed rows.
    cloudpickle = pytest.importorskip("cloudpickle")
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    holding(store, "data/numbers", parquet(100))

    worker = Broker.embedded({"w1": Worker.generic(mode="network")})
    g = Graph.somatize(
        Parquet(store, "data/numbers").named("sms")
        >> Counts().named("counts").at("w1")
    )

    assert g.forward({"at": 0, "take": 32}, broker=worker) == 32.0


# ── A column, without a dataframe library ──


def test_a_column_comes_over_as_plain_python_values(store):
    holding(store, "sms/train", parquet(6))

    class Reads(Node):
        def forward(self, frame, ctx):
            return frame.column("n")

    got = Graph.somatize(
        Parquet(store, "sms/train").named("sms") >> Reads().named("reads")
    ).forward({"at": 2, "take": 3})

    assert got == [2.0, 3.0, 4.0]


def test_and_a_column_nobody_has_says_which_ones_there_are(store):
    holding(store, "sms/train", parquet(6))
    source = Parquet(store, "sms/train")

    class Asks(Node):
        def forward(self, frame, ctx):
            return frame.column("nope")

    with pytest.raises(ValueError, match="no column `nope`"):
        Graph.somatize(source.named("sms") >> Asks().named("asks")).forward(
            {"at": 0, "take": 2}
        )
