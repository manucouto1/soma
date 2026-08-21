"""The store, opened by hand.

Until now it was a **string** you handed the engine — `forward(store=...)` — and
a place nobody else could open. That is enough while the only thing kept is what
the engine decided to keep, and it stops being enough the moment a training run
has something of its own to write down and another machine has to read it.

Two ways of asking and one directory: `put`/`get`/`bind`/`resolve` deal in bytes
and are `soma_next_store::Store` one for one; `keep`/`recall` deal in values,
tensors included, and are what an export needs.
"""

import subprocess
import sys

import pytest

from soma_next import Store


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


# ── Bytes, by what they are ──


def test_what_it_saves_it_gives_back(store):
    digest = store.put(b"a tensor, once")

    assert store.get(digest) == b"a tensor, once"


def test_the_same_bytes_twice_are_the_same_bytes_once(store):
    # What content addressing is for, and why a round of federated training that
    # changed nothing costs nothing.
    assert store.put(b"the same") == store.put(b"the same")


def test_different_bytes_are_different_names(store):
    assert store.put(b"one") != store.put(b"other")


def test_asking_for_something_that_is_not_here_is_not_a_failure(store):
    assert store.get("sha256:" + "0" * 64) is None


# ── Names, which point at them ──


def test_a_name_points_at_bytes_and_carries_what_was_said_beside_it(store):
    digest = store.put(b"weights")

    store.bind("round/3", digest, {"round": "3", "clients": "4"})
    found = store.resolve("round/3")

    assert found.name == "round/3"
    assert found.digest == digest
    assert found.meta == [("round", "3"), ("clients", "4")]
    assert found.when > 0, "a store you cannot sort by time is one you cannot explore"


def test_binding_a_name_again_replaces_it(store):
    # A name is the question and the answer can be refreshed. Rounds go by.
    store.bind("latest", store.put(b"one"))
    store.bind("latest", store.put(b"other"))

    assert store.get(store.resolve("latest").digest) == b"other"


def test_a_name_nobody_bound_resolves_to_nothing(store):
    assert store.resolve("never") is None


def test_everything_bound_can_be_looked_at(store):
    for which in range(3):
        store.bind(f"round/{which}", store.put(f"{which}".encode()))

    assert sorted(b.name for b in store.bound()) == ["round/0", "round/1", "round/2"]


def test_what_is_remembered_beside_a_name_is_text_to_text(store):
    with pytest.raises(ValueError, match="text to text"):
        store.bind("n", store.put(b"x"), {"round": 3})


# ── Values, which is what anybody actually has ──


def test_a_map_of_tensors_goes_in_and_comes_out_a_map_of_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    import soma_next.torch  # noqa: F401

    store = Store(str(tmp_path))
    store.keep("round/0", {"body": {"0": torch.ones(2, 3), "1": torch.zeros(3)}})

    back = store.recall("round/0")

    assert torch.equal(back["body"]["0"], torch.ones(2, 3))
    assert torch.equal(back["body"]["1"], torch.zeros(3))


def test_a_bare_tensor_is_kept_although_it_would_not_cross_an_edge(tmp_path):
    # The one difference between an edge and a store, and it is on purpose: on an
    # edge a bare tensor is a mistake with two right answers, so refusing it makes
    # the cost of converting visible. In a store it is bytes either way.
    torch = pytest.importorskip("torch")
    import soma_next.torch  # noqa: F401

    store = Store(str(tmp_path))
    store.keep("just/one", torch.ones(4))

    assert torch.equal(store.recall("just/one"), torch.ones(4))


def test_ordinary_values_do_not_need_a_codec_to_be_kept(store):
    store.keep("a/record", {"loss": 0.25, "rounds": 8.0, "who": "client-2"})

    assert store.recall("a/record") == {"loss": 0.25, "rounds": 8.0, "who": "client-2"}


def test_something_nobody_registered_a_codec_for_says_which_type_it_was(store):
    with pytest.raises(ValueError) as e:
        store.keep("nope", {"x": object()})

    assert "`object`" in str(e.value)
    assert "codec(" in str(e.value)


def test_recalling_what_nobody_kept_is_nothing_and_not_a_failure(store):
    assert store.recall("never") is None


def test_keeping_gives_back_the_digest_so_a_round_can_be_named_by_it(store):
    digest = store.keep("round/0", {"loss": 1.0})

    assert store.resolve("round/0").digest == digest
    assert store.get(digest) is not None


# ── Claiming, which is how work gets handed out ──


def test_a_name_nobody_has_can_be_claimed_and_one_somebody_has_cannot(store):
    mine, theirs = store.put(b"me"), store.put(b"somebody else")

    assert store.claim("the/work", mine) is True
    assert store.claim("the/work", theirs) is False
    assert store.resolve("the/work").digest == mine, "the second one overwrote it"


def test_what_bind_replaces_claim_refuses(store):
    # Next to each other on purpose, and they are not the same question: a name
    # whose answer can be refreshed, and a name that is a piece of work somebody
    # took.
    one, other = store.put(b"one"), store.put(b"other")

    store.bind("latest", one)
    store.bind("latest", other)
    store.claim("taken", one)
    store.claim("taken", other)

    assert store.resolve("latest").digest == other
    assert store.resolve("taken").digest == one


def test_a_claim_carries_what_was_said_beside_it_like_any_other_record(store):
    store.claim("work", store.put(b"me"), {"who": "node3"})

    assert store.resolve("work").meta == [("who", "node3")]


RACER = """
import sys
from soma_next import Store
store = Store(sys.argv[1])
mine = store.put(sys.argv[2].encode())
print("won" if store.claim("one/piece/of/work", mine) else "lost")
"""


def test_eight_processes_at_once_and_exactly_one_of_them_wins(tmp_path):
    # The whole point, and the only way to check it is to really race — and to
    # race **processes**, because that is what this is for: eight Slurm tasks on
    # a folder they all mounted. `resolve` and then `bind` passes every test
    # above and loses this one.
    where = str(tmp_path / "shared")
    Store(where)  # so that the eight of them find it made

    racing = [
        subprocess.Popen(
            [sys.executable, "-c", RACER, where, f"racer {which}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for which in range(8)
    ]
    said = []
    for racer in racing:
        out, err = racer.communicate()
        assert racer.returncode == 0, err
        said.append(out.strip())

    assert said.count("won") == 1, said


def test_and_whoever_was_told_they_won_is_the_one_written_down(tmp_path):
    # Not enough that one wins: the record has to be **that** one's, or the
    # winner does the work and somebody else's name is on it.
    where = str(tmp_path / "shared")
    store = Store(where)

    racing = {
        which: subprocess.Popen(
            [sys.executable, "-c", RACER, where, f"racer {which}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for which in range(8)
    }
    won = [which for which, r in racing.items() if r.communicate()[0].strip() == "won"]

    assert len(won) == 1, won
    digest = store.resolve("one/piece/of/work").digest
    assert store.get(digest) == f"racer {won[0]}".encode()


# ── And the point of all of it: it crosses a process ──


WRITER = """
import sys, torch, soma_next.torch
from soma_next import Graph, Node, Done, Opaque, Store
from soma_next.torch import Trainer, parameters

class Layer(Node):
    def __init__(self):
        self.lin = torch.nn.Linear(4, 2)
    def forward(self, x, ctx):
        return Done(Opaque(self.lin(x)))
    def parameters(self):
        return list(self.lin.parameters())

torch.manual_seed(0)
g = Graph.somatize(Layer().named("body"))
t = Trainer(g, objective=torch.nn.functional.mse_loss,
            optimizer=torch.optim.SGD(parameters(g), lr=0.1))
t.step((torch.randn(8, 4), torch.randn(8, 2)))
digest = Store(sys.argv[1]).keep(sys.argv[2], t.export())
print(float(g.implementation("body").lin.weight.sum()), digest)
"""


def written_by_another_process(where, name):
    """Runs the training run above in a separate interpreter and gives back what
    it ended at and the digest it wrote."""
    said = subprocess.run(
        [sys.executable, "-c", WRITER, where, name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert said.returncode == 0, said.stderr
    sum_, digest = said.stdout.split()
    return float(sum_), digest


def test_a_training_run_written_down_over_there_is_read_back_here(tmp_path):
    # What this whole slice is for. One process trains and writes what it learnt;
    # **another** — a different interpreter, nothing shared but the directory —
    # reads it. That is a federated round's client half, and the only thing
    # between the two is a folder both can see.
    torch = pytest.importorskip("torch")
    import soma_next.torch  # noqa: F401

    where = str(tmp_path / "shared")
    theirs, _ = written_by_another_process(where, "round/0")

    exported = Store(where).recall("round/0")

    assert sorted(exported) == ["body"]
    assert sorted(exported["body"]) == ["0", "1"], "the weights and the bias"
    assert float(exported["body"]["0"].sum()) == pytest.approx(theirs, rel=1e-6)


def test_and_two_processes_that_wrote_the_same_weights_wrote_them_once(tmp_path):
    # Content addressing **across processes**: the digest is a fact about the
    # bytes and about nothing else, so two clients that ended at the same weights
    # cost one copy of them. It is what makes a round that changed nothing free.
    pytest.importorskip("torch")

    where = str(tmp_path / "shared")
    _, first = written_by_another_process(where, "client/0")
    _, second = written_by_another_process(where, "client/1")

    assert first == second
    assert sorted(b.name for b in Store(where).bound()) == ["client/0", "client/1"]
    assert len({b.digest for b in Store(where).bound()}) == 1
