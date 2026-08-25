"""The same store, on a bucket.

`Store(...)` is a directory everybody mounted, and there are clusters where there
is none. `Store.on_bucket(...)` is the other way to have one — S3, MinIO, R2 —
and the point of it is that **nothing above it changes**: `take`, `report`,
`finished` and `gather` take a store and never ask what kind it is.

What the trait promises is checked in Rust, against every implementor there is
(`store/tests/unit/contract.rs`). What is checked here is the half that only
exists in Python: credentials, values with a codec in front of them, and level 3
using the thing by duck.

The bucket half needs something to talk to, so it is opt-in — the same handshake
the Rust contract uses, so one `up -d` serves both::

    docker compose -f store/tests/docker/compose.yaml up -d
    SOMA_S3=http://127.0.0.1:9000 python -m pytest tests/test_bucket.py -q

Without `SOMA_S3` everything that needs a bucket skips, and what does not — an
endpoint that is not there, credentials that are not given — runs anyway.
"""

import os

import pytest

from somatize import Graph, Node, Store
from somatize.study import DONE, RUNNING, Sampler, Space, finished, report, take, trials

ENDPOINT = os.environ.get("SOMA_S3")

needs_a_bucket = pytest.mark.skipif(
    ENDPOINT is None, reason="set SOMA_S3 to run against a real bucket"
)

KEY = os.environ.get("SOMA_S3_KEY", "somanext")
SECRET = os.environ.get("SOMA_S3_SECRET", "somanextsecret")
BUCKET = os.environ.get("SOMA_S3_BUCKET", "soma")


@pytest.fixture
def bucket():
    return Store.on_bucket(ENDPOINT, BUCKET, key=KEY, secret=SECRET)


@pytest.fixture
def mine():
    """Names of this run's own.

    A scratch directory is thrown away between runs and a bucket is not, so
    every name here carries the process that wrote it. The alternative is a test
    that passes once.
    """
    return lambda what: f"test/{os.getpid()}/{what}"


# ── Opening one, which is where the credentials are ──


def test_a_bucket_with_no_credentials_anywhere_says_which_one_it_wanted(monkeypatch):
    # Where everything else looks, so nobody has to pass what `aws` already
    # reads. When it is not there either, the message names the variable.
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)

    with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
        Store.on_bucket("http://127.0.0.1:1", "soma")


def test_an_endpoint_that_is_not_there_says_so_instead_of_being_found_out_later(mine):
    # `on_bucket` talks to the endpoint before it returns, so a wrong address is
    # a failure here and not a study whose first `claim` blows up an hour in.
    with pytest.raises(RuntimeError, match="could not be reached"):
        Store.on_bucket("http://127.0.0.1:1", "soma", key="k", secret="s")


@needs_a_bucket
def test_credentials_are_taken_from_the_environment_when_none_are_given(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)

    assert repr(Store.on_bucket(ENDPOINT, BUCKET)) == f"Store({ENDPOINT}/{BUCKET})"


# ── Bytes and names, which are the same two questions as a directory's ──


@needs_a_bucket
def test_what_it_saves_it_gives_back(bucket):
    digest = bucket.put(b"a tensor, once")

    assert bucket.get(digest) == b"a tensor, once"


@needs_a_bucket
def test_a_name_points_at_bytes_and_carries_what_was_said_beside_it(bucket, mine):
    name = mine("round/3")
    digest = bucket.put(b"weights")

    bucket.bind(name, digest, {"round": "3", "clients": "4"})
    found = bucket.resolve(name)

    assert found.digest == digest
    assert found.meta == [("round", "3"), ("clients", "4")]
    assert found.when > 0


@needs_a_bucket
def test_a_name_nobody_has_can_be_claimed_and_one_somebody_has_cannot(bucket, mine):
    # The one operation a bucket does differently — a conditional PUT rather than
    # a link — and the one everything above it leans on.
    name = mine("the/work")
    ours, theirs = bucket.put(b"me"), bucket.put(b"somebody else")

    assert bucket.claim(name, ours) is True
    assert bucket.claim(name, theirs) is False
    assert bucket.resolve(name).digest == ours, "the second one overwrote it"


# ── Values, which is the layer that only exists here ──


@needs_a_bucket
def test_a_map_of_tensors_goes_in_and_comes_out_a_map_of_tensors(bucket, mine):
    # A codec is a fact about the value and not about where it is going, so this
    # asks nothing new of one — which is the whole reason it is worth checking.
    torch = pytest.importorskip("torch")
    import somatize.torch  # noqa: F401

    bucket.keep(mine("round/0"), {"body": {"0": torch.ones(2, 3)}})

    back = bucket.recall(mine("round/0"))

    assert torch.equal(back["body"]["0"], torch.ones(2, 3))


@needs_a_bucket
def test_recalling_what_nobody_kept_is_nothing_and_not_a_failure(bucket, mine):
    assert bucket.recall(mine("never")) is None


# ── A graph's cache, which until now could only be a disk ──


class Counts(Node):
    """Says how many times it was really asked. A hit is invisible otherwise."""

    def __init__(self):
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return x


@needs_a_bucket
def test_what_a_graph_keeps_can_live_on_a_bucket(bucket, mine):
    # `forward(store=...)` took a path and only a path, so a cache could only be
    # a directory — on a cluster with nothing mounted, no cache at all. It takes
    # a `Store` now, and the same three lines that ran against a disk run here.
    counts = Counts()
    g = Graph.somatize(counts.named("n").frozen().cached(salt=mine("cache")))

    assert g.forward(7.0, store=bucket) == 7.0
    assert g.forward(7.0, store=bucket) == 7.0
    assert counts.calls == 1, "the second answer came out of the bucket"


# ── And the point of all of it: nothing above knows which one it has ──


def searched(store, study):
    """One machine's whole run of a small study, over whatever store it is given.

    Two trials and a straight line for a score: what is being compared is the
    bookkeeping, not the search.
    """
    space = Space().real("lr", 1e-5, 1e-1, log=True)
    sampler = Sampler.sobol(seed=0)
    for trial in range(2):
        point = sampler.ask(space, trial, finished(store, space, study=study))
        assert take(store, point, study=study, trial=trial, me="one"), "nobody else"
        report(
            store,
            point,
            [1.0, 0.5],
            study=study,
            trial=trial,
            me="one",
            state=DONE,
        )
    return space


@needs_a_bucket
def test_a_study_over_a_bucket_and_over_a_directory_answers_the_same(bucket, tmp_path):
    # `take`, `report` and `finished` are not overloaded, not annotated and do no
    # `isinstance`: they were written against a directory and a bucket arrived
    # under them. If that had stopped being true, it would show up here as two
    # different histories.
    study = f"spam-{os.getpid()}"
    directory = Store(str(tmp_path))

    space = searched(bucket, study)
    searched(directory, study)

    assert finished(bucket, space, study=study) == finished(directory, space, study=study)
    assert trials(bucket, space, study=study) == trials(directory, space, study=study)


@needs_a_bucket
def test_a_trial_another_machine_is_holding_is_visible_before_it_finishes(bucket):
    # The liveness of a study over a bucket is the same fact as over a folder:
    # the record is rewritten as it goes, and `running` is in it. Nobody is
    # asked anything.
    study = f"held-{os.getpid()}"
    space = Space().real("lr", 1e-5, 1e-1, log=True)
    point = Sampler.sobol(seed=0).ask(space, 0, [])

    take(bucket, point, study=study, trial=0, me="the other one")

    (seen,) = trials(bucket, space, study=study)
    assert seen["state"] == RUNNING
    assert seen["score"] is None
    assert seen["who"] == "the other one"
