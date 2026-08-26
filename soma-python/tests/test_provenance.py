"""What a kept value says about where it came from.

A store outlives every process that wrote to it. Months later somebody is
holding a hash and a question — *what made this, and can I still compare it with
anything?* — and a key cannot answer either, because a key does not run
backwards. So what cannot be recovered is written down at the moment it is
known, beside the value.

Five things, and the split between them is the point:

| written by | what | recoverable later? |
|---|---|---|
| the engine | the node, the fingerprint of its code | it already was |
| the engine | the input, by the name its content has | **never** — only a keeper can hash a value |
| this layer | the environment | **never** — it is not in any key |
| the caller | whatever they know: a commit, an investigation | not by anybody else |

The last row is why none of the others may depend on a caller who remembered.
"""

import json
import sys
import types

import pytest

from somatize import Graph, Node, Store
from somatize import _environment


class Embed(Node):
    def __init__(self, scale=0.5):
        self.scale = scale

    def forward(self, x, ctx):
        return [len(t) * self.scale for t in x]


@pytest.fixture
def kept(tmp_path):
    """A graph that keeps one thing, and the store it kept it in."""

    def run(**how):
        g = Graph.somatize(Embed().named("embed").frozen().cached())
        g.freeze("embed", "weights-v1")
        g.forward(["hello", "and", "goodbye"], store=str(tmp_path), **how)
        return Store(str(tmp_path))

    return run


def said_of(store, node="embed"):
    """What was written beside the one value this graph keeps."""
    for bound in store.bound():
        meta = dict(bound.meta)
        if meta.get("node") == node:
            return meta
    raise AssertionError(f"nothing kept for `{node}`: {[b.name for b in store.bound()]}")


def test_a_value_says_which_node_and_which_code_made_it(kept):
    said = said_of(kept())

    assert said["node"] == "embed"
    assert said["fingerprint"], said


def test_a_value_says_what_the_graph_was_fed(kept):
    # The one the caller could not supply if they wanted to: only a keeper can
    # hash a value. And the one nothing recovers afterwards.
    said = said_of(kept())

    assert said["input"].startswith("sha256:"), said


def test_a_value_says_what_it_was_produced_against(kept):
    # Not in the key and never will be: a fingerprint stops at what is
    # installed, so the interpreter's version is in no name a run produces.
    # Two interpreters can name the same node identically.
    said = said_of(kept())

    assert said["env"] == _environment.named(_environment.environment())


def test_the_environment_is_written_out_in_full_beside_its_digest(kept):
    # Twelve characters group things and explain nothing. Whoever reads this
    # store back in a year needs both, so the reading is bound once under a
    # name anybody can `cat`.
    store = kept()
    said = said_of(store)

    reading = store.recall(f"{_environment.WHERE}/{said['env']}")
    assert reading["python"], reading
    assert reading == _environment.environment()


def test_nobody_has_to_ask_for_any_of_it(kept):
    # The case this exists for. Provenance that has to be remembered is missing
    # from exactly the runs nobody thought would matter — and a graph run with
    # no experiment tool in sight is most of them.
    said = said_of(kept())

    assert {"node", "fingerprint", "input", "env"} <= set(said), said


def test_what_the_caller_knows_is_written_too(kept):
    # Which commit, which investigation: words the engine does not know and
    # must not learn. They arrive as text and are passed through untouched.
    said = said_of(kept(stamping={"run": "an-investigation/3847d0c1"}))

    assert said["run"] == "an-investigation/3847d0c1"


def test_a_caller_cannot_make_a_value_lie_about_its_node(kept):
    # Refused where somebody is typing, rather than dropped where nobody is
    # looking: stamping `node` is not a typo, it is somebody believing they are
    # saying something. And a value that came back naming another node would be
    # the one mistake this whole mechanism exists to make impossible.
    with pytest.raises(ValueError, match="written by the engine"):
        kept(stamping={"node": "other"})


def test_a_stamp_that_is_not_text_is_refused_rather_than_stringified(kept):
    # `str()` over whatever arrived would write somebody's home directory, or an
    # object's address, into a record that is kept for years and handed back as
    # it was got. Being told now beats finding it out from the record.
    with pytest.raises(ValueError, match="text"):
        kept(stamping={"when": 1750000000})


def test_two_runs_in_one_environment_write_one_reading(kept, tmp_path):
    # Bound and not claimed: the second writes what is already there, and a
    # claim would be asking who won a race with no loser.
    kept()
    store = kept()

    readings = [b for b in store.bound() if b.name.startswith(f"{_environment.WHERE}/")]
    assert len(readings) == 1, [b.name for b in readings]


def test_what_is_installed_is_read_once_and_not_once_per_run(monkeypatch):
    # A reading is written on every `forward` that has a store behind it, and
    # all of what it costs is read off disk: the scan walks the metadata of
    # every distribution in the environment — ~350 ms where torch is one of
    # them — and then each version is read again, one file per distribution.
    #
    # Uncached that was a toll of 358 ms a run, and it drowned the 121 ms of
    # weighing a 19 MB batch that CU24 measured. Nothing went red: what a run
    # says is right either way, and only the clock knew.
    import importlib.metadata as about

    scans, versions = [], []
    scan, version = about.packages_distributions, about.version

    def counting_scan():
        scans.append(1)
        return scan()

    def counting_version(distribution):
        versions.append(distribution)
        return version(distribution)

    monkeypatch.setattr(about, "packages_distributions", counting_scan)
    monkeypatch.setattr(about, "version", counting_version)
    # Something has read it already, so the count only means anything from
    # cold. `cache_clear` is `functools.cache`'s and nothing else's, so asking
    # for it here is half of what this test asserts.
    _environment._installed.cache_clear()
    _environment._version.cache_clear()
    try:
        first, second = _environment.environment(), _environment.environment()
    finally:
        _environment._installed.cache_clear()
        _environment._version.cache_clear()

    assert first == second
    assert len(scans) == 1, f"scanned {len(scans)} times"
    assert sorted(versions) == sorted(set(versions)), f"read twice: {versions}"


def test_a_module_imported_after_the_first_reading_is_still_in_the_next(monkeypatch):
    # Keeping the scan must not freeze the reading. What goes in is what the
    # process reached for, and it reaches for more as it runs; only *what is
    # installed* is what does not move.
    monkeypatch.setattr(_environment, "_installed", lambda: {"nothing": ["pytest"]})

    before = _environment.environment()
    assert "pytest" not in before

    monkeypatch.setitem(sys.modules, "nothing", types.ModuleType("nothing"))
    assert "pytest" in _environment.environment()


def test_the_environment_is_the_same_name_twice(tmp_path):
    # Over the JSON with sorted keys, so it is a function of what is in it and
    # not of how the dictionary was built.
    assert _environment.named(_environment.environment()) == _environment.named(
        _environment.environment()
    )
    assert json.dumps(_environment.environment(), sort_keys=True)


def test_a_run_with_nowhere_to_keep_things_says_nothing_about_anything(tmp_path):
    # No store, nothing kept, nothing to attribute. Writing a reading of the
    # environment there would be filing a label for a value that does not exist.
    g = Graph.somatize(Embed().named("embed").frozen().cached())
    g.freeze("embed", "weights-v1")

    assert g.forward(["word"]) == [2.0]
    assert not list(tmp_path.iterdir())
