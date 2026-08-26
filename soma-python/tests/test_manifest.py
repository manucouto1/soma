"""The `project` artifact: names, versions and state, without code.

Only unit tests here. That the whole thing works against a real worker is in
`test_integration.py`; what is pinned down here is what the artifact promises on
its own — that no code travels, that the version is checked class by class, and
that a class moved between files is still found.
"""

import importlib
import linecache
import sys
import textwrap

import pytest

from somatize._manifest import KIND, DifferentVersion, _find, pack, unpack

BASE = """
    from somatize import Node

    class Filter(Node):
        def __init__(self, threshold):
            self.threshold = threshold

        def forward(self, words, ctx):
            return [w for w in words if len(w) > self.threshold]
"""


@pytest.fixture
def write(tmp_path, monkeypatch):
    """Writes a module and imports it fresh, even under the same name.

    Rewriting a file in place is how a worker with a half-updated clone is
    simulated, and it takes evicting three caches: `sys.modules`, the import
    finders\' and `linecache`\'s — the last is where `inspect.getsource` reads
    from, so without clearing it the fingerprint would come out of the old text.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    written = []

    def do_it(source, name=None):
        name = name or f"proj_{len(written)}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
        written.append(name)
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        linecache.clearcache()
        return importlib.import_module(name)

    return do_it


def test_the_kind_is_what_the_wire_calls_it():
    assert KIND == "project"


def test_the_state_travels_and_the_code_does_not(write):
    # The whole point: a `Filter(5)` is a class name plus a `__dict__`, not
    # bytecode. `cloudpickle` of the same thing goes past ten kilobytes.
    module = write(BASE)
    blob = pack({"f": module.Filter(5)})

    assert len(blob) < 512, f"it weighs {len(blob)} bytes"
    assert b"forward" not in blob, "the body of the method travelled"


def test_the_round_trip_gives_back_the_nodes_with_their_state(write):
    module = write(BASE)
    nodes = unpack(pack({"f": module.Filter(5)}))

    assert list(nodes) == ["f"]
    assert nodes["f"].threshold == 5
    assert type(nodes["f"]) is module.Filter, "it resolved against this clone"


def test_another_version_of_the_class_stops_it(write):
    module = write(BASE)
    blob = pack({"f": module.Filter(5)})

    # The same module rewritten in place: the name resolves, the fingerprint
    # does not match.
    changed = BASE.replace("len(w) > self.threshold", "len(w) >= self.threshold")
    assert changed != BASE
    write(changed, name=module.__name__)

    with pytest.raises(DifferentVersion) as e:
        unpack(blob)

    said = str(e.value)
    assert "Filter(" in said, said
    assert "--lucky" in said, f"it has to say the way out: {said}"


def test_lucky_executes_whatever_it_has_and_warns(write):
    module = write(BASE)
    blob = pack({"f": module.Filter(5)})
    write(BASE.replace("len(w) > self.threshold", "len(w) >= self.threshold"),
          name=module.__name__)

    warnings = []
    nodes = unpack(blob, strict=False, warn=warnings.append)

    assert nodes["f"].threshold == 5
    assert len(warnings) == 1, warnings
    assert "--lucky" in warnings[0] and "Filter(" in warnings[0]


def test_a_class_that_is_not_yours_is_not_versioned(write):
    # A `dict` or a `str` inside the state resolves as always: there is nothing
    # to version in the standard library.
    module = write(BASE)
    node = module.Filter(5)
    node.extra = {"a": [1, 2]}

    assert unpack(pack({"f": node}))["f"].extra == {"a": [1, 2]}


#
# Moving a class between files without touching it does not change its
# fingerprint — on purpose, so it stays the same version. Then the hint stops
# holding and the name still does.


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A package with the class in a module the client would not name."""
    monkeypatch.syspath_prepend(str(tmp_path))
    root = tmp_path / "moved"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "elsewhere.py").write_text("class Thing:\n    pass\n")
    return root


def test_the_sweep_finds_a_class_that_changed_file(package):
    # The client's hint says `moved.gone`, which does not exist here.
    assert _find("moved.gone", "Thing").__name__ == "Thing"


def test_the_fast_path_is_the_hint_when_it_still_holds(package):
    assert _find("moved.elsewhere", "Thing").__name__ == "Thing"


def test_a_worker_without_the_package_says_which_one_and_the_way_out():
    with pytest.raises(DifferentVersion) as e:
        _find("a_package_that_is_not_installed", "Thing")

    said = str(e.value)
    assert "a_package_that_is_not_installed" in said, said
    assert "network" in said, f"it has to say the way out: {said}"


def test_a_name_that_is_nowhere_in_the_package_says_so(package):
    with pytest.raises(DifferentVersion, match="nor anywhere in `moved`"):
        _find("moved.elsewhere", "NotHere")
