"""Which version of the code a class is.

Every test writes a module, imports it, rewrites it and looks again. It is more
cumbersome than defining classes here, and it is the only way to check what
matters: that a change **elsewhere in the file** reaches — or does not reach — a
class's fingerprint.
"""

import itertools
import subprocess
import sys
import textwrap

import pytest

from soma_next._fingerprint import CannotVersion, digest, fingerprint

BASE = """
    from soma_next import Done, Node

    THRESHOLD = 5

    def long_ones(words):
        return [w for w in words if len(w) > THRESHOLD]

    class Common(Node):
        def forward(self, x, ctx):
            return Done(x)

    class Filter(Common):
        def forward(self, words, ctx):
            return Done(long_ones(words))

    class NextDoor(Node):
        def forward(self, x, ctx):
            return Done(x)
"""


NEVER_REPEATED = itertools.count()
"""A counter for the **whole session**, and not for the test.

With one per test, the second test asked for `net_1` again, `import` found it in
`sys.modules` and returned the **previous test's** module. The "this does not
change the fingerprint" checks then compared two different sources and failed for
a reason that had nothing to do with the fingerprint.
"""


@pytest.fixture
def write(tmp_path, monkeypatch):
    """Writes a module, imports it fresh, and returns its `Filter`."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def do_it(source):
        name = f"net_{next(NEVER_REPEATED)}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
        return __import__(name).Filter

    return do_it


def changing(write, old, new):
    """The fingerprint before and after a change in the source.

    It checks that the change **happens** before looking at anything: a search
    string with the indentation miscounted finds nothing, `replace` does nothing,
    and the test ends up comparing the file with itself. The two that expect "no
    change" would always pass. It happened.
    """
    changed = BASE.replace(old, new)
    assert changed != BASE, f"the test is changing nothing: it cannot find {old!r}"
    return digest(write(BASE)), digest(write(changed))


# ── The shape ──


def test_a_fingerprint_is_the_name_and_the_version(write):
    mark = fingerprint(write(BASE))

    assert mark.startswith("Filter(")
    assert mark.endswith(")")
    assert len(mark) == len("Filter()") + 8


def test_the_same_class_always_gives_the_same_fingerprint(write):
    cls = write(BASE)
    assert len({digest(cls) for _ in range(5)}) == 1


def test_the_fingerprint_does_not_depend_on_the_process(tmp_path):
    # What makes it usable as an identity across machines: no `id()`, no
    # `hash()`, no `dict` ordering. Two interpreters, the same number.
    (tmp_path / "net.py").write_text(textwrap.dedent(BASE))
    program = (
        "import sys; sys.path.insert(0, %r)\n"
        "from net import Filter\n"
        "from soma_next._fingerprint import digest\n"
        "print(digest(Filter))" % str(tmp_path)
    )

    outputs = {
        subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(outputs) == 1, outputs


# ── What does change the version ──


def test_changing_the_body_of_forward_changes_it(write):
    before, after = changing(write, "return Done(long_ones(words))", "return Done(words)")
    assert before != after


def test_changing_a_helper_in_the_same_module_changes_it(write):
    # The hole a fingerprint of the bare class would have: `long_ones` is not
    # inside `Filter`, but `Filter` does nothing without it.
    before, after = changing(write, "if len(w) > THRESHOLD", "if len(w) >= THRESHOLD")
    assert before != after


def test_changing_a_module_constant_changes_it(write):
    before, after = changing(write, "THRESHOLD = 5", "THRESHOLD = 7")
    assert before != after


def test_changing_the_base_class_changes_it(write):
    before, after = changing(
        write,
        "class Common(Node):\n        def forward(self, x, ctx):\n            return Done(x)",
        "class Common(Node):\n        def forward(self, x, ctx):\n            return Done(x or [])",
    )
    assert before != after


# ── What does not change it, and matters just as much ──


def test_a_comment_does_not_change_it(write):
    # The reason for comparing the AST and not the text: a comment does not
    # change what the code does, and making it bump the version would be pure
    # noise.
    before, after = changing(
        write,
        "return Done(long_ones(words))",
        "return Done(long_ones(words))  # mind the accents",
    )
    assert before == after


def test_a_docstring_does_not_change_it(write):
    before, after = changing(
        write,
        "    class Filter(Common):\n",
        '    class Filter(Common):\n        """Keeps the long ones."""\n',
    )
    assert before == after


def test_splitting_a_line_in_two_does_not_change_it(write):
    before, after = changing(
        write,
        "return [w for w in words if len(w) > THRESHOLD]",
        "return [\n            w for w in words if len(w) > THRESHOLD\n        ]",
    )
    assert before == after


def test_another_class_in_the_same_file_does_not_change_it(write):
    # If the hash were of the whole module, touching `NextDoor` would invalidate
    # `Filter` and nobody would understand why.
    before, after = changing(
        write,
        "class NextDoor(Node):\n        def forward(self, x, ctx):\n            return Done(x)",
        "class NextDoor(Node):\n        def forward(self, x, ctx):\n            return Done(x * 2)",
    )
    assert before == after


def test_two_different_classes_do_not_share_a_fingerprint(write):
    module = write(BASE)
    import sys as _sys

    next_door = getattr(_sys.modules[module.__module__], "NextDoor")
    assert digest(module) != digest(next_door)


# ── What cannot be versioned ──


def test_a_class_without_source_says_so(write):
    # A notebook cell, or an `exec`. They cannot be resolved from a clone, so
    # versioning them makes no sense either: they travel whole.
    scope = {}
    exec("class MadeUp:\n    def forward(self, x, ctx): ...", scope)  # noqa: S102

    with pytest.raises(CannotVersion, match="notebook"):
        digest(scope["MadeUp"])


def test_a_global_named_like_an_attribute_does_not_get_in(write):
    # `self.model` puts `model` in the code's `co_names`, which mixes globals
    # with attribute names. Read raw, a module that happens to have a global
    # called `model` too had its **value** hashed into the version of a class
    # that never named it. The version then changed on its own: the cache said
    # the code had changed, and a `--strict` worker refused to run over a
    # mismatch that did not exist.
    holds_one = """
    from soma_next import Done, Node

    class Filter(Node):
        def forward(self, words, ctx):
            return Done(self.model(words))
    """
    and_a_global = holds_one + '\n    model = "nobody\'s business"\n'

    assert digest(write(holds_one)) == digest(write(and_a_global))
