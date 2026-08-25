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

from soma_next._fingerprint import CannotVersion, bill, digest, fingerprint

BASE = """
    from soma_next import Node

    THRESHOLD = 5

    def long_ones(words):
        return [w for w in words if len(w) > THRESHOLD]

    class Common(Node):
        def forward(self, x, ctx):
            return x

    class Filter(Common):
        def forward(self, words, ctx):
            return long_ones(words)

    class NextDoor(Node):
        def forward(self, x, ctx):
            return x
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
    before, after = changing(write, "return long_ones(words)", "return words")
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
        "class Common(Node):\n        def forward(self, x, ctx):\n            return x",
        "class Common(Node):\n        def forward(self, x, ctx):\n            return x or []",
    )
    assert before != after


# ── What does not change it, and matters just as much ──


def test_a_comment_does_not_change_it(write):
    # The reason for comparing the AST and not the text: a comment does not
    # change what the code does, and making it bump the version would be pure
    # noise.
    before, after = changing(
        write,
        "return long_ones(words)",
        "return long_ones(words)  # mind the accents",
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
        "class NextDoor(Node):\n        def forward(self, x, ctx):\n            return x",
        "class NextDoor(Node):\n        def forward(self, x, ctx):\n            return x * 2",
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
    from soma_next import Node

    class Filter(Node):
        def forward(self, words, ctx):
            return self.model(words)
    """
    and_a_global = holds_one + '\n    model = "nobody\'s business"\n'

    assert digest(write(holds_one)) == digest(write(and_a_global))


def test_a_class_composed_in_init_changes_it(write):
    # El agujero que tenía una huella leída a través del envoltorio.
    #
    # `Node.__init_subclass__` sustituye el `__init__` de cada nodo por uno que
    # recuerda con qué se construyó, así que **todos** los nodos llegaban aquí
    # envueltos. Los nombres que se leían eran los del envoltorio —`_bound`,
    # `BUILT_WITH`— y no los del cuerpo escrito, de modo que una clase compuesta
    # ahí quedaba fuera: editarla no movía nada.
    #
    # Es justo el caso que este fichero existe para cazar. Un nodo que arma un
    # enrutador en su `__init__` y delega en él calcula otra cosa cuando el
    # enrutador cambia, y la caché acertaba y devolvía lo de antes.
    composes = """
    from soma_next import Node

    class Router:
        def route(self, x):
            return x[:1]

    class Filter(Node):
        def __init__(self):
            self.router = Router()

        def forward(self, words, ctx):
            return self.router.route(words)
    """
    changed = composes.replace("return x[:1]", "return x[:2]")
    assert changed != composes, "the test is changing nothing"
    assert digest(write(composes)) != digest(write(changed))


@pytest.mark.xfail(reason="a decorator's body is not reached from the wrapper", strict=True)
def test_a_decorator_of_your_own_still_gets_in(write):
    """El hueco que queda, escrito como test y no como nota al pie.

    Si el decorador es tuyo, cambiarlo cambia lo que la función hace, y la huella
    debería moverse. No se mueve: `@twice` sale en el AST de la clase —así que
    renombrarlo sí cuenta— pero el envoltorio que devuelve **cierra sobre** la
    función envuelta en vez de nombrarla, y de uno a otro no hay ningún global
    que seguir.

    `strict=True` a propósito: el día que alguien lea los nombres de decorador
    del AST de la clase, esto se pone en rojo por pasar, que es la única forma de
    que un límite conocido no se quede escrito cuando deja de serlo.
    """
    decorated = """
    from soma_next import Node

    def twice(fn):
        def again(self, words, ctx):
            return fn(self, fn(self, words, ctx), ctx)

        again.__wrapped__ = fn
        return again

    class Filter(Node):
        @twice
        def forward(self, words, ctx):
            return words[:1]
    """
    changed = decorated.replace("fn(self, fn(self, words, ctx), ctx)", "fn(self, words, ctx)")
    assert changed != decorated, "the test is changing nothing"
    assert digest(write(decorated)) != digest(write(changed))


# ── What the version was computed over ──


@pytest.fixture
def spread(tmp_path, monkeypatch):
    """Writes several modules at once and returns the class asked for.

    A fixture of its own because the question is a different one: `write` asks
    what a change *elsewhere in the file* does, and this asks what the walk does
    when the network is **not** in one file — which is the case `bill` exists
    for and the one `write` cannot set up.
    """
    monkeypatch.syspath_prepend(str(tmp_path))

    def do_it(files, wanted):
        mark = next(NEVER_REPEATED)
        named = {name: f"{name}_{mark}" for name in files}
        for name, source in files.items():
            # The modules import each other by name, and every name got a
            # suffix so no import lands on a previous test's module.
            written = textwrap.dedent(source)
            for plain, unique in named.items():
                written = written.replace(f"from {plain} import", f"from {unique} import")
            (tmp_path / f"{named[name]}.py").write_text(written)
        module, _, called = wanted.partition(":")
        return getattr(__import__(named[module]), called)

    return do_it


#: One network across three files, assembled in one node. What the panel used to
#: show of it was `node.py` and nothing else.
ACROSS = {
    "parts": """
        WIDTH = 32

        class Router:
            def route(self, x):
                return x[:WIDTH]
    """,
    "head": """
        from parts import Router

        class Head:
            def __init__(self):
                self.router = Router()

            def run(self, x):
                return self.router.route(x)
    """,
    "node": """
        from soma_next import Node
        from head import Head

        class Encoder(Node):
            def __init__(self):
                self.net = Head()

            def forward(self, x, ctx):
                return self.net.run(x)
    """,
}


def yours(billed):
    """What of a bill can be opened, by the name it was called."""
    return {one["called"]: one for one in billed if one["kind"] == "yours"}


def test_the_bill_lists_the_class_itself(write):
    cls = write(BASE)
    mine = yours(bill(cls))

    assert "Filter" in mine
    assert mine["Filter"]["file"].endswith(".py")
    assert mine["Filter"]["lines"] > 0


def test_the_bill_reaches_the_files_a_network_is_spread_across(spread):
    # The case this exists for. `inspect.getsourcefile` says one file, because a
    # node **is** its class; the fingerprint has always walked all three, and
    # until now said so only as eight characters of sha256.
    mine = yours(bill(spread(ACROSS, "node:Encoder")))

    assert set(mine) == {"Encoder", "Head", "Router"}
    assert len({one["file"] for one in mine.values()}) == 3


def test_what_is_installed_is_listed_with_its_version_and_no_file(spread):
    reaches = {
        "node": """
            import re
            from re import findall
            from soma_next import Node

            class Filter(Node):
                def forward(self, words, ctx):
                    return findall(re.escape(words), words)
        """
    }
    billed = bill(spread(reaches, "node:Filter"))

    outside = [one for one in billed if one["kind"] in ("installed", "module")]
    assert outside, billed
    # No file on purpose: this is where the walk **stops**, and offering a path
    # into somebody's `site-packages` would say it did not.
    assert all(one["file"] is None for one in outside)
    assert all(one["version"] for one in outside)


def test_the_bill_and_the_version_walk_the_same_thing(spread):
    # Not two walks that could drift: one, read twice. So a file that stops
    # being reached leaves the bill and moves the version in the same edit.
    alone = dict(ACROSS)
    alone["head"] = """
        class Head:
            def run(self, x):
                return x
    """
    before, after = spread(ACROSS, "node:Encoder"), spread(alone, "node:Encoder")

    assert "Router" in yours(bill(before))
    assert "Router" not in yours(bill(after))
    assert digest(before) != digest(after)


def test_a_class_without_source_cannot_be_billed_either(write):
    scope = {}
    exec("class MadeUp:\n    def forward(self, x, ctx): ...", scope)  # noqa: S102

    with pytest.raises(CannotVersion, match="notebook"):
        bill(scope["MadeUp"])


def test_the_bill_comes_out_in_the_same_order_twice(spread):
    cls = spread(ACROSS, "node:Encoder")
    assert bill(cls) == bill(cls)
