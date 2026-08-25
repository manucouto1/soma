"""Which version of the code a class is: `SVMFilter(a1b2c3d4)`.

It is what lets a worker with a clone of the project know whether its
`SVMFilter` is **the same one** the graph was written against. Without it, a
half-updated repository would execute other code and the result would come out
different without anyone noticing.

## What goes into the fingerprint

The **code with functionality**, and transitively whatever that code names that
is yours:

| change | does the fingerprint change? |
|---|---|
| the body of `forward` | yes |
| a helper in the same module that `forward` calls | **yes** |
| a base class | **yes** |
| a module constant it uses | **yes** |
| a class it composes in `__init__` | **yes** |
| a comment or a docstring | no |
| another class in the same file it does not touch | no |

The last two rows are the reason for comparing the **AST** and not the text: a
comment does not change what the code does, and making it change the version
would turn versioning into noise. Docstrings are stripped for the same reason.

## Where it stops

- At **what is installed**: `numpy` is noted by name and distribution version, a
  standard-library module by its bare name — it is glued to the interpreter,
  which is already compared at the greeting.
- At **what has no source**: an extension module cannot be read, so it is
  treated as installed. `Node` and `Opaque` are among those.
- At **`soma_next`**: it is the framework, not your code. Its version goes into
  the client's identity and not into each class's fingerprint; otherwise
  upgrading the library would invalidate them all at once. It is named
  explicitly because in editable mode it lives outside `site-packages`, and the
  alternative is a fingerprint that depends on how you installed the framework.

## What it does not cover, said before it bites

What the code **does not name**: a data file it opens by path, an environment
variable, a model it downloads. It is the same limit Bazel or Nix have when a
rule declares its inputs badly, and there is no closing it by looking at code.

The **body of a decorator of your own**. `@twice` shows in the class's AST, so
renaming it changes the version; rewriting what `twice` does to the call does
not. The wrapper it returns closes over the wrapped function rather than naming
it, so there is no global to follow from the one to the other — reaching it
means reading the decorator names off the class's AST, which nothing does yet.
Unlike the row above this one is closable, and `test_a_decorator_of_your_own`
is the test that goes green the day somebody closes it.
"""

from __future__ import annotations

import ast
import dis
import hashlib
import inspect
import sys
import sysconfig
import textwrap
import types
from typing import Any, Callable

__all__ = ["CannotVersion", "bill", "digest", "fingerprint"]

LENGTH = 8
"""How many characters of the sha256 are shown: 32 bits, plenty to tell apart
versions of one class and short enough to read aloud."""


class CannotVersion(Exception):
    """There is no source to look at, so there is no version to compute."""


def fingerprint(cls: type) -> str:
    """`SVMFilter(a1b2c3d4)` — the class's name and its version.

    Deterministic across processes: nothing that depends on the run goes in, and
    everything collected is sorted before being mixed.
    """
    return f"{cls.__name__}({digest(cls)})"


def digest(cls: type) -> str:
    """Just the version part. Raises `CannotVersion` if the class has no source
    to read — a notebook, an `exec` — which cannot be resolved from a clone
    anyway."""
    return _mixed(_reached(cls).pieces)


def bill(cls: type) -> list[dict[str, Any]]:
    """**What the version was computed over**, said instead of only hashed.

    The same walk `digest` makes, handed back: every class and function of yours
    the fingerprint reached, transitively, and every distribution it stopped at
    with the version it was pinned to. Sorted, so two processes list it the same
    way.

    It exists because that walk is the answer to a second question nobody could
    ask before: **which files is this node made of?** A network written across
    four modules and assembled in one `__init__` is one class to
    `inspect.getsourcefile` and four files to whoever has to read it — and the
    closure that knows the difference was already being computed here and
    thrown away.

    Each entry says what kind of thing it is, because what you can do with it
    differs:

    | `kind` | what it is | has a file |
    |---|---|---|
    | `yours` | a class or function of the project | yes |
    | `installed` | something from a distribution, by name and version | no |
    | `module` | a module named as a whole — `np.array` reaches `numpy` | no |
    | `value` | a module constant, by the name it was read under | no |

    Only `yours` can be opened, and that is the point of the column: an entry
    without a file is not a gap in the answer, it is where the fingerprint
    deliberately stops.

    Raises `CannotVersion` for the same class `digest` refuses, and for the same
    reason: there is no source to read.
    """
    return sorted(
        _reached(cls).noted.values(),
        key=lambda one: (one["kind"], one["module"] or "", one["called"]),
    )


def _mixed(pieces: set[str]) -> str:
    """The pieces into one version.

    Deterministic across processes: everything is sorted before being mixed, and
    nothing that depends on the run went in.
    """
    return hashlib.sha256("\n".join(sorted(pieces)).encode()).hexdigest()[:LENGTH]


class _Reach:
    """What one walk found: what it hashes, and what it walked over.

    Two containers rather than one derived from the other, because they answer
    to different rules. `pieces` is a **set of strings** and its exact contents
    are the fingerprint — anything added to it moves every version there is.
    `noted` is what those pieces were read off, keyed the way the walk memoizes,
    and adding a field to it costs nothing.
    """

    def __init__(self) -> None:
        self.pieces: set[str] = set()
        self.noted: dict[str, dict[str, Any]] = {}

    def note(self, mark: str, kind: str, called: str, obj: object = None, **rest: Any) -> None:
        """Writes down one thing the walk reached. First writing wins, which is
        the same rule `seen` follows one line above every call to this."""
        if mark in self.noted:
            return
        where = _written_where(obj) if kind == "yours" else None
        self.noted[mark] = {
            "kind": kind,
            "called": called,
            "module": getattr(obj, "__module__", None) if kind != "module" else called,
            "file": where[0] if where else None,
            "line": where[1] if where else 0,
            "lines": where[2] if where else 0,
            "version": None,
            **rest,
        }


def _reached(cls: type) -> _Reach:
    """The whole walk from one class, once."""
    reach = _Reach()
    _collect(cls, reach, set())
    return reach


def _written_where(obj: object) -> tuple[str, int, int] | None:
    """Where it is written: file, first line, how many.

    **Absolute**, as `inspect` gives it. Making it relative needs a directory to
    be relative *to*, and the only one this could pick is the process's — which
    is the wrong answer wherever the caller is not standing in the checkout.
    """
    try:
        file = inspect.getsourcefile(obj)  # type: ignore[arg-type]
        source, line = inspect.getsourcelines(obj)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return None
    return (file, line, len(source)) if file else None


def _collect(obj: object, reach: _Reach, seen: set[str]) -> None:
    """Adds to `reach.pieces` what defines `obj`, and follows what it names.

    Memoized by `module:name` but **noted** by name alone, so moving a class
    between files does not change its version.
    """
    if (mark := _who_is(obj)) in seen:
        return
    seen.add(mark)
    called = _what_it_is_called(obj)
    pieces = reach.pieces

    if isinstance(obj, type):
        # The bases first: changing one changes what is inherited.
        for base in obj.__mro__[1:]:
            if _is_yours(base):
                _collect(base, reach, seen)
        pieces.add(f"{called}={_shape(obj)}")
        reach.note(mark, "yours", called, obj)
        for member in vars(obj).values():
            _follow_names(member, reach, seen)
        return

    if isinstance(obj, types.FunctionType):
        pieces.add(f"{called}={_shape(obj)}")
        reach.note(mark, "yours", called, obj)
        _follow_names(obj, reach, seen)
        return

    if isinstance(obj, types.ModuleType):
        pieces.add(f"module:{obj.__name__}@{_version(obj.__name__)}")
        reach.note(mark, "module", obj.__name__, obj, version=_version(obj.__name__))
        return

    # Anything else: its written form. Module constants are noted by
    # `_follow_names`, which knows the name they were read under.
    pieces.add(f"{called}={obj!r}")
    reach.note(mark, "value", called, obj)


def _follow_names(member: object, reach: _Reach, seen: set[str]) -> None:
    """Follows the global names `member`'s code mentions."""
    pieces = reach.pieces
    function = _function_of(member)
    if function is None:
        return
    globals_ = getattr(function, "__globals__", {})
    for name in sorted(_names(function.__code__)):
        if name not in globals_:
            continue  # an attribute (`np.array` → `array`), or a builtin
        pointed = globals_[name]
        if _is_yours(pointed) or isinstance(pointed, types.ModuleType):
            _collect(pointed, reach, seen)
        elif isinstance(pointed, (type, types.FunctionType)):
            # Something installed: what identifies it is where it comes from and
            # with what version.
            pieces.add(f"outside:{_who_is(pointed)}@{_version_of(pointed)}")
            reach.note(
                f"outside:{_who_is(pointed)}",
                "installed",
                _what_it_is_called(pointed),
                pointed,
                version=_version_of(pointed),
            )
        else:
            # A module constant, by content: by type, `THRESHOLD = 5` and
            # `= 7` would both be `builtins.int`. It happened.
            pieces.add(f"{name}={pointed!r}")
            # Noted under the name it was **read** under, which is all there is:
            # a value carries no module of its own, so `from over_there import
            # THRESHOLD` cannot be told from one written here. That is why it is
            # a `value` and not a file somebody gets to open.
            reach.note(f"value:{name}", "value", name)


#: The instructions that read or write a **global**. Anything else that carries
#: a name — an attribute, above all — is not one.
GLOBALS = ("LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL")


def _names(code: types.CodeType) -> set[str]:
    """The global names this code and everything nested mention — descending
    into comprehensions, lambdas and inner functions, whose names live in `code`
    objects inside `co_consts`.

    Read off the **instructions** and not off `co_names`, which mixes globals
    with attribute names: `self.model` puts `model` in there, and if the module
    happens to have a global called `model` too, its value ends up in the
    fingerprint of a class that never named it. Then the version changes on its
    own, the cache says the code changed, and a `--strict` worker refuses to run
    over a mismatch that does not exist.
    """
    out = {i.argval for i in dis.get_instructions(code) if i.opname in GLOBALS}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            out |= _names(constant)
    return out


def _shape(obj: type | Callable[..., Any]) -> str:
    """The AST of its source, without comments or docstrings: what is versioned
    is what the code **does**, not what it says."""
    try:
        source = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError) as e:
        raise CannotVersion(
            f"`{getattr(obj, '__qualname__', obj)}` has no source to read, so it "
            "cannot be versioned. That happens when it is defined in a notebook or "
            "with `exec`; those cannot be resolved from a clone and have to travel whole"
        ) from e

    tree = ast.parse(source)
    _strip_docstrings(tree)
    return ast.dump(tree, annotate_fields=False)


def _strip_docstrings(node: ast.AST) -> None:
    """Removes the first string literal from the body of each def and each class."""
    for child in ast.walk(node):
        if not isinstance(
            child, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(child, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]


def _function_of(member: object) -> Callable[..., Any] | None:
    """The function behind a method, a `staticmethod` or a `property`.

    **Unwrapped.** A decorator returns a function whose code mentions the
    decorator's own globals and not the ones written in the body it wraps, so
    reading it is reading the wrapper — and every name the real function
    reaches for is invisible.

    That is not a hypothetical: `Node.__init_subclass__` wraps every node's
    `__init__` to remember what it was built with, so *every* node in every
    graph reached this with a wrapper in hand. A node that composes a class of
    yours in `__init__` — a router, a head, an encoder — had that class left
    out of its fingerprint, and editing it moved nothing. Which is the one case
    this file exists to catch: the cache hits and hands back what the old code
    produced.
    """
    if isinstance(member, (staticmethod, classmethod)):
        member = member.__func__
    elif isinstance(member, property):
        member = member.fget
    if not isinstance(member, types.FunctionType):
        return None
    # `inspect.unwrap` follows `__wrapped__`, which `functools.wraps` sets. A
    # decorator that does not use it stays opaque, and there is no reading
    # through that from here.
    #
    # And what it lands on is checked rather than assumed: a node that writes no
    # `__init__` of its own gets `object.__init__` wrapped — a C slot, with no
    # `__code__` to read. Unwrapping into it and handing it back turned every
    # such node into an `AttributeError`.
    inner = inspect.unwrap(member)
    return inner if isinstance(inner, types.FunctionType) else member


def _who_is(obj: object) -> str:
    """With module: for memoizing, and for what is installed."""
    module = getattr(obj, "__module__", None)
    if module and (name := _what_it_is_called(obj)):
        return f"{module}:{name}"
    return _what_it_is_called(obj)


def _what_it_is_called(obj: object) -> str:
    """Without module: what gets noted in the fingerprint."""
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    return name or f"value:{type(obj).__module__}.{type(obj).__name__}"


OURS = "soma_next"
"""The framework is not your code. See why above."""


def _is_yours(obj: object) -> bool:
    """Whether it is project code and not a library's or the framework's."""
    if not isinstance(obj, (type, types.FunctionType)):
        return False
    name = getattr(obj, "__module__", None)
    if not name:
        return False
    root = name.split(".")[0]
    if root in sys.stdlib_module_names or root == OURS:
        return False
    file: str | None = getattr(sys.modules.get(name), "__file__", None)
    # `.py` and nothing else: an extension module has no source to read.
    # `is not None` rather than `bool(file)`: the two agree — an empty path does
    # not end in `.py` either — and only one of them narrows the type away from
    # `None` for the two calls that follow.
    return file is not None and file.endswith(".py") and not _installed(file)


def _installed(file: str) -> bool:
    """Whether the file lives where `pip` leaves things."""
    paths = sysconfig.get_paths()
    where = [paths.get(k) for k in ("purelib", "platlib", "stdlib", "platstdlib")]
    return any(place and file.startswith(place) for place in where)


def _version_of(obj: object) -> str:
    """The version of the distribution it comes from, if it can be worked out."""
    name = getattr(obj, "__module__", None) or ""
    return _version(name.split(".")[0])


def _version(package: str) -> str:
    if package in sys.stdlib_module_names:
        # Glued to the interpreter, whose version is compared at the greeting.
        return "stdlib"
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package)
    except (PackageNotFoundError, ImportError, ValueError):
        return "?"
