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

__all__ = ["CannotVersion", "digest", "fingerprint"]

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
    pieces: set[str] = set()
    _collect(cls, pieces, set())
    return hashlib.sha256("\n".join(sorted(pieces)).encode()).hexdigest()[:LENGTH]


def _collect(obj: object, pieces: set[str], seen: set[str]) -> None:
    """Adds to `pieces` what defines `obj`, and follows what it names.

    Memoized by `module:name` but **noted** by name alone, so moving a class
    between files does not change its version.
    """
    if (mark := _who_is(obj)) in seen:
        return
    seen.add(mark)
    called = _what_it_is_called(obj)

    if isinstance(obj, type):
        # The bases first: changing one changes what is inherited.
        for base in obj.__mro__[1:]:
            if _is_yours(base):
                _collect(base, pieces, seen)
        pieces.add(f"{called}={_shape(obj)}")
        for member in vars(obj).values():
            _follow_names(member, pieces, seen)
        return

    if isinstance(obj, types.FunctionType):
        pieces.add(f"{called}={_shape(obj)}")
        _follow_names(obj, pieces, seen)
        return

    if isinstance(obj, types.ModuleType):
        pieces.add(f"module:{obj.__name__}@{_version(obj.__name__)}")
        return

    # Anything else: its written form. Module constants are noted by
    # `_follow_names`, which knows the name they were read under.
    pieces.add(f"{called}={obj!r}")


def _follow_names(member: object, pieces: set[str], seen: set[str]) -> None:
    """Follows the global names `member`'s code mentions."""
    function = _function_of(member)
    if function is None:
        return
    globals_ = getattr(function, "__globals__", {})
    for name in sorted(_names(function.__code__)):
        if name not in globals_:
            continue  # an attribute (`np.array` → `array`), or a builtin
        pointed = globals_[name]
        if _is_yours(pointed) or isinstance(pointed, types.ModuleType):
            _collect(pointed, pieces, seen)
        elif isinstance(pointed, (type, types.FunctionType)):
            # Something installed: what identifies it is where it comes from and
            # with what version.
            pieces.add(f"outside:{_who_is(pointed)}@{_version_of(pointed)}")
        else:
            # A module constant, by content: by type, `THRESHOLD = 5` and
            # `= 7` would both be `builtins.int`. It happened.
            pieces.add(f"{name}={pointed!r}")


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
    """The function behind a method, a `staticmethod` or a `property`."""
    if isinstance(member, types.FunctionType):
        return member
    if isinstance(member, (staticmethod, classmethod)):
        return member.__func__
    if isinstance(member, property):
        return member.fget
    return None


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
