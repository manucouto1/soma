"""The stub describes the module that was actually built.

`_somatize.pyi` is a promise about a binary it cannot see. The promise rots the
first time somebody adds a `#[pymethods]` and does not think about typing, and
nothing complains: the stub goes on type-checking, it just stops describing
reality. **A stub that lies is worse than no stub**, because what it produces is
a confident wrong answer where `Any` would have produced a shrug.

So it is checked against the compiled module, here. These tests do not verify
that the *types* are right — no test can, a type is a claim about meaning. They
verify that the two have the same **shape**: the same classes, the same methods
and attributes on each, the same parameter names in the same order, and the same
parameters carrying defaults. That is the drift that happens by accident. What is
left — a wrong type on a parameter that does exist — is the part a person has to
get right, and it is the part a reviewer can see in a diff.

Two PyO3 shapes this has to know about, and both bit while it was being written:

- **A `#[new]`'s signature lands on the type**, as `cls.__text_signature__`, and
  not on `__new__`. A class with no `#[new]` has `None` there, which is how
  "constructible" is asked.
- **A `#[getter]` is a `getset_descriptor`**, which is not callable, so it never
  shows up as a method. In the stub the same thing is written `@property def`,
  which *is* a function — so the decorator is what sorts it, not the node type.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import somatize._somatize as ext

STUB = Path(ext.__file__).with_name("_somatize.pyi")

# Every class has these by construction; the stub is not expected to redeclare
# them.
UNIVERSAL = {
    "__doc__",
    "__module__",
    "__weakref__",
    "__dict__",
    "__getattribute__",
    "__delattr__",
    "__setattr__",
    "__dir__",
    "__reduce__",
    "__reduce_ex__",
    "__sizeof__",
    "__format__",
    "__init_subclass__",
    "__subclasshook__",
    "__class_getitem__",
    "__new__",
    "__init__",
}

# What PyO3 fills in for free the moment a class defines `__eq__`: it writes the
# whole `tp_richcompare` slot, so all six names appear in `vars(cls)` and the
# four orderings answer `NotImplemented` — `Space() < Space()` is a `TypeError`.
# They are not surface and the stub does not declare them.
#
# The price, said plainly: if a class ever *does* define an ordering by hand,
# this stops noticing that the stub left it out. Nothing in the extension orders
# anything today, and the day one does, the name goes on the class that has it
# rather than into this set.
DERIVED_FROM_EQ = {"__ne__", "__lt__", "__le__", "__gt__", "__ge__"}


# ── Reading the stub ─────────────────────────────────────────────────────────


def _tree() -> ast.Module:
    return ast.parse(STUB.read_text(), filename=str(STUB))


def _is_property(node: ast.FunctionDef) -> bool:
    return any(
        isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
    )


def _params(node: ast.FunctionDef) -> list[tuple[str, bool]]:
    """`(name, has_default)` per parameter, in order, without self/cls."""
    a = node.args
    positional = a.posonlyargs + a.args
    n_defaults = len(a.defaults)
    out: list[tuple[str, bool]] = [
        (p.arg, i >= len(positional) - n_defaults) for i, p in enumerate(positional)
    ]
    if out and out[0][0] in ("self", "cls"):
        out = out[1:]
    if a.vararg:
        out.append(("*" + a.vararg.arg, False))
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        out.append((p.arg, d is not None))
    if a.kwarg:
        out.append(("**" + a.kwarg.arg, False))
    return out


def _runtime_params(sig: str) -> list[tuple[str, bool]]:
    """The same shape, parsed out of a PyO3 `__text_signature__`."""
    src = "def _f" + sig.replace("$self", "self").replace("$type", "cls") + ": ..."
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    return _params(fn)


def _stub_classes() -> dict[str, ast.ClassDef]:
    return {
        n.name: n
        for n in _tree().body
        if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
    }


def _stub_functions() -> dict[str, ast.FunctionDef]:
    return {
        n.name: n
        for n in _tree().body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    }


def _stub_methods(node: ast.ClassDef) -> set[str]:
    return {
        n.name
        for n in node.body
        if isinstance(n, ast.FunctionDef)
        if not _is_property(n)
        if n.name not in UNIVERSAL
    }


def _stub_attributes(node: ast.ClassDef) -> set[str]:
    """`@property def` in the stub is a `#[getter]` in the extension."""
    return {n.name for n in node.body if isinstance(n, ast.FunctionDef) if _is_property(n)}


def _stub_method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for n in node.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


# ── Reading the built module ─────────────────────────────────────────────────


def _runtime_classes() -> dict[str, type]:
    return {
        n: o
        for n in dir(ext)
        if not n.startswith("_") and inspect.isclass(o := getattr(ext, n))
    }


def _runtime_functions() -> dict[str, object]:
    return {
        n: o
        for n in dir(ext)
        if not n.startswith("_")
        if not inspect.isclass(o := getattr(ext, n))
        if callable(o)
    }


def _own_methods(cls: type) -> set[str]:
    """What the class itself defines, not what it inherits.

    Dunders count when the class defines one: `__len__` on a graph and
    `__getitem__` on a point are surface somebody uses, and a checker needs them
    declared before `len(g)` or `p["lr"]` type-checks.
    """
    derived = DERIVED_FROM_EQ if "__eq__" in vars(cls) else set()
    return {
        n
        for n, v in vars(cls).items()
        if callable(v) or isinstance(v, (staticmethod, classmethod))
        if n not in UNIVERSAL and n not in derived
    }


def _own_attributes(cls: type) -> set[str]:
    """Getters. Not callable, so `_own_methods` does not see them."""
    return {
        n
        for n, v in vars(cls).items()
        if not callable(v)
        if not isinstance(v, (staticmethod, classmethod))
        if not (n.startswith("__") and n.endswith("__"))
    }


# ── What has to hold ─────────────────────────────────────────────────────────


def test_the_stub_is_there_and_the_package_says_it_is_typed():
    assert STUB.is_file(), f"no stub beside the extension: {STUB}"
    marker = STUB.with_name("py.typed")
    assert marker.is_file(), "py.typed is missing, so nothing downstream reads the stub"


def test_every_class_in_the_module_is_in_the_stub():
    assert set(_stub_classes()) == set(_runtime_classes())


def test_every_free_function_in_the_module_is_in_the_stub():
    assert set(_stub_functions()) == set(_runtime_functions())


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_class_declares_the_methods_it_has(name):
    stub = _stub_classes().get(name)
    assert stub is not None, f"{name} is not in the stub"
    assert _stub_methods(stub) == _own_methods(_runtime_classes()[name]), (
        f"{name}: the stub and the built module disagree about which methods exist"
    )


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_class_declares_the_attributes_it_has(name):
    """A getter is surface too — `bound.digest` as much as `store.get()`."""
    stub = _stub_classes()[name]
    assert _stub_attributes(stub) == _own_attributes(_runtime_classes()[name]), (
        f"{name}: the stub and the built module disagree about which attributes exist"
    )


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_class_is_constructible_in_the_stub_only_if_it_is_constructible(name):
    """`Bound`, `Frame` and `Point` have no `#[new]`.

    They arrive as answers — a bound name out of a store, a frame out of a
    source, a point out of a sampler — and calling them raises `TypeError: No
    constructor defined`. So the stub declares no `__init__` for them.

    What this does *not* buy is worth saying: leaving `__init__` out does not
    make a checker reject `Point()`. It falls back to `object.__init__`, which
    takes no arguments, so the no-argument call still passes. What it does buy is
    that the stub never advertises a constructor the extension does not have.
    """
    built = _runtime_classes()[name]
    stub = _stub_classes()[name]
    constructible = getattr(built, "__text_signature__", None) is not None
    declared = _stub_method(stub, "__init__") is not None
    assert declared == constructible, (
        f"{name}: the extension {'has' if constructible else 'has no'} constructor "
        f"and the stub {'declares' if declared else 'declares'} one"
    )


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_constructor_takes_the_parameters_the_stub_says(name):
    sig = getattr(_runtime_classes()[name], "__text_signature__", None)
    if sig is None:
        pytest.skip(f"{name} is not constructible")
    stub = _stub_method(_stub_classes()[name], "__init__")
    assert stub is not None
    assert _params(stub) == _runtime_params("(self, " + sig.lstrip("(")), (
        f"{name}.__init__: parameters differ from the built extension"
    )


def _methods_with_signatures() -> list[tuple[str, str]]:
    out = []
    for cls_name, cls in _runtime_classes().items():
        for m_name, v in vars(cls).items():
            if m_name in UNIVERSAL or m_name.startswith("__"):
                continue
            if getattr(v, "__text_signature__", None) is not None:
                out.append((cls_name, m_name))
    return sorted(out)


@pytest.mark.parametrize("cls_name,m_name", _methods_with_signatures())
def test_a_method_takes_the_parameters_the_stub_says(cls_name, m_name):
    stub = _stub_method(_stub_classes()[cls_name], m_name)
    assert stub is not None, f"{cls_name}.{m_name} is not in the stub"
    sig = getattr(getattr(_runtime_classes()[cls_name], m_name), "__text_signature__")
    assert _params(stub) == _runtime_params(sig), (
        f"{cls_name}.{m_name}: parameters differ from the built extension"
    )


@pytest.mark.parametrize("name", sorted(_runtime_functions()))
def test_a_free_function_takes_the_parameters_the_stub_says(name):
    sig = getattr(_runtime_functions()[name], "__text_signature__", None)
    if sig is None:
        pytest.skip(f"{name} carries no signature")
    stub = _stub_functions()[name]
    assert _params(stub) == _runtime_params(sig), (
        f"{name}: parameters differ from the built extension"
    )
