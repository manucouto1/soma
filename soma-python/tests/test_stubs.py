"""The stub describes the module that is actually built.

A hand-written `.pyi` is a promise about a binary it cannot see. The promise
rots the first time someone adds a `#[pymethods]` and does not think about
typing, and nothing complains: the stub keeps type-checking, it just stops
describing reality. A stub that lies is worse than no stub, because the
error it produces is a confident wrong answer instead of `Any`.

So the stub is checked against the compiled module, here. These tests do not
verify that the *types* are right — no test can, they are a claim about
meaning. They verify that the stub and the extension have the same shape:
the same classes, the same methods on each, the same parameter names in the
same order, and the same parameters carrying defaults.

That covers the drift that happens by accident. What is left — a wrong type
on a parameter that exists — is the part a human has to get right, and it is
the part a reviewer can actually see in a diff.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import soma._soma as _soma

STUB = Path(_soma.__file__).with_name("_soma.pyi")

# Inherited from the exception base or from `object`; the stub is not
# expected to redeclare them.
_INHERITED = {"add_note", "with_traceback", "__init__", "__new__"}


def _stub_tree() -> ast.Module:
    return ast.parse(STUB.read_text(), filename=str(STUB))


def _params(node: ast.FunctionDef) -> list[tuple[str, bool]]:
    """(name, has_default) per parameter, in order, minus self/cls."""
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
    """Same shape, parsed out of a PyO3 `__text_signature__`."""
    src = "def _f" + sig.replace("$self", "self").replace("$type", "cls") + ": ..."
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)
    return _params(fn)


def _stub_classes() -> dict[str, ast.ClassDef]:
    """Public classes. A leading underscore means stub-only (a Protocol)."""
    return {
        n.name: n
        for n in _stub_tree().body
        if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
    }


def _stub_functions() -> dict[str, ast.FunctionDef]:
    return {
        n.name: n
        for n in _stub_tree().body
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    }


def _runtime_classes() -> dict[str, type]:
    return {
        n: o
        for n in dir(_soma)
        if not n.startswith("_") and inspect.isclass(o := getattr(_soma, n))
    }


def _runtime_functions() -> dict[str, object]:
    return {
        n: o
        for n in dir(_soma)
        if not n.startswith("_")
        and not inspect.isclass(o := getattr(_soma, n))
        and callable(o)
    }


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _own_methods(cls: type) -> set[str]:
    """What the class itself defines, not what it inherits.

    Dunders are included when the class defines one itself: `__len__` on a
    graph and `__getitem__` on a trial are surface a caller uses, and a
    checker needs them declared before `len(g)` or `t["lr"]` type-checks.
    The ones every class has by construction are excluded.
    """
    return {
        n
        for n, v in vars(cls).items()
        if callable(v) or isinstance(v, (staticmethod, classmethod))
        if n not in _INHERITED and n not in _UNIVERSAL_DUNDERS
    }


def _own_attributes(cls: type) -> set[str]:
    """Getters. Not callable, so `_own_methods` does not see them."""
    return {
        n
        for n, v in vars(cls).items()
        if not callable(v)
        if not isinstance(v, (staticmethod, classmethod))
        if not _is_dunder(n)
    }


_UNIVERSAL_DUNDERS = {
    "__doc__",
    "__module__",
    "__weakref__",
    "__dict__",
    "__hash__",
    "__getattribute__",
    "__dir__",
    "__reduce__",
    "__reduce_ex__",
    "__sizeof__",
    "__format__",
    "__init_subclass__",
    "__subclasshook__",
    "__class_getitem__",
}


def _stub_methods(node: ast.ClassDef) -> set[str]:
    return {
        n.name
        for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        if n.name not in _INHERITED and n.name not in _UNIVERSAL_DUNDERS
        # An implementation stub under `@overload` forms declare the same name.
    }


def _stub_attributes(node: ast.ClassDef) -> set[str]:
    return {
        t.target.id
        for t in node.body
        if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
    }


def test_the_stub_exists_and_the_package_claims_to_be_typed():
    assert STUB.is_file(), f"no stub next to the extension: {STUB}"
    marker = STUB.with_name("py.typed")
    assert marker.is_file(), "py.typed is missing, so nothing downstream reads the stub"


def test_every_class_in_the_module_is_in_the_stub():
    assert set(_stub_classes()) == set(_runtime_classes())


def test_every_free_function_in_the_module_is_in_the_stub():
    assert set(_stub_functions()) == set(_runtime_functions())


def test_the_module_exports_what_the_stub_declares():
    """`__all__` is maintained by hand in Rust and drifts the same way."""
    exported = set(_soma.__all__) - {"__version__"}
    assert exported == set(_stub_classes()) | set(_stub_functions())


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_class_declares_the_methods_it_actually_has(name):
    stub = _stub_classes().get(name)
    assert stub is not None, f"{name} is not in the stub"
    assert _stub_methods(stub) == _own_methods(_runtime_classes()[name]), (
        f"{name}: the stub and the built module disagree about which methods exist"
    )


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_class_declares_the_attributes_it_actually_has(name):
    """Getters are surface too — `study.best_trial` as much as `study.run()`."""
    stub = _stub_classes()[name]
    assert _stub_attributes(stub) == _own_attributes(_runtime_classes()[name]), (
        f"{name}: the stub and the built module disagree about which attributes exist"
    )


@pytest.mark.parametrize("name", sorted(_runtime_classes()))
def test_a_class_is_constructible_in_the_stub_only_if_it_is_constructible(name):
    """`Trial`, `Run` and `StepCtx` have no `#[new]`.

    They arrive as arguments — a trial from a study's executor, a step
    context from a poll — and calling them raises `TypeError: No constructor
    defined`. So the stub declares no constructor for them, and this pins
    both halves: that they really are not constructible, and that the stub
    does not describe a constructor they do not have.

    Note what this does *not* buy. Omitting `__init__` does not make a
    checker reject `Trial()`: it falls back to `object.__init__`, which
    takes no arguments, so the no-argument call still passes. What it does
    prevent is a stub inventing parameters — `Trial({"lr": 0.1})` is
    rejected, and that is the call someone would plausibly write.
    """
    cls = _runtime_classes()[name]
    if issubclass(cls, BaseException):
        pytest.skip("an exception is constructible through its base, with no signature")
    declared = {
        n.name
        for n in _stub_classes()[name].body
        if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "__new__")
    }
    if cls.__text_signature__ is None:
        with pytest.raises(TypeError):
            cls()
        assert not declared, (
            f"{name} has no constructor, but the stub declares {sorted(declared)}"
        )
    else:
        assert declared, f"{name} is constructible but the stub declares no constructor"


@pytest.mark.parametrize(
    "name",
    sorted(n for n, c in _runtime_classes().items() if c.__text_signature__),
)
def test_a_constructor_in_the_stub_matches_the_one_that_was_compiled(name):
    """PyO3 puts the `#[new]` signature on the type, not on `__new__`."""
    node = next(
        n
        for n in _stub_classes()[name].body
        if isinstance(n, ast.FunctionDef) and n.name in ("__init__", "__new__")
    )
    expected = _runtime_params(_runtime_classes()[name].__text_signature__)
    assert _params(node) == expected, (
        f"{name}: stub says {_params(node)}, the built module says {expected}"
    )


def _signature_cases():
    """Every callable the extension exposes a real signature for.

    PyO3 emits `__text_signature__` for `#[pymethods]`, so almost everything
    is covered. It does *not* emit one for `#[new]` unless the constructor
    carries an explicit `#[pyo3(signature = ...)]` — which is why the
    constructors in this crate carry one.
    """
    for cname, cls in _runtime_classes().items():
        for mname, member in vars(cls).items():
            fn = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
            sig = getattr(fn, "__text_signature__", None)
            # `__new__` reports a useless `(*args, **kwargs)` — PyO3 puts the
            # real constructor signature on the type, which has its own test.
            if sig and not _is_dunder(mname):
                yield f"{cname}.{mname}", cname, mname, sig
    for fname, fn in _runtime_functions().items():
        if sig := getattr(fn, "__text_signature__", None):
            yield fname, None, fname, sig


@pytest.mark.parametrize(
    "case_id,cname,mname,sig",
    list(_signature_cases()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_signature_in_the_stub_matches_the_one_that_was_compiled(case_id, cname, mname, sig):
    if cname is None:
        node = _stub_functions().get(mname)
    else:
        node = next(
            (
                n
                for n in _stub_classes()[cname].body
                if isinstance(n, ast.FunctionDef) and n.name == mname
            ),
            None,
        )
    assert node is not None, f"{case_id} is in the module but not in the stub"

    # `@overload` splits one runtime signature into several narrower ones,
    # so there is nothing to compare parameter-by-parameter: PyO3 reports the
    # flattened `(*args, target=None)` while the stub says `(filter, /)` and
    # `(id, filter, /)`. Only two members are overloaded (`Graph.node` and
    # `tool`), both because they dispatch on arity in Rust; the name check
    # above still covers them.
    if any(
        isinstance(d, ast.Name) and d.id == "overload"
        for d in getattr(node, "decorator_list", [])
    ):
        pytest.skip("overloaded: no single signature to compare against")

    actual = _params(node)
    assert actual == _runtime_params(sig), (
        f"{case_id}: stub says {actual}, the built module says {_runtime_params(sig)}"
    )
