"""Tests for the cloudpickle by-value registration walker.

Regression tests for the bug where:
  - Filter classes defined in importable user submodules (e.g. `src.filters.baselines`)
    were pickled by-reference, failing on workers that couldn't import the module.
  - The registration heuristic (substring 'lib/python' in __file__) incorrectly
    included stdlib modules on Fedora/RHEL (which live under /usr/lib64/python3.13/),
    causing cloudpickle recursion on stdlib modules with circular internal refs.

The fix uses sys.stdlib_module_names (3.10+) + sys.builtin_module_names +
sysconfig.get_paths() to robustly detect stdlib and site-packages across all
platforms (Debian, Fedora/RHEL, conda, virtualenvs, Windows).
"""
import os
import sys
import textwrap
import importlib
import pickle

import pytest
import cloudpickle

from soma import Graph


def _by_value_registry():
    """Return the set of module names currently registered for by-value pickling."""
    # cloudpickle exposes the registry at module level; handle both private API shapes.
    import cloudpickle.cloudpickle as _cp
    reg = getattr(_cp, "_PICKLE_BY_VALUE_MODULES", None)
    if reg is None:
        # Older/newer cloudpickle may store it differently; fall back to empty set.
        reg = set()
    return set(reg)


@pytest.fixture
def user_filter_pkg(tmp_path, monkeypatch):
    """Create an importable user package outside site-packages with a Filter subclass.

    Simulates the real failure mode: project-local `src/filters/baselines.py` style.
    """
    pkg_root = tmp_path / "myproj"
    (pkg_root / "src" / "filters").mkdir(parents=True)
    (pkg_root / "src" / "__init__.py").write_text("")
    (pkg_root / "src" / "filters" / "__init__.py").write_text("")
    (pkg_root / "src" / "filters" / "baselines.py").write_text(textwrap.dedent("""
        from soma import Filter

        class BaselineFilter(Filter):
            _kind = "stateless"

            def __init__(self, factor=2.0):
                super().__init__(factor=factor)

            def forward(self, x, state):
                return [v * self.factor for v in x]
    """))
    monkeypatch.syspath_prepend(str(pkg_root))
    # Ensure a fresh import (previous tests may have imported a stale version)
    for mod_name in list(sys.modules):
        if mod_name.startswith("src."):
            del sys.modules[mod_name]
    yield pkg_root


def test_user_submodule_is_registered_by_value(user_filter_pkg):
    """Filter in an importable user submodule must be registered for by-value pickling."""
    mod = importlib.import_module("src.filters.baselines")
    flt = mod.BaselineFilter(factor=3.0)

    g = Graph()
    g.node(flt)  # triggers PyFilterBridge::new → walker

    reg = _by_value_registry()
    assert "src.filters.baselines" in reg, (
        f"User submodule not registered by-value. Registry: {sorted(reg)}"
    )


def test_stdlib_is_not_registered_by_value(user_filter_pkg):
    """Walker must NOT register stdlib modules (would cause cloudpickle recursion)."""
    mod = importlib.import_module("src.filters.baselines")
    flt = mod.BaselineFilter()

    g = Graph()
    g.node(flt)

    reg = _by_value_registry()
    # These are the stdlib modules that recurse under cloudpickle on Fedora/RHEL:
    # their __file__ paths don't contain 'lib/python' (contain 'lib64/python'),
    # and they hold circular internal refs.
    forbidden = {
        "io", "re", "types", "abc", "copyreg", "enum", "base64", "struct",
        "importlib", "importlib._bootstrap", "importlib._bootstrap_external",
        "os", "sys", "builtins", "collections", "functools", "itertools",
    }
    leaked = reg & forbidden
    assert not leaked, (
        f"Stdlib modules leaked into by-value registry: {sorted(leaked)}. "
        f"This would cause cloudpickle recursion on Fedora/RHEL-like systems."
    )


def test_soma_is_not_registered_by_value(user_filter_pkg):
    """Soma itself is installed on the worker; must not be pickled by-value."""
    mod = importlib.import_module("src.filters.baselines")
    flt = mod.BaselineFilter()

    g = Graph()
    g.node(flt)

    reg = _by_value_registry()
    soma_leaked = {name for name in reg if name == "soma" or name.startswith("soma.")}
    assert not soma_leaked, (
        f"Soma modules leaked into by-value registry: {sorted(soma_leaked)}"
    )


def test_roundtrip_does_not_recurse(user_filter_pkg):
    """After registration, cloudpickle.dumps → pickle.loads must complete without recursion."""
    mod = importlib.import_module("src.filters.baselines")
    flt = mod.BaselineFilter(factor=5.0)

    g = Graph()
    g.node(flt)  # registers modules

    # Full round-trip — this is what the worker does over the wire.
    blob = cloudpickle.dumps(flt)
    restored = pickle.loads(blob)
    assert restored.factor == 5.0
    assert restored.forward([1, 2, 3], None) == [5.0, 10.0, 15.0]


def test_sysconfig_site_packages_detection(user_filter_pkg):
    """site-packages must be detected via sysconfig (not just substring match).

    Guards against regressions on conda / virtualenvs / Windows where the
    substring 'site-packages' might still appear but sysconfig is authoritative.
    """
    import sysconfig, site
    # Pick any installed package (numpy is a dev-dep; fall back to cloudpickle)
    for pkg_name in ("numpy", "cloudpickle"):
        try:
            installed = importlib.import_module(pkg_name)
            break
        except ImportError:
            continue
    else:
        pytest.skip("no installed package available for site-packages test")

    installed_file = getattr(installed, "__file__", None)
    assert installed_file is not None
    site_dirs = {sysconfig.get_paths().get("purelib", ""),
                 sysconfig.get_paths().get("platlib", "")}
    user_site = getattr(site, "getusersitepackages", lambda: None)()
    if user_site:
        site_dirs.add(user_site)
    site_prefixes = tuple(
        os.path.realpath(p) + os.sep for p in site_dirs if p
    )
    real = os.path.realpath(installed_file)
    assert real.startswith(site_prefixes), (
        f"sysconfig/site did not resolve {pkg_name} under any site-packages dir. "
        f"dirs={site_dirs} file={real}"
    )

    # And after a Graph.node() the installed package must NOT be registered by-value
    mod = importlib.import_module("src.filters.baselines")
    flt = mod.BaselineFilter()
    g = Graph()
    g.node(flt)

    reg = _by_value_registry()
    assert pkg_name not in reg, (
        f"Installed package {pkg_name} leaked into by-value registry: {sorted(reg)}"
    )
