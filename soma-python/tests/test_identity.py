"""Deterministic filter identity (Phase 2): keys stable across
processes, sensitive to code changes, with the _cache_version override
and typed errors for unhashable configs."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import soma
from soma import CacheConfigError, Filter, Graph, search
from soma._identity import canonical_config_json, code_fingerprint, filter_identity


FILTER_MODULE = textwrap.dedent(
    """
    from soma import Filter, search

    class Scaler(Filter):
        factor = search(0.5, 2.0, default=1.5)

        def fit(self, x, y=None):
            return {"m": sum(x) / len(x)}

        def forward(self, x, state):
            return [(v - state["m"]) * self.factor for v in x]
    """
)


def _hash_in_subprocess(tmp_path, extra=""):
    """Build the filter in a fresh interpreter and print its node key."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mod = tmp_path / "filters_mod.py"
    mod.write_text(FILTER_MODULE + extra)
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            from filters_mod import Scaler
            from soma._identity import filter_identity
            ident = filter_identity(Scaler())
            print(ident["config_json"])
            print(ident["code_fp"])
            """
        )
    )
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ),
    )
    assert out.returncode == 0, out.stderr
    config_json, code_fp = out.stdout.strip().splitlines()
    return config_json, code_fp


def test_identity_is_stable_across_processes(tmp_path):
    a = _hash_in_subprocess(tmp_path / "run1")
    b = _hash_in_subprocess(tmp_path / "run2")
    assert a == b


def test_editing_source_changes_fingerprint(tmp_path):
    _, fp_before = _hash_in_subprocess(tmp_path / "before")
    # A behavioral edit inside the class body (via a subclass-free module
    # rewrite): change the forward body.
    edited = FILTER_MODULE.replace('(v - state["m"])', '(v + state["m"])')
    mod_dir = tmp_path / "after"
    mod_dir.mkdir()
    (mod_dir / "filters_mod.py").write_text(edited)
    script = mod_dir / "probe.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            from filters_mod import Scaler
            from soma._identity import filter_identity
            print(filter_identity(Scaler())["code_fp"])
            """
        )
    )
    out = subprocess.run(
        [sys.executable, str(script), str(mod_dir)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    fp_after = out.stdout.strip()
    assert fp_after != fp_before, "editing forward() must change the code fingerprint"


def test_cache_version_overrides_source_hash():
    class Pinned(Filter):
        _cache_version = "v7"

        def forward(self, x, state):
            return x

    method, digest = code_fingerprint(Pinned)
    assert method == "cache_version"
    assert digest == "v7"


def test_search_defaults_enter_the_config():
    class WithDefault(Filter):
        lr = search(0.001, 0.1, default=0.01)

        def forward(self, x, state):
            return x

    unset = WithDefault()
    explicit = WithDefault(lr=0.01)
    assert canonical_config_json(unset) == canonical_config_json(explicit)

    different = WithDefault(lr=0.05)
    assert canonical_config_json(unset) != canonical_config_json(different)


def test_qualified_name_disambiguates():
    class Model(Filter):
        def forward(self, x, state):
            return x

    ident = filter_identity(Model())
    assert ident["qualname"].endswith("test_qualified_name_disambiguates.<locals>.Model")
    assert "." in ident["qualname"]


def test_unhashable_attribute_raises_typed_error():
    class BadFilter(Filter):
        def __init__(self):
            super().__init__()
            self.handle = open(os.devnull)  # noqa: SIM115 — deliberately unserializable

        def forward(self, x, state):
            return x

    f = BadFilter()
    with pytest.raises(CacheConfigError, match="handle"):
        canonical_config_json(f)
    f.handle.close()


def test_soma_config_hook_wins():
    class Custom(Filter):
        def __init__(self):
            super().__init__()
            self._engine = object()  # private, excluded anyway
            self.threshold = 0.5

        def __soma_config__(self):
            return {"threshold": self.threshold, "profile": "fast"}

        def forward(self, x, state):
            return x

    cfg = canonical_config_json(Custom())
    assert '"profile":"fast"' in cfg
    assert '"threshold":0.5' in cfg


def test_graph_node_uses_new_identity(tmp_path, monkeypatch):
    """End-to-end: two graphs in one process, same filter class and
    config → same cached fit (counter file written once)."""
    counters = tmp_path / "counters.txt"

    class Probe(Filter):
        _counters = str(counters)

        def fit(self, x, y=None):
            with open(self._counters, "a") as fh:
                fh.write("fit\n")
            return {"n": len(x)}

        def forward(self, x, state):
            return x

    # monkeypatch restores conftest's session-scoped SOMA_CACHE_DIR on
    # teardown — a plain os.environ.pop would leak the developer's real
    # ~/.soma/cache into every later test.
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))
    for _ in range(2):
        g = Graph()
        g.node("probe", Probe())
        g.fit([1.0, 2.0, 3.0])

    assert counters.read_text().count("fit") == 1
