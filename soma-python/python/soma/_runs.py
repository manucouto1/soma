"""Read-side API over run directories (``.soma/runs/<run_id>/``).

``soma.runs()`` lists every tracked run; ``soma.RunView`` reads one run's
logs and aggregates them into chart-ready data. Readers never write —
they consume the files ``graph.track_run(...)`` / ``Study.run(...)``
produce, so they work on live, finished, and crashed runs alike::

    for run in soma.runs():
        print(run.id, run.state, run.name)

    run = soma.RunView(".soma/runs/train_20260729T101500_ab12")
    run.metric_series("val_f1")   # [{ts, name, value, step, ...}, ...]
    run.node_timings()            # per-node spans (gantt substrate)
    run.cache_activity()          # {"hits": ..., "misses": ..., "by_node": ...}

Wall-clock times come from the ``ts`` the event sink stamps at emit
time; a ``running`` run whose heartbeat went stale is reported as
``crashed``.
"""

from __future__ import annotations

import json

from soma import _soma


class RunView:
    """Reader over one run directory."""

    def __init__(self, path: str, _info: dict | None = None):
        self._dir = str(path)
        self._info = _info or json.loads(_soma.run_info_json(self._dir))

    # ── identity ──

    @property
    def dir(self) -> str:
        """Absolute path of the run directory."""
        return self._dir

    @property
    def id(self) -> str:
        return self._info["run_id"]

    @property
    def name(self) -> str:
        return self._info["name"]

    @property
    def kind(self) -> str:
        return self._info["kind"]

    @property
    def state(self) -> str:
        """``running`` | ``completed`` | ``failed`` | ``crashed``."""
        return self._info["state"]

    @property
    def info(self) -> dict:
        """Listing entry: id, kind, name, state, created_at, duration_ms, tags."""
        return dict(self._info)

    def refresh(self) -> "RunView":
        """Re-read status (state/heartbeat) for a live run."""
        self._info = json.loads(_soma.run_info_json(self._dir))
        return self

    # ── raw logs ──

    def manifest(self) -> dict:
        """The run's ``manifest.json`` (environment, git, graph summary)."""
        return json.loads(_soma.run_manifest_json(self._dir))

    def events(self) -> list[dict]:
        """All parseable event envelopes (``{seq, ts, event_type, ...}``),
        in log order. Torn or unknown lines are skipped; gaps in ``seq``
        reveal the skips."""
        return json.loads(_soma.run_events_json(self._dir))

    # ── aggregations (chart-ready) ──

    def metric_series(self, name: str | None = None) -> list[dict]:
        """Metric points ``{ts, name, value, step, trial_id, node_id}``,
        optionally filtered by metric name."""
        return json.loads(_soma.run_metric_series_json(self._dir, name=name))

    def node_timings(self) -> list[dict]:
        """Per-node execution spans ``{node_id, started_ts, finished_ts,
        duration_ms, outcome, cache_tier, error}`` in event order."""
        return json.loads(_soma.run_node_timings_json(self._dir))

    def cache_activity(self) -> dict:
        """Cache hit/miss counts: ``{hits, misses, by_node}``."""
        return json.loads(_soma.run_cache_activity_json(self._dir))

    def health_flags(self) -> list[dict]:
        """HealthFlag events ``{ts, node_id, step, flag, detail}``."""
        return json.loads(_soma.run_health_flags_json(self._dir))

    def trial_timeline(self) -> list[dict]:
        """Trial lifetimes from ``study.json`` (empty for non-study runs)."""
        return json.loads(_soma.run_trial_timeline_json(self._dir))

    # ── architecture rendering ──

    def _module_trees(self) -> list[dict]:
        """Inner-architecture snapshots written by ``gradient_audit
        (inside=...)`` — one ``diagnostics/modules/<node>.json`` per
        scoped node: ``{node, graph, order, params, ids, mermaid_ids}``.
        Identified by the ``node`` field, never the filename."""
        import pathlib

        root = pathlib.Path(self._dir) / "diagnostics" / "modules"
        if not root.is_dir():
            return []
        trees = []
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except ValueError:
                continue
            if isinstance(data, dict) and "node" in data:
                trees.append(data)
        return trees

    def overlay(self) -> dict:
        """Per-node rendering annotations aggregated from this run's
        events: status, total duration, cache tier, health flags. Feed
        it to ``graph.to_mermaid(overlay=...)`` or use ``to_mermaid()``
        directly."""
        return json.loads(_soma.run_overlay_json(self._dir))

    def to_mermaid(self, overlay: bool = True, node: str | None = None) -> str:
        """Mermaid diagram of the graph this run executed, annotated
        with per-node timing/cache/health (``overlay=False`` for the
        plain topology).

        ``node="encoder"`` renders that node's *inner* architecture
        instead — the submodule chain snapshotted by
        ``gradient_audit(inside=...)``, annotated with parameter counts,
        mean output-gradient norms, and per-submodule health flags."""
        if node is None:
            return _soma.run_to_mermaid(self._dir, overlay=overlay)

        tree = next((t for t in self._module_trees() if t["node"] == node), None)
        if tree is None:
            available = sorted(t["node"] for t in self._module_trees())
            raise ValueError(
                f"run {self.id} has no inner-architecture snapshot for "
                f"{node!r} (scoped nodes: {available}) — audit it with "
                f"gradient_audit(inside=...) under track_run"
            )
        graph_json = json.dumps(tree["graph"])
        if not overlay:
            return _soma.graph_json_to_mermaid(graph_json)
        return _soma.graph_json_to_mermaid(
            graph_json, overlay=self._inner_overlay(tree)
        )

    def _inner_overlay(self, tree: dict) -> dict:
        """Per-submodule overlay for one inner-architecture tree:
        health flags (from HealthFlag events on hierarchical ids) plus
        a `param count · mean |out-grad|` label line (from
        diagnostics/report.json)."""
        import pathlib

        id_by_path = tree["ids"]
        mermaid_by_path = tree["mermaid_ids"]

        report_metrics: dict[str, dict] = {}
        report_path = pathlib.Path(self._dir) / "diagnostics" / "report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                report_metrics = {
                    f["filter"]: f.get("metrics", {})
                    for f in report.get("filters", [])
                }
            except ValueError:
                pass

        flags_by_id: dict[str, list[str]] = {}
        for f in self.health_flags():
            flags_by_id.setdefault(f["node_id"], [])
            family = f["flag"].split("(")[0]
            if family not in flags_by_id[f["node_id"]]:
                flags_by_id[f["node_id"]].append(family)

        def _fmt_params(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1e6:.1f}M θ"
            if n >= 1_000:
                return f"{n / 1e3:.1f}k θ"
            return f"{n} θ"

        nodes: dict[str, dict] = {}
        for path, child_id in id_by_path.items():
            parts = []
            n_params = tree.get("params", {}).get(path)
            if n_params:
                parts.append(_fmt_params(n_params))
            grad = report_metrics.get(child_id, {}).get("out_grad_norm")
            if isinstance(grad, (int, float)) and grad == grad:
                parts.append(f"|∂| {grad:.2e}")
            entry: dict = {}
            if parts:
                entry["sublabel"] = " · ".join(parts)
            flags = flags_by_id.get(child_id)
            if flags:
                entry["flags"] = flags
            if entry:
                nodes[mermaid_by_path[path]] = entry
        return {"nodes": nodes}

    def to_graphviz(self, overlay: bool = True) -> str:
        """Graphviz DOT of the graph this run executed, annotated with
        per-node timing/cache/health."""
        return _soma.run_to_graphviz(self._dir, overlay=overlay)

    def __repr__(self) -> str:
        return f"RunView({self.id!r}, state={self.state!r}, name={self.name!r})"


def runs(root: str = ".soma") -> list[RunView]:
    """All runs under ``<root>/runs/``, newest first. Directories
    without a readable manifest are skipped."""
    infos = json.loads(_soma.list_runs_json(root=root))
    return [RunView(info["dir"], _info=info) for info in infos]
