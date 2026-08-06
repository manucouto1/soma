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

from typing import Any

import json

from soma import _soma


class RunView:
    """Reader over one run directory."""

    def __init__(self, path: str, _info: dict | None = None):
        self._dir = str(path)
        self._info = _info or self._section("info")

    def _section(self, name: str, **kwargs):
        """One section of the run, read through the one reader.

        Every accessor below used to have an FFI function of its own, and
        each of those opened its own reader and re-parsed
        ``events.jsonl``. They are sections of one object now, so reading
        three of them is one call and one parse."""
        return json.loads(_soma.run_sections_json(self._dir, [name], **kwargs))[name]

    def sections(self, *names: str, metric: str | None = None) -> dict:
        """Several sections at once, in one call.

        ``view.sections("node_timings", "cache_activity", "health_flags")``
        is what a report or a dashboard wants: the same parse serves all
        of them. The individual accessors are the one-section case."""
        return json.loads(
            _soma.run_sections_json(self._dir, list(names), metric=metric)
        )

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
        self._info = self._section("info")
        return self

    # ── raw logs ──

    def manifest(self) -> dict:
        """The run's ``manifest.json`` (environment, git, graph summary)."""
        return self._section("manifest")

    def events(self) -> list[dict]:
        """All parseable event envelopes (``{seq, ts, event_type, ...}``),
        in log order. Torn or unknown lines are skipped; gaps in ``seq``
        reveal the skips."""
        return self._section("events")

    # ── aggregations (chart-ready) ──

    def metric_series(self, name: str | None = None) -> list[dict]:
        """Metric points ``{ts, name, value, step, trial_id, node_id}``,
        optionally filtered by metric name."""
        return self._section("metric_series", metric=name)

    def node_timings(self) -> list[dict]:
        """Per-node execution spans ``{node_id, started_ts, finished_ts,
        duration_ms, outcome, cache_tier, error}`` in event order."""
        return self._section("node_timings")

    def cache_activity(self) -> dict:
        """Cache hit/miss counts: ``{hits, misses, by_node}``."""
        return self._section("cache_activity")

    def health_flags(self) -> list[dict]:
        """HealthFlag events ``{ts, node_id, step, flag, detail}``."""
        return self._section("health_flags")

    def trial_timeline(self) -> list[dict]:
        """Trial lifetimes from ``study.json`` (empty for non-study runs)."""
        return self._section("trial_timeline")

    def agentic_activity(self) -> dict:
        """Agent-step activity: run totals (``turns``, ``input_tokens``,
        ``output_tokens``, ``effects``, ``replayed``, ``tool_calls``,
        ``steps_completed``, ``steps_failed``, ``suspensions``) plus
        ``by_node`` with the same breakdown per step node. Empty
        ``by_node`` means the run had no agent steps."""
        return self._section("agentic_activity")

    def agentic_timeline(self) -> list[dict]:
        """Per-effect execution spans ``{node_id, turn, effect,
        started_ts, finished_ts, duration_ms, replayed, is_error,
        outcome}`` in event order — the gantt substrate for agent runs.
        An unclosed span means the run died mid-effect."""
        return self._section("agentic_timeline")

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
        return self._section("overlay")

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

    def to_svg(self, overlay: bool = True, node: str | None = None) -> str:
        """Self-contained SVG diagram (no JavaScript — displays inline
        anywhere). Same semantics as ``to_mermaid``: the run's graph
        with its overlay, or one node's inner architecture with
        ``node="encoder"``."""
        if node is None:
            return _soma.run_to_svg(self._dir, overlay=overlay)
        tree = next((t for t in self._module_trees() if t["node"] == node), None)
        if tree is None:
            available = sorted(t["node"] for t in self._module_trees())
            raise ValueError(
                f"run {self.id} has no inner-architecture snapshot for "
                f"{node!r} (scoped nodes: {available})"
            )
        graph_json = json.dumps(tree["graph"])
        if not overlay:
            return _soma.graph_json_to_svg(graph_json)
        return _soma.graph_json_to_svg(graph_json, overlay=self._inner_overlay(tree))

    def __repr__(self) -> str:
        return f"RunView({self.id!r}, state={self.state!r}, name={self.name!r})"

    def _repr_html_(self) -> str:
        """Card shown when a RunView is the cell result in a notebook."""
        info = self._info
        rows = "".join(
            f"<tr><td style='color:#898781;padding:1px 12px 1px 0'>{k}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{v}</td></tr>"
            for k, v in (
                ("state", _state_chip(info["state"])),
                ("kind", info["kind"]),
                ("created", (info.get("created_at") or "")[:19].replace("T", " ")),
                ("duration", _fmt_ms_html(info.get("duration_ms"))),
                ("tags", ", ".join(info.get("tags") or []) or "—"),
                ("dir", f"<code>{self._dir}</code>"),
            )
        )
        return (
            f"<div style='font-family:system-ui;font-size:13px;line-height:1.5'>"
            f"<b>{info['name']}</b> "
            f"<span style='color:#898781'>({info['run_id']})</span>"
            f"<table style='border-collapse:collapse;margin-top:4px'>{rows}</table></div>"
        )

    # ── Figures and tables (soma.viz) ──
    #
    # Declared here rather than assigned onto the class from
    # `soma.viz._install()` at import time, which meant `help(RunView)`
    # and every IDE showed a class without them. `soma.viz` imports
    # cleanly without plotly or pandas — each figure imports its backend
    # inside the call — so naming them here costs the base install
    # nothing, and calling one without `somatize[viz]` still raises the
    # same helpful error.

    def plot_metrics(self, names: list[str] | None = None) -> Any:
        """The metric series logged during the run."""
        from soma import viz

        return viz.plot_metrics(self, names)

    def plot_gantt(self) -> Any:
        """When each node ran, and for how long."""
        from soma import viz

        return viz.plot_gantt(self)

    def plot_agentic(self) -> Any:
        """When each agent effect ran (model calls, tools, sub-graphs),
        with replays hatched and errors outlined."""
        from soma import viz

        return viz.plot_agentic(self)

    def plot_health(self) -> Any:
        """The health flags raised during the run."""
        from soma import viz

        return viz.plot_health(self)

    def plot_audit(
        self,
        metric: str = "out_grad.norm",
        filters: list[str] | None = None,
        node: str | None = None,
    ) -> Any:
        """A gradient-audit metric per step. Submodule series are hidden unless `node` names one."""
        from soma import viz

        return viz.plot_audit(self, metric, filters, node)

    def plot_module_flow(self, node_id: str | None = None, metric: str = "out_grad.norm") -> Any:
        """The same metric down one node's submodule tree."""
        from soma import viz

        return viz.plot_module_flow(self, node_id, metric)

    def plot_channels(self, filter_id: str, step: int | None = None) -> Any:
        """Per-channel diagnostics for one filter at one step."""
        from soma import viz

        return viz.plot_channels(self, filter_id, step)

    def plot_channel_evolution(self, filter_id: str | None = None) -> Any:
        """How the per-channel picture moved over training."""
        from soma import viz

        return viz.plot_channel_evolution(self, filter_id)

    def metrics_dataframe(self, name: str | None = None) -> Any:
        """The metric series as a pandas DataFrame."""
        from soma import viz

        return viz.metrics_dataframe(self, name)

    def to_html(self, path: str | None = None, inline: bool = False) -> str:
        """One self-contained HTML report for the run."""
        from soma import viz

        return viz.to_html(self, path, inline)


_STATE_COLORS = {
    "completed": "#0ca30c",
    "failed": "#d03b3b",
    "crashed": "#d03b3b",
    "running": "#b07600",
}


def _state_chip(state: str) -> str:
    color = _STATE_COLORS.get(state, "#898781")
    return f"<span style='color:{color};font-weight:600'>{state}</span>"


def _fmt_ms_html(ms) -> str:
    if ms is None:
        return "—"
    seconds = ms / 1000
    if seconds < 120:
        return f"{seconds:.1f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class RunList(list):
    """``list[RunView]`` that renders as a table in notebooks."""

    def _repr_html_(self) -> str:
        if not self:
            return "<i>no runs</i>"
        head = "".join(
            f"<th style='text-align:left;padding:2px 14px 2px 0;color:#898781;"
            f"font-size:11px;text-transform:uppercase'>{h}</th>"
            for h in ("run id", "kind", "state", "created", "duration", "name")
        )
        body = "".join(
            "<tr>"
            f"<td style='padding:2px 14px 2px 0'><code>{r.id}</code></td>"
            f"<td style='padding:2px 14px 2px 0'>{r.kind}</td>"
            f"<td style='padding:2px 14px 2px 0'>{_state_chip(r.state)}</td>"
            f"<td style='padding:2px 14px 2px 0;font-variant-numeric:tabular-nums'>"
            f"{(r.info.get('created_at') or '')[:19].replace('T', ' ')}</td>"
            f"<td style='padding:2px 14px 2px 0'>{_fmt_ms_html(r.info.get('duration_ms'))}</td>"
            f"<td style='padding:2px 14px 2px 0'>{r.name}</td>"
            "</tr>"
            for r in self
        )
        return (
            "<table style='border-collapse:collapse;font-family:system-ui;"
            f"font-size:13px'><tr>{head}</tr>{body}</table>"
        )


def runs(root: str = ".soma") -> "RunList":
    """All runs under ``<root>/runs/``, newest first (renders as a
    table in notebooks). Directories without a readable manifest are
    skipped."""
    infos = json.loads(_soma.list_runs_json(root=root))
    return RunList(RunView(info["dir"], _info=info) for info in infos)
