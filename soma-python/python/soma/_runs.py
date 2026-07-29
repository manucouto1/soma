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

    def overlay(self) -> dict:
        """Per-node rendering annotations aggregated from this run's
        events: status, total duration, cache tier, health flags. Feed
        it to ``graph.to_mermaid(overlay=...)`` or use ``to_mermaid()``
        directly."""
        return json.loads(_soma.run_overlay_json(self._dir))

    def to_mermaid(self, overlay: bool = True) -> str:
        """Mermaid diagram of the graph this run executed, annotated
        with per-node timing/cache/health (``overlay=False`` for the
        plain topology)."""
        return _soma.run_to_mermaid(self._dir, overlay=overlay)

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
