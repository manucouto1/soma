"""DataFrame projections of runs, studies, and the experiments journal.

pandas is imported lazily (part of the ``somatize[viz]`` extra), same
pattern as ``AuditReport.dataframe()``.
"""

from __future__ import annotations


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - exercised without pandas
        raise RuntimeError(
            "dataframes need pandas — install it with: pip install 'somatize[viz]'"
        ) from e
    return pd


def trials_dataframe(study):
    """One row per trial: id, state, timing, ``param_*`` and
    ``metric_*`` columns (last value per metric)."""
    pd = _pandas()
    rows = []
    for t in study.trials:
        row = {
            "trial_id": t["id"],
            "state": t["state"],
            "started_at": t["started_at"],
            "finished_at": t["finished_at"],
            "duration_ms": t["duration_ms"],
        }
        for k, v in t["params"].items():
            row[f"param_{k}"] = v
        for k, v in t["metrics"].items():
            row[f"metric_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("started_at", "finished_at"):
        if col in df:
            df[col] = pd.to_datetime(df[col], format="ISO8601")
    return df


def metrics_dataframe(run_view, name: str | None = None):
    """The run's metric log as a tidy frame:
    ``ts, name, value, step, trial_id, node_id``."""
    pd = _pandas()
    df = pd.DataFrame(
        run_view.metric_series(name),
        columns=["ts", "name", "value", "step", "trial_id", "node_id"],
    )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    return df


def experiments_dataframe(root: str = ".soma"):
    """The experiments journal as one row per record, with ``metric_*``
    and ``param_*`` columns flattened."""
    from soma._experiments import experiments

    pd = _pandas()
    rows = []
    for rec in experiments(root):
        row = {
            k: rec.get(k)
            for k in ("id", "name", "research_line", "hypothesis", "timestamp", "tags")
        }
        for k, v in (rec.get("metrics") or {}).items():
            row[f"metric_{k}"] = v
        for k, v in (rec.get("params") or {}).items():
            row[f"param_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df
