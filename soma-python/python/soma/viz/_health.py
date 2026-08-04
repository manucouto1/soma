"""Architecture-health figures: HealthFlag events, gradient-audit time
series, and per-channel diagnostics snapshots.

Data sources (all inside a tracked run's directory):

- ``events.jsonl`` — ``HealthFlag`` events (via ``RunView.health_flags``)
- ``diagnostics/audit_steps.jsonl`` — per-step, per-filter scalars from
  ``graph.gradient_audit(...)``
- ``diagnostics/channels/index.jsonl`` + ``<filter>/step_*.safetensors``
  — channel correlation/CKA snapshots (reading tensors needs
  ``safetensors``, an optional dependency of the audit system)
"""

from __future__ import annotations

import json
import pathlib

from soma.viz import _figures, _theme

# Diverging scale for correlations (-1..1): blue ↔ neutral gray ↔ red.
_DIVERGING = [
    [0.0, "#0d366b"],
    [0.25, "#6da7ec"],
    [0.5, "#f0efec"],
    [0.75, "#ec835a"],
    [1.0, "#7a1f1f"],
]

_DEAD_CHANNEL_FRAC = 0.95  # mirrors ChannelConfig.dead_channel_frac


def _audit_steps(run_view) -> list[dict]:
    path = pathlib.Path(run_view.dir) / "diagnostics" / "audit_steps.jsonl"
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            lines.append(json.loads(raw))
        except ValueError:
            continue  # torn tail
    return lines


def _channel_index(run_view) -> list[dict]:
    path = pathlib.Path(run_view.dir) / "diagnostics" / "channels" / "index.jsonl"
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except ValueError:
            continue
    return out


def _load_snapshot(run_view, entry: dict) -> dict:
    try:
        from safetensors.numpy import load_file
    except ImportError as e:
        raise RuntimeError(
            "channel snapshots need safetensors — pip install safetensors"
        ) from e
    path = pathlib.Path(run_view.dir) / "diagnostics" / "channels" / entry["file"]
    return load_file(str(path))


def plot_health(run_view):
    """Health flags over training: one mark per ``HealthFlag`` event,
    node × step, colored by flag family. Empty-flag runs return a
    figure that says so — absence of flags is the result."""
    go = _figures._go()
    flags = run_view.health_flags()

    fig = go.Figure()
    if not flags:
        fig.add_annotation(
            text="No health flags — every audited step stayed within thresholds.",
            showarrow=False,
            font={"color": _theme.INK_SECONDARY, "size": 14},
        )
        fig.update_layout(template="soma", title=f"Health flags — {run_view.name}")
        return fig

    families = []
    for f in flags:
        fam = f["flag"].split("(")[0]
        if fam not in families:
            families.append(fam)
    for i, fam in enumerate(families):
        rows = [f for f in flags if f["flag"].split("(")[0] == fam]
        fig.add_trace(
            go.Scatter(
                x=[f["step"] for f in rows],
                y=[f["node_id"] for f in rows],
                customdata=[f"{f['flag']} — {f['detail']}" for f in rows],
                mode="markers",
                name=fam,
                marker={
                    "size": 12,
                    "symbol": "square",
                    "color": _theme.CATEGORICAL[i % 8],
                },
                hovertemplate="%{y} @ step %{x}<br>%{customdata}<extra></extra>",
            )
        )
    fig.update_layout(
        template="soma",
        title=f"Health flags — {run_view.name}",
        xaxis_title="step",
        yaxis_title="node",
    )
    return fig


def plot_audit(
    run_view,
    metric: str = "out_grad.norm",
    filters: list[str] | None = None,
    node: str | None = None,
):
    """One line per filter of an audit scalar over steps, from
    ``diagnostics/audit_steps.jsonl``. `metric` is dotted, e.g.
    ``"act.zero_frac"``, ``"out_grad.norm"``, ``"param.grad_param_ratio"``.
    Norm/ratio metrics get a log y-axis.

    By default only node-level (root) series are drawn; submodule
    series recorded by ``gradient_audit(inside=...)`` appear when you
    select them — ``node="encoder"`` draws that node plus its
    submodules in execution order, ``filters=[...]`` picks ids
    explicitly."""
    go = _figures._go()
    steps = _audit_steps(run_view)
    if not steps:
        raise ValueError(f"run {run_view.id} has no diagnostics/audit_steps.jsonl")

    group, _, field = metric.partition(".")
    by_filter: dict[str, list[tuple[int, float]]] = {}
    for line in steps:
        value = line.get(group, {}).get(field) if field else line.get(group)
        if value is None or isinstance(value, bool):
            continue
        by_filter.setdefault(line["filter"], []).append((line["step"], value))
    if not by_filter:
        raise ValueError(f"metric {metric!r} not present in audit steps")

    trees = {t["node"]: t for t in run_view._module_trees()}
    if filters is not None:
        selected = filters
    elif node is not None:
        tree = trees.get(node)
        if tree is None and node not in by_filter:
            raise ValueError(
                f"no audit series for node {node!r} "
                f"(scoped nodes: {sorted(trees)}; roots: {sorted(by_filter)})"
            )
        selected = [node] if node in by_filter else []
        if tree is not None:
            selected += [
                tree["ids"][path]
                for path in tree["order"]
                if tree["ids"][path] in by_filter
            ]
    else:
        child_ids = {cid for t in trees.values() for cid in t["ids"].values()}
        selected = sorted(fid for fid in by_filter if fid not in child_ids)

    if len(selected) > 8:
        import warnings

        warnings.warn(
            f"plot_audit: showing first 8 of {len(selected)} series; "
            f"dropped {selected[8:]} — pass filters=[...] or node=... to choose",
            stacklevel=2,
        )
        selected = selected[:8]

    fig = go.Figure()
    for i, fid in enumerate(selected):
        series = by_filter.get(fid, [])
        fig.add_trace(
            go.Scatter(
                x=[s for s, _ in series],
                y=[v for _, v in series],
                mode="lines",
                name=fid,
                line={"width": 2, "color": _theme.CATEGORICAL[i % 8]},
                hovertemplate=fid + "<br>step %{x} · %{y:.4g}<extra></extra>",
            )
        )
    log_y = field in ("norm", "grad_norm", "grad_max", "max", "grad_param_ratio")
    fig.update_layout(
        template="soma",
        title=f"Gradient audit: {metric} — {run_view.name}",
        xaxis_title="step",
        yaxis={"title": metric, "type": "log" if log_y else "linear"},
        showlegend=len(selected) > 1,
    )
    return fig


def plot_module_flow(
    run_view, node_id: str | None = None, metric: str = "out_grad.norm"
):
    """Gradient flow *through* one node's inner architecture: x =
    submodules in real execution order, y = the audit metric (log scale
    for norms). Vanishing gradients read as a staircase falling toward
    the input; exploding, rising. Traces: mean across steps with a
    min–max band, plus the final recorded step dashed.

    Needs a run audited with ``gradient_audit(inside=...)``."""
    go = _figures._go()
    trees = {t["node"]: t for t in run_view._module_trees()}
    if not trees:
        raise ValueError(
            f"run {run_view.id} has no inner-architecture snapshots — "
            f"audit with gradient_audit(inside=...) under track_run"
        )
    if node_id is None:
        if len(trees) > 1:
            raise ValueError(
                f"multiple scoped nodes {sorted(trees)} — pass node_id=..."
            )
        node_id = next(iter(trees))
    tree = trees.get(node_id)
    if tree is None:
        raise ValueError(
            f"no inner-architecture snapshot for {node_id!r} "
            f"(scoped nodes: {sorted(trees)})"
        )

    group, _, field = metric.partition(".")
    series: dict[str, dict[int, float]] = {}
    for line in _audit_steps(run_view):
        value = line.get(group, {}).get(field) if field else line.get(group)
        if value is None or isinstance(value, bool):
            continue
        series.setdefault(line["filter"], {})[line["step"]] = value

    order = [p for p in tree["order"] if tree["ids"][p] in series]
    if not order:
        raise ValueError(
            f"metric {metric!r} not recorded for any submodule of {node_id!r}"
        )
    per_module = [series[tree["ids"][p]] for p in order]
    means = [sum(v.values()) / len(v) for v in per_module]
    lows = [min(v.values()) for v in per_module]
    highs = [max(v.values()) for v in per_module]
    last_step = max(s for v in per_module for s in v)
    finals = [v.get(last_step) for v in per_module]

    fig = go.Figure()
    # min–max band across steps.
    fig.add_trace(
        go.Scatter(
            x=order + order[::-1],
            y=highs + lows[::-1],
            fill="toself",
            fillcolor="rgba(42, 120, 214, 0.12)",
            line={"width": 0},
            name="min–max",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=order,
            y=means,
            mode="lines+markers",
            name="mean over steps",
            line={"width": 2, "color": _theme.CATEGORICAL[0]},
            marker={"size": 8},
            hovertemplate="%{x}<br>mean " + metric + " = %{y:.3e}<extra></extra>",
        )
    )
    if any(v is not None for v in finals):
        fig.add_trace(
            go.Scatter(
                x=order,
                y=finals,
                mode="lines",
                name=f"step {last_step}",
                line={"width": 1, "color": _theme.SEQUENTIAL[5], "dash": "dash"},
                hovertemplate="%{x}<br>step "
                + str(last_step)
                + " = %{y:.3e}<extra></extra>",
            )
        )
    log_y = field in ("norm", "grad_norm", "grad_max", "max", "grad_param_ratio")
    fig.update_layout(
        template="soma",
        title=f"Gradient flow through {node_id} — {run_view.name}",
        xaxis={"title": "submodule (execution order)", "type": "category"},
        yaxis={"title": metric, "type": "log" if log_y else "linear"},
        hovermode="x unified",
    )
    return fig


def plot_channels(run_view, filter_id: str, step: int | None = None):
    """Channel-correlation heatmap for one filter at one snapshot
    (default: the last). Dead channels (zero fraction above the audit
    threshold) are called out in the title and hover."""
    go = _figures._go()
    entries = [e for e in _channel_index(run_view) if e["filter"] == filter_id]
    if not entries:
        raise ValueError(
            f"no channel snapshots for filter {filter_id!r} "
            f"(available: {sorted({e['filter'] for e in _channel_index(run_view)})})"
        )
    entry = (
        min(entries, key=lambda e: abs(e["step"] - step))
        if step is not None
        else entries[-1]
    )
    tensors = _load_snapshot(run_view, entry)
    corr = tensors["corr"]
    n = corr.shape[0]

    dead = []
    if "act_zero_frac" in tensors:
        zero_frac = tensors["act_zero_frac"]
        dead = [i for i in range(len(zero_frac)) if zero_frac[i] > _DEAD_CHANNEL_FRAC]
    labels = [f"ch{i}" + (" †" if i in dead else "") for i in range(n)]

    fig = go.Figure(
        go.Heatmap(
            z=corr,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=_DIVERGING,
            colorbar={"title": "corr"},
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        )
    )
    dead_note = f" · {len(dead)} dead channel(s) †" if dead else ""
    fig.update_layout(
        template="soma",
        title=(
            f"Channel correlation — {filter_id} @ step {entry['step']}"
            f" (eff. rank {entry['eff_rank']:.1f}){dead_note}"
            if entry.get("eff_rank") is not None
            else f"Channel correlation — {filter_id} @ step {entry['step']}{dead_note}"
        ),
        yaxis={"autorange": "reversed"},
    )
    return fig


def plot_channel_evolution(run_view, filter_id: str | None = None):
    """Effective rank and max inter-group CKA per snapshot over
    training — the collapse/leakage trajectory. One filter, or all
    filters when `filter_id` is None."""
    go = _figures._go()
    entries = _channel_index(run_view)
    if filter_id is not None:
        entries = [e for e in entries if e["filter"] == filter_id]
    if not entries:
        raise ValueError("no channel snapshots recorded")

    by_filter: dict[str, list[dict]] = {}
    for e in entries:
        by_filter.setdefault(e["filter"], []).append(e)

    fig = go.Figure()
    for i, (fid, snaps) in enumerate(sorted(by_filter.items())):
        color = _theme.CATEGORICAL[i % 8]
        ranks = [(s["step"], s["eff_rank"]) for s in snaps if s.get("eff_rank") is not None]
        if ranks:
            fig.add_trace(
                go.Scatter(
                    x=[s for s, _ in ranks],
                    y=[v for _, v in ranks],
                    mode="lines+markers",
                    name=f"{fid} eff. rank",
                    line={"width": 2, "color": color},
                    marker={"size": 8},
                    hovertemplate=fid + "<br>step %{x} · eff. rank %{y:.2f}<extra></extra>",
                )
            )
        ckas = [
            (s["step"], max(s["cka"].values()))
            for s in snaps
            if s.get("cka")
        ]
        if ckas:
            fig.add_trace(
                go.Scatter(
                    x=[s for s, _ in ckas],
                    y=[v for _, v in ckas],
                    mode="lines+markers",
                    name=f"{fid} max CKA",
                    line={"width": 2, "color": color, "dash": "dash"},
                    marker={"size": 8, "symbol": "diamond"},
                    hovertemplate=fid + "<br>step %{x} · max CKA %{y:.3f}<extra></extra>",
                )
            )
    fig.update_layout(
        template="soma",
        title=f"Channel evolution — {run_view.name}",
        xaxis_title="step",
        yaxis_title="eff. rank (solid) / max CKA (dashed)",
    )
    return fig
