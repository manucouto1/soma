"""Plotly figure builders over Soma's run/study data.

Every function returns a ``plotly.graph_objects.Figure`` (interactive in
notebooks via ``_repr_html_``; serializable with ``fig.to_json()`` for
the HTML report). Figures are thin skins: all aggregation happens in the
Rust readers, so the same data shapes feed any future front-end.

plotly is an optional dependency::

    pip install 'somatize[viz]'
"""

from __future__ import annotations

from soma.viz import _theme


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover - exercised without plotly
        raise RuntimeError(
            "soma.viz needs plotly — install it with: pip install 'somatize[viz]'"
        ) from e
    _theme.soma_template()
    return go


def _primary_objective(study) -> tuple[str, str]:
    """(metric, direction) of the study's first objective."""
    objectives = study.objectives
    if not objectives:
        raise ValueError("study has no declared objectives")
    return objectives[0]


def _objective_values(trials: list[dict], metric: str) -> list[tuple[dict, float]]:
    """(trial, value) for completed trials that recorded `metric`."""
    out = []
    for t in trials:
        if t["state"] == "completed" and metric in t["metrics"]:
            out.append((t, t["metrics"][metric]))
    return out


# ── Study figures (Optuna-aligned names) ────────────────────────────


def plot_optimization_history(study, metric: str | None = None):
    """Objective value per trial with the running best overlaid."""
    go = _go()
    obj_metric, direction = _primary_objective(study)
    metric = metric or obj_metric

    completed = _objective_values(study.trials, metric)
    xs = [i for i, _ in enumerate(completed)]
    ids = [t["id"] for t, _ in completed]
    ys = [v for _, v in completed]

    best_so_far: list[float] = []
    for v in ys:
        if not best_so_far:
            best_so_far.append(v)
        elif direction == "minimize":
            best_so_far.append(min(best_so_far[-1], v))
        else:
            best_so_far.append(max(best_so_far[-1], v))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            customdata=ids,
            mode="markers",
            name=metric,
            marker={
                "size": 9,
                "color": _theme.CATEGORICAL[0],
                "line": {"width": 1.5, "color": "#ffffff"},
            },
            hovertemplate="%{customdata}<br>" + metric + " = %{y:.5g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=best_so_far,
            mode="lines",
            name="best so far",
            line={"width": 2.5, "color": _theme.SEQUENTIAL[5], "shape": "hv"},
            hovertemplate="best = %{y:.5g}<extra></extra>",
        )
    )
    fig.update_layout(
        template="soma",
        title=f"Optimization history — {study.name}",
        xaxis_title="completed trial #",
        yaxis_title=metric,
        hovermode="x unified",
    )
    return fig


def plot_intermediate_values(study, metric: str | None = None):
    """Per-trial learning curves from ``trial.report(...)``. The best
    trial is highlighted; pruned trials are dashed and muted."""
    go = _go()
    obj_metric, _ = _primary_objective(study)
    metric = metric or obj_metric
    best = study.best_trial
    best_id = best["id"] if best else None

    fig = go.Figure()
    for trial in study.trials:
        series = [m for m in trial["series"] if m["name"] == metric]
        if not series:
            continue
        pruned = trial["state"] == "pruned"
        is_best = trial["id"] == best_id
        if is_best:
            color, width, dash = _theme.CATEGORICAL[0], 3, None
        elif pruned:
            color, width, dash = _theme.STATE_COLORS["pruned"], 1, "dash"
        else:
            color, width, dash = _theme.BASELINE, 1, None
        line = {"width": width, "color": color}
        if dash:
            line["dash"] = dash
        fig.add_trace(
            go.Scatter(
                x=[m["step"] for m in series],
                y=[m["value"] for m in series],
                mode="lines",
                name=trial["id"],
                line=line,
                showlegend=False,
                hovertemplate=(
                    trial["id"]
                    + ("(best)" if is_best else " (pruned)" if pruned else "")
                    + "<br>step %{x} · "
                    + metric
                    + " = %{y:.5g}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="soma",
        title=f"Intermediate values — {study.name}"
        + (f" (best: {best_id})" if best_id else ""),
        xaxis_title="step",
        yaxis_title=metric,
    )
    return fig


def plot_parallel_coordinate(study, params: list[str] | None = None):
    """One line per completed trial across parameter axes, colored by
    objective value on a perceptually-uniform gradient (Viridis; bright
    = better). Log-spanning numeric params get log₁₀ axes automatically,
    and brushing an axis dims the unselected lines."""
    go = _go()
    metric, direction = _primary_objective(study)
    completed = _objective_values(study.trials, metric)
    if not completed:
        raise ValueError("no completed trials to plot")

    all_params = params or sorted({k for t, _ in completed for k in t["params"]})
    values = [v for _, v in completed]

    dimensions = [
        _parcoords_dim(name, [t["params"].get(name) for t, _ in completed])
        for name in all_params
    ]
    dimensions.append(_parcoords_dim(metric, values))

    fig = go.Figure(
        go.Parcoords(
            line={
                "color": values,
                "colorscale": _theme.MAGNITUDE,
                # Bright end of the gradient = better.
                "reversescale": direction == "minimize",
                "showscale": True,
                "colorbar": {
                    "title": {"text": metric, "side": "right"},
                    "thickness": 12,
                    "outlinewidth": 0,
                    "len": 0.85,
                    "tickfont": {"size": 11, "color": _theme.INK_SECONDARY},
                },
            },
            dimensions=dimensions,
            labelfont={"color": _theme.INK, "size": 13},
            tickfont={"color": _theme.INK_SECONDARY, "size": 11},
            rangefont={"color": _theme.MUTED, "size": 9},
            # Brushing an axis dims everything else to a faint gray —
            # the selection pops, W&B-style.
            unselected={"line": {"color": _theme.GRID, "opacity": 0.35}},
        )
    )
    fig.update_layout(
        template="soma",
        title=f"Parallel coordinates — {study.name}",
        margin={"l": 90, "r": 90, "t": 90, "b": 40},
    )
    return fig


_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _parcoords_dim(name: str, raw: list) -> dict:
    """One parallel-coordinates axis: numeric, log₁₀ (auto-detected for
    positive values spanning ≥ ~2 decades, e.g. learning rates), or
    categorical with labeled ticks."""
    numeric = all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw
    )
    if not numeric:
        seen: dict = {}
        idx = [seen.setdefault(str(v), len(seen)) for v in raw]
        return {
            "label": name,
            "values": idx,
            "tickvals": list(seen.values()),
            "ticktext": list(seen.keys()),
        }
    if all(v > 0 for v in raw) and max(raw) / min(raw) >= 50:
        import math

        logs = [math.log10(v) for v in raw]
        exponents = list(range(math.floor(min(logs)), math.ceil(max(logs)) + 1))
        return {
            "label": name,
            "values": logs,
            "tickvals": exponents,
            "ticktext": [f"10{str(e).translate(_SUPERSCRIPTS)}" for e in exponents],
        }
    return {"label": name, "values": raw}


def plot_param_importances(study):
    """Spearman rank correlation between each parameter and the
    objective, over completed trials. An honest, dependency-free
    importance measure — |ρ| ranks, sign colors. (fANOVA is a
    documented future upgrade.)"""
    go = _go()
    metric, _ = _primary_objective(study)
    completed = _objective_values(study.trials, metric)
    if len(completed) < 3:
        raise ValueError("need at least 3 completed trials for importances")

    values = [v for _, v in completed]
    names = sorted({k for t, _ in completed for k in t["params"] if k != "seed"})

    def _rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=xs.__getitem__)
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(order):  # average ties
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    def _spearman(xs: list[float], ys: list[float]) -> float:
        rx, ry = _rank(xs), _rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        dx = sum((a - mx) ** 2 for a in rx) ** 0.5
        dy = sum((b - my) ** 2 for b in ry) ** 0.5
        return num / (dx * dy) if dx and dy else 0.0

    rows = []
    for name in names:
        raw = [t["params"].get(name) for t, _ in completed]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
            continue  # rank correlation needs ordered values
        rows.append((name, _spearman([float(v) for v in raw], values)))
    rows.sort(key=lambda r: abs(r[1]))

    fig = go.Figure(
        go.Bar(
            x=[abs(rho) for _, rho in rows],
            y=[name for name, _ in rows],
            orientation="h",
            marker={"color": _theme.CATEGORICAL[0]},
            customdata=[rho for _, rho in rows],
            hovertemplate="%{y}: ρ = %{customdata:.3f}<extra></extra>",
            text=[f"ρ={rho:+.2f}" for _, rho in rows],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="soma",
        title=f"Parameter importance (|Spearman ρ| vs {metric}) — {study.name}",
        xaxis_title=f"|rank correlation with {metric}|",
        xaxis_range=[0, 1.15],
        bargap=0.45,
    )
    return fig


def plot_timeline(study):
    """Gantt of trial lifetimes, colored by terminal state."""
    go = _go()
    trials = [t for t in study.trials if t["started_at"] and t["finished_at"]]
    if not trials:
        raise ValueError("no timestamped trials to plot")

    fig = go.Figure()
    shown_states: set[str] = set()
    for t in trials:
        state = t["state"]
        color = _theme.STATE_COLORS.get(state, _theme.MUTED)
        fig.add_trace(
            go.Bar(
                base=[t["started_at"]],
                x=[t["duration_ms"] or 0],
                y=[t["id"]],
                orientation="h",
                name=state,
                legendgroup=state,
                showlegend=state not in shown_states,
                marker={"color": color},
                hovertemplate=(
                    f"{t['id']} ({state})<br>"
                    f"{(t['duration_ms'] or 0) / 1000.0:.2f}s<extra></extra>"
                ),
            )
        )
        shown_states.add(state)
    fig.update_layout(
        template="soma",
        title=f"Trial timeline — {study.name}",
        # base = start timestamp, x = duration in ms: needs a date axis.
        xaxis={"type": "date", "title": "wall time"},
        yaxis={"autorange": "reversed"},
        barmode="overlay",
        bargap=0.3,
    )
    return fig


def plot_pareto_front(study):
    """Scatter of the first two objectives; non-dominated trials
    highlighted. Requires a multi-objective study."""
    go = _go()
    objectives = study.objectives
    if len(objectives) < 2:
        raise ValueError("pareto front needs at least two objectives")
    (mx, dx), (my, dy) = objectives[0], objectives[1]

    pts = []
    for t in study.trials:
        if t["state"] == "completed" and mx in t["metrics"] and my in t["metrics"]:
            pts.append((t["id"], t["metrics"][mx], t["metrics"][my]))
    if not pts:
        raise ValueError("no completed trials with both objectives")

    def _better(a: float, b: float, direction: str) -> bool:
        return a <= b if direction == "minimize" else a >= b

    def _dominated(p) -> bool:
        return any(
            q is not p
            and _better(q[1], p[1], dx)
            and _better(q[2], p[2], dy)
            and (q[1] != p[1] or q[2] != p[2])
            for q in pts
        )

    front = [p for p in pts if not _dominated(p)]
    rest = [p for p in pts if _dominated(p)]

    fig = go.Figure()
    if rest:
        fig.add_trace(
            go.Scatter(
                x=[p[1] for p in rest],
                y=[p[2] for p in rest],
                customdata=[p[0] for p in rest],
                mode="markers",
                name="dominated",
                marker={
                    "size": 8,
                    "color": _theme.BASELINE,
                    "line": {"width": 1.5, "color": "#ffffff"},
                },
                hovertemplate="%{customdata}<br>%{x:.5g}, %{y:.5g}<extra></extra>",
            )
        )
    front.sort(key=lambda p: p[1])
    fig.add_trace(
        go.Scatter(
            x=[p[1] for p in front],
            y=[p[2] for p in front],
            customdata=[p[0] for p in front],
            mode="lines+markers",
            name="pareto front",
            marker={
                "size": 10,
                "color": _theme.CATEGORICAL[0],
                "line": {"width": 1.5, "color": "#ffffff"},
            },
            line={"width": 2, "color": _theme.CATEGORICAL[0], "dash": "dot"},
            hovertemplate="%{customdata}<br>%{x:.5g}, %{y:.5g}<extra></extra>",
        )
    )
    fig.update_layout(
        template="soma",
        title=f"Pareto front — {study.name}",
        xaxis_title=f"{mx} ({dx})",
        yaxis_title=f"{my} ({dy})",
    )
    return fig


# ── Run figures ─────────────────────────────────────────────────────

_MAX_METRIC_SERIES = 8  # categorical palette cap — beyond it, fold


def plot_metrics(run_view, names: list[str] | None = None):
    """Line per metric over steps, from the run's metric log."""
    go = _go()
    points = run_view.metric_series()
    if not points:
        raise ValueError(f"run {run_view.id} recorded no metrics")

    by_name: dict[str, list[dict]] = {}
    for p in points:
        by_name.setdefault(p["name"], []).append(p)
    selected = names or sorted(by_name)
    dropped = selected[_MAX_METRIC_SERIES:]
    if dropped:  # never silently truncate
        import warnings

        warnings.warn(
            f"plot_metrics: showing first {_MAX_METRIC_SERIES} metrics; "
            f"dropped {dropped} — pass names=[...] to choose",
            stacklevel=2,
        )
        selected = selected[:_MAX_METRIC_SERIES]

    fig = go.Figure()
    for i, name in enumerate(selected):
        series = by_name.get(name, [])
        fig.add_trace(
            go.Scatter(
                x=[p["step"] for p in series],
                y=[p["value"] for p in series],
                mode="lines+markers" if len(series) < 50 else "lines",
                name=name,
                line={"width": 2, "color": _theme.CATEGORICAL[i % 8]},
                marker={"size": 8},
                hovertemplate=name + "<br>step %{x} · %{y:.5g}<extra></extra>",
            )
        )
    fig.update_layout(
        template="soma",
        title=f"Metrics — {run_view.name}",
        xaxis_title="step",
        yaxis_title="value",
        showlegend=len(selected) > 1,
        hovermode="x unified",
    )
    return fig


def plot_gantt(run_view):
    """Waterfall of node execution spans (wall time from the event
    envelopes), colored by outcome — where the run's time went."""
    go = _go()
    spans = run_view.node_timings()
    spans = [s for s in spans if s["started_ts"]]
    if not spans:
        raise ValueError(f"run {run_view.id} has no node timing events")

    fig = go.Figure()
    shown: set[str] = set()
    for i, s in enumerate(spans):
        outcome = s["outcome"]
        color = _theme.STATE_COLORS.get(outcome, _theme.MUTED)
        label = f"{s['node_id']} #{i}" if len(spans) > len({x['node_id'] for x in spans}) else s["node_id"]
        duration = s["duration_ms"] or 0
        fig.add_trace(
            go.Bar(
                base=[s["started_ts"]],
                x=[max(duration, 1)],  # sub-ms spans stay visible
                y=[label],
                orientation="h",
                name=outcome,
                legendgroup=outcome,
                showlegend=outcome not in shown,
                marker={"color": color},
                hovertemplate=(
                    f"{s['node_id']} ({outcome}"
                    + (f", {s['cache_tier']}" if s.get("cache_tier") else "")
                    + f")<br>{duration}ms<extra></extra>"
                ),
            )
        )
        shown.add(outcome)
    fig.update_layout(
        template="soma",
        title=f"Node timeline — {run_view.name}",
        # base = start timestamp, x = duration in ms: needs a date axis.
        xaxis={"type": "date", "title": "wall time"},
        yaxis={"autorange": "reversed"},
        barmode="overlay",
        bargap=0.3,
    )
    return fig


# Effect kinds in fixed categorical order — never cycled, never rank-
# dependent. Anything unrecognized folds into "other" (muted), it does
# not invent a ninth hue.
_EFFECT_SLOTS = {"llm": 0, "tool": 1, "graph": 2, "sleep": 3, "custom": 4}


def plot_agentic(run_view):
    """Waterfall of an agent run's effect executions (model calls, tools,
    sub-graphs), colored by effect kind. Replayed effects — served from
    the journal instead of performed — carry a hatch pattern; a failed
    effect gets a status-red outline and says so on hover."""
    go = _go()
    spans = run_view.agentic_timeline()
    spans = [s for s in spans if s["started_ts"]]
    if not spans:
        raise ValueError(f"run {run_view.id} has no agent effect events")

    fig = go.Figure()
    shown: set[str] = set()
    for i, s in enumerate(spans):
        kind = s["effect"].split(":", 1)[0]
        slot = _EFFECT_SLOTS.get(kind)
        color = _theme.CATEGORICAL[slot] if slot is not None else _theme.MUTED
        duration = s["duration_ms"] or 0
        replay = s["replayed"]
        marker = {"color": color}
        if replay:
            # Secondary encoding, not a new hue: same identity, hatched.
            marker["pattern"] = {"shape": "/", "fgcolor": _theme.SURFACE, "solidity": 0.35}
        if s["is_error"]:
            marker["line"] = {"color": _theme.STATE_COLORS["failed"], "width": 2}
        fig.add_trace(
            go.Bar(
                base=[s["started_ts"]],
                x=[max(duration, 1)],  # sub-ms spans stay visible
                y=[f"{s['node_id']} · t{s['turn']} #{i}"],
                orientation="h",
                name=kind,
                legendgroup=kind,
                showlegend=kind not in shown,
                marker=marker,
                hovertemplate=(
                    f"{s['node_id']} turn {s['turn']}<br>{s['effect']}"
                    + (" (replayed)" if replay else "")
                    + (" — ERROR" if s["is_error"] else "")
                    + f"<br>{duration}ms<extra></extra>"
                ),
            )
        )
        shown.add(kind)
    fig.update_layout(
        template="soma",
        title=f"Agent activity — {run_view.name}",
        xaxis={"type": "date", "title": "wall time"},
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        barmode="overlay",
        bargap=0.3,
        showlegend=len(shown) > 1,
    )
    return fig
