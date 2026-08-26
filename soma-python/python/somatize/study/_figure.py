"""A study, drawn: the trials, which knob mattered, and where the good ones are.

    from somatize.study import coordinates, influence, table

    table(store, space, study="widths")
    influence(store, space, study="widths")
    coordinates(store, space, study="widths")

All three read a `Store` and nothing else, so a machine that ran none of these
trials draws them, and all three are **one scan and no fetches**.

`table` shows pruned trials too, with their state; `influence` and `coordinates`
use only what ran to the end, for the same reason `finished` leaves the others
out — a pruned score was measured after fewer epochs.

**The direction comes from the study and is never guessed.** Getting it backwards
is the quietest lie a figure can tell: everything is drawn, nothing raises, and
the region you read as promising is the one to stay away from. When nobody says,
`table` gives up the claim and falls back to the order the trials ran in, and
`coordinates` raises — a colour scale has two ends, and drawing one is saying
which is good.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from somatize._typing import Figure

if TYPE_CHECKING:
    from somatize._somatize import Space, Store
    from somatize.study._run import Trial

import math

from somatize import _theme
from somatize.study._run import (
    DONE,
    MAX,
    MIN,
    POINT,
    SCORE,
    STATE,
    direction,
    importance,
    trials,
)

__all__ = ["coordinates", "influence", "table"]

BIG = 50
"""Past this ratio between a knob's largest and smallest value, its axis is
drawn in log. The original's rule, and it is measured rather than declared
because a `Space` does not say how a knob was searched."""


def table(
    store: "Store", space: "Space", *, study: str, goal: str | None = None
) -> Figure:
    """Every scored trial, best first, with the configuration that got it. One
    scan and no fetches. Pruned trials are here too and say so.

    *Best* needs a direction, which comes from the record; `goal` overrides it.
    When neither says, the trials come back **in the order they were run** and
    the title says so — a table headed *best first* sorted the wrong way round is
    worse than one that is not sorted.
    """
    go = _theme.plotly()
    goal = _goal(store, study=study, goal=goal)
    scored = _scored(store, space, study=study, goal=goal)
    knobs = list(space.names())
    columns = ["trial", "state", *knobs, "score"]
    rows = [
        [one["trial"] for one in scored],
        [one[STATE] for one in scored],
        *[[_said(one[POINT][knob]) for one in scored] for knob in knobs],
        [f"{one[SCORE]:.4g}" for one in scored],
    ]
    done = sum(one[STATE] == DONE for one in scored)
    order = "best first" if goal else "in the order they ran — no direction recorded"
    return go.Figure(
        go.Table(
            columnwidth=[0.6, 0.8] + [1.0] * len(knobs) + [0.9],
            header={
                "values": [f"<b>{name}</b>" for name in columns],
                "fill_color": _theme.RAISED,
                "line_color": _theme.EDGE,
                "font": {"color": _theme.INK, "size": 12},
                "align": "left",
                "height": 30,
            },
            cells={
                "values": rows,
                "fill_color": _theme.GROUND,
                "line_color": _theme.EDGE,
                "font": {"color": _theme.INK, "size": 11},
                "align": "left",
                "height": 26,
            },
        )
    ).update_layout(
        **_theme.layout(
            title=_theme.titled(
                f"{study} — {len(scored)} scored, {done} finished, {order}"
            ),
            height=90 + 26 * (len(scored) + 1),
            margin={"l": 24, "r": 24, "t": 52, "b": 16},
        )
    )


def influence(store: "Store", space: "Space", *, study: str) -> Figure:
    """How decisive each knob was: |rho| against the score, biggest first.

    A rank correlation, so it says *this knob orders the results* and not *this
    knob is worth these many points*.
    """
    go = _theme.plotly()
    mattered = list(reversed(importance(store, space, study=study)))
    return go.Figure(
        go.Bar(
            x=[value for _, value in mattered],
            y=[knob for knob, _ in mattered],
            orientation="h",
            marker={
                "color": _theme.SERIES["loss"],
                "line": {"color": _theme.EDGE, "width": 1},
            },
            text=[f"{value:.2f}" for _, value in mattered],
            textposition="outside",
            textfont={"color": _theme.MUTED, "size": 11},
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>|rho| %{x:.3f}<extra></extra>",
        )
    ).update_layout(
        **_theme.layout(
            title=_theme.titled(f"{study} — |rho| against the score"),
            height=100 + 40 * len(mattered),
            showlegend=False,
            bargap=0.4,
        )
    ).update_xaxes(**_theme.axis(range=[0, 1.05], title_text="decisive ->")).update_yaxes(
        **_theme.axis(showgrid=False)
    )


def coordinates(
    store: "Store",
    space: "Space",
    *,
    study: str,
    goal: str | None = None,
) -> Figure:
    """Every finished trial as a **curve** across the knobs, coloured by score.
    The one picture that shows a *region* of the space rather than one knob at a
    time.

    Drawn by hand out of splines rather than with plotly's `Parcoords`, which
    only draws straight segments. That costs its brushing and buys a trial
    reading as one continuous thing. A curve is an interpolation **between axes,
    where there is nothing to be wrong about** — a point exists only where it
    crosses an axis — and still drawn gently, because a curve bulging past the
    top of an axis reads as a value beyond its range.

    No colour scale beside it: the score is the last axis. Unlike `table` there
    is no fallback, so a study that recorded no direction and a caller that names
    none is an error rather than a guess.
    """
    go = _theme.plotly()
    from plotly.colors import sample_colorscale

    goal = _goal(store, study=study, goal=goal, needed=True)
    scored = [
        one
        for one in _scored(store, space, study=study, goal=goal)
        if one[STATE] == DONE
    ]
    knobs = list(space.names())
    if not scored:
        return _nothing(go, study)

    axes = [_axis(knob, [one[POINT][knob] for one in scored]) for knob in knobs]
    scores = [one[SCORE] for one in scored]
    axes.append(_axis("score", scores))

    low, high = min(scores), max(scores)
    spread = (high - low) or 1.0
    # Reversed for a goal of `min`, because the eye reads bright as good.
    shades = sample_colorscale(
        "Viridis",
        [1 - (s - low) / spread if goal == MIN else (s - low) / spread for s in scores],
    )

    figure = go.Figure()
    for which, one in enumerate(scored):
        values = [one[POINT][knob] for knob in knobs] + [one[SCORE]]
        figure.add_trace(
            go.Scatter(
                x=list(range(len(axes))),
                y=[axis["at"](value) for axis, value in zip(axes, values)],
                mode="lines",
                # Gently: a spline overshoots, and a curve bulging past the top
                # of an axis reads as a value beyond its range. Curved enough to
                # follow one trial through the bundle, not enough to invent a
                # maximum nobody measured.
                line={
                    "color": shades[which],
                    "width": 1.6,
                    "shape": "spline",
                    "smoothing": 0.45,
                },
                opacity=0.85,
                hovertemplate=(
                    f"<b>trial {one['trial']}</b><br>"
                    + "<br>".join(
                        f"{knob} {_said(one[POINT][knob])}" for knob in knobs
                    )
                    + f"<br>score {one[SCORE]:.4g}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    shapes, notes = [], []
    for at, axis in enumerate(axes):
        shapes.append(
            {
                "type": "line",
                "x0": at, "x1": at, "y0": 0.0, "y1": 1.0,
                "xref": "x", "yref": "y",
                "line": {"color": _theme.EDGE, "width": 1.2},
            }
        )
        notes.append(_label(at, 1.06, axis["name"], _theme.INK))
        for value, place in axis["ticks"]:
            notes.append(_label(at, place, value, _theme.MUTED, size=10))

    return figure.update_layout(
        shapes=shapes,
        annotations=notes,
        **_theme.layout(
            title=_theme.titled(f"{study} — {len(scored)} finished trials"),
            height=420,
            # Room above for the axis names, which sit outside the plot.
            margin={"l": 56, "r": 56, "t": 58, "b": 28},
            xaxis={"visible": False, "range": [-0.35, len(axes) - 0.65]},
            yaxis={"visible": False, "range": [-0.09, 1.13]},
            showlegend=False,
        ),
    )


def _axis(name: str, values: Sequence[Any]) -> dict[str, Any]:
    """One axis: how to place a value on `[0, 1]`, and what to write beside it.

    Three shapes, and the difference is not cosmetic. A **categorical** knob is
    placed by its options in order, with every option written. A knob whose
    values span more than `BIG` is placed in **log**, which is how it was almost
    certainly searched. Anything else is linear.
    """
    if any(isinstance(one, str) for one in values):
        seen = sorted({str(one) for one in values})
        # `float`, because the other two branches below reuse the name for a
        # span that is one: all three are the divisor that puts a value on
        # `[0, 1]`, and only the categorical one happens to be whole.
        spread: float = max(len(seen) - 1, 1)
        return {
            "name": name,
            "at": lambda what: seen.index(str(what)) / spread,
            "ticks": [(one, i / spread) for i, one in enumerate(seen)],
        }
    low, high = min(values), max(values)
    logged = low > 0 and high / low >= BIG
    if logged:
        low, high = math.log10(low), math.log10(high)
    spread = (high - low) or 1.0
    place = (lambda what: (math.log10(what) - low) / spread) if logged else (
        lambda what: (what - low) / spread
    )
    unlog = (lambda x: 10**x) if logged else (lambda x: x)
    return {
        "name": f"{name} (log)" if logged else name,
        "at": place,
        "ticks": [(_said(unlog(low + spread * f)), f) for f in (0.0, 0.5, 1.0)],
    }


def _said(what: Any) -> str:
    """A value as it goes on a figure: short, and never in scientific notation
    when it does not have to be."""
    if isinstance(what, float):
        return f"{what:.3g}"
    return str(what)


def _label(
    x: float,
    y: float,
    text: str,
    colour: str,
    size: int = 11,
) -> dict[str, Any]:
    return {
        "x": x, "y": y, "xref": "x", "yref": "y",
        "text": text, "showarrow": False,
        "font": {"size": size, "color": colour, "family": _theme.FONT},
        "xanchor": "center", "yanchor": "middle",
        "bgcolor": _theme.GROUND,
    }


def _goal(
    store: "Store", *, study: str, goal: str | None, needed: bool = False
) -> str | None:
    """Which way is better here: what the caller said, else what the record says.

    The caller wins because the record is history and an override is the only
    way to draw a study that was run before the direction was written down.
    """
    said = goal if goal is not None else direction(store, study=study)
    if said is None and needed:
        raise ValueError(
            f"`{study}` does not say which way is better, so this figure cannot "
            f"know which end of its colour scale is good. Pass `goal=\"{MIN}\"` "
            f"or `goal=\"{MAX}\"`, and pass the same to `report` so the study "
            f"says it itself"
        )
    if said is not None and said not in (MIN, MAX):
        raise ValueError(
            f"`{said}` does not say which way is better: write `{MIN}` for a loss "
            f"or `{MAX}` for an accuracy"
        )
    return said


def _scored(
    store: "Store", space: "Space", *, study: str, goal: str | None
) -> list["Trial"]:
    """Every trial with a score, best first. Pruned ones included.

    `goal` decides which end *best* is. `None` means nobody said, and then the
    order is the one the trials were run in — a fact, rather than an ascending
    sort called *best first*.
    """
    scored = [
        one
        for one in trials(store, space, study=study)
        if one[SCORE] is not None and one[POINT] is not None
    ]
    if goal is None:
        return scored
    return sorted(scored, key=lambda one: one[SCORE], reverse=goal == MAX)


def _nothing(go: Any, study: str) -> Figure:
    """A study nobody has scored yet is a statement, not an exception."""
    return go.Figure().update_layout(
        annotations=[_label(0.5, 0.5, f"{study} — nothing has finished yet", _theme.MUTED)],
        **_theme.layout(
            xaxis={"visible": False, "range": [0, 1]},
            yaxis={"visible": False, "range": [0, 1]},
            height=160,
        ),
    )
