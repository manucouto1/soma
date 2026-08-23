"""A study, drawn: the trials, which knob mattered, and where the good ones are.

    from soma_next.study import coordinates, influence, table

    table(store, space, study="widths")
    influence(store, space, study="widths")
    coordinates(store, space, study="widths", goal="min")

All three read a `Store` and nothing else, so a machine that ran none of these
trials draws them — which is the point of a study handed out of a folder.

## What each one costs

`table` and `influence` are **one scan and no fetches**: the configuration and
the score are both in the record, which is the shape CU18 gave it. `coordinates`
is the same scan. Nothing here reads a blob; a pruner's curves are the only
thing that does.

## Pruned and finished are not ranked together

`table` shows both, with their state. `influence` and `coordinates` use only the
trials that ran to the end, for the same reason `finished` leaves the others
out: a pruned score is real and was measured after fewer epochs, so ranking the
two together says a trial that was stopped early did badly, when all that is
known is that it was stopped.

## `goal` decides which end of the colour scale is good

It is a parameter and not a guess. Getting it backwards is the quietest kind of
lie a figure can tell: everything is drawn, nothing raises, and the region you
read as promising is the one to stay away from.
"""

from __future__ import annotations

import math

from soma_next import _theme
from soma_next.study._run import DONE, POINT, SCORE, STATE, importance, trials

__all__ = ["coordinates", "influence", "table"]

BIG = 50
"""Past this ratio between a knob's largest and smallest value, its axis is
drawn in log. The original's rule, and it is measured rather than declared
because a `Space` does not say how a knob was searched."""


def table(store, space, *, study):
    """Every scored trial, best first, with the configuration that got it.

    One scan and no fetches. Pruned trials are here too and say so: they are
    what the study spent its time on, and hiding them would make a run of
    thirty look like a run of fourteen.
    """
    go = _theme.plotly()
    scored = _scored(store, space, study=study)
    knobs = list(space.names())
    columns = ["trial", "state", *knobs, "score"]
    rows = [
        [one["trial"] for one in scored],
        [one[STATE] for one in scored],
        *[[_said(one[POINT][knob]) for one in scored] for knob in knobs],
        [f"{one[SCORE]:.4g}" for one in scored],
    ]
    done = sum(one[STATE] == DONE for one in scored)
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
            title=_theme.titled(f"{study} — {len(scored)} scored, {done} finished"),
            height=90 + 26 * (len(scored) + 1),
            margin={"l": 24, "r": 24, "t": 52, "b": 16},
        )
    )


def influence(store, space, *, study):
    """How decisive each knob was: |rho| against the score, biggest first.

    A rank correlation, so it says *this knob orders the results* and not *this
    knob is worth these many points*. One bar near zero is a knob you can stop
    searching; one near one is the knob the study is actually about.
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


def coordinates(store, space, *, study, goal="min"):
    """Every finished trial as a **curve** across the knobs, coloured by score.

    The one picture that shows a *region* of the space rather than one knob at a
    time: where the good curves bunch together is where to look next.

    Drawn by hand out of splines rather than with plotly's `Parcoords`, which
    only draws straight segments. What that costs is `Parcoords`' brushing —
    dragging a range on an axis to filter — and what it buys is that a trial
    reads as one continuous thing instead of a zigzag, which is what makes a
    bundle of them visible as a bundle.

    A curve is an interpolation and it is between axes, where **there is nothing
    to be wrong about**: a point exists only where it crosses an axis, and it
    crosses at the value it has. Nothing is being claimed about the space in
    between, and that is exactly what a rolling mean over a loss could not say.
    It is still drawn gently, because a curve bulging past the top of an axis
    reads as a value beyond its range even when it means nothing at all.

    There is no colour scale beside it: the score is the last axis, so the
    colour is the same fact read twice. What it is for is making a bundle
    visible as a bundle.
    """
    go = _theme.plotly()
    from plotly.colors import sample_colorscale

    scored = [one for one in _scored(store, space, study=study) if one[STATE] == DONE]
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
        [1 - (s - low) / spread if goal == "min" else (s - low) / spread for s in scores],
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


def _axis(name, values):
    """One axis: how to place a value on `[0, 1]`, and what to write beside it.

    Three shapes, and the difference is not cosmetic. A **categorical** knob is
    placed by its options in order, with every option written. A knob whose
    values span more than `BIG` is placed in **log**, which is how it was almost
    certainly searched. Anything else is linear.
    """
    if any(isinstance(one, str) for one in values):
        seen = sorted({str(one) for one in values})
        spread = max(len(seen) - 1, 1)
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


def _said(what):
    """A value as it goes on a figure: short, and never in scientific notation
    when it does not have to be."""
    if isinstance(what, float):
        return f"{what:.3g}"
    return str(what)


def _label(x, y, text, colour, size=11):
    return {
        "x": x, "y": y, "xref": "x", "yref": "y",
        "text": text, "showarrow": False,
        "font": {"size": size, "color": colour, "family": _theme.FONT},
        "xanchor": "center", "yanchor": "middle",
        "bgcolor": _theme.GROUND,
    }


def _scored(store, space, *, study):
    """Every trial with a score, best first. Pruned ones included."""
    return sorted(
        (
            one
            for one in trials(store, space, study=study)
            if one[SCORE] is not None and one[POINT] is not None
        ),
        key=lambda one: one[SCORE],
    )


def _nothing(go, study):
    """A study nobody has scored yet is a statement, not an exception."""
    return go.Figure().update_layout(
        annotations=[_label(0.5, 0.5, f"{study} — nothing has finished yet", _theme.MUTED)],
        **_theme.layout(
            xaxis={"visible": False, "range": [0, 1]},
            yaxis={"visible": False, "range": [0, 1]},
            height=160,
        ),
    )
