"""A run, drawn: while it is going and after it is over.

    from somatize.record import Live, progress, spent

    live = Live()                      # in a notebook cell, on its own
    t = Trainer(g, ..., watching=[Recorder(store, summarising=["loss"]), live])

    progress(store, run="tuesday")     # afterwards, or another machine's
    spent(store, run="tuesday")

## The same picture from two sources, which is the point

`progress` reads a store and `Live` is handed facts as they happen, and they draw
**the same figure** — because a fact read back is the very dict a watcher was
given. So there is one drawing function here and two ways to fill it, rather than
a live view and a report that slowly stop agreeing.

## Why the smooth line is a mean and not a spline through the points

A spline drawn through measured values invents the values between them, and an
overshoot on a loss curve can dip below a minimum that never happened. This
project's rule about figures is that they may simplify and may not lie.

So the bold line is a **rolling mean**, which is a stated transformation, and the
raw series stays underneath it, thin and faint. Nothing is hidden by the
smoothing: what was measured is on the figure, and what is easy to read is
admitted to be an average. It is also what anybody who has watched a loss curve
expects to see.

## What is drawn, and what is deliberately not

Progress and cost: the loss, how long each `forward` took, and which of them
broke. That last one is the only judgement on the figure and it is not one — a
`forward` that broke is a fact in the record.

Nothing here says whether a number is *bad*. A gradient that is dying, a layer
that is saturated, a rate that has stalled: those are opinions with arguable
thresholds, they are CU21, and the invariant is that they have to be reachable
from the stored record without training again. When they exist they get a channel
of their own; they do not get to recolour these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from somatize._typing import Fact, Figure

if TYPE_CHECKING:
    from somatize._somatize import Store
    from somatize.record._read import Row

#: One bar of a timeline: what it is called, when it began and how long it
#: took, in milliseconds, plus whether it happened elsewhere.
Bar = dict[str, Any]

from somatize import _theme
from somatize.record._read import facts, fleet, forwards, nodes

__all__ = ["Live", "gantt", "progress", "spent"]

SMOOTH_OVER = 15
"""A rolling mean spans `len // SMOOTH_OVER` points, so that a run of forty and
a run of ten thousand are smoothed by eye to the same degree rather than one of
them being flattened into a straight line."""


def progress(
    store: "Store",
    *,
    run: str,
    smooth: int | None = None,
) -> Figure:
    """How a run went, step by step: the loss, the time, and what broke.

    One scan when the recorder was told to summarise the loss, and a fetch per
    `forward` when it was not — `curve_costs` says which.

    `smooth` is the window of the rolling mean, in `forward`s. `0` draws only
    what was measured.
    """
    return _drawn(forwards(store, run=run), title=run, smooth=smooth)


def gantt(store: "Store", *, run: str, forward: int = 0) -> Figure:
    """One `forward` on a timeline: what ran when, and what was waiting.

    The picture `spent` cannot draw. A total says a node cost four hundred
    milliseconds; this says whether those four hundred were **beside** the rest
    of the graph or in front of it, which is the difference between a slow node
    and a bottleneck.

    Every fact carries `began_us` — how far into the run it started — so a
    `Wave` shows as overlapping bars and a `Sequence` as a staircase. A slice
    that ran on another machine counts from **its own** start, so its bars are
    shifted by the `left` they arrived under. That is not a fudge: an offset
    into a slice is a fact about the slice, and two wall clocks would not have
    composed at all.

    One `forward` and not an average of them, because an average of timelines
    is not a timeline.
    """
    go = _theme.plotly()
    bars = _bars(facts(store, run=run, forward=forward) or [])
    if not bars:
        return _drawn([], title=f"{run} — nothing ran in forward {forward}")

    figure = go.Figure()
    for which, one in enumerate(bars):
        figure.add_trace(
            go.Bar(
                x=[one["took"]],
                y=[one["name"]],
                base=[one["began"]],
                orientation="h",
                marker={
                    "color": _theme.SERIES["recalled"] if one["remote"] else _theme.SERIES["took"],
                    "line": {"color": _theme.EDGE, "width": 1},
                },
                hovertemplate=(
                    f"<b>{one['name']}</b><br>starts at %{{base:.2f}} ms"
                    f"<br>takes %{{x:.2f}} ms<br>{one['where']}<extra></extra>"
                ),
                showlegend=False,
                name=one["name"],
            )
        )
    return figure.update_layout(
        **_theme.layout(
            title=_theme.titled(f"{run} — forward {forward}, on a timeline"),
            height=max(180, 34 * len(bars) + 110),
            barmode="overlay",
            bargap=0.35,
        )
    ).update_xaxes(**_theme.axis(title_text="ms into the forward")).update_yaxes(
        **_theme.axis(showgrid=False, autorange="reversed", automargin=True)
    )


def _bars(seen: Sequence[Fact]) -> list[Bar]:
    """The facts of one `forward` as `(name, began, took)`, in the order they
    started.

    A `left` carries the slice it framed, so what ran over there is shifted onto
    this timeline by adding its offset — and drawn in its own colour, because
    *this happened elsewhere* is the thing a timeline of a distributed graph is
    for.
    """
    bars: list[Bar] = []
    waiting: dict[str, list[tuple[Fact, float, float]]] = {}
    for fact in seen:
        began, took = _read(fact, "began_us"), _read(fact, "took_us")
        if began is None or took is None:
            continue
        if fact["fact"] == "ran" and "host" in fact:
            # It arrived **before** the `left` that frames it — a relay hands a
            # fact over while the dispatch is still in flight — so it waits here
            # until the offset it has to be shifted by turns up.
            waiting.setdefault(fact["host"], []).append((fact, began, took))
        elif fact["fact"] == "ran":
            bars.append(_bar(fact.get("node", "?"), began, took, remote=False, where="here"))
        elif fact["fact"] == "left":
            host = fact.get("host", "?")
            bars.append(
                _bar(
                    f"→ {host}",
                    began,
                    took,
                    remote=True,
                    where=f"the whole round trip to {host}",
                )
            )
            for one, at,lasted in waiting.pop(host, []):
                bars.append(
                    _bar(one.get("node", "?"), began + at, lasted, remote=True, where=f"on {host}")
                )
    # A slice that never came back leaves its facts unframed: they are still
    # what happened, and dropping them would be the timeline hiding the failure.
    for host, held in waiting.items():
        for one, at, lasted in held:
            bars.append(_bar(one.get("node", "?"), at, lasted, remote=True, where=f"on {host}"))
    return sorted(bars, key=lambda one: one["began"])


def _bar(
    name: str,
    began: float,
    took: float,
    *,
    remote: bool,
    where: str,
) -> Bar:
    return {
        "name": name,
        "began": began / 1000.0,
        "took": took / 1000.0,
        "remote": remote,
        "where": where,
    }


def _read(fact: Fact, name: str) -> float | None:
    try:
        return float(fact[name])
    except (KeyError, TypeError, ValueError):
        return None


def spent(store: "Store", *, run: str, last: int | None = None) -> Figure:
    """Where the time went, added up per node — the aggregate view.

    Bars are coloured by **where** the node ran, which is the same table the
    graph is drawn with: a device is green, another machine is orange.

    It costs a fetch per `forward`, because which node did what is in the blobs.
    `last=N` reads only the last N, which is the question worth asking of a run
    that is ten thousand steps long.
    """
    return _spent(nodes(store, run=run, last=last), title=run)


class Live:
    """A run drawn while it happens. Hand it to `watching=`.

        live = Live()
        live                                   # the cell shows it
        g.forward(x, watching=live)

    In a notebook with `ipywidgets` it redraws in place; without one it still
    tallies everything and `figure()` gives the same picture at any moment.
    Either way it holds only the summary of each `forward` — one row of numbers —
    so watching a run for an afternoon does not grow with it.

    It keeps nothing on disk. A `Live` beside a `Recorder` is the normal pairing:
    one to look at now, one to have afterwards.
    """

    def __init__(
        self,
        *,
        title: str = "live",
        smooth: int | None = None,
        every: int = 1,
    ) -> None:
        self.title = title
        self.smooth = smooth
        #: How many finished `forward`s go by between redraws. A figure redrawn
        #: a thousand times a second is a figure nobody sees and a notebook
        #: nobody can type in.
        self.every = every
        self.rows: list[dict[str, Any]] = []
        self._pending: dict[str, Any] = {"nodes": 0}
        self._widget: Any = None
        self._since = 0

    def __call__(self, fact: Fact) -> None:
        """One fact, from the engine or from a level above it."""
        kind = fact.get("fact")
        if kind == "ran":
            self._pending["nodes"] += 1
        elif kind in ("finished", "broke"):
            self._pending["forward"] = len(self.rows)
            self._pending["state"] = "ok" if kind == "finished" else "broke"
            self._pending["took_us"] = int(fact.get("took_us", 0))
            self.rows.append(self._pending)
            self._pending = {"nodes": 0}
            self._redrawn()
        elif kind is not None and self.rows:
            # Whatever else was said — a loss, and whatever a level above
            # invents next. It lands on the `forward` that just ended, exactly
            # as the recorder files it, and under the same `<kind>.<field>` name
            # so that this figure and one read back off a store are one figure.
            for name, what in fact.items():
                if name != "fact":
                    self.rows[-1][f"{kind}.{name}"] = what
            self._redrawn(anyway=True)

    def figure(self) -> Figure:
        """The same figure `progress` draws, from what has arrived so far."""
        return _drawn(self.rows, title=self.title, smooth=self.smooth)

    def widget(self) -> Any:
        """A `FigureWidget` that redraws in place, or `None` without ipywidgets.

        Asked for by hand rather than made on construction: building one imports
        ipywidgets, and a `Live` used from a script has no business needing it.
        """
        if self._widget is None:
            go = _theme.plotly()
            try:
                self._widget = go.FigureWidget(self.figure())
            except Exception:
                return None
        return self._widget

    def _redrawn(self, anyway: bool = False) -> None:
        """Push what has arrived into the widget, if there is one and it is due."""
        self._since += 1
        if not anyway and self._since < self.every:
            return
        self._since = 0
        if self._widget is None:
            return
        fresh = self.figure()
        with self._widget.batch_update():
            for there, here in zip(self._widget.data, fresh.data):
                there.x, there.y = here.x, here.y

    def _repr_mimebundle_(
        self,
        include: object = None,
        exclude: object = None,
        **rest: Any,
    ) -> dict[str, Any] | None:
        """In a notebook cell: the figure, and nothing about this object.

        The same wall CU19 walked into — a figure reaches a cell through the
        mimebundle it publishes, not through hand-written HTML — and the same
        way out. An empty bundle has to become `None` or the cell shows neither.
        """
        try:
            bundle = self.figure()._repr_mimebundle_(include, exclude, **rest)
        except RuntimeError:
            return None
        return bundle or None


def _drawn(
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    smooth: int | None = None,
) -> Figure:
    """The two-row figure: what it cost, over what it learnt.

    Both callers land here, which is what keeps a live view and a report from
    drifting apart.
    """
    go = _theme.plotly()
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.66, 0.34],
    )
    steps = [row["forward"] for row in rows]
    losses = [_number(row.get("loss.value")) for row in rows]
    took = [row.get("took_us", 0) / 1000.0 for row in rows]

    # What was measured: thin and faint, and never covered up by the mean drawn
    # over it.
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=losses,
            name="loss",
            mode="lines",
            line={"color": _theme.SERIES["loss"], "width": 1},
            opacity=0.32,
            hovertemplate="forward %{x}<br>loss %{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=_smoothed(losses, smooth),
            name="loss, smoothed",
            mode="lines",
            line={"color": _theme.SERIES["loss"], "width": 2.4, "shape": "spline"},
            hovertemplate="forward %{x}<br>mean %{y:.4g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=steps,
            y=took,
            name="took",
            mode="lines",
            line={"color": _theme.SERIES["took"], "width": 1.6, "shape": "spline"},
            fill="tozeroy",
            fillcolor="rgba(79,195,176,0.10)",
            hovertemplate="forward %{x}<br>%{y:.1f} ms<extra></extra>",
        ),
        row=2,
        col=1,
    )
    # The only red on the figure, and it marks a fact: this one did not finish.
    broke = [row["forward"] for row in rows if row.get("state") == "broke"]
    figure.add_trace(
        go.Scatter(
            x=broke,
            y=[_number(rows[0].get("loss.value")) if rows else 0] * len(broke),
            name="broke",
            mode="markers",
            marker={"color": _theme.SERIES["alarm"], "size": 9, "symbol": "x"},
            showlegend=bool(broke),
            hovertemplate="forward %{x} broke<extra></extra>",
        ),
        row=1,
        col=1,
    )

    figure.update_layout(
        **_theme.layout(
            title=_theme.titled(_said(title, rows)),
            height=440,
            hovermode="x unified",
        )
    )
    figure.update_yaxes(_theme.axis(title_text="loss"), row=1, col=1)
    figure.update_yaxes(_theme.axis(title_text="ms", rangemode="tozero"), row=2, col=1)
    figure.update_xaxes(_theme.axis(showticklabels=False), row=1, col=1)
    figure.update_xaxes(_theme.axis(title_text="forward"), row=2, col=1)
    return figure


def machines(store: "Store", *, run: str, last: int | None = None) -> Figure:
    """The run, per machine: what each one worked and what it was waited on.

    The inverse of `spent`, and the split is the whole of it. A bar is the round
    trip and it comes in two parts — the time a machine spent **working**, and
    the time it was **waited on**: the wire, the queue and the codec. Neither
    half belongs to a node, which is why no per-node view can draw this and why
    it is worth its own figure.

    It is the answer to *was sending it worth it*, and the answer is often no
    and obvious the moment it is a picture: a slice with sixty microseconds of
    work behind a second of round trip is a slice that should have stayed here.

    Costs what `fleet` costs — a scan and a fetch per `forward`.
    """
    return _machines(fleet(store, run=run, last=last), title=run)


def _machines(tally: Sequence["Row"], *, title: str) -> Figure:
    """Working against waited-on, per machine, the longest trip at the top."""
    go = _theme.plotly()
    tally = list(reversed(tally))
    figure = go.Figure()
    # Working is teal because teal is what `took` is everywhere else here, and
    # waited-on is the **remote** outline because that colour already means
    # *another machine* on the graph. Neither is a new colour: one table.
    for name, of, fill in (
        ("working", "took_us", _theme.SERIES["took"]),
        ("waited on", "waiting_us", _theme.PALETTE["remote"][1]),
    ):
        figure.add_trace(
            go.Bar(
                x=[one[of] / 1000.0 for one in tally],
                y=[one["host"] for one in tally],
                orientation="h",
                name=name,
                marker={"color": fill, "line": {"color": fill, "width": 1.0}},
                customdata=[
                    (one["slices"], one["ran"], ", ".join(one["nodes"]) or "—", _itself(one))
                    for one in tally
                ],
                hovertemplate=(
                    f"<b>%{{y}}</b> — {name}<br>%{{x:.1f}} ms"
                    "<br>%{customdata[0]} slices, %{customdata[1]} runs"
                    "<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
    # The round trip written on the end of the bar, because the working half is
    # often microseconds against seconds and comes out invisible — which is the
    # finding, but a reader has to be able to tell *invisible* from *absent*.
    figure.add_trace(
        go.Scatter(
            x=[(one["took_us"] + one["waiting_us"]) / 1000.0 for one in tally],
            y=[one["host"] for one in tally],
            mode="text",
            text=[f"  {(one['took_us'] + one['waiting_us']) / 1000.0:,.1f} ms"
                  for one in tally],
            textposition="middle right",
            textfont={"color": _theme.MUTED, "size": 11},
            cliponaxis=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_layout(
        barmode="stack",
        **_theme.layout(
            title=_theme.titled(f"{title} — what each machine did"),
            height=max(180, 40 * len(tally) + 130),
            bargap=0.4,
        ),
    )
    # In the order they stack, so the legend reads the way the bar does. The
    # theme owns everything else about a legend; this is the one thing that is
    # this figure's.
    figure.update_layout(legend_traceorder="normal")
    # Room on the right for the total, which is written past the end of the bar
    # and would otherwise be cut off by the axis on the longest one — the very
    # row somebody is looking at.
    widest = max((one["took_us"] + one["waiting_us"]) / 1000.0 for one in tally) if tally else 1.0
    figure.update_xaxes(_theme.axis(title_text="ms", range=[0, widest * 1.22]))
    figure.update_yaxes(_theme.axis(automargin=True, showgrid=False))
    return figure


def _itself(one: "Row") -> str:
    """What the machine said about itself, for the hover.

    On the hover and not on the bar, because the bar is time and this is not:
    one fact per channel, the same rule the graph's fill obeys. A machine that
    said nothing says so — `None` is *nobody asked it*, and on this end that is
    every machine but the ones being sent work.
    """
    said = []
    if one.get("busy") is not None:
        cores = f" of {one['cores']:.0f}" if one.get("cores") else ""
        said.append(f"{one['busy']:.0%} busy{cores}")
    if one.get("memory") is not None:
        said.append(f"{one['memory']:.0%} memory")
    if one.get("up_us") is not None:
        said.append(f"up {one['up_us'] / 1e6:,.0f}s")
    return " · ".join(said) or "said nothing about itself"


def _spent(tally: Sequence["Row"], *, title: str) -> Figure:
    """Time per node, longest at the top, coloured by where it ran."""
    go = _theme.plotly()
    tally = list(reversed(tally))
    figure = go.Figure(
        go.Bar(
            x=[one["took_us"] / 1000.0 for one in tally],
            y=[one["node"] for one in tally],
            orientation="h",
            marker={
                "color": [_theme.PALETTE[_family(one)][0] for one in tally],
                "line": {
                    "color": [_theme.PALETTE[_family(one)][1] for one in tally],
                    "width": 1.2,
                },
            },
            text=[f"{one['took_us'] / 1000.0:,.1f} ms" for one in tally],
            textposition="outside",
            textfont={"color": _theme.MUTED, "size": 11},
            cliponaxis=False,
            customdata=[
                (one["ran"], one.get("recalled", 0), one["mean_us"] / 1000.0, _where(one))
                for one in tally
            ],
            hovertemplate=(
                "<b>%{y}</b><br>%{x:.1f} ms in total"
                "<br>%{customdata[0]} runs, %{customdata[1]} read back"
                "<br>%{customdata[2]:.2f} ms each<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_theme.layout(
            title=_theme.titled(f"{title} — where the time went"),
            height=max(180, 34 * len(tally) + 110),
            showlegend=False,
            bargap=0.35,
        )
    )
    figure.update_xaxes(_theme.axis(title_text="ms"))
    # No grid across the bars: the number is written on each one, so the lines
    # would only be something to read through.
    figure.update_yaxes(_theme.axis(automargin=True, showgrid=False))
    return figure


def _family(one: "Row") -> str:
    """Which row of the one table this node is drawn with.

    Where it ran, and nothing else — the same rule the graph's fill obeys. A
    node that ran on another machine is that first: which device it used over
    there is on the hover, where a second fact belongs.
    """
    if one["hosts"]:
        return "remote"
    if any(device.startswith("cuda") for device in one["devices"]):
        return "cuda"
    if any(device == "meta" for device in one["devices"]):
        return "meta"
    return "cpu"


def _where(one: "Row") -> str:
    """Where a node ran, for the hover: the hosts and devices it was seen on."""
    said = ", ".join(one["hosts"] + one["devices"])
    return said or "here"


def _said(title: str, rows: Sequence[dict[str, Any]]) -> str:
    """The title, with what the figure is actually showing beside it."""
    broke = sum(row.get("state") == "broke" for row in rows)
    said = f"{title} — {len(rows)} forward{'s' if len(rows) != 1 else ''}"
    return said + (f", {broke} broke" if broke else "")


def _number(what: Any) -> float | None:
    """A field as a number, or `None` where there was none.

    `None` and not zero: a `forward` with no loss said about it is a gap in the
    line, and plotly draws a gap. Zero would draw a step down to nothing.
    """
    try:
        return float(what)
    except (TypeError, ValueError):
        return None


def _smoothed(
    values: list[float | None],
    window: int | None = None,
) -> list[float | None]:
    """A **centred** rolling mean, ignoring the gaps.

    Centred and not trailing: a trailing mean is the same curve shifted to the
    right, and drawn on top of the raw series that shift reads as the smoothing
    disagreeing with the measurement. Nothing is being predicted here — every
    point of the run is already in hand — so there is no reason to lag it.

    The window scales with the run so that forty points and ten thousand are
    smoothed by eye to the same degree. `window=0` gives back what it was given,
    which is how you ask for no smoothing at all.

    Prefix sums, so it stays linear: a live view redraws this on every step, and
    a window of five hundred over ten thousand points done the naive way is five
    million additions a frame.
    """
    if window == 0:
        return values
    if window is None:
        window = max(1, len(values) // SMOOTH_OVER)
    if window <= 1:
        return values
    total, count = [0.0], [0]
    for value in values:
        total.append(total[-1] + (value or 0.0))
        count.append(count[-1] + (value is not None))
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        seen = count[hi] - count[lo]
        out.append((total[hi] - total[lo]) / seen if seen else None)
    return out
