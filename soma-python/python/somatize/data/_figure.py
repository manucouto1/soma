"""What the model leaned on, drawn.

    from somatize.data import contribution, leaned

    leaned(contribution(g, batches, objective=mse))

One bar per input, as a **share** of what all of them together are worth, so
the picture answers *how much of what matters is this* rather than a number in
units nobody remembers. What tripped is in the alarm colour, with the flag on
the hover — this and `somatize.health` are the only figures here where a
colour is allowed to mean bad, because they are the only ones drawing opinions.

A **negative** share is drawn and not clamped. It means the model does better
without that input, which is a real and useful thing to find out, and hiding it
would be the figure being tidier than the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from somatize._typing import Figure

if TYPE_CHECKING:
    from somatize._somatize import Thresholds

from somatize import _theme
from somatize._somatize import about
from somatize.data._ablation import leaning, shares

__all__ = ["leaned"]


def leaned(
    drops: dict[str, float],
    *,
    thresholds: "Thresholds | None" = None,
    title: str = "what it leaned on",
) -> Figure:
    """The share each input was worth, biggest first."""
    go = _theme.plotly()
    said = shares(drops, thresholds)
    flags = leaning(drops, thresholds)
    if not said:
        return _nothing(go, "nothing was ablated, so nothing can be said")

    order = sorted(said, key=lambda one: said[one])
    return go.Figure(
        go.Bar(
            x=[said[one] for one in order],
            y=order,
            orientation="h",
            marker={
                "color": [
                    _theme.SERIES["alarm"] if flags.get(one) else _theme.SERIES["took"]
                    for one in order
                ],
                "line": {"color": _theme.EDGE, "width": 1},
            },
            text=[f"{said[one]:.0%}" for one in order],
            textposition="outside",
            textfont={"color": _theme.MUTED, "size": 11},
            cliponaxis=False,
            customdata=[
                (
                    ", ".join(flags.get(one, [])) or "nothing tripped",
                    drops[one],
                )
                for one in order
            ],
            hovertemplate=(
                "<b>%{y}</b><br>%{x:.1%} of what matters"
                "<br>the score is %{customdata[1]:.4g} worse without it"
                "<br>%{customdata[0]}<extra></extra>"
            ),
        )
    ).update_layout(
        **_theme.layout(
            title=_theme.titled(title),
            height=110 + 40 * len(order),
            showlegend=False,
            bargap=0.4,
        )
    ).update_xaxes(
        **_theme.axis(title_text="share of what matters", tickformat=".0%")
    ).update_yaxes(**_theme.axis(showgrid=False, automargin=True))


def _nothing(go: Any, what: str) -> Figure:
    """Nothing to draw is a statement and not an exception."""
    return go.Figure().update_layout(
        annotations=[
            {
                "x": 0.5, "y": 0.5, "xref": "paper", "yref": "paper",
                "text": what, "showarrow": False,
                "font": {"size": 12, "color": _theme.MUTED, "family": _theme.FONT},
            }
        ],
        **_theme.layout(xaxis={"visible": False}, yaxis={"visible": False}, height=140),
    )


def _about(flag: str) -> str:
    """What a flag means, for whoever composes something else out of this."""
    return about(flag)
