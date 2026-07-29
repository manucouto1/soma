"""Soma's chart theme: a validated palette + a Plotly template.

Color is assigned by job, not by taste:

- **Categorical** (series identity): eight hues in a fixed order that
  passes colorblind-safety checks for adjacent pairs; never cycled
  past eight — extra series fold into "other".
- **Sequential** (magnitude): one hue (blue), light→dark.
- **Status** (execution state): a reserved set shared with the graph
  overlays (completed/cached/failed/running/pruned), never reused for
  ordinary series.

Everything here is plain data — importable without plotly. The template
builder is called lazily by the figure functions.
"""

from __future__ import annotations

# Categorical slots, fixed order (light-surface steps).
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# Sequential ramp (blue, light→dark) for continuous magnitude.
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

# Execution-state colors — consistent with the mermaid/graphviz overlay
# classes so a run reads the same in every rendering.
STATE_COLORS = {
    "completed": "#0ca30c",
    "cache_hit": "#2a78d6",
    "cached": "#2a78d6",
    "failed": "#d03b3b",
    "running": "#fab219",
    "pruned": "#898781",
    "pending": "#c3c2b7",
}

# Chart chrome (light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def soma_template():
    """Build (once) and return the registered "soma" Plotly template."""
    import plotly.graph_objects as go
    import plotly.io as pio

    if "soma" not in pio.templates:
        pio.templates["soma"] = go.layout.Template(
            layout={
                "colorway": CATEGORICAL,
                "font": {"family": FONT, "color": INK, "size": 13},
                "paper_bgcolor": SURFACE,
                "plot_bgcolor": SURFACE,
                "xaxis": {
                    "gridcolor": GRID,
                    "linecolor": BASELINE,
                    "zerolinecolor": BASELINE,
                    "tickcolor": BASELINE,
                    "tickfont": {"color": MUTED, "size": 12},
                    "title": {"font": {"color": INK_SECONDARY, "size": 13}},
                },
                "yaxis": {
                    "gridcolor": GRID,
                    "linecolor": BASELINE,
                    "zerolinecolor": BASELINE,
                    "tickcolor": BASELINE,
                    "tickfont": {"color": MUTED, "size": 12},
                    "title": {"font": {"color": INK_SECONDARY, "size": 13}},
                },
                "title": {"font": {"color": INK, "size": 16}},
                "legend": {
                    "font": {"color": INK_SECONDARY, "size": 12},
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "left",
                    "x": 0,
                },
                "hoverlabel": {"font": {"family": FONT, "size": 12}},
                "margin": {"l": 64, "r": 24, "t": 56, "b": 48},
            },
        )
    return pio.templates["soma"]
