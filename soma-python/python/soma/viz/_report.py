"""Self-contained HTML report for a run or study.

One file per run: manifest header, the architecture DAG annotated with
timing/cache/health, metric curves, the node-time gantt, HPO charts and
the trial table (for studies), and the health section.

**Front-end contract**: every dataset the report renders is embedded as
``<script type="application/json" id="soma-data-…">`` blobs with stable
ids (documented in ``docs design/tracking.md``). A future live UI reads
the same shapes from the run directory; this report is just the static
packaging of that contract.

Charts are Plotly figure JSON in ``soma-fig-…`` blobs, drawn by a small
bootstrap script. By default plotly.js/mermaid.js load from their CDNs
(pinned versions); ``inline=True`` embeds plotly.js from the installed
plotly package so the file works with no network — in that mode the DAG
ships as a mermaid source block instead of a rendered diagram.
"""

from __future__ import annotations

import datetime
import html
import json
import pathlib

PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.0.1.min.js"
MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

_CSS = """
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --warn: #fab219; --bad: #d03b3b; --accent: #2a78d6;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 40px 0 12px; color: var(--ink); }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
         font-size: 12px; font-weight: 600; margin-right: 6px;
         border: 1px solid var(--border); color: var(--ink-2); }
.badge.state-completed { color: #006300; border-color: #0ca30c55; }
.badge.state-failed, .badge.state-crashed { color: var(--bad); border-color: #d03b3b55; }
.badge.state-running { color: #b07600; border-color: #fab21955; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1px; background: var(--grid); border: 1px solid var(--grid);
        border-radius: 8px; overflow: hidden; margin-top: 16px; }
.meta > div { background: var(--surface); padding: 10px 14px; }
.meta .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
           color: var(--muted); margin-bottom: 2px; }
.meta .v { font-size: 13px; color: var(--ink-2); overflow-wrap: anywhere; }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 16px; margin-bottom: 16px; overflow-x: auto; }
.fig { min-height: 320px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 12px 18px; min-width: 130px; }
.tile .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
           color: var(--muted); }
.tile .v { font-size: 22px; font-weight: 600; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.04em; padding: 6px 10px;
     border-bottom: 1px solid var(--grid); position: sticky; top: 0;
     background: var(--surface); }
td { padding: 6px 10px; border-bottom: 1px solid var(--grid); color: var(--ink-2);
     font-variant-numeric: tabular-nums; white-space: nowrap; }
tr:hover td { background: #f4f4f1; }
.table-wrap { max-height: 420px; overflow: auto; }
pre.mermaid-src { background: var(--surface); padding: 16px; overflow-x: auto;
                  font-size: 12px; line-height: 1.5; margin: 0; }
.note { color: var(--muted); font-size: 12px; }
footer { margin-top: 48px; color: var(--muted); font-size: 12px; }
"""

_BOOT_JS = """
document.querySelectorAll('script[type="application/json"][id^="soma-fig-"]').forEach(el => {
  const spec = JSON.parse(el.textContent);
  const div = document.getElementById(el.id.replace('soma-fig-', 'fig-'));
  if (div && window.Plotly) {
    spec.layout = spec.layout || {};
    spec.layout.autosize = true;
    Plotly.newPlot(div, spec.data, spec.layout, {responsive: true, displaylogo: false});
  }
});
if (window.mermaid) {
  mermaid.initialize({ startOnLoad: true, theme: 'neutral' });
}
"""


def _esc(text) -> str:
    return html.escape(str(text if text is not None else "—"))


def _data_blob(blob_id: str, payload) -> str:
    body = json.dumps(payload).replace("</", "<\\/")
    return f'<script type="application/json" id="soma-data-{blob_id}">{body}</script>'


def _fig_block(name: str, fig, title: str | None = None) -> str:
    spec = fig.to_json().replace("</", "<\\/")
    heading = f"<h2>{_esc(title)}</h2>" if title else ""
    return (
        f"{heading}"
        f'<div class="card"><div class="fig" id="fig-{name}"></div></div>'
        f'<script type="application/json" id="soma-fig-{name}">{spec}</script>'
    )


def _try_fig(builder, *args, **kwargs):
    try:
        return builder(*args, **kwargs)
    except (ValueError, RuntimeError):
        return None


def _fmt_ms(ms) -> str:
    if ms is None:
        return "—"
    seconds = ms / 1000
    if seconds < 120:
        return f"{seconds:.1f}s"
    if seconds < 7200:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _trials_table(trials: list[dict], objective: str | None) -> str:
    if not trials:
        return ""
    param_names = sorted({k for t in trials for k in t["params"]})
    head = (
        "<tr><th>trial</th><th>state</th>"
        + "".join(f"<th>{_esc(p)}</th>" for p in param_names)
        + (f"<th>{_esc(objective)}</th>" if objective else "")
        + "<th>duration</th></tr>"
    )
    rows = []
    for t in trials:
        cells = [f"<td>{_esc(t['id'])}</td>", f"<td>{_esc(t['state'])}</td>"]
        for p in param_names:
            v = t["params"].get(p)
            cells.append(f"<td>{_esc(f'{v:.4g}' if isinstance(v, float) else v)}</td>")
        if objective:
            v = t["metrics"].get(objective)
            cells.append(f"<td>{_esc(f'{v:.5g}' if isinstance(v, float) else v)}</td>")
        cells.append(f"<td>{_esc(_fmt_ms(t['duration_ms']))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<h2>Trials</h2><div class='card table-wrap'><table>"
        + head
        + "".join(rows)
        + "</table></div>"
    )


def _script_tags(inline: bool, need_mermaid: bool) -> str:
    if not inline:
        tags = f'<script src="{PLOTLY_CDN}"></script>'
        if need_mermaid:
            tags += f'<script src="{MERMAID_CDN}"></script>'
        return tags
    import plotly

    bundle = (
        pathlib.Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    ).read_text()
    # plotly.min.js is MIT-licensed (header preserved inside the bundle).
    return f"<script>{bundle}</script>"


def build_report(run_view, inline: bool = False) -> str:
    """Render one run directory into a self-contained HTML string."""
    from soma import __version__
    from soma.viz import _figures, _health

    info = run_view.info
    manifest = run_view.manifest()
    overlay = run_view.overlay()
    node_timings = run_view.node_timings()
    cache = run_view.cache_activity()
    metrics = run_view.metric_series()
    flags = run_view.health_flags()
    trials = run_view.trial_timeline()

    study = None
    if (pathlib.Path(run_view.dir) / "study.json").exists():
        from soma import Study

        study = Study.load(run_view.dir)

    sections: list[str] = []
    blobs: list[str] = [
        _data_blob("info", info),
        _data_blob("manifest", manifest),
        _data_blob("overlay", overlay),
        _data_blob("node-timings", node_timings),
        _data_blob("cache", cache),
        _data_blob("metrics", metrics),
        _data_blob("health-flags", flags),
        _data_blob("trial-timeline", trials),
    ]

    # ── header ──
    git = manifest.get("git") or {}
    git_str = "—"
    if git.get("sha"):
        git_str = git["sha"][:10] + (" (dirty)" if git.get("dirty") else "")
        if git.get("branch"):
            git_str += f" · {git['branch']}"
    meta_cells = {
        "created": (info.get("created_at") or "")[:19].replace("T", " "),
        "duration": _fmt_ms(info.get("duration_ms")),
        "git": git_str,
        "host": manifest.get("hostname"),
        "soma": manifest.get("soma_version"),
        "tags": ", ".join(manifest.get("tags") or []) or "—",
    }
    sections.append(
        f"<h1>{_esc(info['name'])}</h1>"
        f'<div class="sub">{_esc(info["run_id"])}</div>'
        f'<div><span class="badge state-{_esc(info["state"])}">{_esc(info["state"])}</span>'
        f'<span class="badge">{_esc(info["kind"])}</span></div>'
        + '<div class="meta">'
        + "".join(
            f'<div><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
            for k, v in meta_cells.items()
        )
        + "</div>"
    )

    # ── architecture ──
    try:
        mermaid_src = run_view.to_mermaid()
    except RuntimeError:
        mermaid_src = None
    if mermaid_src:
        sections.append("<h2>Architecture</h2>")
        if inline:
            sections.append(
                f'<div class="card"><pre class="mermaid-src">{_esc(mermaid_src)}</pre>'
                '<div class="note">Offline report: paste this into mermaid.live to '
                "render, or regenerate without --inline.</div></div>"
            )
        else:
            sections.append(
                f'<div class="card"><pre class="mermaid">{_esc(mermaid_src)}</pre></div>'
            )

    # ── efficiency ──
    total_node_ms = sum(s["duration_ms"] or 0 for s in node_timings)
    tiles = [
        ("nodes run", len(node_timings)),
        ("node compute", _fmt_ms(total_node_ms) if node_timings else "—"),
        ("cache hits", cache["hits"]),
        ("cache misses", cache["misses"]),
    ]
    sections.append(
        "<h2>Efficiency</h2><div class='tiles'>"
        + "".join(
            f"<div class='tile'><div class='k'>{_esc(k)}</div><div class='v'>{_esc(v)}</div></div>"
            for k, v in tiles
        )
        + "</div>"
    )
    gantt = _try_fig(_figures.plot_gantt, run_view)
    if gantt:
        sections.append(_fig_block("gantt", gantt))

    # ── metrics ──
    if metrics:
        fig = _try_fig(_figures.plot_metrics, run_view)
        if fig:
            sections.append(_fig_block("metrics", fig, "Metrics"))

    # ── optimization ──
    if study is not None:
        objective = study.objectives[0][0] if study.objectives else None
        sections.append("<h2>Optimization</h2>")
        for name, builder in (
            ("history", _figures.plot_optimization_history),
            ("intermediate", _figures.plot_intermediate_values),
            ("parallel-coords", _figures.plot_parallel_coordinate),
            ("importances", _figures.plot_param_importances),
            ("timeline", _figures.plot_timeline),
            ("pareto", _figures.plot_pareto_front),
        ):
            fig = _try_fig(builder, study)
            if fig:
                sections.append(_fig_block(name, fig))
        sections.append(_trials_table(study.trials, objective))

    # ── health ──
    sections.append("<h2>Health</h2>")
    sections.append(_fig_block("health", _health.plot_health(run_view)))
    audit_fig = _try_fig(_health.plot_audit, run_view)
    if audit_fig:
        sections.append(_fig_block("audit", audit_fig))
    evolution = _try_fig(_health.plot_channel_evolution, run_view)
    if evolution:
        sections.append(_fig_block("channels", evolution))

    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    need_mermaid = bool(mermaid_src) and not inline
    return (
        "<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>soma report — {_esc(info['name'])}</title>\n"
        f"<style>{_CSS}</style>\n"
        f"{_script_tags(inline, need_mermaid)}\n"
        "</head>\n<body>\n<main>\n"
        + "\n".join(sections)
        + "\n"
        + "\n".join(blobs)
        + f"\n<footer>generated by soma {_esc(__version__)} · {generated}</footer>\n"
        f"<script>{_BOOT_JS}</script>\n"
        "</main>\n</body>\n</html>\n"
    )


def to_html(run_view, path: str | None = None, inline: bool = False) -> str:
    """Build the report; optionally write it to `path`. Returns the HTML."""
    doc = build_report(run_view, inline=inline)
    if path:
        pathlib.Path(path).write_text(doc)
    return doc


def study_to_html(study, path: str | None = None, inline: bool = False) -> str:
    """Report for a study's run directory (requires tracking enabled)."""
    from soma._runs import RunView

    run_dir = study.run_dir
    if not run_dir:
        raise ValueError("study has no run directory (tracking disabled or never run)")
    return to_html(RunView(run_dir), path=path, inline=inline)
