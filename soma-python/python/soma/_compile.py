"""Readable ``g.compile()`` output.

``Graph.compile`` returns a :class:`CompileInfo` — still a plain dict
(``info["plan_text"]``, ``info["diagnostics"]`` keep working) — that
renders in notebooks as summary tiles, per-level diagnostic callouts,
and the execution plan drawn as an SVG diagram.
"""

from __future__ import annotations

import html

from soma._soma import Graph as _RustGraph

_LEVEL_STYLES = {
    # (border, chip background, chip ink) — same family as the report.
    "warning": ("#fab219", "#fff8e1", "#b07600"),
    "error": ("#d03b3b", "#ffebee", "#b71c1c"),
    "info": ("#c3c2b7", "#f4f4f1", "#52514e"),
}


class CompileInfo(dict):
    """``g.compile()`` result: a dict that displays itself."""

    def _repr_html_(self) -> str:
        tiles = "".join(
            f"<div style='border:1px solid rgba(11,11,11,0.10);border-radius:8px;"
            f"padding:8px 16px;min-width:110px'>"
            f"<div style='font-size:11px;text-transform:uppercase;color:#898781'>{label}</div>"
            f"<div style='font-size:20px;font-weight:600'>{self.get(key, '—')}</div></div>"
            for key, label in (
                ("total_nodes", "nodes"),
                ("cached_nodes", "cached"),
                ("parallel_branches", "parallel branches"),
            )
        )

        callouts = []
        for diag in self.get("diagnostics", []):
            if not isinstance(diag, dict):  # tolerate legacy string form
                callouts.append(
                    f"<div style='margin:6px 0;color:#52514e'>{html.escape(str(diag))}</div>"
                )
                continue
            level = diag.get("level", "info")
            border, chip_bg, chip_ink = _LEVEL_STYLES.get(level, _LEVEL_STYLES["info"])
            callouts.append(
                f"<div style='border-left:3px solid {border};padding:6px 12px;"
                f"margin:6px 0;background:#fcfcfb;border-radius:0 6px 6px 0'>"
                f"<span style='background:{chip_bg};color:{chip_ink};font-size:11px;"
                f"font-weight:600;padding:1px 8px;border-radius:999px;"
                f"text-transform:uppercase'>{html.escape(level)}</span> "
                f"<code style='font-size:12px'>{html.escape(diag.get('node', ''))}</code>"
                f"<div style='margin-top:3px;color:#0b0b0b'>"
                f"{html.escape(diag.get('message', ''))}</div></div>"
            )
        diagnostics = (
            "<div style='margin-top:10px'>" + "".join(callouts) + "</div>"
            if callouts
            else "<div style='margin-top:10px;color:#898781'>no diagnostics</div>"
        )

        plan_svg = self.get("plan_svg", "")
        plan_text = html.escape(self.get("plan_text", ""))
        plan = (
            f"<div style='margin-top:10px'>{plan_svg}</div>"
            f"<details style='margin-top:6px'><summary style='cursor:pointer;"
            f"color:#898781;font-size:12px'>plan como texto</summary>"
            f"<pre style='font-family:ui-monospace,monospace;font-size:12px;"
            f"background:#fcfcfb;padding:10px;border-radius:6px'>{plan_text}</pre>"
            f"</details>"
        )

        return (
            "<div style='font-family:system-ui;font-size:13px'>"
            f"<div style='display:flex;gap:10px;flex-wrap:wrap'>{tiles}</div>"
            f"{diagnostics}{plan}</div>"
        )


_rust_compile = _RustGraph.compile


def _compile(self, mode: str = "inference") -> CompileInfo:
    return CompileInfo(_rust_compile(self, mode=mode))


_compile.__doc__ = _rust_compile.__doc__
_RustGraph.compile = _compile
