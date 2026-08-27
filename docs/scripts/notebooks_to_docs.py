#!/usr/bin/env python3
"""Render the executed notebooks as Starlight pages.

A notebook is JSON, so this needs no nbconvert and the docs build gains no
dependency — python3 is already on every runner that builds this site.

The pages are GENERATED, not committed: `notebooks/` is the single source of
truth, and a page that could drift from its notebook would eventually drift.
`docs/.gitignore` covers both output directories, and the npm `pre*` hooks
run this before dev, build and check alike.

Figures are stored twice in these notebooks — Plotly for a live kernel, PNG
so they also render on GitHub. A static site takes the PNG, which is why
this can be a pure-stdlib text transform.
"""

from __future__ import annotations

import json
import re
import sys
from base64 import b64decode
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _output import rewrite  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
NOTEBOOKS = DOCS.parent / "examples"
OUT = DOCS / "src/content/docs/tutorials"
ASSETS = DOCS / "src/assets/notebooks"
# From src/content/docs/tutorials/<page>.md up to src/, where assets live.
# Astro resolves and optimizes relative image references in a collection.
ASSETS_REL = "../../../assets/notebooks"
REPO = "https://github.com/manucouto1/soma/blob/main/examples"

# First match wins. PNG before the Plotly JSON that sits beside it; HTML
# before text/plain, because `_repr_html_` is the interesting one (the SVG
# diagrams and the run tables both arrive as text/html).
MIME_ORDER = ("image/png", "image/svg+xml", "text/html", "text/plain")


def slug_of(path: Path) -> str:
    return path.stem.replace("_", "-")


def yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def source_of(cell: dict) -> str:
    src = cell.get("source", "")
    return ("".join(src) if isinstance(src, list) else src).rstrip()


def payload_of(data: dict, mime: str) -> str:
    value = data[mime]
    return "".join(value) if isinstance(value, list) else value


def title_and_body(cells: list[dict], number: str) -> tuple[str, str, int]:
    """Pull the H1 out of the first markdown cell — Starlight renders the
    frontmatter title as the page heading, so leaving it would duplicate it.
    Returns (title, description, index of the consumed cell)."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        lines = source_of(cell).split("\n")
        for j, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                # The notebook numbers itself with one digit and the file
                # with two; the sidebar reads ragged unless they agree, so the
                # file's number wins and a title with none is given one.
                title = re.sub(r"^\d{1,2}\s*[—-]\s*", "", title)
                title = f"{number} — {title}"
                rest = [x for x in lines[j + 1 :] if x.strip()]
                desc = re.sub(r"[*`\[\]]|\(http[^)]*\)", "", rest[0]).strip() if rest else ""
                desc = desc.rstrip(":").rstrip()
                del lines[j]
                cell["source"] = "\n".join(lines)
                return title, desc[:160], i
    raise SystemExit(f"notebook has no H1: {number}")


def render_outputs(
    cell: dict, slug: str, index: int, assets: dict[str, bytes]
) -> list[str]:
    out: list[str] = []
    stream = ""
    for k, output in enumerate(cell.get("outputs", [])):
        kind = output.get("output_type")
        if kind == "stream":
            stream += payload_of(output, "text")
            continue
        if kind == "error":
            trace = "\n".join(output.get("traceback", []))
            out.append("```text\n" + re.sub(r"\x1b\[[0-9;]*m", "", trace) + "\n```")
            continue
        data = output.get("data", {})
        mime = next((m for m in MIME_ORDER if m in data), None)
        if mime is None:
            continue
        if mime == "image/png":
            name = f"cell-{index:02d}-{k}.png"
            assets[f"{slug}/{name}"] = b64decode(payload_of(data, mime))
            out.append(f"![Figure from cell {index}]({ASSETS_REL}/{slug}/{name})")
        else:
            # Raw SVG and `_repr_html_` pass through: these files are .md,
            # not .mdx, so braces in an inline style are not expressions.
            # The blank lines are what keeps markdown from wrapping it in a
            # paragraph.
            body = payload_of(data, mime).strip()
            if mime == "text/plain":
                out.append("```text\n" + body + "\n```")
            else:
                out.append('<div class="nb-output">\n\n' + body + "\n\n</div>")
    if stream.strip():
        out.insert(0, "```text\n" + stream.rstrip() + "\n```")
    return out


def convert(path: Path, assets: dict[str, bytes]) -> tuple[str, str]:
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]
    slug = slug_of(path)
    number = path.stem[:2]
    title, description, consumed = title_and_body(cells, number)

    parts = [
        "---",
        f"title: {yaml_quote(title)}",
        f"description: {yaml_quote(description)}",
        "---",
        "",
        ":::note[Generated from a notebook]",
        f"This page is [`examples/{path.name}`]({REPO}/{path.name}), executed, "
        "with its outputs. Run it yourself with "
        "`jupyter lab examples/` after `pip install 'somatize[viz]'`. "
        "Interactive Plotly figures show here as the PNG the notebook also stores.",
        ":::",
        "",
    ]
    for i, cell in enumerate(cells):
        body = source_of(cell)
        if cell["cell_type"] == "markdown":
            if body:
                parts.append(body)
        elif cell["cell_type"] == "code":
            if body:
                parts.append("```python\n" + body + "\n```")
            parts.extend(render_outputs(cell, slug, i, assets))
    return slug, "\n\n".join(parts) + "\n"


def main() -> None:
    pages: dict[str, str | bytes] = {}
    assets: dict[str, bytes] = {}
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        slug, page = convert(path, assets)
        pages[f"{slug}.md"] = page

    rewrite(OUT, pages)
    rewrite(ASSETS, dict(assets))
    print(f"notebooks → docs: {len(pages)} pages, {len(assets)} figures")


if __name__ == "__main__":
    main()
