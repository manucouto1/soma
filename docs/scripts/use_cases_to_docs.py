#!/usr/bin/env python3
"""Render `docs/use-cases.md` as one Starlight page per slice.

That file is the project's requirements document: 38 sections, one per use
case, each recording what was settled at the moment it closed and carrying the
questionnaire it had to answer. It is written as one file because it is written
by hand, in order, and a section is added by typing at the bottom. It is *read*
one slice at a time, which is a different shape.

So the pages are GENERATED, exactly as the tutorials are generated from
`examples/`: the file is the single source of truth, and a committed copy of it
would eventually disagree with the file it came from. `docs/.gitignore` covers
the output and the npm `pre*` hooks run this before dev, build and check.

Splitting is on `^## ` outside fenced code — the file has 38 of those and none
inside a fence, which is checked here rather than assumed, because a `# comment`
line inside a python block looks exactly like a heading to a regex.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _output import rewrite  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
SOURCE = DOCS / "use-cases.md"
OUT = DOCS / "src/content/docs/use-cases"
REPO = "https://github.com/manucouto1/soma/blob/main/docs/use-cases.md"

# Long enough to stay readable in a URL, short enough that a heading of
# fourteen words does not become a path nobody can type. Cut on a word.
SLUG_MAX = 56


def slug_of(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    if len(slug) <= SLUG_MAX:
        return slug
    return slug[:SLUG_MAX].rsplit("-", 1)[0]


def yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def description_of(body: str) -> str:
    """The first line of prose, with the markup taken out. Headings, fences,
    rules and list bullets are not a description of anything."""
    for line in body.split("\n"):
        line = line.strip()
        if not line or line.startswith(("#", "```", "---", "|", "- ", "> ")):
            continue
        line = re.sub(r"\*\*|\*|`", "", line)
        line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
        return line[:160]
    return ""


def split(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(everything before the first heading, [(heading, body), ...])."""
    head: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    fenced = False
    for line in text.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), []))
            continue
        (sections[-1][1] if sections else head).append(line)
    if fenced:
        raise SystemExit("use-cases.md ends inside a fenced block")
    return "\n".join(head).strip(), [(h, "\n".join(b).strip()) for h, b in sections]


def page(title: str, body: str, order: int, note: str) -> str:
    return "\n".join(
        [
            "---",
            f"title: {yaml_quote(title)}",
            f"description: {yaml_quote(description_of(body))}",
            "sidebar:",
            f"  order: {order}",
            "---",
            "",
            ":::note[Generated from the requirements document]",
            note,
            ":::",
            "",
            body,
            "",
        ]
    )


def main() -> None:
    head, sections = split(SOURCE.read_text())
    if not sections:
        raise SystemExit("use-cases.md has no `## ` sections")

    # The first section is the map of all the others, so it belongs on the page
    # somebody lands on rather than one click away from it.
    (index_title, index_body), rest = sections[0], sections[1:]
    note = (
        f"This section is part of [`docs/use-cases.md`]({REPO}), the document "
        "the project is written from. Each section records a decision at the "
        "moment it was taken: what it says is what was true when that slice "
        "closed."
    )
    pages = {
        "index.md": page("Use cases", f"{head}\n\n## {index_title}\n\n{index_body}", 0, note)
    }
    for order, (heading, body) in enumerate(rest, start=1):
        name = f"{slug_of(heading)}.md"
        if name in pages:
            raise SystemExit(f"two sections slug to {name!r}")
        pages[name] = page(heading, body, order, note)

    rewrite(OUT, pages)
    print(f"use-cases → docs: {len(pages)} pages")


if __name__ == "__main__":
    main()
