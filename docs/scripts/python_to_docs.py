#!/usr/bin/env python3
"""Render `docs/python-surface.json` as one Starlight page per module.

The third generator here, and the same bargain as the other two: the source is
the truth and a committed copy of the output would drift from it. What is
different is where the truth lives — `use-cases.md` and `examples/*.ipynb` are
files, and this one is the installed package's own docstrings, dumped by
`python_surface.py` because reading them needs an interpreter with the
extension built and this runs on a runner that has neither.

So nothing here imports somatize, and nothing here writes a sentence. Every
paragraph on these pages is the one the package carries, which is what makes
the reference incapable of describing a different library — the failure legacy's
876 hand-written lines of `Filter` and `board` actually had.

Two things are decided rather than copied, and both are layout:

- **Members are grouped by shape** — constructors, then methods, then
  properties — because `dir()` order puts `Graph.forward`, the reason anybody
  is on the page, between `fingerprints` and `foreseen_json`. Nothing is hidden;
  a list of what to leave out would be a second declaration of the surface, and
  the point of generating is to have only one.
- **An inherited member is named where it is defined and linked, not repeated.**
  `.at()`, `.on()` and `.cached()` reach `Node`, `Parquet`, `Learning` and
  `Split` alike; writing them out on each would be the same 1.5 KB four times.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _output import rewrite  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent
SOURCE = DOCS / "python-surface.json"
OUT = DOCS / "src/content/docs/reference/python"
BASE = "/soma"


def page_of(module: str) -> str:
    """`somatize` → `somatize`, `somatize.health` → `health`."""
    return module.split(".", 1)[1] if "." in module else module


def anchor(text: str) -> str:
    """What Starlight's slugger makes of a heading, for the links built here."""
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).strip().replace(" ", "-")


def yaml_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def description_of(doc: str) -> str:
    for line in doc.split("\n"):
        line = line.strip()
        if not line or line.startswith(("#", "```", "---", "|", "- ", "> ")):
            continue
        line = re.sub(r"\*\*|\*|`", "", line)
        return line[:160]
    return ""


SHELL = ("python ", "python3 ", "pip ", "uv ", "cargo ", "docker ", "maturin ", "$ ")


def language(block: list[str]) -> str:
    """`bash` or `python`, from the first line that says anything.

    Three of the twenty-five examples in these docstrings are commands and the
    rest are code. Highlighting a shell line as Python paints `--listen` as an
    operator, which is a small lie told in colour.
    """
    first = next((b.strip() for b in block if b.strip()), "")
    return "bash" if first.startswith(SHELL) else "python"


def fenced(doc: str) -> str:
    """The examples in a docstring, as markdown fences.

    Twenty-two of them are introduced the way Python's own convention does — a
    line ending in `::`, then an indented block — which markdown renders as a
    paragraph with two stray colons above a block it does not highlight. Three
    more are indented with nothing announcing them at all. The example is the
    most useful half of those docstrings, so both shapes are worth the pass.

    A block already inside a fence is left alone: one docstring writes its own,
    and converting the inside of it would nest a fence in a fence. So is an
    indented line under a bullet or a quote, which is a continuation and not an
    example — the one way this pass could eat prose rather than mark it up.
    """
    lines = doc.split("\n")
    out: list[str] = []
    i, in_fence = 0, False
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        above = next((o for o in reversed(out) if o.strip()), "")
        introduced = not in_fence and line.rstrip().endswith("::")
        starts_bare = (
            not in_fence
            and not introduced
            and bool(line.strip())
            and line[:1] in " \t"
            and not (out and out[-1].strip())  # a blank line separates it
            and not above.lstrip().startswith(("-", "*", ">", "|"))
            and above[:1] not in " \t"
        )
        if not introduced and not starts_bare:
            out.append(line)
            i += 1
            continue

        if introduced:
            head = line.rstrip()[:-2].rstrip()
            if head:
                out.append(head + ":")
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
        body: list[str] = []
        while i < len(lines) and (not lines[i].strip() or lines[i][:1] in " \t"):
            body.append(lines[i])
            i += 1
        while body and not body[-1].strip():
            body.pop()
        if not body:
            continue
        margin = min(len(b) - len(b.lstrip()) for b in body if b.strip())
        out += ["", f"```{language(body)}"] + [b[margin:] for b in body] + ["```", ""]
    return "\n".join(out)


def member(qualified: str, entry: dict, depth: int) -> list[str]:
    """One callable, as a heading, a signature and its docstring.

    The signature goes in a block under the heading rather than in it: a heading
    carrying twelve keyword arguments is an anchor that changes whenever a
    default does, and every link into this page would rot on a shrug.
    """
    body = ["", f"{'#' * depth} `{qualified}`", ""]
    signature = entry.get("signature")
    if signature is not None:
        body += ["```python", f"{qualified}{signature}", "```", ""]
    if entry.get("doc"):
        body += [fenced(entry["doc"]), ""]
    return body


def inherited_note(entries: list[dict]) -> list[str]:
    """The ones defined elsewhere, named and linked in a line each."""
    by_home: dict[str, list[str]] = {}
    for entry in entries:
        by_home.setdefault(entry["inherited_from"], []).append(entry["name"])
    out: list[str] = []
    for home, names in sorted(by_home.items()):
        module, _, cls = home.rpartition(".")
        link = f"{BASE}/reference/python/{page_of(module)}/#{anchor(cls)}"
        listed = ", ".join(f"`{n}`" for n in names)
        out += ["", f"Also from [`{home}`]({link}): {listed}.", ""]
    return out


def klass(entry: dict, depth: int) -> list[str]:
    name = entry["name"]
    body = ["", f"{'#' * depth} `{name}`", ""]
    if entry["signature"] is not None:
        body += ["```python", f"{name}{entry['signature']}", "```", ""]
    else:
        # A fact, and the one a reader most needs: `Sampler()` raises
        # `TypeError: No constructor defined`. What is missing is said, because
        # an empty `()` would be an invitation to call something that throws.
        built = ", ".join(f"`{c['name']}`" for c in entry["constructors"])
        body += [
            f"Not constructed directly — {'use ' + built if built else 'handed to you'}."
            if not entry["exception"]
            else "Raised, not constructed.",
            "",
        ]
    if entry["doc"]:
        body += [fenced(entry["doc"]), ""]

    groups = [
        ("Constructors", entry["constructors"]),
        ("Methods", entry["methods"]),
        ("Properties", entry["properties"]),
    ]
    filled = [(label, items) for label, items in groups if items]
    for label, items in filled:
        own = [m for m in items if "inherited_from" not in m]
        borrowed = [m for m in items if "inherited_from" in m]
        # A label only earns a line when there is more than one group to tell
        # apart. A class with nothing but methods says so by having them.
        #
        # Bold and not a heading, which the anchor check is what taught: every
        # class on a page would contribute a `Methods`, and Starlight would hand
        # out `methods-1`, `methods-2` — anchors that renumber whenever a class
        # is added above. These are signposts, and nothing links to a signpost.
        if len(filled) > 1 and (own or borrowed):
            body += ["", f"**{label}**", ""]
        for m in own:
            body += member(f"{name}.{m['name']}", m, depth + 1)
        body += inherited_note(borrowed)
    return body


def constant(entry: dict) -> str:
    if "items" in entry:
        return f"- `{entry['name']}` — " + ", ".join(f"`{x}`" for x in entry["items"])
    return f"- `{entry['name']}` — `{entry['repr']}`"


def render(module: dict, version: str, order: int) -> str:
    name = module["name"]
    note = (
        f"Rendered from the docstrings of `somatize` {version} itself, so every "
        "signature and every paragraph below is the one the installed package "
        "carries. Nothing on this page is written twice."
    )
    body = [
        "---",
        f"title: {name}",
        f"description: {yaml_quote(description_of(module['doc']))}",
        "sidebar:",
        f"  label: {name}",
        f"  order: {order}",
        "---",
        "",
        ":::note[Generated from the package]",
        note,
        ":::",
        "",
        fenced(module["doc"]),
        "",
    ]
    if module["classes"]:
        body += ["", "## Classes", ""]
        for entry in module["classes"]:
            body += klass(entry, depth=3)
    if module["functions"]:
        body += ["", "## Functions", ""]
        for entry in module["functions"]:
            body += member(entry["name"], entry, depth=3)
    if module["constants"]:
        body += ["", "## Constants", ""]
        body += [constant(entry) for entry in module["constants"]]
    body.append("")

    # Assembled from lists that each pad themselves, so the seams double up.
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(body))
    # Two headings that slug the same would give one of them an anchor nobody
    # can link to and the other a `-1` suffix that moves when a name is added.
    anchors = [anchor(h) for h in re.findall(r"^#{2,5} `?([^`\n]+)`?$", text, re.M)]
    doubled = {a for a in anchors if anchors.count(a) > 1}
    if doubled:
        raise SystemExit(f"{name}: two headings slug the same: {sorted(doubled)}")
    return text


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(
            f"{SOURCE} is missing. It is committed; run "
            "`python docs/scripts/python_surface.py` where somatize is installed."
        )
    dump = json.loads(SOURCE.read_text())
    pages = {
        f"{page_of(module['name'])}.md": render(module, dump["version"], order)
        for order, module in enumerate(dump["modules"])
    }
    rewrite(OUT, pages)
    print(f"python surface → docs: {len(pages)} pages")


if __name__ == "__main__":
    main()
