#!/usr/bin/env python3
"""Dump the public Python surface to `docs/python-surface.json`.

The reference for Python is generated, and the reason is a measurement rather
than a preference: the package carries **110 public names and not one of them is
undocumented**, with `Trainer`'s docstring at 1.6 KB and each module's own
between 1.1 and 2.3 KB — worked example, table of flags and all. A hand-written
reference would be a second copy of prose that already exists, and legacy's was
exactly that: 876 lines describing `Filter` and `board`, an API this framework
does not have. Two records that must agree stop agreeing.

**Why this is two scripts and not one.** Reading a docstring needs the package
imported, which needs the extension built; `python_to_docs.py` runs from the npm
`pre*` hook on a runner that has nothing but a stdlib `python3`. So this half
runs where somatize is installed and writes JSON that is **committed**, and the
rendering half reads that file and imports nothing.

That makes the JSON a copy, which is the very thing this design avoids
everywhere else — so it comes with the check that prose cannot have. `--check`
re-derives it and fails on any difference, and CI runs that in the job that has
already built and installed the extension. A copy a machine compares is not the
same risk as a copy a person maintains.

    python docs/scripts/python_surface.py            # rewrite the dump
    python docs/scripts/python_surface.py --check     # fail if it is stale
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent.parent / "python-surface.json"


def modules() -> list[str]:
    """`somatize` and every submodule of it that is not private.

    Discovered rather than listed, so a tenth submodule needs no edit here. It
    does need one in the sidebar, and `check-sidebar` is what says so — which is
    the same trade the tutorials make and the right way round: a new page SHOULD
    stop the build until somebody decides where it goes.
    """
    import somatize

    found = [m.name for m in pkgutil.iter_modules(somatize.__path__)]
    return ["somatize"] + sorted(f"somatize.{n}" for n in found if not n.startswith("_"))


def is_own_doc(obj: Any) -> bool:
    """Whether a docstring belongs to the object or to its type.

    `inspect.getdoc` walks to the type when an object has none, so
    `somatize.study.DONE` — the string `"done"` — answers with the whole of
    `str.__doc__`. A constant documented with 400 lines of `str` is worse than
    one documented with nothing, so the test is whether the two are the *same*
    text.
    """
    doc = inspect.getdoc(obj)
    return bool(doc) and doc != inspect.getdoc(type(obj))


def doc_of(obj: Any) -> str:
    return inspect.getdoc(obj) or "" if is_own_doc(obj) else ""


def annotation(text: Any) -> str:
    """An annotation as somebody would type it.

    `from __future__ import annotations` leaves every annotation a string, and
    `inspect` renders a string with `repr` — so a parameter typed `Store` comes
    out `'Store'`, and one typed `'Frame'` because it was a forward reference
    comes out `"'Frame'"`. Two layers of quotes around a type nobody wrote in
    quotes. Peeled here rather than regexed out of the finished signature,
    where a quote inside a default value would be indistinguishable.
    """
    if not isinstance(text, str):
        return inspect.formatannotation(text)
    out = text.strip()
    while len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    # And the ones left *inside*, from a forward reference written mid-expression:
    # `dict[str, 'Learning']` is one type, not a type and a string. Held back
    # from a `Literal[...]`, which is the one annotation where a quote carries
    # meaning — there is none in this package, and this is what keeps the rule
    # from silently becoming wrong if one arrives.
    if "Literal[" not in out:
        out = out.replace("'", "").replace('"', "")
    return out


def signature_of(obj: Any) -> str | None:
    """The call signature as it would be typed, or `None` when there is none.

    `Sampler()` raises `TypeError: No constructor defined` — you build one with
    `Sampler.tpe(...)`. `inspect` reports that as an unknown signature, and the
    page says so rather than inventing `()`: not knowing how to construct
    something is the fact a reader most needs.

    `self` goes, and so does the `/` that PyO3 leaves behind it. A page writes
    `space.read(said)`, which is how it is called.
    """
    try:
        parameters = list(inspect.signature(obj).parameters.values())
        returns = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return None

    if parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
        if parameters and parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY:
            parameters = [p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in parameters]

    out: list[str] = []
    starred = False
    for i, p in enumerate(parameters):
        if p.kind is p.KEYWORD_ONLY and not starred:
            out.append("*")
            starred = True
        if p.kind is p.VAR_POSITIONAL:
            starred = True
        text = ("**" if p.kind is p.VAR_KEYWORD else "*" if p.kind is p.VAR_POSITIONAL else "") + p.name
        if p.annotation is not p.empty:
            text += f": {annotation(p.annotation)}"
        if p.default is not p.empty:
            text += f" = {p.default!r}" if p.annotation is not p.empty else f"={p.default!r}"
        out.append(text)
        after = parameters[i + 1] if i + 1 < len(parameters) else None
        if p.kind is p.POSITIONAL_ONLY and (after is None or after.kind is not p.POSITIONAL_ONLY):
            out.append("/")

    rendered = f"({', '.join(out)})"
    if returns is not inspect.Signature.empty:
        rendered += f" -> {annotation(returns)}"
    return rendered


def takes_self(obj: Any) -> bool:
    """Whether the first parameter is the instance.

    The one test that separates a constructor from a method across both kinds of
    class here. `Worker.at(addr, ...)` builds a worker and `Node.at(self, host)`
    places a node — same name, and only the receiver tells them apart. Reading
    `staticmethod` off the class works for the Python half and not for the PyO3
    half, where a static method is a plain builtin function.
    """
    try:
        first = next(iter(inspect.signature(obj).parameters))
    except (TypeError, ValueError, StopIteration):
        return False
    return first in ("self", "cls")


def owner_of(cls: type, name: str) -> type | None:
    """The class that actually defines `name`, walking up the MRO."""
    return next((base for base in cls.__mro__ if name in vars(base)), None)


def home_of(cls: type, name: str, documented: set[type]) -> type | None:
    """The documented class a member should be written up under.

    `.at()`, `.on()` and `.cached()` are defined on `_dsl.Topology`, which is
    private, and reached by `Node`, `Parquet`, `Learning` and `Split` alike.
    Writing each of them out four times would be the same 1.5 KB paragraph four
    times over; attributing them to the private base would send a reader to a
    page that does not exist. So the home is the **furthest documented class up
    the MRO that defines it from the same place** — `Node`, here — and the other
    three link to it.

    `None` means nothing in this package owns it: `add_note` on an exception is
    `BaseException`'s, and belongs to Python rather than here.

    Asking who defines a name rather than comparing what `getattr` hands back,
    because a `classmethod` builds a fresh bound object on every access — which
    is how `Graph.somatize`, the way a graph is made, went missing from the
    first draft of this page with nothing to say it had.
    """
    owner = owner_of(cls, name)
    if owner is None or not owner.__module__.startswith("somatize"):
        return None
    home = None
    for base in cls.__mro__:
        if base in documented and owner_of(base, name) is owner:
            home = base
    return home


def qualified(cls: type, by_class: dict[type, str]) -> str:
    return by_class[cls]


def members_of(cls: type, documented: set[type], by_class: dict[type, str]) -> dict[str, list[dict]]:
    """A class's members, split the three ways a page shows them."""
    out: dict[str, list[dict]] = {"constructors": [], "methods": [], "properties": []}
    for name, _ in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        home = home_of(cls, name, documented)
        if home is None:
            continue  # Python's, not ours.

        obj = getattr(cls, name)
        static = inspect.getattr_static(cls, name, None)
        if isinstance(static, property) or (not inspect.isroutine(obj) and inspect.isdatadescriptor(static)):
            where, entry = "properties", {"name": name, "doc": doc_of(static) or doc_of(obj)}
        elif inspect.isroutine(obj):
            where = "constructors" if not takes_self(obj) else "methods"
            entry = {"name": name, "signature": signature_of(obj), "doc": doc_of(obj)}
        else:
            continue

        if home is not cls:
            entry = {"name": name, "inherited_from": qualified(home, by_class)}
        out[where].append(entry)
    return out


def constant(name: str, obj: Any) -> dict:
    """A value rather than a callable: its type, and what it is.

    The items are kept apart from the repr when it is a tuple or list of
    strings, because `FINDINGS` and `KINDS` are vocabularies — six words a
    reader wants to read down a column, not across a line.
    """
    entry = {"name": name, "type": type(obj).__name__, "repr": repr(obj)}
    if isinstance(obj, (tuple, list)) and obj and all(isinstance(x, str) for x in obj):
        entry["items"] = list(obj)
    return entry


def surface() -> dict:
    loaded = {name: importlib.import_module(name) for name in modules()}

    # Two passes: what counts as a documented class has to be known before any
    # class is walked, because attribution runs up the MRO into other modules —
    # `somatize.data.Parquet` lands on `somatize.Node`.
    by_class: dict[type, str] = {}
    for mod_name, mod in loaded.items():
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            if inspect.isclass(obj):
                by_class.setdefault(obj, f"{mod_name}.{name}")
    documented = set(by_class)

    out = {"version": loaded["somatize"].__version__, "modules": []}
    for mod_name, mod in loaded.items():
        exported = getattr(mod, "__all__", None)
        if exported is None:
            raise SystemExit(f"{mod_name} declares no __all__; the surface has to be said, not guessed")

        entry: dict[str, Any] = {
            "name": mod_name,
            "doc": inspect.getdoc(mod) or "",
            "classes": [],
            "functions": [],
            "constants": [],
        }
        undocumented = []
        for name in exported:
            obj = getattr(mod, name)
            if inspect.isclass(obj):
                entry["classes"].append(
                    {
                        "name": name,
                        # `-> None` on a class is `__init__`'s return showing
                        # through. What a constructor returns is the class.
                        "signature": (signature_of(obj) or "").removesuffix(" -> None") or None,
                        "doc": doc_of(obj),
                        "exception": issubclass(obj, BaseException),
                        **members_of(obj, documented, by_class),
                    }
                )
            elif inspect.isroutine(obj):
                entry["functions"].append(
                    {"name": name, "signature": signature_of(obj), "doc": doc_of(obj)}
                )
            else:
                entry["constants"].append(constant(name, obj))
                continue
            if not doc_of(obj):
                undocumented.append(name)

        # The bar this generation rests on. A public name with no docstring
        # renders as a heading with nothing under it, which is the shape a wrong
        # reference starts as. It has never happened — 110 for 110 — and this is
        # what keeps it that way.
        if undocumented:
            raise SystemExit(f"{mod_name}: exported with no docstring: {undocumented}")
        out["modules"].append(entry)
    return out


def main() -> None:
    fresh = json.dumps(surface(), indent=2, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        if not OUT.exists():
            raise SystemExit(f"{OUT} does not exist; run this script without --check")
        if OUT.read_text() != fresh:
            raise SystemExit(
                f"{OUT.name} is stale: the installed somatize no longer matches it.\n"
                f"Run `python docs/scripts/python_surface.py` and commit the result."
            )
        print(f"python surface OK — {OUT.name} matches the installed somatize")
        return
    OUT.write_text(fresh)
    counted = sum(
        len(m["classes"]) + len(m["functions"]) + len(m["constants"])
        for m in json.loads(fresh)["modules"]
    )
    print(f"python surface → {OUT.name}: {len(json.loads(fresh)['modules'])} modules, {counted} names")


if __name__ == "__main__":
    main()
