"""What a run was executed against, and the short name it goes by.

A fingerprint stops at what is installed — the interpreter is compared at the
greeting rather than hashed into every class — so the version of Python a value
was produced under is **not** in its key.

That is the right call for naming and the wrong one for provenance. A store
outlives every process that wrote to it, and *what was this produced against* is
a question asked months later by somebody holding a hash and nothing else; a key
does not run backwards. So it is written beside the value instead: the digest on
each blob, and the reading of it once per environment, under a name anybody can
`cat`.

What goes in is what the process actually reached for — the distributions behind
whatever is in `sys.modules` — and not the whole of `pip list`, so installing
something unrelated does not move it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys

__all__ = ["WHERE", "environment", "named"]

WHERE = "env"
"""The prefix a reading is filed under: `env/<digest>`.

Text and a constant rather than a formatting call at each site, because whoever
reads this store back is another tool and this is the whole of the agreement.
"""

LENGTH = 12
"""How much of the sha256 names an environment. Longer than a fingerprint's
eight: this is a name in a store shared by an entire investigation, where a
collision is two runs quietly filed as one."""


@functools.cache
def _installed() -> dict[str, list[str]]:
    """Which distribution each importable top-level name comes from.

    A reading of what is **installed**, which does not change while a process
    runs, and the whole cost of `environment()`: the scan walks the metadata of
    every distribution in the environment and takes ~350 ms where torch is one
    of them. Since a reading is written on every `forward` that has a store
    behind it, scanning once is the difference between a label and a toll —
    it was 358 ms a forward, and it drowned what CU24 measured.

    Cached and not read once at import: an environment nobody writes down
    should not pay for one.
    """
    import importlib.metadata as about

    return about.packages_distributions()


@functools.cache
def _version(distribution: str) -> str | None:
    """What that distribution says its version is, or `None` if it says nothing.

    Read from its metadata on disk, and kept for the same reason as the scan:
    it is a fact about what is **installed**, which a running process does not
    change. Eighteen of these is 5 ms, and a reading is written on every
    `forward` that has a store behind it — small enough to look free and paid
    often enough not to be.

    `None` and not an exception, so the caller stays a dictionary comprehension
    and the absence is cached too: imported and not installed — a path entry, a
    source checkout — is a stable answer, not a failure to retry.
    """
    import importlib.metadata as about

    try:
        return about.version(distribution)
    except about.PackageNotFoundError:
        return None


def environment() -> dict[str, str]:
    """The interpreter and the version of every distribution this reached for.

    Sorted, so two processes that ran against the same thing say it the same
    way — the same rule the fingerprint follows, and for the same reason.

    `sys.modules` is read afresh every time even though the scan behind it
    is kept, so something imported after the first reading is still in the
    next one.
    """
    said = {"python": ".".join(str(n) for n in sys.version_info[:3])}
    known = _installed()
    for module in {top.split(".")[0] for top in sys.modules}:
        for distribution in known.get(module, ()):
            # Imported and not installed: a path entry, a source checkout.
            # Absent rather than guessed at.
            if (version := _version(distribution)) is not None:
                said[distribution] = version
    # The framework itself, which an editable install leaves out of that map,
    # and which is deliberately absent from every fingerprint. It belongs here
    # and nowhere else.
    if (mine := _version("somatize")) is not None:
        said["somatize"] = mine
    return dict(sorted(said.items()))


def named(said: dict[str, str]) -> str:
    """The short name an environment goes by.

    Over the JSON with sorted keys, so the name is a function of what is in it
    and not of how the dictionary was built.
    """
    written = json.dumps(said, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(written.encode()).hexdigest()[:LENGTH]
