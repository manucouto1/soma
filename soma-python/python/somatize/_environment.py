"""What a run was executed against, and the short name it goes by.

## Why this is not in a key, and is written down anyway

A fingerprint stops at what is installed: `numpy` is noted by name and
distribution version, and a standard-library module by its bare name, because
the interpreter is compared at the greeting rather than hashed into every class.
So the version of Python a value was produced under is **not** in its key, and
two interpreters can produce the same name for the same node.

That is the right call for naming and the wrong one for provenance. A store
outlives every process that wrote to it, and *what was this produced against* is
a question asked months later by somebody holding a hash and nothing else. The
key cannot answer it and never will, because a key does not run backwards.

So it is written beside the value instead: the digest on each blob, and the
reading of it once per environment, under a name anybody can `cat`.

## What goes in

What the process actually reached for — the distributions behind whatever is in
`sys.modules` — and not the whole of `pip list`. Installing something unrelated
does not move it, which is what makes two runs comparable at all.
"""

from __future__ import annotations

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


def environment() -> dict[str, str]:
    """The interpreter and the version of every distribution this reached for.

    Sorted, so two processes that ran against the same thing say it the same
    way — the same rule the fingerprint follows, and for the same reason.
    """
    import importlib.metadata as about

    said = {"python": ".".join(str(n) for n in sys.version_info[:3])}
    known = about.packages_distributions()
    for module in {top.split(".")[0] for top in sys.modules}:
        for distribution in known.get(module, ()):
            try:
                said[distribution] = about.version(distribution)
            except about.PackageNotFoundError:
                # Imported and not installed: a path entry, a source checkout.
                # Absent rather than guessed at.
                pass
    # The framework itself, which an editable install leaves out of that map,
    # and which is deliberately absent from every fingerprint. It belongs here
    # and nowhere else.
    try:
        said["somatize"] = about.version("somatize")
    except about.PackageNotFoundError:
        pass
    return dict(sorted(said.items()))


def named(said: dict[str, str]) -> str:
    """The short name an environment goes by.

    Over the JSON with sorted keys, so the name is a function of what is in it
    and not of how the dictionary was built.
    """
    written = json.dumps(said, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(written.encode()).hexdigest()[:LENGTH]
