"""Who registers each codec this library ships with, for the side that did not
ask for it.

A codec is registered by importing whoever calls `codec(...)` — the client does
it by writing `import somatize.torch`. The **worker** never imports it: it starts
empty, and the nodes that arrive may not mention `torch` themselves while a
tensor goes past them all the same.

So the import is asked for at the one moment it is known to be needed: something
written by that codec has arrived and nothing here reads it. Not on standing up —
a worker with `torch` installed and nothing to do with it would pay two seconds
per worker to find that out.

Only this library's own codecs are in here. A codec a user registers is theirs to
import, in the process that reads it: we do not know what their `kind` means.
"""

from __future__ import annotations

import importlib

SHIPPED = {"torch.Tensor": "somatize.torch"}
"""`kind` written beside the bytes → the module whose import registers it."""


def summon(kind: str) -> None:
    """Imports whoever registers `kind`, if it is one of ours.

    Says nothing and raises nothing: whatever made this necessary is about to
    fail with the name of what is missing in front of it.
    """
    if (module := SHIPPED.get(kind)) is None:
        return
    try:
        importlib.import_module(module)
    except ImportError:
        pass
