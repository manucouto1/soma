"""The `project` artifact: names and state, without code.

It is what gets sent to a worker that **already has the project**. Instead of
serializing your whole objects, what travels is:

- the **state** of each node and of the driver, with a plain `pickle` — which is
  a reference to the class by name, plus its `__dict__`. Your hyperparameters go
  in there;
- a **manifest** saying which version of the code the client expected for each
  class: `net:Filter → Filter(43b0bf6e)`.

Two nodes are forty-eight bytes. Compared with `cloudpickle`, which also puts in
the classes' bytecode and their closures, there is no contest — and in exchange
there is no coupling to the interpreter, because not one line of code travels.

## Where versioning comes in

Through `find_class`, which `pickle` calls when it finds a reference to a class.
The name is resolved against the worker's clone, its fingerprint compared against
the manifest's, and the decision made. The whole policy lives in one method and
`pickle` does the rest — including the objects nested inside the state.

## What to know before trusting it

`pickle` **does not call `__init__`** when rebuilding: it creates the object
empty and sets the state on it. A node whose `__init__` opens a connection or
loads a model has to say how that is redone in `__getstate__`/`__setstate__`,
just as it would to be saved to disk.
"""

from __future__ import annotations

import importlib
import io
import json
import pickle
import pkgutil
import struct
import sys

from soma_next._fingerprint import CannotVersion, fingerprint

__all__ = ["KIND", "DifferentVersion", "pack", "unpack"]

KIND = "project"
"""What this kind of artifact is called on the wire."""


class DifferentVersion(Exception):
    """The worker has a different version of the code than the client expected."""


class _Notes(pickle.Pickler):
    """A `Pickler` that notes down the classes of yours it leaves by reference.

    `reducer_override` is called for every object serialized, and returning
    `NotImplemented` lets `pickle` carry on: the hook is used to look, not to
    substitute.
    """

    def __init__(self, output):
        super().__init__(output)
        self.classes = {}

    def reducer_override(self, obj):
        if isinstance(obj, type):
            from soma_next._fingerprint import _is_yours

            if _is_yours(obj):
                self.classes[f"{obj.__module__}:{obj.__qualname__}"] = fingerprint(obj)
        return NotImplemented


def pack(nodes, driver=None):
    """The nodes and the driver as a `project` artifact.

    The driver is packed exactly like a node — same pickler, so the same classes
    get noted and versioned — because how it travels is not what tells them
    apart.

    Raises `CannotVersion` if some class has no source to read — a notebook, an
    `exec` — since those cannot be resolved from a clone at all.
    """
    state = io.BytesIO()
    noting = _Notes(state)
    noting.dump((nodes, driver))

    manifest = json.dumps({"classes": noting.classes}, sort_keys=True).encode()
    return struct.pack("<I", len(manifest)) + manifest + state.getvalue()


class _Resolves(pickle.Unpickler):
    """An `Unpickler` that looks each class up here and checks its version."""

    def __init__(self, input_, classes, strict, warn):
        super().__init__(input_)
        self.classes = classes
        self.strict = strict
        self.warn = warn

    def find_class(self, module, name):
        expected = self.classes.get(f"{module}:{name}")
        if expected is None:
            # Not yours: nothing to version, imported as always.
            return super().find_class(module, name)

        cls = _find(module, name)
        try:
            here = fingerprint(cls)
        except CannotVersion:
            here = f"{name}(no source)"
        if here == expected:
            return cls

        if self.strict:
            raise DifferentVersion(
                f"the client wrote the graph against `{expected}` and here there is `{here}`. "
                "Update this worker's clone, or stand it up with `--lucky` if you "
                "want it to execute whatever it has"
            )
        self.warn(f"--lucky: `{name}` here is `{here}` and the client expected `{expected}`")
        return cls


def _find(module, name):
    """The class, by the module hint the client left and failing that by
    sweeping its package — one package, because importing has effects.

    The sweep is needed because moving a class between files does not change its
    fingerprint, on purpose, so the hint stops holding while the name does not.
    """
    try:
        return _inside(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        pass

    root = module.split(".")[0]
    try:
        package = importlib.import_module(root)
    except ImportError as e:
        raise DifferentVersion(
            f"this worker does not have `{module}`, so it cannot resolve `{name}`. "
            "Either update its clone of the project, or send it the code with "
            "`mode='network'`"
        ) from e

    for found in pkgutil.walk_packages(getattr(package, "__path__", []), f"{root}."):
        try:
            candidate = _inside(importlib.import_module(found.name), name)
        except (ImportError, AttributeError):
            continue
        return candidate

    raise DifferentVersion(
        f"`{name}` is neither in `{module}` nor anywhere in `{root}` on this worker"
    )


def _inside(module, name):
    """`getattr` that understands a dotted `qualname`."""
    obj = module
    for piece in name.split("."):
        obj = getattr(obj, piece)
    return obj


def unpack(blob, strict=True, warn=None):
    """The `(nodes, driver)` of a `project` artifact, resolved against this clone.

    Raises `DifferentVersion` if `strict` and some class here is not the one the
    client expected, or if there is no way of finding it.
    """
    (length,) = struct.unpack("<I", blob[:4])
    manifest = json.loads(blob[4 : 4 + length])
    warn = warn or (lambda line: print(line, file=sys.stderr, flush=True))

    return _Resolves(
        io.BytesIO(blob[4 + length :]),
        manifest.get("classes", {}),
        strict,
        warn,
    ).load()
