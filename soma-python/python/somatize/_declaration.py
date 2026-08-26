"""What a node was built with: `Embed(dim=512, dropout=0.1)`, and its digest.

`Embed(512)` and `Embed(64)` are one class, one identity and **two different
answers**. Without this they share a name in a store and the second is handed the
first one's — no error, no warning, the wrong tensor.

The declaration is **what was handed to `__init__`**, captured there, and not
what the object turns out to be holding later: a node that counts its calls or
caches a client has attributes that move **while it runs**. Bound against the
signature with defaults filled in, so `Layer(64, 32)` and `Layer(in_=64, out=32)`
are one declaration.

Why this is in a key and the code's fingerprint is not: saying `Embed(512)` is a
**decision** readable in a microsecond, while editing a `forward` is not a
decision about this node and costs an AST to read. There is nothing to compare on
here anyway — there is no hit, there is a **collision**.

A key is computed on the client and **again** on a worker, so two failures, and
they are not symmetric. **Unstable** — one declaration, two texts, an address —
costs a cache that misses forever and never says why. **Lossy** — two
declarations, one text, a truncated tensor or a scrubbed `<Helper>` — costs **the
wrong value served in silence**. So neither is accepted: what cannot be written
faithfully *and* identically twice raises `CannotDeclare`.

What catches an address is **not** a test on the type: a **list** of
address-bearing objects has `list.__repr__`, which is defined. So containers are
walked rather than repr'd, a trusted `__repr__` has its text checked afterwards,
and a `set` is sorted first, its repr order depending on hash randomisation.
A type whose `repr` is faithful and still says the wrong thing — a `Store`,
which answers *where* — declares itself with `__soma_declared__`. Beyond that,
the answer is `salt=`.
"""

from __future__ import annotations

import enum
import functools
import hashlib
import inspect
import re
import types
from typing import Any, Callable, Iterable, NoReturn

__all__ = ["CannotDeclare", "digest", "remember_what_built_it", "written"]

BUILT_WITH = "_soma_built_with"
"""Where the arguments are kept on the instance. One name, so a subclass calling
up to its base does not overwrite what the subclass was called with."""

DECLARED = "__soma_declared__"
"""How a type says it is written down, when that is not what its `repr` says.

For the things whose `repr` answers **where** rather than **what**. A `Store`
is the case this exists for: `Store(/mnt/data/runs)` is a faithful text and a
stable one, so the rule below trusts it — and then the same bytes in two
directories name everything differently, and moving a store loses every hit it
had. Which is the failure the address rule stops, one rung further out: a path
is not the run, it is the machine.

Read off the **type** and not the instance, since it is a statement about the
kind: nobody declares a location per object.
"""

DEEP = 8
"""How far in to follow what a node holds before giving up. A declaration that
nests deeper than this is one nobody is reading either."""

AN_ADDRESS = re.compile(r"\bat 0x[0-9a-fA-F]+")
"""How CPython writes a thing it cannot write: an object, a function, a method,
a module. The address is the run, so anything carrying one is not the same text
twice."""

PLAIN = (type(None), bool, int, float, complex, str, bytes, bytearray)
"""What is its own written form already."""

CALLABLE = (types.FunctionType, types.BuiltinFunctionType)
"""What is named rather than written: what it *does* is the fingerprint's, the
same way a class is noted by name and not by its source."""


class CannotDeclare(Exception):
    """This node holds something that cannot be written down faithfully and the
    same way in another process. Says which attribute, because *some node holds
    something* is not an answer when it holds twenty."""


def remember_what_built_it(cls: type[Any]) -> None:
    """Wraps a class's `__init__` so every instance keeps what it was built with.
    `somatize.Node.__init_subclass__` calls it, so a user writes nothing. Only
    the **outermost** `__init__` records: a subclass calling up to its base would
    otherwise have the base's empty arguments overwrite its own.
    """
    built = cls.__init__
    if getattr(built, "_soma_remembers", False):
        return

    @functools.wraps(built)
    def remembering(self: Any, *args: Any, **said: Any) -> Any:
        try:
            if BUILT_WITH not in self.__dict__:
                object.__setattr__(self, BUILT_WITH, _bound(built, args, said))
        except AttributeError:
            # A node with `__slots__` has nowhere to keep it. It falls through
            # to being read off its attributes, like anything else that is not
            # a node.
            pass
        return built(self, *args, **said)

    # `setattr` and not `remembering._soma_remembers = True`: a function object
    # takes any attribute at run time and a checker knows only the ones on
    # `Callable`, so the assignment reads as an error where the intent is a mark.
    setattr(remembering, "_soma_remembers", True)
    cls.__init__ = remembering


def _bound(
    built: Callable[..., Any],
    args: tuple[Any, ...],
    said: dict[str, Any],
) -> dict[str, Any]:
    """The arguments by the name each was declared under, defaults included. A
    signature that will not take them is not this module's to report — the call
    is about to fail with the error the user needs — but it must not lose them,
    so the raw call is kept as a fallback.
    """
    try:
        binding = inspect.signature(built).bind(None, *args, **said)
    except (TypeError, ValueError):
        return {"*": args, "**": said}
    binding.apply_defaults()
    taken = dict(list(binding.arguments.items())[1:])
    # An empty `*rest` or `**said` is nothing, and a class with no `__init__` of
    # its own binds against `object.__init__(*args, **kwargs)` and is *all*
    # nothing. Writing that down says `Head(args=(), kwargs={})`, which is noise
    # in the one place somebody reads to find out why a key moved.
    spread = {
        name
        for name, one in inspect.signature(built).parameters.items()
        if one.kind in (one.VAR_POSITIONAL, one.VAR_KEYWORD)
    }
    return {
        name: one
        for name, one in taken.items()
        if name not in spread or one
    }


def digest(obj: object) -> str:
    """The declaration, hashed. The whole sha256 and not a prefix of it: this
    goes **into a key**, and a truncated digest is a collision waiting for a
    store big enough."""
    return hashlib.sha256(written(obj).encode()).hexdigest()


def written(obj: object) -> str:
    """`Embed(dim=512, dropout=0.1)` — what this node was built with, as text.

    Public because when a key moves and nobody knows why, this is the answer.
    Raises `CannotDeclare` naming the attribute that could not be written.
    """
    return _written(obj, type(obj).__name__, DEEP, frozenset())


def _written(value: object, where: str, deep: int, seen: frozenset[int]) -> str:
    """One value, at `where`, with `deep` levels left and `seen` holding what is
    already being written, so a cycle is refused rather than followed.

    The order of the rungs is the design: anything that wears angle brackets and
    is nonetheless stable comes out before the rule that refuses them, and
    containers before the rule that trusts a `__repr__`.
    """
    if isinstance(value, PLAIN):
        return repr(value)
    built = value.__dict__.get(BUILT_WITH) if hasattr(value, "__dict__") else None
    if built is not None:
        inside = ", ".join(
            f"{name}={_written(one, f'{where}.{name}', deep - 1, seen | {id(value)})}"
            for name, one in sorted(built.items())
        )
        return f"{type(value).__name__}({inside})"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, type):
        return value.__qualname__
    if isinstance(value, CALLABLE):
        named = getattr(value, "__qualname__", None)
        # A lambda has a name and it is everybody's. Two of them written down
        # the same way is the collision this module exists to stop, so the one
        # callable that cannot be named is refused rather than named badly.
        if not named or named.endswith("<lambda>"):
            return _refuse(
                where,
                "is a lambda, and every lambda is written down under the same "
                "name: give it a `def`, or say `salt=`",
            )
        return named
    if isinstance(value, types.MethodType):
        # Bound, so which instance it is bound to is half of which method it is.
        held = _written(value.__self__, f"{where}.__self__", deep - 1, seen)
        return f"{held}.{value.__func__.__name__}"

    if deep <= 0:
        return _refuse(where, f"is nested deeper than {DEEP} levels")
    if id(value) in seen:
        return _refuse(where, "holds itself")
    seen = seen | {id(value)}

    # Before its repr is trusted: something with a shape and a dtype is data,
    # and every library writes data with an ellipsis in the middle once it is
    # big — two different tensors, one text.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _refuse(
            where,
            f"is a {type(value).__name__}, which is data and not a declaration: "
            "settle it with `somatize.torch.freeze`",
        )

    if isinstance(value, (list, tuple)):
        each = _each(value, where, deep, seen)
        if not isinstance(value, tuple):
            return f"[{', '.join(each)}]"
        # As Python writes one: `()`, `(1,)`, `(1, 2)`. A tuple written `(,)`
        # is not something anybody can paste back.
        return f"({each[0]},)" if len(each) == 1 else f"({', '.join(each)})"
    if isinstance(value, (set, frozenset)):
        # Sorted on the text each item wrote, because a set has no order of its
        # own and the one its repr shows is the process's.
        inside = ", ".join(sorted(_each(value, where, deep, seen)))
        return f"{type(value).__name__}{{{inside}}}"
    if isinstance(value, dict):
        # `pairs` and not `written`: that name is this module's own function,
        # and rebinding it here made the two impossible to tell apart.
        pairs = [
            (
                _written(name, f"{where}[{name!r}]", deep - 1, seen),
                _written(one, f"{where}[{name!r}]", deep - 1, seen),
            )
            for name, one in value.items()
        ]
        # Sorted too, and on the key alone: a mapping built in another order is
        # the same mapping, and two values are not always comparable.
        inside = ", ".join(
            f"{name}: {one}" for name, one in sorted(pairs, key=_first)
        )
        return f"{{{inside}}}"

    # Its own say before its repr, and checked all the same: a type saying how
    # it is declared is still a `__repr__` somebody wrote.
    own = getattr(type(value), DECLARED, None)
    if own is not None:
        return _trusted(own(value), where)

    if type(value).__repr__ is not object.__repr__:
        return _trusted(repr(value), where)

    # `attributes` and not `held`: that name is the bound method's `__self__`
    # further up, and one name for two things is what made this branch
    # unreadable.
    attributes = getattr(value, "__dict__", None)
    if attributes is None:
        return _refuse(
            where,
            f"is a {type(value).__name__}, which has neither a `__repr__` of its "
            "own nor attributes to read",
        )
    inside = ", ".join(
        f"{name}={_written(one, f'{where}.{name}', deep - 1, seen)}"
        for name, one in sorted(attributes.items())
    )
    return f"{type(value).__name__}({inside})"


def _each(
    values: Iterable[Any],
    where: str,
    deep: int,
    seen: frozenset[int],
) -> list[str]:
    """Every item of a container, in order, each knowing where it sits."""
    return [
        _written(one, f"{where}[{at}]", deep - 1, seen) for at, one in enumerate(values)
    ]


def _first(pair: tuple[str, str]) -> str:
    """The key a mapping's items are sorted on."""
    return pair[0]


def _trusted(text: str, where: str) -> str:
    """A `__repr__` somebody else wrote, checked rather than believed. An address
    is the run; angle brackets are CPython saying *this has no faithful repr*,
    and the stable ones that wear them were taken out above.
    """
    if AN_ADDRESS.search(text):
        return _refuse(
            where,
            "writes itself with the address it happens to live at, which is a "
            "different text in every process",
        )
    if text.startswith("<") and text.endswith(">"):
        return _refuse(
            where, f"writes itself as `{text}`, which says nothing to compare"
        )
    return text


def _refuse(where: str, why: str) -> NoReturn:
    """`NoReturn`, which is what makes `return _refuse(...)` legal inside a
    function that answers `str`: it never comes back, so there is nothing to
    return of the wrong type."""
    raise CannotDeclare(f"`{where}` {why}")
