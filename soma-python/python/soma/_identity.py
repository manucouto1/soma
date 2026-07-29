"""Deterministic filter identity for cache keys.

A filter's cache identity must be stable across processes and machines,
and must change when the filter's behavior changes. It is built from:

- the **qualified class name** (module + qualname — two ``Model`` classes
  in different modules must never collide),
- the **canonical config**: public instance attributes merged over
  search-descriptor defaults, JSON-encoded with sorted keys,
- a **code fingerprint**, resolved through a soundness ladder:

  1. an explicit ``_cache_version`` class attribute (soundest — bump it
     to invalidate; survives refactors and helper-module edits),
  2. a hash of the class source text via ``inspect.getsource`` (default —
     editing the class invalidates; edits to helpers in other modules do
     NOT, declare those with ``_cache_version``),
  3. a cloudpickle hash with a loud :class:`UserWarning` (last resort —
     not stable across Python/cloudpickle versions).

Unhashable configs raise :class:`CacheConfigError` — "uncacheable" is
always an explicit, visible state, never a silent random key (the
HuggingFace-datasets silent-random-fingerprint failure mode).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import textwrap
import warnings


class CacheConfigError(TypeError):
    """A filter attribute cannot participate in the cache key.

    Fix by prefixing the attribute with ``_`` (excluded from identity),
    or by defining ``__soma_config__(self) -> dict`` returning the
    JSON-serializable subset that defines the filter's behavior.
    """


def _encode_attr(cls_name: str, name: str, value) -> str:
    def np_scalar(o):
        # numpy scalars (np.float64, np.int32, ...) → native Python.
        if type(o).__module__ in ("numpy", "numpy.core.multiarray") and hasattr(o, "item"):
            return o.item()
        raise TypeError(type(o).__qualname__)

    try:
        return json.dumps(value, sort_keys=True, default=np_scalar)
    except (TypeError, ValueError) as e:
        raise CacheConfigError(
            f"attribute {name!r} of {cls_name!r} (type {type(value).__qualname__!r}) "
            f"cannot enter the cache key: {e}. Prefix it with '_' to exclude it, or "
            f"define __soma_config__() returning a JSON-serializable dict."
        ) from None


def canonical_config_json(obj) -> str:
    """Public attrs merged over search-descriptor defaults, sorted keys.

    An unset searchable parameter still shapes behavior through its
    default, so it must be part of the identity — a filter constructed
    with ``lr=0.1`` and one relying on ``lr = search(default=0.1)``
    hash identically, as they should.
    """
    cls = type(obj)

    hook = getattr(obj, "__soma_config__", None)
    if callable(hook):
        config = dict(hook())
    else:
        config = {}
        try:
            from soma.search import SearchDescriptor

            for klass in reversed(cls.__mro__):
                for name, attr in vars(klass).items():
                    if isinstance(attr, SearchDescriptor) and not name.startswith("_"):
                        config[name] = attr.default
        except ImportError:  # pragma: no cover — search always importable in-tree
            pass
        for name, value in vars(obj).items():
            if not name.startswith("_"):
                config[name] = value

    encoded = {name: _encode_attr(cls.__qualname__, name, value) for name, value in config.items()}
    return "{" + ",".join(f"{json.dumps(k)}:{encoded[k]}" for k in sorted(encoded)) + "}"


def code_fingerprint(cls) -> tuple[str, str]:
    """Resolve the code-identity ladder. Returns ``(method, digest)``."""
    version = getattr(cls, "_cache_version", None)
    if version is not None:
        return "cache_version", str(version)

    try:
        source = textwrap.dedent(inspect.getsource(cls))
        return "source", hashlib.sha256(source.encode()).hexdigest()
    except (OSError, TypeError):
        pass

    try:
        import cloudpickle

        blob = cloudpickle.dumps(cls)
    except Exception as e:  # noqa: BLE001 — converted to a typed error
        raise CacheConfigError(
            f"cannot fingerprint {cls.__qualname__!r}: source unavailable and "
            f"cloudpickle failed ({e}). Set `_cache_version = \"...\"` on the class."
        ) from None
    warnings.warn(
        f"soma: cannot read the source of {cls.__qualname__!r}; using a "
        "cloudpickle-based cache identity, which is NOT stable across Python or "
        'cloudpickle versions. Set `_cache_version = "..."` on the class for a '
        "stable cache key.",
        UserWarning,
        stacklevel=3,
    )
    return "cloudpickle", hashlib.sha256(blob).hexdigest()


def filter_identity(obj) -> dict:
    """The full identity record consumed by the Rust bridge."""
    cls = type(obj)
    method, digest = code_fingerprint(cls)
    return {
        "qualname": f"{cls.__module__}.{cls.__qualname__}",
        "config_json": canonical_config_json(obj),
        "code_fp": f"{method}:{digest}",
    }
