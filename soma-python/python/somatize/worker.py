"""The generic worker: `pip install somatize` and nothing else.

    python -m somatize.worker --listen 127.0.0.1:7000 [--store /scratch/soma]

An independent process on whatever machine. It starts **empty** — it does not
know what `tokenize` is — and waits for someone to connect and send it what to
build its catalog from. There is no way of giving it one by hand, on purpose.

| artifact kind | what arrives | what the worker supplies |
|---|---|---|
| `project` *(default)* | names, versions and state | **the code**, from its clone |
| `pickle` | the code and the state | nothing |

`project` is what you want when the worker runs in a clone of the project: tens
of bytes per node, no coupling between interpreters, and it **checks the
version** — `--strict` *(default)* stops with both versions in front of you and
`--lucky` runs anyway and says so on `stderr`.

**Whoever reaches this port runs code here**, as their user. There is no
authentication and there is not going to be one — that is `srun`'s and `ssh`'s
job. Bind to `127.0.0.1` and tunnel, or to a private interface inside a cluster.

It does not solve the **environment**: `cloudpickle` moves your objects, not
`torch`. That belongs to whoever stands the worker up, and putting it in here
cost the original 420 lines and a hot `pip install`. It fits in one file::

    #!/usr/bin/env -S uv run --script
    # /// script
    # requires-python = "==3.13.*"
    # dependencies = ["somatize[remote]", "torch==2.10.0"]
    # ///
    from somatize import worker
    worker.listen("0.0.0.0:7000", store="/scratch/soma")

`stdout` **is** the wire, so at startup `sys.stdout` is redirected to `stderr`: a
stray `print()` in one of your nodes would otherwise break the protocol.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload, runtime_checkable

from somatize import _manifest
from somatize._somatize import listen_provisioned as _listen_provisioned
from somatize._somatize import serve_provisioned as _serve_provisioned

if TYPE_CHECKING:
    from collections.abc import Callable

    from somatize._somatize import Store

__all__ = [
    "Pickles",
    "Project",
    "Provision",
    "Strategies",
    "listen",
    "runtime",
    "serve_provisioned",
]


@runtime_checkable
class Provision(Protocol):
    """How a worker turns an artifact into a catalog. A `Protocol` because there
    are three of these here and none inherits from the others. It is the seam
    `wire`'s `Provision` trait is filled from, and a user with a fourth way of
    packing nodes writes one and passes it to `listen`.
    """

    def accepts(self, client: str, kind: str) -> str | None:
        """`None` to accept, or **how this worker identifies itself** so the
        client can see what it disagrees with."""

    def provide(self, kind: str, blob: bytes) -> dict[str, Any]:
        """The nodes that artifact unpacks to, by id."""


def runtime() -> str:
    """How this process identifies itself, so a mismatch is refused on connect
    instead of surfacing inside a `loads`. `somatize`'s version goes here, which
    is why it does not go into each class's fingerprint.
    """
    import cloudpickle

    from somatize import __version__

    v = sys.version_info
    return (
        f"cpython-{v.major}.{v.minor}"
        f"/cloudpickle-{cloudpickle.__version__}"
        f"/soma-{__version__}"
    )


class Pickles:
    """Turns `pickle` artifacts into nodes. Python's `Provision`:
    `accepts(client, kind)` returns `None` or **how this worker identifies
    itself**, and `provide(kind, blob)` returns the nodes as a dict."""

    def accepts(self, client: str, kind: str) -> str | None:
        del kind  # `Pickles` only opens one, and checks it in `catalog`
        mine = runtime()
        return None if client == mine else mine

    def provide(self, kind: str, blob: bytes) -> dict[str, Any]:
        if kind != "pickle":
            raise ValueError(
                f"this worker only knows how to open `pickle` artifacts, and a `{kind}` arrived"
            )
        import cloudpickle

        try:
            sent = cloudpickle.loads(blob)
        except ModuleNotFoundError as e:
            # The likeliest failure: the nodes came from a client module that
            # does not exist here, so cloudpickle stored a reference.
            raise ModuleNotFoundError(
                f"this worker does not have the module `{e.name}`, so it cannot open "
                f"what it was sent. Either install it here, or make it travel inside "
                f'the artifact: Worker.generic(..., send=["{(e.name or "").split(".")[0]}"])'
            ) from e

        return _nodes(sent, "pickle")


class Project:
    """Opens `project` artifacts: resolves the classes against **this** clone.
    No code comes over the wire, so neither `cloudpickle` nor matching
    interpreters are needed."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def accepts(self, client: str, kind: str) -> str | None:
        del client, kind
        # Nothing to refuse: no code arrives, so the interpreter does not
        # matter. The version is checked on opening, class by class.
        return None

    def provide(self, kind: str, blob: bytes) -> dict[str, Any]:
        if kind != _manifest.KIND:
            raise ValueError(f"this does not open `{kind}` artifacts")
        return _nodes(_manifest.unpack(blob, strict=self.strict), _manifest.KIND)


class Strategies:
    """Several ways of building the catalog, asked by `kind` and not tried in a
    chain to see which sticks."""

    def __init__(self, **by_kind: Provision) -> None:
        self.by_kind = by_kind

    def _who(self, kind: str) -> Provision:
        if (which := self.by_kind.get(kind)) is None:
            raise ValueError(
                f"this worker cannot open `{kind}` artifacts; it can open: "
                + ", ".join(sorted(self.by_kind))
            )
        return which

    def accepts(self, client: str, kind: str) -> str | None:
        return self._who(kind).accepts(client, kind)

    def provide(self, kind: str, blob: bytes) -> dict[str, Any]:
        return self._who(kind).provide(kind, blob)


def _nodes(sent: Any, kind: str) -> dict[str, Any]:
    """What an artifact has to unpack to: a dict of id → node."""
    if not isinstance(sent, dict):
        raise TypeError(
            f"a `{kind}` artifact is a dict of id → node, and a "
            f"`{type(sent).__name__}` arrived"
        )
    said: dict[str, Any] = sent
    return said


def default(strict: bool = True) -> Strategies:
    """What a worker knows how to open if you do not say otherwise."""
    both: dict[str, Provision] = {_manifest.KIND: Project(strict), "pickle": Pickles()}
    return Strategies(**both)


def serve_provisioned(
    provision: Provision | None = None,
    store: "Store | str | None" = None,
    reporting: float | None = None,
) -> None:
    """Serves slices with the catalog the client sends, until the client closes.
    Without an argument it opens both kinds."""
    _hush()
    return _serve_provisioned(
        default() if provision is None else provision, store=store, reporting=reporting
    )


def _hush() -> None:
    """Leaves `stdout` free for the protocol and sends what is printed to `stderr`."""
    sys.stdout = sys.stderr


def listen(
    addr: str,
    provision: Provision | None = None,
    opened: Callable[[str], None] | None = None,
    store: "Store | str | None" = None,
    reporting: float | None = None,
) -> None:
    """Stands on `addr` and serves whoever connects. It does not return.

    `provision` says what the implementations are resolved with. `stdout` is not
    touched — here the wire is the socket — and `opened` is called once with the
    real address, so port `0` can be asked for. `store` answers two questions
    that stay two: an artifact it already has is not sent again, and a node whose
    answer is there is not run again.
    """
    return _listen_provisioned(
        addr, default() if provision is None else provision, opened, store=store,
        reporting=reporting
    )


def main(argv: list[str] | None = None) -> None:
    """`python -m somatize.worker [--listen HOST:PORT] [--store DIR] [--lucky]`.

    Without `--listen` it talks over standard input, which is for testing.
    `--reporting SECONDS` writes a reading of this machine into the store on a
    clock — it goes to the store because an idle worker's connection is one
    nobody is reading. Off unless asked for, and it needs a `--store`.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    _read_the_codecs()
    which = default(strict="--lucky" not in argv)
    store = _after("--store", argv)
    every = _after("--reporting", argv)
    if "--listen" in argv:
        return listen(
            _after("--listen", argv, needed=True),
            provision=which,
            opened=lambda where: print(f"listening on {where}", flush=True),
            store=store,
            reporting=float(every) if every else None,
        )
    return serve_provisioned(
        which, store=store, reporting=float(every) if every else None
    )


def _read_the_codecs() -> None:
    """Registers what this worker knows how to read, **before** it serves.

    A codec has to be there before a value arrives. Discovered instead, the first
    import of a large native extension happens inside a request thread while the
    GIL is handed back and forth, and with torch that is not an error but **heap
    corruption**: a worker that dies with no traceback, and a client told only
    that it closed without answering.

    Not the worker learning about your project: `somatize.torch` is soma's own,
    and one that has no torch carries no tensors, which is a worker too.
    """
    try:
        import somatize.torch  # noqa: F401
    except ImportError:
        pass


@overload
def _after(flag: str, argv: list[str], needed: Literal[True]) -> str: ...


@overload
def _after(flag: str, argv: list[str], needed: bool = False) -> str | None: ...


def _after(flag: str, argv: list[str], needed: bool = False) -> str | None:
    """What comes after a flag, or `None` if the flag is not there.

    `needed` says the flag is not optional, and the two overloads above are it
    said in types: asking for one that has to be there answers a `str` and not a
    `str | None`, so the caller has nothing to check.
    """
    if flag not in argv:
        if needed:
            raise SystemExit(f"`{flag}` is required: {flag} 127.0.0.1:7000")
        return None
    after = argv.index(flag) + 1
    if after >= len(argv):
        raise SystemExit(f"`{flag}` needs a value: {flag} 127.0.0.1:7000")
    return argv[after]


if __name__ == "__main__":
    main()
