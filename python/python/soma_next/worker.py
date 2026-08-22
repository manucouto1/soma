"""The generic worker: `pip install soma-next` and nothing else.

    python -m soma_next.worker --listen 127.0.0.1:7000 [--store /scratch/soma]

An independent process, in the background, on whatever machine. It starts
**empty** — it does not know what `tokenize` is — and waits for someone to
connect and send it what to build its catalog from. There is no way of giving it
a catalog by hand, and that is on purpose: **a worker receives the plan and
resolves the implementations**.

## The strategies, and how they differ

| artifact kind | what arrives | what the worker supplies |
|---|---|---|
| `project` *(default)* | names, versions and state | **the code**, from its clone |
| `pickle` | the code and the state | nothing |

`project` is the one you want when the worker runs in a clone of the project: it
sends tens of bytes per node, couples no interpreters, and **checks the version**
— if your clone is half-updated, it finds out before executing. `pickle` removes
all friction when the worker does not have your code: a bare node, a
`pip install`, and that is it. A worker accepts both, by `kind`.

With `project`, if the class here is not the version the graph was written
against, `--strict` *(default)* stops and says so with both versions in front of
you, and `--lucky` executes whatever it has and **reports it on `stderr`** —
running a different version silently is what gets discovered three days later.

## The port belongs on a network you trust

A worker executes what it is sent, and that is the whole point of it — but say
it plainly: `pickle` artifacts are opened with `cloudpickle.loads`, and `project`
ones resolve classes out of this clone, so **whoever reaches this port runs code
here**, as their user. There is no authentication and there is not going to be
one: this is `srun`'s and `ssh`'s job, not a framework's.

Bind it to `127.0.0.1` and reach it through a tunnel, or to a private interface
inside a cluster. `0.0.0.0` on a machine with a public address is handing out a
shell.

## What this module does NOT solve

The **environment**. `cloudpickle` moves your objects, it does not move `torch`.
If the dependencies your nodes import are not installed here, the `loads` fails
with an ordinary `ModuleNotFoundError`. Installing them belongs to whoever stands
the worker up, and putting it in here cost the original soma 420 lines of
environment manager and a hot `pip install`.

Which is a **recipe and not a mechanism**, and it fits in one file::

    #!/usr/bin/env -S uv run --script
    # /// script
    # requires-python = "==3.13.*"
    # dependencies = ["soma-next[remote]", "torch==2.10.0"]
    # ///
    from soma_next import worker
    worker.listen("0.0.0.0:7000", store="/scratch/soma")

`uv lock --script` leaves the resolution beside it, so the machine that claims a
trial from a shared folder gets the environment the one that wrote it had —
without a shared env on NFS and without a `module load`. Nothing in here knows
what `uv` is, and that is the point: it is the same file the worker already was.

## `stdout` is the wire

The protocol's messages go over this process's standard output, so at startup
`sys.stdout` is redirected to `sys.stderr`: a stray `print()` in one of your
nodes would otherwise break them. A library that writes to file descriptor 1
directly would still break it; that is rare and we do not paper over it.
"""

from __future__ import annotations

import sys

from soma_next import _manifest
from soma_next._soma_next import listen_provisioned as _listen_provisioned
from soma_next._soma_next import serve_provisioned as _serve_provisioned

__all__ = [
    "Pickles",
    "Project",
    "Strategies",
    "listen",
    "runtime",
    "serve_provisioned",
]


def runtime():
    """How this process identifies itself to the other side, so a mismatch is
    refused on connect instead of surfacing inside a `loads`.

    `soma_next`'s version goes in here, and that is why it does not go into each
    class's fingerprint.
    """
    import cloudpickle

    from soma_next import __version__

    v = sys.version_info
    return (
        f"cpython-{v.major}.{v.minor}"
        f"/cloudpickle-{cloudpickle.__version__}"
        f"/soma-next-{__version__}"
    )


class Pickles:
    """Turns `pickle` artifacts into nodes. Python's `Provision`:
    `accepts(client, kind)` returns `None` or **how this worker identifies
    itself**, and `provide(kind, blob)` returns the nodes as a dict."""

    def accepts(self, client, kind):
        del kind  # `Pickles` only opens one, and checks it in `catalog`
        mine = runtime()
        return None if client == mine else mine

    def provide(self, kind, blob):
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

    def __init__(self, strict=True):
        self.strict = strict

    def accepts(self, client, kind):
        del client, kind
        # Nothing to refuse: no code arrives, so the interpreter does not
        # matter. The version is checked on opening, class by class.
        return None

    def provide(self, kind, blob):
        if kind != _manifest.KIND:
            raise ValueError(f"this does not open `{kind}` artifacts")
        return _nodes(_manifest.unpack(blob, strict=self.strict), _manifest.KIND)


class Strategies:
    """Several ways of building the catalog, asked by `kind` and not tried in a
    chain to see which sticks."""

    def __init__(self, **by_kind):
        self.by_kind = by_kind

    def _who(self, kind):
        if (which := self.by_kind.get(kind)) is None:
            raise ValueError(
                f"this worker cannot open `{kind}` artifacts; it can open: "
                + ", ".join(sorted(self.by_kind))
            )
        return which

    def accepts(self, client, kind):
        return self._who(kind).accepts(client, kind)

    def provide(self, kind, blob):
        return self._who(kind).provide(kind, blob)


def _nodes(sent, kind):
    """What an artifact has to unpack to: a dict of id → node."""
    if not isinstance(sent, dict):
        raise TypeError(
            f"a `{kind}` artifact is a dict of id → node, and a "
            f"`{type(sent).__name__}` arrived"
        )
    return sent


def default(strict=True):
    """What a worker knows how to open if you do not say otherwise."""
    return Strategies(**{_manifest.KIND: Project(strict), "pickle": Pickles()})


def serve_provisioned(provision=None, store=None):
    """Serves slices with the catalog the client sends, until the client closes.
    Without an argument it opens both kinds."""
    _hush()
    return _serve_provisioned(
        default() if provision is None else provision, store=store
    )


def _hush():
    """Leaves `stdout` free for the protocol and sends what is printed to `stderr`."""
    sys.stdout = sys.stderr


def listen(addr, provision=None, opened=None, store=None):
    """Stands on `addr` and serves whoever connects. It does not return.

    `provision` says what the implementations are resolved with; by default,
    `project` and `pickle`. `stdout` is not touched — here the wire is the
    socket — and `opened` is called once with the real address, so port `0` can
    be asked for.

    `store` is a directory, and it answers two questions that stay two: an
    artifact it already has is **not sent again**, and a node whose answer is
    already there is **not run again**. Shared between workers — a mount, a
    network folder — the second one to be stood up starts warm.
    """
    return _listen_provisioned(
        addr, default() if provision is None else provision, opened, store=store
    )


def main(argv=None):
    """`python -m soma_next.worker [--listen HOST:PORT] [--store DIR] [--lucky]`.

    Without `--listen` it talks over standard input, which is for testing.
    `--lucky` executes even if its code is not the version the graph was written
    against; by default it stops. `--store` is a directory to keep things in.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    _read_the_codecs()
    which = default(strict="--lucky" not in argv)
    store = _after("--store", argv)
    if "--listen" in argv:
        return listen(
            _after("--listen", argv, needed=True),
            provision=which,
            opened=lambda where: print(f"listening on {where}", flush=True),
            store=store,
        )
    return serve_provisioned(which, store=store)


def _read_the_codecs():
    """Registers what this worker knows how to read, **before** it serves.

    A codec has to be there before a value arrives and not be discovered while
    one is being unpacked. Discovered, the first import of a large native
    extension happens inside a request thread while the GIL is being handed
    back and forth, and with torch that is not an error but **heap corruption**:
    `free(): chunks in smallbin corrupted`, a worker that dies with no
    traceback, and a client told only that it closed without answering.

    It hid for a long time because the first job a worker got was usually a
    small one, which imported torch harmlessly; it surfaced the day a second
    torch worker was stood up whose very first job was a training run.

    This is not the worker learning about your project. `soma_next.torch` is
    soma-next's own, and a worker that can carry a tensor is one that has read
    the codec for it. One that has no torch carries no tensors, and that is a
    worker too.
    """
    try:
        import soma_next.torch  # noqa: F401
    except ImportError:
        pass


def _after(flag, argv, needed=False):
    """What comes after a flag, or `None` if the flag is not there."""
    if flag not in argv:
        return None
    after = argv.index(flag) + 1
    if after >= len(argv):
        raise SystemExit(f"`{flag}` needs a value: {flag} 127.0.0.1:7000")
    return argv[after]


if __name__ == "__main__":
    main()
