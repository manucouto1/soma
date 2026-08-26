"""How N clients that never speak to each other finish a round together.

`fedavg` is the arithmetic and this is the meeting: everybody writes what they
learnt into a directory they all mounted, and everybody leaves with the average.
No server, no port, no protocol — a folder, and `claim`::

    # the same script on every machine; Slurm gives out `mine`
    for r in range(rounds):
        trainer.fit(my_data)
        trainer.load(gather(store, trainer.export(), run="cifar", round=r,
                            clients=4, mine=int(os.environ["SLURM_PROCID"])))

A round needs somebody to wait for all of it and average it, and a coordinator is
a process that has to stay alive — and a run that hangs over a weekend when it
does not. So **whoever finds the round complete claims the averaging**, and
exactly one can win that, because that is what a claim is.

```text
<run>/round/<r>/client/<k>    what client k learnt   (its size in the record)
<run>/round/<r>/averaging     who is doing the mean  (claimed, so exactly one)
<run>/round/<r>/average       the mean               (what everybody leaves with)
```

Without a deadline the round waits for ever, so running out of it says **which
clients are missing** by name. Waiting longer is a number; going on without them
is a policy and not this function's to make.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from somatize._somatize import Bound, Store

import time

from somatize.torch._federated import fedavg

SIZE = "size"
"""What a client's record says about how much data it saw, so that the one doing
the averaging can weigh by it without anybody being asked."""


def gather(
    store: "Store",
    what: Any,
    *,
    run: str,
    round: int,
    clients: int,
    mine: int,
    size: float | None = None,
    within: float = 600.0,
    asking: float = 1.0,
) -> Any:
    """Puts this client's round in, waits for everybody else's, and gives back the
    average. `run` names the training run, `round` which round, `clients` how
    many there are and `mine` which one this is. `size` is how much data this
    client saw, and it travels in the record so whoever averages can weigh by it.
    Raises `TimeoutError` after `within` seconds, naming who never turned up.
    """
    store.keep(_client(run, round, mine), what, {SIZE: str(size)} if size else None)
    me = store.put(f"{run}/{round}/{mine}".encode())

    giving_up = time.monotonic() + within
    while True:
        average = store.recall(_average(run, round))
        if average is not None:
            return average
        missing = _who_is_missing(store, run, round, clients)
        # Everybody is in. One of us does the arithmetic, and which one is
        # settled by the only thing that can settle it between processes.
        if not missing and store.claim(_averaging(run, round), me):
            return _the_mean(store, run, round, clients)
        if time.monotonic() >= giving_up:
            raise TimeoutError(_never_turned_up(run, round, missing, within))
        time.sleep(asking)


def _the_mean(store: "Store", run: str, round: int, clients: int) -> Any:
    """The average of what everybody put in, published for them to find.

    Written **before** it is returned: the client that does this is also a
    client, and if it went on with an average nobody else could see, the round
    would have happened for one of them.
    """
    puts = [store.resolve(_client(run, round, which)) for which in range(clients)]
    exports = []
    for which in range(clients):
        one = store.recall(_client(run, round, which))
        if one is None:
            # `_who_is_missing` asked whether the **record** was there, and the
            # record and the bytes are two things — the same split `_blob` says
            # out loud everywhere else. Named, rather than an `AttributeError`
            # from inside the arithmetic.
            raise RuntimeError(
                f"client {which} of round {round} of `{run}` has a record and no "
                f"bytes behind it, so there is nothing of its to average"
            )
        exports.append(one)
    said = [_size_in(put) for put in puts]
    # Every client has to have said, or none of them weighs: a round where one
    # size is missing is a round weighted by a number nobody wrote down.
    sizes = [one for one in said if one is not None]
    average = fedavg(exports, sizes=sizes if len(sizes) == clients else None)
    store.keep(_average(run, round), average)
    return average


def _who_is_missing(store: "Store", run: str, round: int, clients: int) -> list[int]:
    """Which clients have not put their round in yet, in order."""
    names = [_client(run, round, which) for which in range(clients)]
    return [
        which
        for which, name in enumerate(names)
        if store.resolve(name) is None
    ]


def _size_in(put: "Bound | None") -> float | None:
    """How much data that client saw, if its record says."""
    said = dict(put.meta).get(SIZE) if put is not None else None
    try:
        return float(said) if said is not None else None
    except ValueError:
        return None


def _never_turned_up(
    run: str,
    round: int,
    missing: list[int],
    within: float,
) -> str:
    """Why it gave up, and the two reasons are not the same reason.

    Somebody missing is a client that never started. **Nobody** missing is worse
    and rarer: everybody put their round in, so whoever won the averaging died
    holding it — and no one else will try, because a claim is a claim.
    """
    if not missing:
        return (
            f"round {round} of `{run}` waited {within:g}s: every client put its "
            f"round in and no average was ever published, so whoever claimed the "
            f"averaging did not finish it. Nobody else will try — that is what a "
            f"claim is — so the round has to be started again"
        )
    who = ", ".join(str(which) for which in missing)
    return (
        f"round {round} of `{run}` waited {within:g}s and client {who} never "
        f"turned up. Whoever wants a round without them says so themselves: "
        f"`fedavg` takes whatever list it is handed"
    )


def _client(run: str, round: int, which: int) -> str:
    return f"{run}/round/{round}/client/{which}"


def _averaging(run: str, round: int) -> str:
    return f"{run}/round/{round}/averaging"


def _average(run: str, round: int) -> str:
    return f"{run}/round/{round}/average"
