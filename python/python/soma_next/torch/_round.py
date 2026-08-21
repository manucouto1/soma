"""How N clients that never speak to each other finish a round together.

`fedavg` is the arithmetic and this is the meeting: everybody writes what they
learnt into a directory they all mounted, and everybody leaves with the average
of it. There is no server, no port and no protocol — a folder, and `claim`.

```python
# the same script on every machine; Slurm gives out `mine`
for r in range(rounds):
    trainer.fit(my_data)
    trainer.load(gather(store, trainer.export(), run="cifar", round=r,
                        clients=4, mine=int(os.environ["SLURM_PROCID"])))
```

# Nobody is in charge, and that is what `claim` is for

A round needs somebody to wait for all of it and average it, and the obvious
answer — a coordinator process — is a thing that has to stay alive, and a run
that hangs on a weekend when it does not. So instead: **whoever finds the round
complete claims the averaging**, and exactly one of them can win that, because
that is what a claim is. The winner averages and publishes; the others find the
average on their next look.

The cost of it is that every client runs the same script and one of them does a
little more work than the others, once a round. The saving is a process nobody
has to babysit.

# What a round is made of, on disk

```text
<run>/round/<r>/client/<k>    what client k learnt   (its size in the record)
<run>/round/<r>/averaging     who is doing the mean  (claimed, so exactly one)
<run>/round/<r>/average       the mean               (what everybody leaves with)
```

`run` is there so two training runs sharing a directory are two training runs.

# When somebody does not turn up

A node falls over, Slurm has no room, a client is simply slow: without an answer
the round waits for ever, and a run that hangs is worse than one that stops. So
there is a deadline, and running out of it says **which clients are missing** by
name — which is the difference between "it hung" and "node 7 never started".

Waiting longer is a number; deciding to go on without them is a policy, and it is
not this function's to make. `fedavg` takes whatever list you hand it, so whoever
wants a round of three out of four says so themselves.
"""

from __future__ import annotations

import time

from soma_next.torch._federated import fedavg

SIZE = "size"
"""What a client's record says about how much data it saw, so that the one doing
the averaging can weigh by it without anybody being asked."""


def gather(
    store,
    what,
    *,
    run,
    round,
    clients,
    mine,
    size=None,
    within=600.0,
    asking=1.0,
):
    """Puts this client's round in, waits for everybody else's, and gives back
    the average of all of them.

    `run` names this training run, `round` which round it is, `clients` how many
    there are and `mine` which one this is — whatever Slurm handed out. `size` is
    how much data this client saw, and it travels in the record so that whoever
    ends up averaging can weigh by it.

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


def _the_mean(store, run, round, clients):
    """The average of what everybody put in, published for them to find.

    Written **before** it is returned, and not after: the client that does this
    is also a client, and if it went on to the next round with an average nobody
    else could see, the round would have happened for one of them.
    """
    puts = [store.resolve(_client(run, round, which)) for which in range(clients)]
    exports = [store.recall(_client(run, round, which)) for which in range(clients)]
    sizes = [_size_in(put) for put in puts]
    average = fedavg(exports, sizes=sizes if all(sizes) else None)
    store.keep(_average(run, round), average)
    return average


def _who_is_missing(store, run, round, clients):
    """Which clients have not put their round in yet, in order."""
    names = [_client(run, round, which) for which in range(clients)]
    return [
        which
        for which, name in enumerate(names)
        if store.resolve(name) is None
    ]


def _size_in(put):
    """How much data that client saw, if its record says."""
    said = dict(put.meta).get(SIZE) if put is not None else None
    try:
        return float(said) if said is not None else None
    except ValueError:
        return None


def _never_turned_up(run, round, missing, within):
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


def _client(run, round, which):
    return f"{run}/round/{round}/client/{which}"


def _averaging(run, round):
    return f"{run}/round/{round}/averaging"


def _average(run, round):
    return f"{run}/round/{round}/average"
