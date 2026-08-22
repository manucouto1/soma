"""How N machines that never speak to each other search one space together.

`Sampler` says where to look and `Pruner` when to stop looking; this is the
meeting. Everybody runs the same script over a directory they all mounted, and
`claim` is what hands out the work. There is no server, no port and no protocol.

```python
# the same script on every machine; Slurm gives out `me`
store, space = Store("/scratch/spam"), Space().real("lr", 1e-5, 1e-1, log=True)
sampler = Sampler.sobol(seed=0)

for trial in range(100):
    point = sampler.ask(space, trial, finished(store, space, study="spam"))
    if not take(store, point, study="spam", trial=trial, me=me):
        continue                                   # somebody else has that one
    ...
    report(store, point, reports, study="spam", trial=trial, me=me, state="done")
```

# Why there is no queue, and nothing is sent anywhere

A trial is a number. `ask` is a function of that number and not of what was asked
before, so a machine that claims trial 7 works out where to look **on its own**,
without replaying the first six and without asking anybody. That is why handing
out work costs no message: the state *is* the queue, and a claim is a claim, so
exactly one machine gets each number. Nothing can be lost in flight because
nothing is in flight.

The exception is `Sampler.tpe`, which is guided and so does depend on what has
already finished. It gets that from `finished` below — the same scan, no
coordinator — and two machines asking at the same moment see the same history
and may well propose neighbouring points. That is the known cost of running a
guided search in parallel and it is not solved here.

# What a trial is, on disk

```text
<study>/trial/<n>/<attempt>
```

In the **record**, which a scan already carries:

```text
state = running | done | pruned | failed
point = lr=0.001,batch=32,opt=adam
score = 0.0837                         (absent while it is still running)
who   = whoever claimed it
```

In the **blob**, for whoever wants the detail: the whole curve, and why it
stopped.

The split is the reason it scales: rebuilding the history a sampler asks for
costs **one scan and not one fetch per trial**, because the configuration and
the score are both in the record. Only a pruner comparing curves pays for blobs.

One record per trial, rewritten as it goes — and not five events. The
`TrialStarted`/`TrialPruned`/`TrialCompleted` of a bus are the *diff* of this
record: from a state the events can be derived, and from a lossy stream the
state cannot.

# The attempt, which is a segment nothing reads yet

`claim` is a link, so a trial whose machine died stays claimed for ever and
rescuing it with a plain write would be a race. A retry will be a claim of the
next attempt, and whoever reads keeps the highest. It is paid for now because
**the name is the one part of this that cannot be refactored later**: changing
it means migrating directories belonging to people with studies running.
"""

from __future__ import annotations

import json

RUNNING = "running"
"""Claimed, and still going."""

DONE = "done"
"""Ran to the end, and its score means what every other `done` score means."""

PRUNED = "pruned"
"""Given up on. It has a score, and it is **not** comparable with a `done` one:
it was measured after a different number of epochs."""

FAILED = "failed"
"""Blew up. It says so rather than looking like a trial that scored badly."""

STATE, POINT, SCORE, WHO = "state", "point", "score", "who"


def take(store, point, *, study, trial, me, attempt=0):
    """Claims the `trial`-th trial of `study`. `True` when it is this machine's.

    `False` means somebody else got there first, and the loop simply goes on to
    the next number — that is the whole of how the work is handed out.
    """
    digest = store.put(_blob(point, [], RUNNING, None, None))
    return store.claim(
        _trial(study, trial, attempt),
        digest,
        {STATE: RUNNING, POINT: str(point), WHO: str(me)},
    )


def report(
    store,
    point,
    reports,
    *,
    study,
    trial,
    me,
    attempt=0,
    state=RUNNING,
    score=None,
    because=None,
    took=None,
):
    """Writes down where this trial has got to.

    Called as often as there is something to say — once an epoch is the useful
    rate, and it is what makes a curve watchable from another machine while it
    is still being drawn.

    Only the machine that claimed the trial writes to it. Nothing enforces that
    and nothing needs to: nobody else has a reason to, because nobody else could
    have got the claim.
    """
    if score is None and reports and state != RUNNING:
        score = reports[-1]
    said = {STATE: state, POINT: str(point), WHO: str(me)}
    if score is not None:
        said[SCORE] = repr(float(score))
    digest = store.put(_blob(point, reports, state, because, took))
    store.bind(_trial(study, trial, attempt), digest, said)


def finished(store, space, *, study):
    """Every trial that ran to the end, as `(point, score)` — what `ask` wants.

    **One scan and no fetches**: the configuration and the score are both in the
    record, so a machine that ran none of these trials rebuilds the whole history
    without reading a single blob.

    Pruned trials are left out on purpose. A pruned score is real but it is not
    comparable with a finished one — it was measured after fewer epochs — and a
    sampler that treats it as a bad configuration learns something that is not
    true.
    """
    history = []
    for record in _latest(store, study):
        said = dict(record.meta)
        if said.get(STATE) != DONE or SCORE not in said:
            continue
        history.append((space.read(said[POINT]), float(said[SCORE])))
    return history


def curves(store, *, study):
    """The reports of every trial that ran to the end — what a `Pruner` wants.

    This is the reader that pays: a curve grows, so it lives in the blob, and
    this is a scan **plus one fetch per trial**. It is the price of pruning
    against trials that other machines ran.
    """
    drawn = []
    for record in _latest(store, study):
        if dict(record.meta).get(STATE) != DONE:
            continue
        drawn.append(_read(store, record)["reports"])
    return drawn


def trials(store, space, *, study):
    """Every trial of this study, whatever state it is in, as records.

    The one for looking rather than deciding: a notebook drawing what is going
    on, and the answer to "is this study done". `state` says which are still
    running, which is also the list of what another machine is holding.
    """
    seen = []
    for record in _latest(store, study):
        said = dict(record.meta)
        seen.append(
            {
                "trial": _numbered(record.name, study)[0],
                STATE: said.get(STATE),
                POINT: space.read(said[POINT]) if POINT in said else None,
                SCORE: float(said[SCORE]) if SCORE in said else None,
                WHO: said.get(WHO),
            }
        )
    return seen


def _latest(store, study):
    """One record per trial — the highest attempt of each — in trial order."""
    best = {}
    for record in store.bound():
        numbered = _numbered(record.name, study)
        if numbered is None:
            continue
        trial, attempt = numbered
        if trial not in best or best[trial][0] < attempt:
            best[trial] = (attempt, record)
    return [record for _, (_, record) in sorted(best.items())]


def _numbered(name, study):
    """The `(trial, attempt)` that name is, or `None` if it is not one of ours.

    A store holds whatever anybody put in it — a cache, another study, an
    artifact — so this has to be a question and not an assumption.
    """
    prefix = f"{study}/trial/"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :].split("/")
    if len(rest) != 2:
        return None
    try:
        return int(rest[0]), int(rest[1])
    except ValueError:
        return None


def _read(store, record):
    """The blob that record points at."""
    bytes_ = store.get(record.digest)
    if bytes_ is None:
        raise RuntimeError(
            f"`{record.name}` points at `{record.digest}` and this store does not "
            f"have it: the record and the bytes are two things, and one of them "
            f"is missing"
        )
    return json.loads(bytes_)


def _blob(point, reports, state, because, took):
    """What is kept beside the record: the curve, and why it stopped.

    JSON and not a pickle, because whoever reads this is another process — often
    another machine, sometimes a notebook — and none of them should need this
    library's version of anything to look at a study.
    """
    return json.dumps(
        {
            "point": str(point),
            "reports": [float(one) for one in reports],
            "state": state,
            "because": because,
            "took": took,
        },
        sort_keys=True,
    ).encode()


def _trial(study, trial, attempt):
    return f"{study}/trial/{trial}/{attempt}"
