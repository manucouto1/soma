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
    if not take(store, point, study="spam", trial=trial, me=me, goal="min"):
        continue                                   # somebody else has that one
    ...
    report(store, point, reports, study="spam", trial=trial, me=me,
           state="done", goal="min")
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
goal  = min | max                      (absent when nobody said)
```

In the **blob**, for whoever wants the detail: the whole curve, and why it
stopped.

The split is the reason it scales: rebuilding the history a sampler asks for
costs **one scan and not one fetch per trial**, because the configuration and
the score are both in the record. Only a pruner comparing curves pays for blobs.

# Why the direction is written down, when nothing here reads it

`0.0837` is a good score or a bad one and the number does not say which. Inside
a loop that never matters: the `Goal` was handed to the sampler and the pruner
at the top of the script, and they are the only things comparing anything.

It matters to **whoever reads the study without the script**, which is the case
a study handed out of a folder exists for — a notebook opened over somebody
else's run, another tool reading the same directory. Without it they have the
score and no way to know which end is good, and the ways of getting that wrong
are all silent: a table that says *best first* listing the worst, a colour scale
with the good region drawn as the one to avoid.

So it goes where `score` goes. That is the same call CU18 already made and for
the same reason: what a scan carries is what a reader can use, and a number
whose meaning lives in somebody else's script is not readable.

It is per trial and not per study, which **is** denormalised — the direction
belongs to the study and it is written N times. The alternative is a name of its
own, which costs a fetch and a missing case, and buys a normalisation nobody was
going to query. What the denormalised shape gets right for free: it records what
was meant **at the time**, so a study whose direction was changed halfway does
not retell the old trials.

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

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from somatize._somatize import Bound, Point, Space, Store

#: One trial as `trials` reports it: its number, its state, the point it
#: ran, the score it reached and which way better was. `Any` for the same
#: reason the record's rows are: the columns are named in the docstring and
#: the store never learned what any of them mean.
Trial = dict[str, Any]

import json
import math

RUNNING = "running"
"""Claimed, and still going."""

DONE = "done"
"""Ran to the end, and its score means what every other `done` score means."""

PRUNED = "pruned"
"""Given up on. It has a score, and it is **not** comparable with a `done` one:
it was measured after a different number of epochs."""

FAILED = "failed"
"""Blew up. It says so rather than looking like a trial that scored badly."""

STATE, POINT, SCORE, WHO, GOAL = "state", "point", "score", "who", "goal"

MIN = "min"
"""Smaller is better: a loss, an error, a runtime. The word `Goal` reads, so the
record and the enum spell it the same and neither has to translate."""

MAX = "max"
"""Larger is better: an accuracy, an F1, a reward."""


def take(
    store: "Store",
    point: "Point",
    *,
    study: str,
    trial: int,
    me: object,
    attempt: int = 0,
    goal: str | None = None,
) -> bool:
    """Claims the `trial`-th trial of `study`. `True` when it is this machine's.

    `False` means somebody else got there first, and the loop simply goes on to
    the next number — that is the whole of how the work is handed out.

    `goal` is `"min"` or `"max"` and is what the sampler and the pruner were
    given. It is written here as well as on every `report` so that a trial which
    was claimed and never reported still says which way it was looking.
    """
    said = _said(RUNNING, point, me, goal)
    digest = store.put(_blob(point, [], RUNNING, None, None))
    return store.claim(_trial(study, trial, attempt), digest, said)


def report(
    store: "Store",
    point: "Point",
    reports: Sequence[float],
    *,
    study: str,
    trial: int,
    me: object,
    attempt: int = 0,
    state: str = RUNNING,
    score: float | None = None,
    because: str | None = None,
    took: float | None = None,
    goal: str | None = None,
) -> None:
    """Writes down where this trial has got to.

    Called as often as there is something to say — once an epoch is the useful
    rate, and it is what makes a curve watchable from another machine while it
    is still being drawn.

    Only the machine that claimed the trial writes to it. Nothing enforces that
    and nothing needs to: nobody else has a reason to, because nobody else could
    have got the claim.

    `goal` is `"min"` or `"max"` — the direction the sampler and the pruner were
    given — and it is written beside the score because a score without it is not
    readable by anybody who does not have this script. Leaving it out is allowed
    and costs exactly that: `direction` answers `None` and a figure has to be
    told.
    """
    if score is None and reports and state != RUNNING:
        score = reports[-1]
    said = _said(state, point, me, goal)
    if score is not None:
        said[SCORE] = repr(float(score))
    digest = store.put(_blob(point, reports, state, because, took))
    store.bind(_trial(study, trial, attempt), digest, said)


def _said(state: str, point: "Point", me: object, goal: str | None) -> dict[str, str]:
    """What both writers put in the record, so neither can forget half of it.

    `goal` is left out rather than guessed when nobody said: a default written
    into the store is a guess that reads exactly like a fact.
    """
    said = {STATE: state, POINT: str(point), WHO: str(me)}
    if goal is not None:
        said[GOAL] = _direction(goal)
    return said


def _direction(goal: str) -> str:
    """`min` or `max`, or the error where the typo was typed.

    The same two words `Goal` parses, checked here rather than at the far end:
    a study that wrote `minimize` into two thousand records is not a figure that
    draws badly, it is a directory to migrate.
    """
    if goal not in (MIN, MAX):
        raise ValueError(
            f"`{goal}` does not say which way is better: write `{MIN}` for a loss "
            f"or `{MAX}` for an accuracy"
        )
    return goal


def finished(
    store: "Store",
    space: "Space",
    *,
    study: str,
) -> list[tuple["Point", float]]:
    """Every trial that ran to the end, as `(point, score)` — what `ask` wants.

    **One scan and no fetches**: the configuration and the score are both in the
    record, so a machine that ran none of these trials rebuilds the whole history
    without reading a single blob.

    Pruned trials are left out on purpose. A pruned score is real but it is not
    comparable with a finished one — it was measured after fewer epochs — and a
    sampler that treats it as a bad configuration learns something that is not
    true.
    """
    history: list[tuple["Point", float]] = []
    for _, _, record in _latest(store, study):
        said = dict(record.meta)
        if said.get(STATE) != DONE or SCORE not in said:
            continue
        history.append((space.read(said[POINT]), float(said[SCORE])))
    return history


def direction(store: "Store", *, study: str) -> str | None:
    """Which way is better in this study — `"min"`, `"max"`, or `None`.

    **One scan and no fetches**, like everything else the record carries. It is
    what makes a score readable by somebody who does not have the script that
    produced it, which is the whole case for a study handed out of a folder.

    `None` means no trial said, and it is the honest answer rather than `"min"`:
    a study written before this was recorded, or a loop that never passed
    `goal`. Whoever is drawing has to be told, and that is better than being
    told wrong.

    # When the records disagree

    The newest one wins, because the direction is not a fact about a trial — it
    is what the person running the study currently means, and the only reason
    two records differ is that they changed their mind halfway. The old trials
    keep saying what they were run for, which is worth having and is not what a
    figure needs: drawing needs one answer.

    Ties are broken by the **higher trial number**, and that is not a detail: a
    study writes its first records inside the same second, so on a stamp alone
    two directions would be settled by whichever the scan happened to reach
    first. A trial number is a total order and it is the order they were run in.
    """
    said: tuple[tuple[int, int], str] | None = None
    for trial, _, record in _latest(store, study):
        goal = dict(record.meta).get(GOAL)
        if goal is not None and (said is None or said[0] < (record.when, trial)):
            said = ((record.when, trial), goal)
    return said[1] if said else None


def curves(store: "Store", *, study: str) -> list[list[float]]:
    """The reports of every trial that ran to the end — what a `Pruner` wants.

    This is the reader that pays: a curve grows, so it lives in the blob, and
    this is a scan **plus one fetch per trial**. It is the price of pruning
    against trials that other machines ran.
    """
    drawn: list[list[float]] = []
    for _, _, record in _latest(store, study):
        if dict(record.meta).get(STATE) != DONE:
            continue
        drawn.append(_read(store, record)["reports"])
    return drawn


def trials(store: "Store", space: "Space", *, study: str) -> list[Trial]:
    """Every trial of this study, whatever state it is in, as records.

    The one for looking rather than deciding: a notebook drawing what is going
    on, and the answer to "is this study done". `state` says which are still
    running, which is also the list of what another machine is holding.
    """
    seen: list[Trial] = []
    for trial, _, record in _latest(store, study):
        said = dict(record.meta)
        seen.append(
            {
                "trial": trial,
                STATE: said.get(STATE),
                POINT: space.read(said[POINT]) if POINT in said else None,
                SCORE: float(said[SCORE]) if SCORE in said else None,
                WHO: said.get(WHO),
                GOAL: said.get(GOAL),
            }
        )
    return seen


STALE = 3600.0
"""How far behind the rest of the study a record may fall before whoever wrote
it is taken to have stopped. Generous on purpose: being early costs one point of
the space, being late costs a little more of the same, and neither is worth a
tight number."""


def in_flight(
    store: "Store",
    space: "Space",
    *,
    study: str,
    stale: float = STALE,
) -> list[tuple["Point", float | None]]:
    """The trials another machine is holding, **each with no score**.

    Hand these to a sampler beside `finished` and a guided one stops proposing
    next to what somebody else is already trying::

        point = sampler.ask(space, trial,
                            finished(store, space, study=STUDY)
                            + in_flight(store, space, study=STUDY))

    That is *constant liar* (Ginsbourger, Le Riche and Carraro, 2010), and it is
    what parallel Bayesian optimisation needs to stop being worse than random:
    two machines asking at the same moment see the same history, propose almost
    the same point, and spend two trials learning one thing.

    # It is not actually a lie, and that was measured

    The name comes from handing the sampler a made-up bad score. Doing that here
    **backfires**, and not slightly: `Tpe` sizes the pile it imitates as a share
    of everything it is handed, so one more point raises the quota and promotes a
    trial out of the bad pile into the good one. If that trial sat in the same
    region as the one in flight, the warning pulls the search **towards** it.
    Counted over two hundred proposals: one landed on the occupied region
    without the warning, thirty-nine with it.

    So nothing is made up. A score that is `None` says *running*, `ask` puts it
    in the pile to keep away from, and it does not vote on how big the other pile
    is. The four schemes that look at nothing ignore the whole argument, so
    passing this to any sampler is safe.

    # What it costs

    One scan and **no fetches**, the same as `finished`: the configuration is in
    the record and `state = running` is right beside it. Knowing what the other
    machines are looking at is free, and that is the shape the record was given.

    # When somebody stopped writing

    A record is rewritten on every `report` and the store stamps the time on
    every write, so a `running` trial that has not moved is a machine that has
    stopped. There is nobody to ask — this design has no server, no port and no
    protocol — so liveness is not "does it answer" but "is it still writing".

    `stale` is how far behind it may fall, and it is measured **against the
    newest write in this study and not against this machine's clock**. Those are
    two clocks on two machines sharing a folder, and on a cluster they disagree
    by minutes as a matter of course; comparing writers with writers makes the
    drift cancel.

    Which leaves one honest hole: a study where **everything** stopped has no
    newest write to be behind, so nothing looks stale. That costs nothing — if
    nobody is writing, nobody is asking this either.
    """
    running: list[tuple[int, "Point"]] = []
    newest = 0
    for _, _, record in _latest(store, study):
        said = dict(record.meta)
        newest = max(newest, record.when)
        if said.get(STATE) == RUNNING and POINT in said:
            running.append((record.when, space.read(said[POINT])))
    return [(point, None) for when, point in running if newest - when <= stale]


def abandoned(
    store: "Store",
    *,
    study: str,
    stale: float = STALE,
) -> list[tuple[int, int]]:
    """Which trials have stopped moving, as `(trial, attempt)` pairs.

    It **decides nothing**, which is the whole of its contract: reclaiming one
    spends a machine's afternoon, and whether a trial that went quiet is dead, is
    preempted or is merely on a very long epoch is not something a folder can
    tell. So this reports and the loop chooses::

        for trial, attempt in abandoned(store, study=STUDY):
            take(store, point, study=STUDY, trial=trial, me=me, attempt=attempt + 1)

    The same division as a pruner: it answers, and the caller acts. And the same
    reason `claim` uses a link — reclaiming by writing over the old record would
    be a race, taking the next attempt is not.

    Being wrong is cheap in both directions: too eager is a trial run twice, and
    a claim still cannot collide, so it is wasted work and not a wrong answer.
    """
    quiet: list[tuple[int, tuple[int, int]]] = []
    newest = 0
    for trial, attempt, record in _latest(store, study):
        newest = max(newest, record.when)
        if dict(record.meta).get(STATE) == RUNNING:
            quiet.append((record.when, (trial, attempt)))
    return [
        numbered for when, numbered in quiet if newest - when > stale
    ]



def importance(
    store: "Store",
    space: "Space",
    *,
    study: str,
) -> list[tuple[str, float]]:
    """How decisive each knob was, as `(name, |rho|)`, biggest first.

    **Spearman's rho**, which is a rank correlation: how well the score follows a
    knob monotonically, without assuming a shape. It is what the original soma
    actually has — its documentation names fANOVA and says it was deferred, and
    it never arrived — and it is thirty lines of plain Python, so it stays here
    rather than becoming a dependency.

    Ranks and not values, so a knob searched in log needs no special case.

    **Only the trials that ran to the end**, for the same reason `finished`
    leaves pruned ones out: a pruned score is real and was measured after fewer
    epochs, so ranking the two together says a trial that was stopped early did
    badly when all that is known is that it was stopped.

    A categorical knob is ranked by its own options in order, which is honest for
    two of them and gets thin beyond that — a rank correlation over unordered
    categories is a number not to lean on. It is answered anyway, because leaving
    it out would be this deciding what you may look at.

    `0.0` where a knob never varied: no evidence, which is not the same as no
    effect, and a study of one point says nothing about anything.
    """
    scored = [one for one in trials(store, space, study=study)
              if one[STATE] == DONE and one[SCORE] is not None]
    if len(scored) < 2:
        return [(name, 0.0) for name in space.names()]
    scores: list[float] = [one[SCORE] for one in scored]
    said = []
    for name in space.names():
        values: list[Any] = [one[POINT][name] for one in scored]
        if any(isinstance(one, str) for one in values):
            seen = sorted({str(one) for one in values})
            values = [seen.index(str(one)) for one in values]
        said.append((name, abs(_rho(values, scores))))
    return sorted(said, key=lambda one: -one[1])


def _rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman's rho of two equally long lists."""
    a, b = _ranked(xs), _ranked(ys)
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    top = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    below = math.sqrt(
        sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    )
    return top / below if below else 0.0


def _ranked(values: Sequence[Any]) -> list[float]:
    """Ranks, **averaging ties** — which is what makes it Spearman rather than
    Pearson over whatever order the list happened to arrive in."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _latest(store: "Store", study: str) -> list[tuple[int, int, "Bound"]]:
    """One record per trial — the highest attempt of each — in trial order, as
    `(trial, attempt, record)`.

    The numbers come back with the record because working them out is how the
    highest attempt is picked in the first place. Throwing them away left two
    callers parsing the name a second time, and one of those indexed straight
    into what `_numbered` answers when a name is not one of ours.
    """
    best: dict[int, tuple[int, "Bound"]] = {}
    for record in store.bound():
        numbered = _numbered(record.name, study)
        if numbered is None:
            continue
        trial, attempt = numbered
        if trial not in best or best[trial][0] < attempt:
            best[trial] = (attempt, record)
    return [(trial, attempt, record) for trial, (attempt, record) in sorted(best.items())]


def _numbered(name: str, study: str) -> tuple[int, int] | None:
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


def _read(store: "Store", record: "Bound") -> dict[str, Any]:
    """The blob that record points at."""
    bytes_ = store.get(record.digest)
    if bytes_ is None:
        raise RuntimeError(
            f"`{record.name}` points at `{record.digest}` and this store does not "
            f"have it: the record and the bytes are two things, and one of them "
            f"is missing"
        )
    read: dict[str, Any] = json.loads(bytes_)
    return read


def _blob(
    point: "Point",
    reports: Sequence[float],
    state: str,
    because: str | None,
    took: float | None,
) -> bytes:
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


def _trial(study: str, trial: int, attempt: int) -> str:
    return f"{study}/trial/{trial}/{attempt}"
