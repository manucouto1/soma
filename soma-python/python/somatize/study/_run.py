"""How N machines that never speak to each other search one space together.

`Sampler` says where to look and `Pruner` when to stop; this is the meeting.
Everybody runs the same script over a directory they all mounted, and `claim`
hands out the work — no server, no port, no protocol::

    for trial in range(100):
        point = sampler.ask(space, trial, finished(store, space, study="spam"))
        if not take(store, point, study="spam", trial=trial, me=me, goal="min"):
            continue                               # somebody else has that one
        ...
        report(store, point, reports, study="spam", trial=trial, me=me,
               state="done", goal="min")

**Handing out work costs no message because nothing is handed out.** A trial is a
number and `ask` is a function of that number, so a machine that claims trial 7
derives its point without replaying six: the state *is* the queue, and a claim is
exactly-once by construction. `Sampler.tpe` is the exception, being guided, and
`in_flight` is what keeps it from proposing next to what somebody else is trying.

A trial lives at `<study>/trial/<n>/<attempt>`. In the **record**, which a scan
already carries: `state`, `point`, `score`, `who`, `goal`. In the **blob**: the
whole curve and why it stopped. That split is the cost model — a sampler's whole
history is one scan and zero fetches.

`goal` is written per trial, which **is** denormalised: a score is good or bad
and the number does not say which, and whoever reads the study without this
script has no other way to find out. Per trial rather than per study because it
records what was meant **at the time**.

One record rewritten as it goes and not five events: a bus's
`TrialStarted`/`TrialPruned`/`TrialCompleted` are the *diff* of this record, and
from a lossy stream the state cannot be derived.
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
    """Claims the `trial`-th trial of `study`. `True` when it is this machine's;
    `False` means somebody else got there first and the loop goes on to the next
    number. `goal` is written here as well as on every `report`, so a trial
    claimed and never reported still says which way it was looking.
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
    """Writes down where this trial has got to, as often as there is something to
    say — once an epoch makes a curve watchable from another machine while it is
    still being drawn.

    Only the machine that claimed it writes, and nothing has to enforce that:
    nobody else could have got the claim. `goal` goes beside the score because a
    score without it is not readable by anybody without this script.
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
    """`min` or `max`, or the error where the typo was typed. Checked here rather
    than at the far end: a study that wrote `minimize` into two thousand records
    is a directory to migrate, not a figure that draws badly.
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
    """Every trial that ran to the end, as `(point, score)` — what `ask` wants,
    in **one scan and no fetches**.

    Pruned trials are left out on purpose: a pruned score is real but was
    measured after fewer epochs, so a sampler that treats it as a bad
    configuration learns something untrue.
    """
    history: list[tuple["Point", float]] = []
    for _, _, record in _latest(store, study):
        said = dict(record.meta)
        if said.get(STATE) != DONE or SCORE not in said:
            continue
        history.append((space.read(said[POINT]), float(said[SCORE])))
    return history


def direction(store: "Store", *, study: str) -> str | None:
    """Which way is better in this study — `"min"`, `"max"`, or `None`. One scan
    and no fetches.

    `None` means no trial said, and it is the honest answer rather than `"min"`.
    When records disagree the newest wins — the direction is what the person
    running the study currently means. Ties break by the **higher trial number**,
    because a study writes its first records inside the same second.
    """
    said: tuple[tuple[int, int], str] | None = None
    for trial, _, record in _latest(store, study):
        goal = dict(record.meta).get(GOAL)
        if goal is not None and (said is None or said[0] < (record.when, trial)):
            said = ((record.when, trial), goal)
    return said[1] if said else None


def curves(store: "Store", *, study: str) -> list[list[float]]:
    """The reports of every trial that ran to the end — what a `Pruner` wants.
    The reader that pays: a curve grows, so it lives in the blob and this is a
    scan **plus one fetch per trial**.
    """
    drawn: list[list[float]] = []
    for _, _, record in _latest(store, study):
        if dict(record.meta).get(STATE) != DONE:
            continue
        drawn.append(_read(store, record)["reports"])
    return drawn


def trials(store: "Store", space: "Space", *, study: str) -> list[Trial]:
    """Every trial of this study, whatever state it is in, as records. The one for
    looking rather than deciding: what is still running, and whether the study is
    done.
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

    That is *constant liar* (Ginsbourger, Le Riche and Carraro, 2010) without the
    lie, and the difference was measured. Handing the sampler a made-up bad score
    **backfires**: `Tpe` sizes the pile it imitates as a share of everything it is
    handed, so one more point raises the quota and can promote a trial out of the
    bad pile — one proposal in two hundred landed on the occupied region without
    it, thirty-nine with it. So `None` says *running* and does not vote.

    One scan and **no fetches**. `stale` is how far behind a `running` trial may
    fall before it counts as stopped, measured **against the newest write in this
    study and not against this machine's clock** — two machines sharing a folder
    are two clocks that disagree by minutes.
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

    It **decides nothing**: whether a quiet trial is dead, preempted or on a very
    long epoch is not something a folder can tell. So this reports and the loop
    chooses::

        for trial, attempt in abandoned(store, study=STUDY):
            take(store, point, study=STUDY, trial=trial, me=me, attempt=attempt + 1)

    Taking the next attempt rather than writing over the old record, for the same
    reason `claim` uses a link. Being wrong is cheap: too eager is a trial run
    twice, and a claim still cannot collide.
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

    **Spearman's rho**: how well the score follows a knob monotonically, without
    assuming a shape. Ranks rather than values, so a knob searched in log needs
    no special case, and only the trials that ran to the end.

    A categorical knob is ranked by its own options in order, which is honest for
    two and thin beyond that — answered anyway, because leaving it out would be
    this deciding what you may look at. `0.0` where a knob never varied: no
    evidence, which is not no effect.
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
    `(trial, attempt, record)`. The numbers come back because working them out is
    how the highest is picked; throwing them away left two callers parsing the
    name again.
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
    another machine — and none of them should need this library's version of
    anything to look at a study.
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
