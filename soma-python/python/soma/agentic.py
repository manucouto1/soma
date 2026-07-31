"""The named agentic patterns, as graph constructors.

Every function here returns an ordinary :class:`soma.Graph`. There is no
second engine and no new node type: a pattern is a shape you can also build
by hand out of ``node``, ``connect``, ``branch`` and ``loop``. These are the
shapes that recur, written once so nobody writes them again.

That is deliberate. A framework whose patterns are enum variants has to grow
its core for every idea anyone has; the dead node types in every such
framework are the evidence. A pattern that is a function costs nothing to add
and nothing to keep.

Because the result is a plain graph, everything else Soma does applies to it
unchanged — schema checking on the edges, the persistent cache, ``search()``
over a node's attributes, ``Study`` with pruning, the run tracker and the
experiment pool.

    from soma.agentic import react, route, refine, debate, parallel_vote

    g = refine(
        worker=soma.Agent(model="ollama/qwen2.5", system="Write a haiku."),
        judge=soma.Judge(model="ollama/qwen2.5", rubric="Exactly 5-7-5."),
        max_rounds=4,
    )
    print(g.forward("a haiku about compilers"))
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from soma._soma import Graph
from soma.filter import Filter

__all__ = [
    "Revise",
    "react",
    "route",
    "refine",
    "debate",
    "parallel_vote",
    "orchestrate",
]


class Revise(Filter):
    """Turns a verdict into the instruction the next round reads.

    A refine loop carries the judge's verdict back to the top, and a verdict
    is a mapping — which is not something you can hand an agent, because an
    agent's input is a conversation. Converting one to the other is a real
    step in the loop, so it is a node you can see and replace rather than
    leniency buried in the agent.

    The first round has no verdict yet; the request passes through untouched.
    """

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, instruction: str = "Revise it."):
        self.instruction = instruction

    def forward(self, x, state):
        if not isinstance(x, dict):
            return x

        previous = x.get("value", "")
        if isinstance(previous, (dict, list)):
            previous = json.dumps(previous)
        reason = x.get("reason") or "No reason was given."
        score = x.get("score")

        scored = f"It scored {score}. " if score is not None else ""
        return (
            f"Your previous attempt:\n\n{previous}\n\n"
            f"{scored}{reason}\n\n{self.instruction}"
        )


def _graph(provider: str | None, cache: str | None) -> Graph:
    g = Graph() if cache is None else Graph(cache=cache)
    if provider is not None:
        g.use_provider(provider)
    return g


def react(
    model: str,
    tools: Iterable[Any] = (),
    *,
    system: str | None = None,
    max_turns: int | None = None,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """One agent that thinks, calls tools, and answers.

    The reason-act loop from Yao et al.: the model may call any of ``tools``,
    sees each result, and keeps going until it answers in prose. The turn
    budget is a guard against a model that never stops, not a target.
    """
    import soma

    g = _graph(provider, cache)
    kwargs: dict[str, Any] = {"model": model, "tools": list(tools)}
    if system is not None:
        kwargs["system"] = system
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    g.node("agent", soma.Agent(**kwargs))
    return g


def route(
    classifier: Any,
    arms: Mapping[str, Any],
    *,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """Send each request to exactly one of several handlers.

    ``classifier`` names the arm — as a bare string, a bool, or a mapping with
    a ``branch`` key. It decides *where* the request goes, not *what* goes:
    the chosen arm receives the original request, not the label. An arm
    called ``default`` (or ``else``) catches anything unmatched; without one,
    a label matching no arm is an error rather than a silent drop.
    """
    if not arms:
        raise ValueError("route() needs at least one arm")

    g = _graph(provider, cache)
    g.branch("router", classifier, dict(arms))
    return g


def refine(
    worker: Any,
    judge: Any,
    *,
    max_rounds: int = 3,
    revise: Any | None = None,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """Draft, grade, redraft — until the grade is good enough.

    The evaluator-optimizer pattern. ``judge`` must report whether the work
    is done; :class:`soma.Judge` does, and so does any filter returning a
    mapping with a ``done`` key. Its verdict is what the next round reads, so
    the worker sees both the critique and its own previous attempt.

    ``max_rounds`` is a ceiling, not a plan: a loop that keeps going because
    the judge never passes is spending real money.

    The loop is ``revise → worker → judge``. :class:`Revise` is what turns
    the carried verdict into something the worker can read; pass your own
    ``revise=`` to word the instruction differently. The loop's result is
    the last verdict, since that is where the score is.
    """
    if max_rounds < 1:
        raise ValueError("refine() needs at least one round")

    g = _graph(provider, cache)
    g.node("revise", revise if revise is not None else Revise())
    g.node("worker", worker)
    g.node("judge", judge)
    g.connect("revise", "worker")
    g.connect("worker", "judge")
    g.loop("refine", body="revise", until="judge", max_iterations=max_rounds)
    return g


def debate(
    agents: Iterable[Any],
    *,
    rounds: int = 2,
    judge: Any | None = None,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """Several agents answer in turn, each seeing what came before.

    Each round runs the agents in sequence, so a later agent can disagree
    with an earlier one; the next round starts from where the last ended. A
    ``judge``, if given, reads the final exchange and decides — and if it
    reports ``done`` the debate stops early, which is the usual case when the
    agents converge before the round budget runs out.
    """
    agents = list(agents)
    if len(agents) < 2:
        raise ValueError("debate() needs at least two agents")
    if rounds < 1:
        raise ValueError("debate() needs at least one round")

    g = _graph(provider, cache)
    ids = [g.node(f"agent_{i}", a) for i, a in enumerate(agents)]
    for a, b in zip(ids, ids[1:]):
        g.connect(a, b)

    if judge is None:
        # No judge: nothing can say the argument is settled, so run the full
        # round budget. The last agent's answer is what the debate produced.
        g.loop("debate", body=ids[0], until=False, max_iterations=rounds)
    else:
        judge_id = g.node("judge", judge)
        g.connect(ids[-1], judge_id)
        g.loop("debate", body=ids[0], until=judge_id, max_iterations=rounds)
    return g


def parallel_vote(
    agents: Iterable[Any],
    aggregator: Any,
    *,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """Ask several agents the same thing at once, then reconcile.

    The agents share no state, which is the point: independent attempts
    disagree in informative ways. ``aggregator`` receives a mapping from node
    id to that agent's answer — a judge, a majority vote, or a filter that
    concatenates them all, whatever reconciliation the task wants.
    """
    agents = list(agents)
    if len(agents) < 2:
        raise ValueError("parallel_vote() needs at least two agents")

    g = _graph(provider, cache)
    ids = [g.node(f"voter_{i}", a) for i, a in enumerate(agents)]
    agg_id = g.node("aggregate", aggregator)
    for node_id in ids:
        g.connect(node_id, agg_id)
    return g


def orchestrate(
    planner: Any,
    worker: Any,
    synthesizer: Any,
    *,
    n_workers: int = 3,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """A planner breaks the work up, workers do it, a synthesizer joins it.

    The worker pool is a fixed ``n_workers`` — every worker sees the whole
    plan and takes the part addressed to it. Soma cannot yet size the pool
    from the plan at runtime (that needs the dynamic fan-out of
    ``Transition::Spawn``, which no Python step can produce yet), so pick a
    pool wide enough for the work and expect idle workers on small plans.
    """
    if n_workers < 1:
        raise ValueError("orchestrate() needs at least one worker")

    g = _graph(provider, cache)
    plan_id = g.node("planner", planner)
    synth_id = g.node("synthesize", synthesizer)
    for i in range(n_workers):
        worker_id = g.node(f"worker_{i}", worker)
        g.connect(plan_id, worker_id)
        g.connect(worker_id, synth_id)
    return g
