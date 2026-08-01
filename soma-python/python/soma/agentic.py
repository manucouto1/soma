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

    from soma.agentic import react, route, refine, debate, board, parallel_vote

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
    "Validate",
    "Brief",
    "MajorityVote",
    "Fanout",
    "Done",
    "Await",
    "Spawn",
    "Goto",
    "Suspend",
    "Sleep",
    "Custom",
    "Run",
    "Llm",
    "ToolCall",
    "react",
    "route",
    "refine",
    "debate",
    "board",
    "self_consistency",
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


# ── Writing a step in Python ──
#
# A step is any object with `poll(ctx)`. It returns one of these — plain
# dicts, so that what crosses into Rust is data rather than a class
# hierarchy, and so a step can be written without importing anything but
# these five names.
#
#     class Fanout:
#         _cache_version = "1"
#         def poll(self, ctx):
#             if ctx.turn == 0:
#                 return Spawn([Run("worker", item) for item in ctx.input])
#             return Done([r["output"] for r in ctx.results])
#
# `poll` must be deterministic given the same context: that is what makes
# replay work. Put anything that is not — a model call, a tool, a clock —
# in an effect, where the journal records it once and replays it after.


def Done(value=None):  # noqa: N802 — these read as constructors
    """Finished, with this output."""
    return {"transition": "done", "value": value}


def Await(*effects):  # noqa: N802
    """Perform these effects concurrently, then poll again with the results
    in `ctx.results`, in the order asked."""
    if len(effects) == 1 and isinstance(effects[0], (list, tuple)):
        effects = tuple(effects[0])
    return {"transition": "await", "effects": list(effects)}


def Spawn(specs, join="all"):  # noqa: N802
    """Run these nodes now, then poll again with their outputs.

    The map half of map-reduce: the fan-out width comes from the data, not
    from the graph, which is the one thing a static topology cannot express.
    `join` is ``"all"`` (a failure fails the join), ``"all_settled"`` (keep
    whatever succeeded) or ``"first"``.
    """
    return {"transition": "spawn", "specs": list(specs), "join": join}


def Goto(target, carry=None):  # noqa: N802
    """Hand control to another node. This step is done."""
    return {"transition": "goto", "target": target, "carry": carry}


def Suspend(reason="waiting"):  # noqa: N802
    """Stop the run and persist it; resuming replays to here and continues."""
    return {"transition": "suspend", "reason": reason}


def Sleep(seconds):  # noqa: N802
    """Wait. Journaled like anything else, so a replay does not sleep again."""
    return {"effect": "sleep", "seconds": seconds}


def Custom(kind, payload=None):  # noqa: N802
    """An effect this runtime does not know, for a host-supplied handler."""
    return {"effect": "custom", "kind": kind, "payload": payload}


def Run(runs, input=None, label=None):  # noqa: A002, N802
    """One spawned node: which registered step to run, and on what."""
    return {"runs": runs, "input": input, "label": label}


def Llm(model, prompt, system=None):  # noqa: N802
    """Ask a model. The answer arrives as ``{"kind": "llm", "text": ...}``."""
    return {"effect": "llm", "model": model, "prompt": prompt, "system": system}


def ToolCall(name, args=None):  # noqa: N802
    """Call a registered tool by name."""
    return {"effect": "tool", "name": name, "args": args or {}}


#: Separates the question from the panel report inside a brief. The chair
#: splits on it to recover the bare question, so that a brief built from a
#: brief does not nest: without this the question grows by a whole report
#: every round and the members read an ever-longer accretion of themselves.
PANEL_MARKER = "Other members of the panel answered as follows:"


class Brief(Filter):
    """Turns the board's last verdict into the next round's brief.

    Round one has no verdict, so the question passes through untouched.
    After that the members are shown what the board recorded — the tallied
    answers and, when the chair wrote one, its summary — and asked to answer
    again. This is the message Du et al. inject between rounds; giving every
    member the same summarized message rather than each member the verbatim
    text of every other is the summarizer variant the paper introduces for
    larger panels, and it is what makes a moderator a moderator.
    """

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, instruction: str = "Answer the question again."):
        self.instruction = instruction

    def forward(self, x, state):
        if not isinstance(x, dict):
            return x  # round one: the question itself

        question = x.get("question") or ""
        summary = x.get("summary")
        if not summary:
            # The members' *reasoning*, not a tally of their conclusions.
            # Passing only "one member answered 18" strips exactly what makes
            # a debate work — an agent cannot be persuaded by a number it
            # cannot check, and a panel shown only counts converges on
            # whichever wrong answer was most popular. Du et al. hand each
            # agent the other agents' full solutions.
            responses = x.get("responses") or []
            summary = "\n\n".join(
                f"One agent solution: ```{r}```" for r in responses
            ) or "nothing"

        return (
            f"{question}\n\n"
            f"{PANEL_MARKER}\n{summary}\n\n"
            f"Using their answers as additional information, {self.instruction}"
        )


class MajorityVote(Filter):
    """The board as a count rather than a model.

    Reads the fan-in mapping every member wrote into and returns the most
    common answer — the aggregator Du et al. use to close a debate. Ties go
    to whichever answer was seen first, which is arbitrary but at least
    deterministic.

    ``done`` is true once the members agree unanimously. That is what lets a
    board stop at round two instead of buying every round it was allowed:
    once a panel has converged, further rounds cost tokens and change
    nothing.

    The chair also republishes ``question`` so the next brief still knows
    what is being decided — a vote that forgets the question cannot run a
    second round.
    """

    _kind = "stateless"
    _cache_version = "1"

    #: Answers are read as the last ``\boxed{...}`` in the text, else the
    #: last number in it. Both conventions come from the GSM8K literature.
    def __init__(self, brief_node: str = "brief", mode: str = "number"):
        if mode not in ("number", "text"):
            raise ValueError("mode must be 'number' or 'text'")
        self.brief_node = brief_node
        #: ``"number"`` reads a final answer out of a worked solution, the
        #: GSM8K convention. ``"text"`` compares whole replies, normalized —
        #: which is what a panel answering in prose needs, and where reading
        #: "the last number" would vote on a page reference.
        self.mode = mode

    def read(self, text) -> str | None:
        """The answer this vote counts, under the configured mode."""
        if self.mode == "text":
            from soma.library import normalize_answer

            normalized = normalize_answer(text)
            return normalized or None
        return self.extract(text)

    @staticmethod
    def extract(text: str) -> str | None:
        """The final answer in a worked solution, or None if there isn't one."""
        import re

        if not isinstance(text, str):
            text = str(text)
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
        candidate = boxed[-1] if boxed else None
        if candidate is None:
            numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
            candidate = numbers[-1] if numbers else None
        if candidate is None:
            return None
        cleaned = candidate.replace(",", "").replace("$", "").strip().rstrip(".")
        # Normalize 18.0 and 18 to the same vote; they are the same answer.
        try:
            number = float(cleaned)
            return str(int(number)) if number == int(number) else str(number)
        except ValueError:
            return cleaned or None

    def forward(self, x, state):
        from collections import Counter

        if not isinstance(x, dict):
            x = {"member": x}

        brief = x.get(self.brief_node)
        # Round one's brief is the bare question; later ones are the question
        # plus the report this chair wrote. Take the part before the marker so
        # the question republished here is the same string every round.
        question = brief.split(PANEL_MARKER)[0].strip() if isinstance(brief, str) else ""

        members = {k: v for k, v in sorted(x.items()) if k != self.brief_node}
        responses = [v if isinstance(v, str) else str(v) for v in members.values()]
        answers = {node: self.read(text) for node, text in members.items()}
        # An unreadable answer is not a vote. The reference implementation of
        # this paper scores unparseable output as *correct*, which inflates
        # every number it reports; dropping it is the honest reading.
        votes = Counter(a for a in answers.values() if a is not None)

        if not votes:
            return {
                "question": question,
                "answer": None,
                "votes": {},
                "agreement": 0.0,
                "done": False,
                "value": None,
                "responses": responses,
            }

        answer, count = votes.most_common(1)[0]
        return {
            "question": question,
            "answer": answer,
            "votes": dict(votes),
            "agreement": count / sum(votes.values()),
            "done": len(votes) == 1,
            "value": answer,
            # Verbatim, so the next round argues with reasoning rather than
            # with a scoreboard.
            "responses": responses,
        }


class Validate(Filter):
    """Check a value against a JSON Schema, and say what is wrong.

    Returns a verdict rather than raising, so the invalid case goes
    somewhere that handles it instead of costing another model call to find
    out what went wrong:

        {"ok": bool, "errors": [...], "value": ..., "branch": "ok"|"invalid"}

    The ``branch`` key is the label a :meth:`Graph.branch` reads — the
    contract a branch condition already has, rather than a new convention
    invented for this filter:

        g.branch("check", Validate(SCHEMA), {"ok": Use(), "invalid": Repair()})

    Uses the ``jsonschema`` package when it is installed. Without it, falls
    back to checking the root type, required fields and declared property
    types — which catches what actually goes wrong (prose instead of JSON, a
    missing field the caller is about to index) and says so rather than
    pretending to be a full implementation.
    """

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, schema: Mapping[str, Any], *, strict: bool = False):
        self.schema = dict(schema)
        self.strict = strict

    def forward(self, x, state):
        value = x
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return self._verdict(x, ["the value is not JSON"])
        return self._verdict(value, _schema_errors(self.schema, value))

    def _verdict(self, value, errors):
        if errors and self.strict:
            raise ValueError(f"validation failed: {'; '.join(errors)}")
        return {
            "ok": not errors,
            "errors": errors,
            "value": value,
            "branch": "invalid" if errors else "ok",
        }


def _schema_errors(schema: Mapping[str, Any], value: Any) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return _structural_errors(schema, value)

    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    ]


_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
}


def _structural_errors(schema: Mapping[str, Any], value: Any) -> list[str]:
    """Root type, required fields, declared property types.

    Permissive on purpose: a violation missed costs a consumer an error it
    would have hit anyway, while a violation invented sends a correct answer
    back for a "fix".
    """
    errors: list[str] = []

    expected = schema.get("type")
    if expected and not _is_type(expected, value):
        # With the root type wrong, every field check below would fire too
        # and bury the one that matters.
        return [f"expected {expected}, got {type(value).__name__}"]

    if not isinstance(value, dict):
        return errors

    for key in schema.get("required", []):
        if key not in value:
            errors.append(f"missing required field `{key}`")

    for key, spec in (schema.get("properties") or {}).items():
        declared = spec.get("type") if isinstance(spec, Mapping) else None
        if key in value and declared and not _is_type(declared, value[key]):
            errors.append(
                f"field `{key}` should be {declared}, got {type(value[key]).__name__}"
            )

    return errors


def _is_type(expected: str, value: Any) -> bool:
    if expected == "integer":
        # A JSON Schema `integer` accepts 2.0; booleans are not integers.
        return isinstance(value, int) and not isinstance(value, bool) or (
            isinstance(value, float) and value.is_integer()
        )
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    python_type = _JSON_TYPES.get(expected)
    # A type this check does not model is not a type it may reject.
    if python_type is None:
        return True
    if python_type is not bool and isinstance(value, bool):
        return False
    return isinstance(value, python_type)


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


def board(
    members: Iterable[Any],
    chair: Any | None = None,
    *,
    rounds: int = 2,
    brief: Any | None = None,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """A panel answers, a chair moderates, and the panel answers again.

    This is the multi-agent debate of Du et al. (ICML 2024) with the
    summarizer variant that paper introduces for larger panels: every member
    answers the question independently, the ``chair`` reads all of their
    answers and records a decision, and the next round shows every member
    what the chair recorded. The loop is ``brief → members → chair``, and
    the chair is what stops it — the panel converging is the exit condition,
    not a round budget running out.

    The default chair is :class:`MajorityVote`, which is the aggregator the
    paper actually uses to close a debate and which costs no tokens. Pass a
    :class:`soma.Judge` or an agent instead to have a model moderate, in
    which case it must report ``done`` the way a judge does.

    ``rounds`` is a ceiling. Du et al. report accuracy saturating at three to
    four rounds, so paying for more buys very little.

    The members run in parallel and share no state: they influence each other
    only through the chair, which is what makes the chair worth having. The
    chair also reads the brief, so it can republish the question and the
    round after it still knows what is being decided.

        board([solver, solver, solver], rounds=2)

    Every member being the same agent is normal — sampling makes them
    disagree, and disagreement is what a panel is for.
    """
    members = list(members)
    if len(members) < 2:
        raise ValueError("board() needs at least two members")
    if rounds < 1:
        raise ValueError("board() needs at least one round")

    g = _graph(provider, cache)
    g.node("brief", brief if brief is not None else Brief())
    chair_id = g.node("chair", chair if chair is not None else MajorityVote())

    for i, member in enumerate(members):
        member_id = g.node(f"member_{i}", member)
        g.connect("brief", member_id)
        g.connect(member_id, chair_id)
    g.connect("brief", chair_id)

    g.loop("board", body="brief", until=chair_id, max_iterations=rounds)
    return g


def self_consistency(
    agent: Any,
    *,
    n: int = 5,
    aggregator: Any = None,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """Ask one agent the same thing `n` times, and take the majority.

    Self-consistency (Wang et al.): sampling the same model repeatedly and
    voting beats its single best answer, because the wrong answers disagree
    with each other while the right one keeps coming back.

    Distinct from :func:`parallel_vote` and :func:`board`, which need `n`
    *different* agents written out. Here the diversity comes from sampling,
    which is the version the paper actually measures — and the one that was
    impossible to express before, since every copy of one agent would have
    been the same node.

    The default aggregator counts whole answers; pass
    ``MajorityVote(mode="number")`` for worked solutions that end in one.

        self_consistency(soma.Agent(model="ollama/qwen2.5"), n=5)

    Note that identical samples of a deterministic model are exactly what
    the cache collapses — this is worth its tokens only against a model
    sampling at a temperature above zero.
    """
    if n < 2:
        raise ValueError("self_consistency() needs at least two samples")

    g = _graph(provider, cache)
    voter = aggregator if aggregator is not None else MajorityVote(mode="text")
    vote_id = g.node("vote", voter)

    for i in range(n):
        # One node per sample, because a node is what the runtime schedules:
        # they run in parallel, cache separately, and appear separately in
        # the report.
        sample_id = g.node(f"sample_{i}", agent)
        g.connect(sample_id, vote_id)
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


class Fanout:
    """Spawns one worker per item of its input, then hands back the results.

    The width comes from the data, which is the one thing a static topology
    cannot express: a plan with two tasks runs two workers and a plan with
    nine runs nine, decided while the graph is running.

    Its input can be a list, a mapping with a ``tasks`` key, or the plain
    text a planner agent wrote — in which case each non-empty line is a task,
    with any bullet or numbering stripped.
    """

    _cache_version = "1"

    def __init__(self, runs: str = "worker", max_workers: int = 16):
        self.runs = runs
        self.max_workers = max_workers

    @staticmethod
    def tasks(value) -> list:
        if isinstance(value, dict):
            value = value.get("tasks", value.get("plan", []))
        if isinstance(value, str):
            lines = [line.strip(" -*\t") for line in value.splitlines()]
            return [
                line.split(". ", 1)[-1] if line[:1].isdigit() else line
                for line in lines
                if line.strip()
            ]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def poll(self, ctx):
        if ctx.turn == 0:
            work = self.tasks(ctx.input)[: self.max_workers]
            if not work:
                # Spawning nothing would spin. An empty plan has an answer.
                return Done([])
            return Spawn(
                [Run(self.runs, task, label=f"w{i}") for i, task in enumerate(work)]
            )
        return Done([r.get("output") for r in ctx.results])


def orchestrate(
    planner: Any,
    worker: Any,
    synthesizer: Any,
    *,
    max_workers: int = 16,
    provider: str | None = None,
    cache: str | None = None,
) -> Graph:
    """A planner breaks the work up, workers do it, a synthesizer joins it.

    The worker pool is sized from the plan at runtime: `planner → fanout →
    synthesize`, where ``fanout`` spawns one ``worker`` per task the planner
    named. This is the orchestrator-workers pattern with the fan-out actually
    dynamic — the shape of the graph is not known until the planner has
    spoken.

    ``worker`` is registered rather than wired, because a spawned node is
    named by the step that spawns it. A node with no edges would also be a
    root and run once on the graph's own input; registering it keeps it
    spawnable and nothing else.

    ``max_workers`` is a ceiling on how far a plan can fan out — a planner
    that emits four hundred lines should not open four hundred conversations.
    """
    if max_workers < 1:
        raise ValueError("orchestrate() needs to allow at least one worker")

    g = _graph(provider, cache)
    g.node("planner", planner)
    g.node("fanout", Fanout(runs="worker", max_workers=max_workers))
    g.node("synthesize", synthesizer)
    g.register_step("worker", worker)
    g.connect("planner", "fanout")
    g.connect("fanout", "synthesize")
    return g
