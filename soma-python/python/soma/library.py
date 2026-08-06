"""Filters worth having, so nobody writes them twice.

The design says behaviour is library rather than node types in the core.
That claim is only worth anything if the library exists — otherwise every
user writes their own scorer, their own retriever, and their own way of
keeping a transcript from growing without bound, and each one is subtly
different when the results are compared later.

These are ordinary :class:`soma.Filter` subclasses. Nothing here has
privileged access; anything they do, your own filter can do.

    from soma.library import Eval, Accumulator, Retriever, Compact

They live in Python because that is where the primary interface is and
because they are thin. A Rust user does not get them.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Sequence

from soma.filter import Filter

__all__ = ["Eval", "Accumulator", "Retriever", "Compact", "normalize_answer"]


# ── Scoring ──────────────────────────────────────────────────


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize_answer(text: Any) -> str:
    """Lowercase, drop articles and punctuation, collapse whitespace.

    The SQuAD normalization, and the reason ``"The answer is 42."`` and
    ``"answer is 42"`` count as agreeing. Comparing raw strings instead
    makes a scorer measure formatting.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _PUNCT.sub(" ", text.lower())
    text = _ARTICLES.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def _token_f1(prediction: Any, reference: Any) -> float:
    """Token-overlap F1 — SQuAD's, for answers that are phrases."""
    from collections import Counter

    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        # Two empty answers agree; one empty and one not do not.
        return float(predicted == expected)

    shared = sum((Counter(predicted) & Counter(expected)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(predicted)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def _exact(prediction: Any, reference: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _accuracy(prediction: Any, reference: Any) -> float:
    """Equality, without the string normalization — for labels and numbers,
    where ``1`` and ``"1"`` should still agree but ``"The 1"`` should not."""
    if isinstance(prediction, str) != isinstance(reference, str):
        return float(str(prediction).strip() == str(reference).strip())
    return float(prediction == reference)


class Eval(Filter):
    """Score predictions against references, with real metrics.

    The counterpart to :class:`soma.Judge`: a judge asks a model whether the
    work is good, which costs tokens and returns a different number each
    time. When there is a reference to compare against, a metric answers
    exactly, for free, and reproducibly. Use the judge for what has no
    reference.

    Input is a mapping holding both sides — ``{"prediction": …,
    "reference": …}`` by default, or the node ids of a fan-in::

        g.edge("model", "score")
        g.edge("truth", "score")
        g.node("score", Eval(metrics=["accuracy"], prediction="model",
                             reference="truth"))

    Both sides may be single values or equal-length sequences. Output is a
    mapping of metric name to score, plus ``n``, which is what a ``Study``
    objective reads.

    Available metrics:

    - ``accuracy`` — equality, for labels and numbers
    - ``exact_match`` — equality after SQuAD normalization, for text
    - ``f1`` — token-overlap F1, for phrases
    - ``top_k`` — the reference appears in the first ``k`` predictions

    ``missing`` decides what to do with a pair where either side is absent:
    ``"error"`` (the default — a scorer that quietly ignores half its data
    reports a number nobody can interpret), ``"skip"``, or ``"zero"``.
    """

    _kind = "stateless"
    _cache_version = "1"

    _METRICS = {
        "accuracy": _accuracy,
        "exact_match": _exact,
        "f1": _token_f1,
    }

    def __init__(
        self,
        metrics: Iterable[str] = ("accuracy",),
        *,
        prediction: str = "prediction",
        reference: str = "reference",
        k: int = 1,
        missing: str = "error",
    ):
        self.metrics = list(metrics)
        self.prediction = prediction
        self.reference = reference
        self.k = k
        self.missing = missing

        unknown = [m for m in self.metrics if m not in self._METRICS and m != "top_k"]
        if unknown:
            raise ValueError(
                f"unknown metric(s) {unknown}. Available: "
                f"{sorted([*self._METRICS, 'top_k'])}"
            )
        if missing not in ("error", "skip", "zero"):
            raise ValueError("missing must be 'error', 'skip' or 'zero'")

    def forward(self, x, state):
        predictions, references = self._sides(x)
        pairs = self._pairs(predictions, references)

        if not pairs:
            # An accuracy computed over nothing is not zero, it is undefined,
            # and reporting 0.0 puts a fabricated number in a record someone
            # will compare against later.
            raise ValueError(
                "Eval has nothing to score: no prediction/reference pairs "
                "survived. Check the node names or `missing=`."
            )

        scored = {"n": len(pairs)}
        for name in self.metrics:
            if name == "top_k":
                scored["top_k"] = sum(
                    self._top_k_hit(p, r) for p, r in pairs
                ) / len(pairs)
            else:
                fn = self._METRICS[name]
                scored[name] = sum(fn(p, r) for p, r in pairs) / len(pairs)
        return scored

    def _sides(self, x) -> tuple[Any, Any]:
        if not isinstance(x, Mapping):
            raise ValueError(
                f"Eval expects a mapping holding `{self.prediction}` and "
                f"`{self.reference}`, got {type(x).__name__}"
            )
        for key in (self.prediction, self.reference):
            if key not in x:
                raise ValueError(
                    f"Eval: no `{key}` in the input. Present: {sorted(x)}"
                )
        return x[self.prediction], x[self.reference]

    def _pairs(self, predictions, references) -> list[tuple[Any, Any]]:
        if not _is_sequence(predictions):
            predictions, references = [predictions], [references]
        elif not _is_sequence(references):
            raise ValueError("Eval: predictions are a sequence but references are not")

        predictions, references = list(predictions), list(references)
        if len(predictions) != len(references):
            raise ValueError(
                f"Eval: {len(predictions)} prediction(s) against "
                f"{len(references)} reference(s) — they must line up"
            )

        pairs = []
        for prediction, reference in zip(predictions, references):
            if prediction is None or reference is None:
                if self.missing == "error":
                    raise ValueError(
                        "Eval: a prediction or reference is missing. Pass "
                        "`missing='skip'` to drop the pair or "
                        "`missing='zero'` to score it wrong."
                    )
                if self.missing == "skip":
                    continue
                prediction = None if prediction is None else prediction
            pairs.append((prediction, reference))
        return pairs

    def _top_k_hit(self, prediction, reference) -> float:
        candidates = prediction if _is_sequence(prediction) else [prediction]
        wanted = normalize_answer(reference)
        return float(
            any(normalize_answer(c) == wanted for c in list(candidates)[: self.k])
        )


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


# ── Remembering ──────────────────────────────────────────────


class Accumulator(Filter):
    """Keep every value that passed through, not just the last one.

    A loop carries one value from each pass to the next, which is what a
    refine loop needs and not enough for a Reflexion one: to stop repeating
    a mistake, the worker has to see *all* its previous attempts, not the
    most recent.

    Put it at the end of the body, so its output becomes the carry::

        g.edge("judge", "remember")
        g.loop("refine", body="revise", until="judge", max_iterations=5)

    Output is the value it was handed, plus ``history`` (oldest first).

    **This filter holds state, which is the one thing Soma's design pushes
    back on.** There is no stateless way to do it with the current loop: the
    accumulated list would have to travel through every node in the body,
    and the nodes in between — a model, a judge — do not carry values they
    were not asked about. So it declares ``_deterministic = False`` and is
    excluded from output caching, which is correct but means the usual "same
    input, same answer" guarantee does not hold here.

    The list is per-instance, not per-run: a second ``forward()`` on the same
    graph keeps accumulating. Call :meth:`reset` between runs, or build a
    fresh graph.
    """

    _kind = "stateless"
    _deterministic = False
    _cache_version = "1"

    def __init__(self, key: str = "history", limit: int | None = None):
        self.key = key
        self.limit = limit
        self._seen: list[Any] = []

    def reset(self) -> None:
        """Forget everything. Call between runs of the same graph."""
        self._seen = []

    @property
    def history(self) -> list[Any]:
        return list(self._seen)

    def forward(self, x, state):
        self._seen.append(x)
        if self.limit is not None:
            # Keeping the most recent is the useful direction: a worker
            # improves against what it just did, not what it did first.
            self._seen = self._seen[-self.limit :]

        out = dict(x) if isinstance(x, Mapping) else {"value": x}
        out[self.key] = list(self._seen)
        return out


class Retriever(Filter):
    """Look up what has already been tried that bears on this.

    Over the experiment pool — the same ``.soma/experiments.jsonl`` a
    human's runs write to. There is no separate agent memory: an agent
    remembers what was run because running it recorded it.

    Input is the query (a string, or a mapping with ``query``). Output is
    that input with ``retrieved`` attached: a list of ``{"name", "score",
    "hypothesis", "metrics", "conclusion"}``, most relevant first, trimmed
    to what fits in a prompt rather than the whole record.

    Failures are retrievable on purpose, and heavily: not repeating a dead
    end saves as much time as repeating a success.
    """

    _kind = "stateless"
    _deterministic = False  # the pool grows underneath it
    _cache_version = "1"

    def __init__(
        self,
        *,
        k: int = 5,
        query: str = "query",
        research_line: str | None = None,
        root: str = ".soma",
    ):
        self.k = k
        self.query = query
        self.research_line = research_line
        self.root = root

    def forward(self, x, state):
        from soma._lineage import find_similar

        text = x[self.query] if isinstance(x, Mapping) and self.query in x else x
        if not isinstance(text, str):
            text = json.dumps(text, default=str)

        try:
            hits = find_similar(
                text,
                limit=self.k,
                research_line=self.research_line,
                root=self.root,
            )
        except Exception as e:  # noqa: BLE001
            # An empty or unreadable pool is the normal state of a new
            # project. Saying "nothing found" beats ending a run over it.
            hits, note = [], str(e)
        else:
            note = None

        out = dict(x) if isinstance(x, Mapping) else {self.query: x}
        out["retrieved"] = [self._summarize(h) for h in hits]
        if note:
            out["retrieval_error"] = note
        return out

    @staticmethod
    def _summarize(hit: Mapping[str, Any]) -> dict:
        record = hit.get("record", {})
        return {
            "name": record.get("name"),
            "score": round(float(hit.get("score", 0.0)), 4),
            "hypothesis": record.get("hypothesis"),
            "metrics": record.get("metrics", {}),
            "conclusion": (record.get("conclusion") or {}).get("summary")
            if isinstance(record.get("conclusion"), Mapping)
            else record.get("conclusion"),
        }


# ── Keeping a transcript from eating the context window ──────


class Compact(Filter):
    """Cut a transcript down before it stops fitting.

    Context blow-up — a loop pastes its whole history into the next prompt
    until the window overflows — is a named failure mode, and the shape that
    causes it is the one Soma makes easiest: a ``board`` or a ``debate``
    transcript grows by a full round every iteration.

    By default a sliding window: keep the head (usually the question, which
    everything else refers to) and the most recent tail, and say in between
    how much was dropped. That costs nothing.

    Pass ``summarizer=`` — an agent, or any filter — to have the middle
    summarized instead of discarded. That costs a model call per compaction
    and is worth it when the middle carries decisions rather than working.

    **Turning this on invalidates the replay of earlier runs**, and
    correctly so: the model is being shown a different prompt, and a journal
    keyed on what was sent should not pretend otherwise.
    """

    _kind = "stateless"
    _cache_version = "1"

    def __init__(
        self,
        max_chars: int = 8000,
        *,
        head_chars: int | None = None,
        summarizer: Any = None,
    ):
        # A fraction rather than a constant: an absolute default would
        # contradict any budget smaller than itself, and the caller who
        # asked for a small budget did not ask for that argument.
        if head_chars is None:
            head_chars = max(1, max_chars // 8)
        if head_chars >= max_chars:
            raise ValueError("head_chars must leave room for a tail")
        self.max_chars = max_chars
        self.head_chars = head_chars
        self.summarizer = summarizer

    def forward(self, x, state):
        text = x if isinstance(x, str) else json.dumps(x, default=str)
        if len(text) <= self.max_chars:
            return x  # untouched, so a short run is bit-identical

        head = text[: self.head_chars]
        tail_chars = self.max_chars - self.head_chars
        middle, tail = text[self.head_chars : -tail_chars], text[-tail_chars:]

        if self.summarizer is not None:
            bridge = self.summarizer.forward(
                f"Summarize what happened here, in a few sentences:\n\n{middle}",
                None,
            )
            joint = f"\n\n[…summary of {len(middle)} characters…]\n{bridge}\n\n"
        else:
            joint = f"\n\n[…{len(middle)} characters dropped…]\n\n"

        return head + joint + tail
