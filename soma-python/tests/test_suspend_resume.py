"""Pausing for a person, and picking the run back up.

The pieces existed — a step could return `Suspend`, the driver journaled
it, `EffectDriver::resume_with` could answer it — but not one of them was
reachable from Python, which is the only place a graph with steps in it
can be run. Three things were missing: a typed exception carrying the
site of the pause, a way to file the answer, and a way to re-run under
the *same* run id, without which the journal replays nothing.
"""

import pytest

import soma
from soma.agentic import Done, Suspend


class NeedsApproval:
    """Stops for a person, then reports what they said."""

    _cache_version = "v1"

    def poll(self, ctx):
        if not ctx.results:
            return Suspend("approve this?")
        answer = ctx.results[0]
        return Done(f"decided: {answer.get('output')}")


def _graph():
    g = soma.Graph()
    g.node("approve", NeedsApproval())
    return g


def test_a_suspension_arrives_as_a_typed_exception():
    with pytest.raises(soma.SomaSuspended) as caught:
        _graph().forward("please")

    err = caught.value
    assert err.node_id == "approve"
    assert err.turn == 0
    assert err.kind == "human"
    assert err.reason["prompt"] == "approve this?"
    assert err.run_id


def test_a_suspended_run_resumes_where_it_stopped():
    g = _graph()

    with pytest.raises(soma.SomaSuspended) as caught:
        g.forward("please")
    err = caught.value

    g.resume(err.run_id, err.node_id, err.turn, err.reason, "granted")

    # Same run id: the journal replays to the pause and finds the answer.
    out = g.forward("please", run_id=err.run_id)
    assert "granted" in str(out)


def test_resuming_needs_the_run_id_it_was_given():
    g = _graph()

    with pytest.raises(soma.SomaSuspended) as caught:
        g.forward("please")
    err = caught.value
    g.resume(err.run_id, err.node_id, err.turn, err.reason, "granted")

    # A fresh run is a fresh journal site, so it pauses again rather than
    # picking up an answer that was filed against a different run.
    with pytest.raises(soma.SomaSuspended):
        g.forward("please")


def test_a_graph_without_steps_cannot_be_resumed():
    g = soma.Graph()
    g.node("plain", lambda x: x)

    with pytest.raises(RuntimeError, match="no effectful nodes"):
        g.resume("r", "plain", 0, {"type": "human", "prompt": "?"}, "x")
