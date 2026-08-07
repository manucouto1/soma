"""Crash recovery across a real process boundary.

An agentic run journals every effect under the graph's cache dir; replaying
the same run id serves recorded results instead of calling the model. That
is the crash-recovery story — and every in-process replay test shares one
driver, so none of them proves the journal survives the death of the
process that wrote it. These do: each scenario writes the journal in one
process and reads it from another, with a scripted HTTP provider in the
parent counting exactly how many times a model was actually asked.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from conftest import MockProvider, Reply


def _write_providers(tmp_path: pathlib.Path, base_url: str) -> pathlib.Path:
    """A providers.toml naming the parent's mock endpoint, with fast retries
    (mirrors the `providers_file` fixture, which cannot reach a child's env)."""
    path = tmp_path / "providers.toml"
    path.write_text(
        textwrap.dedent(
            f"""
            [providers.mock]
            base_url = "{base_url}"
            auth = {{ type = "none" }}

            [providers.mock.retry]
            base_ms = 1
            max_ms = 5
            jitter = false
            """
        ).lstrip()
    )
    return path


def _child_env(tmp_path: pathlib.Path, providers: pathlib.Path | None = None) -> dict:
    """Everything a child needs: the shared cache dir (where the journal
    lives) and, when a model is involved, the provider catalog."""
    env = dict(os.environ, SOMA_CACHE_DIR=str(tmp_path / "cache"))
    if providers is not None:
        env["SOMA_PROVIDERS"] = str(providers)
    return env


def _run(script: pathlib.Path, args: list[str], env: dict) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=script.parent,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


# ── 1. Replay without a crash: a second process asks nothing ──


ONE_AGENT_SCRIPT = textwrap.dedent(
    """
    import sys
    import soma

    RUN_ID = sys.argv[1]

    g = soma.Graph()  # persistent cache: the journal must live on disk
    g.node("assistant", soma.Agent(model="mock/any-model"))
    print(g.forward("what is soma?", run_id=RUN_ID))
    """
)


@pytest.mark.slow
def test_a_new_process_replays_instead_of_calling(tmp_path):
    """Two processes, one run id, one model call.

    The first process performs the LLM effect and journals it under the
    shared cache dir; the second, given the same run id, must serve the
    recorded answer without the model ever hearing from it. This is the
    contract every 'resume after a crash' promise rests on, checked where
    users feel it: across a process boundary, not inside one driver.
    """
    script = tmp_path / "child.py"
    script.write_text(ONE_AGENT_SCRIPT)

    with MockProvider([Reply.says("a graph runtime")]) as provider:
        env = _child_env(tmp_path, _write_providers(tmp_path, provider.base_url))

        first = _run(script, ["replay-across-processes"], env)
        assert provider.hits == 1

        second = _run(script, ["replay-across-processes"], env)
        # Same answer, from the journal: the model was asked exactly once.
        assert first.stdout.strip() == second.stdout.strip() == "a graph runtime"
        assert provider.hits == 1, (
            "the second process reached the model instead of replaying "
            f"({provider.hits} calls total)"
        )


# ── 2. SIGKILL mid-run: the finished half replays, the rest completes ──


TWO_AGENT_SCRIPT = textwrap.dedent(
    """
    import os, sys, time
    import soma
    from soma.agentic import Done

    RUN_ID = sys.argv[1]
    MARKER = sys.argv[2]


    class Gate:
        '''First run: signal the parent, then block until SIGKILLed.
        Re-run (the marker exists): pass the input straight through.'''

        _cache_version = "1"

        def poll(self, ctx):
            if not os.path.exists(MARKER):
                with open(MARKER, "w") as f:
                    f.write("first agent is done")
                time.sleep(120)  # the parent kills us here
            return Done(ctx.input)


    g = soma.Graph()  # persistent: the journal survives the kill
    g.node("first", soma.Agent(model="mock/any-model"))
    g.node("gate", Gate())
    g.node("second", soma.Agent(model="mock/any-model"))
    g.edge("first", "gate")
    g.edge("gate", "second")
    print(g.forward("start", run_id=RUN_ID))
    """
)


@pytest.mark.slow
def test_sigkill_mid_agentic_run_then_replay(tmp_path):
    """kill -9 between the first model call and the second; re-running the
    same run id must replay the first effect (its hit count stays at 1)
    and go on to finish the run.

    This pins 'a crash after the fourth experiment replays the first
    three' (graph_handler.rs) at the level users feel it: real process
    death, journal on disk, a provider that would notice a repeat call.
    """
    script = tmp_path / "child.py"
    script.write_text(TWO_AGENT_SCRIPT)
    marker = tmp_path / "first-agent-answered"

    with MockProvider(
        [Reply.says("the first answer"), Reply.says("the second answer")]
    ) as provider:
        env = _child_env(tmp_path, _write_providers(tmp_path, provider.base_url))

        proc = subprocess.Popen(
            [sys.executable, str(script), "crash-mid-run", str(marker)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            cwd=tmp_path,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if marker.exists():
                    break
                if proc.poll() is not None:
                    pytest.fail(
                        f"child exited early: {proc.stderr.read().decode()}"
                    )
                time.sleep(0.05)
            else:
                pytest.fail("the first agent never finished")

            assert provider.hits == 1, "only the first agent should have called"
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()

        # Re-run under the same run id. The marker now exists, so the gate
        # passes through — and the first agent's effect must come from the
        # journal, not from the wire.
        rerun = _run(script, ["crash-mid-run", str(marker)], env)

        assert rerun.stdout.strip() == "the second answer"
        assert provider.hits == 2, (
            "the first effect was re-performed after the crash "
            f"({provider.hits} calls total; the journal should hold it at 2)"
        )


# ── 3. Suspend in one process, resume in another ──


SUSPEND_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import soma
    from soma.agentic import Done, Suspend


    class NeedsApproval:
        _cache_version = "1"

        def poll(self, ctx):
            if not ctx.results:
                return Suspend("approve this?")
            return Done("decided: " + str(ctx.results[0].get("output")))


    g = soma.Graph()
    g.node("approve", NeedsApproval())
    try:
        g.forward("please")
    except soma.SomaSuspended as e:
        print(json.dumps({"run_id": e.run_id, "node_id": e.node_id,
                          "turn": e.turn, "reason": e.reason}))
        sys.exit(0)
    sys.exit(1)  # completed without suspending: not the scenario under test
    """
)

RESUME_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import soma
    from soma.agentic import Done, Suspend


    class NeedsApproval:
        _cache_version = "1"

        def poll(self, ctx):
            if not ctx.results:
                return Suspend("approve this?")
            return Done("decided: " + str(ctx.results[0].get("output")))


    site = json.loads(sys.argv[1])
    g = soma.Graph()
    g.node("approve", NeedsApproval())
    g.resume(site["run_id"], site["node_id"], site["turn"], site["reason"],
             "granted")
    print(g.forward("please", run_id=site["run_id"]))
    """
)


@pytest.mark.slow
def test_suspend_in_one_process_resume_in_another(tmp_path):
    """A human-in-the-loop pause is worthless if the answer must arrive
    before the Python process dies. Process A suspends and exits, printing
    the site of the pause; process B — sharing nothing but the on-disk
    journal — files the answer and finishes the run.
    """
    suspender = tmp_path / "suspend.py"
    suspender.write_text(SUSPEND_SCRIPT)
    resumer = tmp_path / "resume.py"
    resumer.write_text(RESUME_SCRIPT)
    env = _child_env(tmp_path)

    paused = _run(suspender, [], env)
    site = json.loads(paused.stdout.strip().splitlines()[-1])
    assert site["node_id"] == "approve"
    assert site["reason"]["prompt"] == "approve this?"

    resumed = _run(resumer, [json.dumps(site)], env)
    assert "decided: granted" in resumed.stdout, (
        "the resumed run should see the answer filed by the other process, "
        f"got: {resumed.stdout!r}"
    )
