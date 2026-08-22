"""The whole example, running the way a user would run it.

What these tests prove and the others do not: that there are **two programs**.
The tests in `test_remote.py` stand up workers from inside pytest's own
interpreter, so they share its `sys.path`, its imports and its directory. Not
here: each client is a real `python client_x.py`, and whatever does not arrive
over the wire does not arrive.

The files are in `tests/integration/` and read as an example:

    net.py                the nodes and the network. The client imports it
    client_child.py       the client, starting the workers itself
    client_generic.py     the client, sending the code with `cloudpickle`
    client_connects.py    the client, against an **already standing** worker
    client_project.py     the client, against a worker that **has the code**
    client_whole.py       the same graph undistributed, for comparison

What is no longer there, and it is the change that counts most: a worker file. A
worker is not written — it is stood up with `python -m soma_next.worker` and
receives the plan. That it can execute `tokenize` is a matter of how it
resolves, not of someone having handed it a dictionary.
"""

import ast
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("cloudpickle")

HERE = Path(__file__).parent / "integration"
DEADLINE = 60


def run(program, *args, cwd=None, expect_failure=False):
    """Launches a client as a program.

    `cwd` is the working directory, and it is **not a detail**: the worker is a
    child process and inherits it, so launching from the directory where `net.py`
    lives gifts it the import. The clients that depend on the code travelling are
    launched from somewhere else, or the test would pass for the wrong reason. It
    happened: the first three attempts at this went green without `send=` doing
    anything.
    """
    done = subprocess.run(
        [sys.executable, str(HERE / program), *args],
        cwd=cwd or HERE,
        capture_output=True,
        text=True,
        timeout=DEADLINE,
    )
    if expect_failure:
        assert done.returncode != 0, f"it had to fail and it did not:\n{done.stdout}"
    else:
        assert done.returncode == 0, f"it failed:\n{done.stdout}\n{done.stderr}"
    return done


def tagged(output, tag):
    """What the client printed after `TAG `."""
    for line in output.splitlines():
        if line.startswith(f"{tag} "):
            return line[len(tag) + 1 :]
    raise AssertionError(f"`{tag}` was not printed in:\n{output}")


# ── The whole graph here: the reference everything is compared against ──


def test_undistributed_everything_runs_in_the_clients_process():
    done = run("client_whole.py")
    output = ast.literal_eval(tagged(done.stdout, "OUTPUT"))
    here = float(tagged(done.stdout, "HERE"))

    assert output["how_many"] == 4.0
    assert output["long_ones"] == ["quickly"]
    assert output["pids"] == [here], "something ran away without anyone sending it"


# ── Distributing, with the workers started by the client ──


def test_distributing_across_two_workers_gives_the_result_and_two_processes():
    done = run("client_child.py")
    output = ast.literal_eval(tagged(done.stdout, "OUTPUT"))
    here = float(tagged(done.stdout, "HERE"))

    assert output["how_many"] == 4.0
    assert output["long_ones"] == ["quickly"]
    assert here not in output["pids"], "something ran in the client"
    assert len(output["pids"]) == 2, "the two hosts had to be two processes"


def test_the_plan_shows_the_three_trips():
    # Two hosts and a wave: `tokenize` goes to w1, and then `count` and
    # `oddities` leave at the same time for w1 and w2. The shape is visible
    # before anything executes.
    plan = tagged(run("client_child.py").stdout, "PLAN")

    assert plan.count("Remote") == 3, plan
    assert "Wave" in plan, plan
    assert 'Host("w1")' in plan and 'Host("w2")' in plan, plan


def test_what_is_declared_is_what_comes_out_of_hosts():
    hosts = ast.literal_eval(tagged(run("client_child.py").stdout, "HOSTS"))

    assert hosts == {"tokenize": "w1", "count": "w1", "oddities": "w2"}


def test_distributed_gives_the_same_as_whole():
    # The invariant that makes distributing a decision and not a change of
    # semantics. Same expression, same input text, same result.
    whole = ast.literal_eval(tagged(run("client_whole.py").stdout, "OUTPUT"))
    distributed = ast.literal_eval(tagged(run("client_child.py").stdout, "OUTPUT"))

    assert distributed["how_many"] == whole["how_many"]
    assert distributed["long_ones"] == whole["long_ones"]


# ── Sending the code, for a worker that does not have the project ──


def test_a_generic_worker_receives_the_network_and_executes_it(tmp_path):
    # From a directory where `net.py` is not: whatever reaches the worker got
    # there over the wire.
    done = run("client_generic.py", cwd=tmp_path)
    output = ast.literal_eval(tagged(done.stdout, "OUTPUT"))
    here = float(tagged(done.stdout, "HERE"))

    assert output["how_many"] == 4.0
    assert output["long_ones"] == ["quickly"]
    assert here not in output["pids"]
    assert len(output["pids"]) == 2


def test_without_send_the_generic_worker_cannot_open_what_it_is_sent(tmp_path):
    # The other half of the test above, and the one that gives it teeth: without
    # `send`, cloudpickle stores a **reference** to `net` and the worker does not
    # have it. That this one fails is what shows the one above does not pass by
    # accident.
    done = run("client_generic.py", "--no-send", cwd=tmp_path, expect_failure=True)

    assert "net" in done.stderr, done.stderr
    assert "send=" in done.stderr, f"the message has to say what to do:\n{done.stderr}"


# ── What the worker cannot do ──


def test_the_worker_does_not_write_to_its_stdout():
    # That is where the wire runs when the worker is a child. If one of your
    # nodes — or a library on import — printed something, the client would read
    # it as a length; that is why the worker redirects `sys.stdout` to `stderr`.
    done = run("client_child.py")

    assert "OUTPUT" in done.stdout
    assert done.stderr == "", f"the worker spoke where it should not:\n{done.stderr}"


# ── The use case: an independent worker, in the background ──
#
# Everything above starts the worker from the client, so it dies with it. Here
# the worker stands up first, on its own, from another directory, and the client
# only connects. It is the only thing that shows a worker is a process and not a
# subprocess.


@contextmanager
def standing_worker(tmp_path, clone=None, lucky=False):
    """Stands up `python -m soma_next.worker --listen` and says where it landed.

    Port `0` and it says which one it got: picking a fixed number is asking for
    two concurrent runs to collide. And it stands up from `tmp_path`, where there
    is neither `net.py` nor anything of ours — whatever it uses, it got over the
    wire.
    """
    command = [sys.executable, "-m", "soma_next.worker", "--listen", "127.0.0.1:0"]
    if lucky:
        command.append("--lucky")
    # `clone` is what this worker has of the project. Without it, it has nothing,
    # and can only serve artifacts that bring the code inside.
    environment = dict(os.environ)
    if clone is not None:
        environment["PYTHONPATH"] = str(clone)
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    said = []
    try:
        line = process.stdout.readline()
        assert line.startswith("listening on "), f"it did not say where it listens: {line!r}"
        yield Standing(line[len("listening on ") :].strip(), said)
    finally:
        process.kill()
        # What the worker said on `stderr` can only be read once it is not going
        # to write any more. That is where `--lucky`'s warnings come out.
        said.append(process.communicate(timeout=DEADLINE)[1])


class Standing:
    """A standing worker: its address and, at the end, what it said on `stderr`."""

    def __init__(self, addr, said):
        self.addr = addr
        self._said = said

    def __str__(self):
        return self.addr

    @property
    def said(self):
        assert self._said, "it can only be read once the worker has finished"
        return self._said[0]


def test_a_separately_stood_up_worker_serves_a_client_that_connects(tmp_path):
    with standing_worker(tmp_path) as w:
        done = run("client_connects.py", w.addr, cwd=tmp_path)

    output = ast.literal_eval(tagged(done.stdout, "OUTPUT"))
    here = float(tagged(done.stdout, "HERE"))

    assert output["how_many"] == 4.0
    assert output["long_ones"] == ["quickly"]
    assert here not in output["pids"], "something ran in the client"


def test_the_worker_stays_standing_when_the_client_leaves(tmp_path):
    # What separates a worker from a subprocess. Two clients in a row, one after
    # the other, against the same process: if it died with the first, the second
    # could not even connect.
    with standing_worker(tmp_path) as w:
        first = run("client_connects.py", w.addr, cwd=tmp_path)
        second = run("client_connects.py", w.addr, cwd=tmp_path)

    pids = ast.literal_eval(tagged(first.stdout, "OUTPUT"))["pids"]
    others = ast.literal_eval(tagged(second.stdout, "OUTPUT"))["pids"]

    assert pids == others, "it is not the same process from one client to the next"


def test_the_two_hosts_can_be_the_same_worker(tmp_path):
    # `w1` and `w2` point at the same process: two names, one destination. The
    # graph does not find out, which is what having a host be a name buys.
    with standing_worker(tmp_path) as w:
        done = run("client_connects.py", w.addr, cwd=tmp_path)

    pids = ast.literal_eval(tagged(done.stdout, "OUTPUT"))["pids"]
    assert len(pids) == 1, f"a single process was expected on the other side: {pids}"


# ── The `project` strategy: the worker supplies the code ──


def test_a_worker_with_the_project_receives_only_names_and_state(tmp_path):
    with standing_worker(tmp_path, clone=HERE) as w:
        done = run("client_project.py", w.addr, cwd=tmp_path)

    output = ast.literal_eval(tagged(done.stdout, "OUTPUT"))
    assert output["how_many"] == 4.0
    assert output["long_ones"] == ["quickly"]
    assert float(tagged(done.stdout, "HERE")) not in output["pids"]


def test_the_code_does_not_go_over_the_wire(tmp_path):
    # Five nodes in less than this comment takes up. It is the difference between
    # sending names and sending bytecode: `cloudpickle` of the same network goes
    # past ten kilobytes.
    with standing_worker(tmp_path, clone=HERE) as w:
        done = run("client_project.py", w.addr, cwd=tmp_path)

    assert int(tagged(done.stdout, "SIZE")) < 1024


def test_a_worker_without_the_project_says_so_instead_of_guessing(tmp_path):
    # Without a clone it cannot resolve `net:Count`, and saying so is right:
    # importing something else with the same name would be the wrong answer.
    with standing_worker(tmp_path) as w:
        done = run("client_project.py", w.addr, cwd=tmp_path, expect_failure=True)

    assert "net" in done.stderr, done.stderr
    assert "network" in done.stderr, f"the message has to say the way out:\n{done.stderr}"


# ── Versioning ──


def other_version(tmp_path):
    """A clone of the project with `Count` changed."""
    clone = tmp_path / "clone"
    clone.mkdir()
    source = (HERE / "net.py").read_text()
    changed = source.replace(
        'return {"how_many": float(len(words))',
        'return {"how_many": float(len(words) * 100)',
    )
    assert changed != source, "the test is changing nothing"
    (clone / "net.py").write_text(changed)
    return clone


def test_a_worker_with_another_version_of_the_code_stops(tmp_path):
    # The case that motivates versioning: the worker has the repository half
    # updated. Without this it would execute other code and the number would come
    # out different without anyone noticing.
    with standing_worker(tmp_path, clone=other_version(tmp_path)) as w:
        done = run("client_project.py", w.addr, cwd=tmp_path, expect_failure=True)

    assert "Count(" in done.stderr, done.stderr
    assert "--lucky" in done.stderr, f"it has to say the way out:\n{done.stderr}"


def test_with_lucky_it_executes_whatever_it_has_and_reports_it(tmp_path):
    # And it shows in the result: the worker's `Count` multiplies by a hundred.
    with standing_worker(tmp_path, clone=other_version(tmp_path), lucky=True) as w:
        done = run("client_project.py", w.addr, cwd=tmp_path)

    output = ast.literal_eval(tagged(done.stdout, "OUTPUT"))
    assert output["how_many"] == 400.0, "the worker's code did not run"

    # And it says so. Running another version of your code **silently** would be
    # worse than stopping: it is discovered three days later with nowhere to
    # start looking.
    assert "--lucky" in w.said, w.said
    assert "Count(" in w.said, w.said
