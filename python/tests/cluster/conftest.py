"""A cluster of containers, brought up for the session.

Opt-in, with `SOMA_CLUSTER=1`, because building the images the first time takes
minutes and `pytest tests/` has to keep taking seconds::

    SOMA_CLUSTER=1 python -m pytest tests/cluster -q

Without it every test here skips, which is the same thing the machines without
docker see.

The store volume is **wiped on every session**: half of what is tested is what a
worker does the first time it sees something, and a warm store would answer for
a run that never happened.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
COMPOSE = HERE.parents[2] / "docker" / "compose.yaml"

# At import time and not in a fixture: the test module asks for `pipeline` while
# it is being collected, which is before any fixture has run. The client imports
# the very copy the workers have mounted, which is what `mode="project"`
# compares.
sys.path.insert(0, str(HERE / "clone"))

#: Which port on the host reaches which worker. The names are what a graph says
#: with `.at(...)`; what they resolve to is said here, which is the whole reason
#: a `Host` is a name.
PORTS = {
    "a": 7001,
    "b": 7002,
    "old": 7003,
    "gpu": 7004,
    "lucky": 7005,
}

#: The ones that come up by default. `worker-gpu` is behind a compose profile:
#: its image carries torch and CUDA and weighs eleven gigabytes, and nothing but
#: the two device tests needs it.
CPU = {name: port for name, port in PORTS.items() if name != "gpu"}

#: How long a container gets to start answering before the test gives up. The
#: images are already built by then: this is a process starting, not a download.
PATIENCE = 60


def compose(*args, check=True, timeout=1800):
    """One `docker compose` call against this project's file."""
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), *args],
        check=check,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def cluster():
    """The workers, up and answering. Yields the ports they are reachable on."""
    if os.environ.get("SOMA_CLUSTER") not in ("1", "build"):
        pytest.skip("set SOMA_CLUSTER=1 to run against real containers")
    if shutil.which("docker") is None:
        pytest.skip("there is no docker here")

    # A clean slate, volumes included: what a worker does on first sight is half
    # of what these tests are about. The containers go with it and come back in
    # seconds; the **images** stay.
    compose("down", "-v", check=False)
    # Never `--build`: `up` builds whatever image is missing, and forcing it
    # re-exports the GPU one — eleven gigabytes — on every session. Ten minutes
    # for nothing. Rebuild on purpose with `SOMA_CLUSTER=build`, which is what
    # you want after touching `python/src` or the Dockerfile.
    rebuild = ["--build"] if os.environ["SOMA_CLUSTER"] == "build" else []
    up = compose("up", "-d", *rebuild, check=False)
    if up.returncode != 0:
        pytest.skip(f"the cluster would not come up:\n{up.stderr[-2000:]}")

    for name, port in CPU.items():
        if not _ready(f"worker-{name}", port):
            pytest.skip(f"`worker-{name}` never opened {port}")
    yield PORTS
    # Left standing on purpose: the next session wipes them, and a cluster that
    # is up is a cluster you can look at when something failed.


@pytest.fixture(scope="session")
def sends_the_code(cluster):
    """A worker that has to be sent everything, by address."""

    def at(which, **kw):
        from soma_next import Worker

        # By its full name, because this directory is a package: `cluster.nodes`
        # is what `sys.modules` has, and what cloudpickle is asked to put **in**
        # the artifact rather than reference.
        return Worker.at(
            f"127.0.0.1:{cluster[which]}", mode="network", send=["cluster.nodes"], **kw
        )

    return at


@pytest.fixture(scope="session")
def has_the_code(cluster):
    """A worker that already has the project, and is sent names and state."""

    def at(which, **kw):
        from soma_next import Worker

        return Worker.at(f"127.0.0.1:{cluster[which]}", mode="project", **kw)

    return at


@pytest.fixture(scope="session")
def gpu(cluster):
    """The worker with the device, up. Skips if it is not there — its image is
    eleven gigabytes and nobody has to have built it."""
    up = compose("--profile", "gpu", "up", "-d", "worker-gpu", check=False)
    if up.returncode != 0 or not _ready("worker-gpu", PORTS["gpu"]):
        pytest.skip("there is no `worker-gpu`: build it with `--profile gpu`")
    return "gpu"


@pytest.fixture(scope="session")
def worker_logs():
    """What a worker said on its `stderr`, which is where a worker talks."""

    def of(service):
        return compose("logs", "--no-log-prefix", service, check=False).stdout

    return of


@pytest.fixture(scope="session")
def in_container():
    """A command run inside one of them, to look at what it has."""

    def run(service, *command):
        return compose("exec", "-T", service, *command, check=False).stdout

    return run


def _ready(service, port, patience=PATIENCE):
    """Whether that worker is really serving yet.

    **Both** conditions, and the first is the one that matters: a published port
    is answered by docker's forwarder as soon as the mapping exists, which can be
    **before** the process behind it is listening. Connecting then succeeds and
    the first message comes back as a broken pipe — a flake that looks like the
    worker died. The worker says `listening on …` when it really is; that line is
    the signal, and the port check only says the mapping got made.
    """
    until = time.monotonic() + patience
    while time.monotonic() < until:
        said = compose("logs", "--no-log-prefix", service, check=False).stdout
        if "listening on" in said and _connects(port):
            return True
        time.sleep(0.2)
    return False


def _connects(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0
