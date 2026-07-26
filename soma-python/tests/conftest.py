"""Shared pytest configuration for the soma test suite."""

from __future__ import annotations

import threading
import time
import urllib.request
import warnings

import pytest


def start_worker_and_wait(worker_factory, port, timeout=30.0):
    """Run a Worker in a daemon thread and block until its /health
    endpoint answers.

    Worker startup includes capability detection (python interpreters,
    conda env listing, nvidia-smi) which can take several seconds on
    cold CI runners — hence the generous timeout. Exceptions inside the
    thread are surfaced instead of dying silently in a daemon."""
    thread_error = []

    def run():
        try:
            worker_factory().serve()
        except BaseException as e:  # noqa: BLE001 — surfaced via pytest.fail
            thread_error.append(e)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if thread_error:
            pytest.fail(f"worker thread crashed on startup: {thread_error[0]!r}")
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            )
            if resp.read() == b"ok":
                return thread
        except Exception:
            time.sleep(0.1)
    pytest.fail(f"worker did not answer /health within {timeout}s")


# PyTorch warns when a full backward hook fires while no input requires
# grad — benign in our fixtures (inputs are data, not parameters). This
# was previously silenced only in test_gradient_audit.py; the audit
# suites in test_diagnostics.py need it too.
@pytest.fixture(autouse=True)
def _silence_torch_hook_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Full backward hook is firing when gradients are computed",
        )
        yield
