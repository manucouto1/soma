"""Shared pytest configuration for the soma test suite."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.request
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Isolate the persistent cache per test session: without this, every
# `Graph()` in the suite would share the developer's real
# `~/.soma/cache`, letting entries from previous runs mask fit/forward
# behavior and creating order-dependent tests. Tests that need a
# specific cache dir override SOMA_CACHE_DIR themselves (subprocesses
# inherit their own env copies).
os.environ.setdefault("SOMA_CACHE_DIR", tempfile.mkdtemp(prefix="soma-test-cache-"))


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


# ── A mock OpenAI-compatible endpoint ────────────────────────
#
# Shared, because four test modules need one and importing a fixture from
# another test module (which is what they used to do) makes the suite's
# collection order load-bearing. It answers from a script, and the script
# can describe failure as easily as success: a client is only as good as
# what it does when the endpoint misbehaves.


class Reply:
    """One scripted answer, successful or not."""

    def __init__(self, status=200, body=None, headers=None, hangup=False, raw=None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.hangup = hangup
        self.raw = raw

    # ── The shapes a working endpoint produces ──

    @staticmethod
    def says(text):
        """A finished answer."""
        return Reply(
            body={
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )

    @staticmethod
    def wants_tool(name, arguments, *, call_id="call_1", finish_reason="tool_calls"):
        """A turn that asks for a tool. `arguments` may be a dict, or a
        string when the test wants to send malformed JSON on purpose."""
        encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
        return Reply(
            body={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": encoded},
                                }
                            ],
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )

    @staticmethod
    def stops(reason, text=""):
        """A turn that ended for a particular reason — `length`,
        `content_filter`, whatever the endpoint decided."""
        return Reply(
            body={
                "choices": [
                    {"message": {"content": text}, "finish_reason": reason}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )

    # ── The shapes a misbehaving one produces ──

    @staticmethod
    def error(status, *, retry_after=None, body=None):
        headers = {}
        if retry_after is not None:
            headers["Retry-After"] = str(retry_after)
        return Reply(
            status=status,
            body=body if body is not None else {"error": {"message": "scripted"}},
            headers=headers,
        )

    @staticmethod
    def garbage(text="<html>502 Bad Gateway</html>", status=200):
        """A 200 carrying something that is not a chat completion — a proxy
        or a captive portal answering for the endpoint."""
        return Reply(status=status, raw=text)

    @staticmethod
    def hangs_up():
        """Accept the request and close without answering."""
        return Reply(hangup=True)


class _Scripted(BaseHTTPRequestHandler):
    """Answers with the next scripted reply, recording what it was sent."""

    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler's naming
        self._serve()

    def do_GET(self):  # noqa: N802 — model listing
        self._serve()

    def _serve(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            self.server.received.append(json.loads(raw or b"{}"))
        except ValueError:
            self.server.received.append({"_unparsed": raw.decode("utf8", "replace")})

        script = self.server.script
        reply = script[min(len(self.server.received) - 1, len(script) - 1)]
        if not isinstance(reply, Reply):
            reply = Reply(body=reply)  # a bare dict is a 200 with that body

        if reply.hangup:
            self.close_connection = True
            self.connection.close()
            return

        payload = (
            reply.raw.encode()
            if reply.raw is not None
            else json.dumps(reply.body).encode()
        )
        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, value in reply.headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass  # keep pytest output clean


class MockProvider:
    """A scripted chat-completions endpoint, as a context manager.

    The script may hold `Reply` objects or bare dicts (a 200 with that
    body). Once exhausted, the last entry repeats — so a permanent failure
    is a one-element script.
    """

    def __init__(self, script):
        self.server = HTTPServer(("127.0.0.1", 0), _Scripted)
        self.server.script = list(script)
        self.server.received = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def received(self):
        return self.server.received

    @property
    def hits(self):
        """How many times the client knocked. The number under test
        whenever retries are involved."""
        return len(self.server.received)


def says(text):
    """The bare body of a finished answer. Kept as a function because most
    tests only care about the text."""
    return Reply.says(text).body


def wants_tool(name, arguments):
    return Reply.wants_tool(name, arguments).body


@pytest.fixture
def providers_file(tmp_path, monkeypatch):
    """Point Soma's provider catalog at a throwaway file.

    Retries default to a fast policy: the real defaults wait half a second
    before the first one, and a suite that sits through that learns nothing
    it would not learn in a millisecond. Tests about *how long* the client
    waits set their own.
    """

    def _write(base_url, *, retry=None, quirks=None):
        policy = {"base_ms": 1, "max_ms": 5, "jitter": False}
        policy.update(retry or {})

        lines = [
            "[providers.mock]",
            f'base_url = "{base_url}"',
            "auth = { type = \"none\" }",
            "",
            "[providers.mock.retry]",
            *(f"{key} = {json.dumps(value)}" for key, value in policy.items()),
        ]
        if quirks:
            lines += [
                "",
                "[providers.mock.quirks]",
                *(f"{key} = {json.dumps(value)}" for key, value in quirks.items()),
            ]

        path = tmp_path / "providers.toml"
        path.write_text("\n".join(lines) + "\n")
        monkeypatch.setenv("SOMA_PROVIDERS", str(path))
        monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))
        return path

    return _write


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
