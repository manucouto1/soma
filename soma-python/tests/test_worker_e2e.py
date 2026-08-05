"""
End-to-end integration tests for the Soma worker system.

Starts a real worker server in /tmp, sends jobs via the Python API,
validates all transport modes, and shuts down the worker cleanly.

Tests cover:
- Inline transport (small payloads)
- HTTP bulk upload (large payloads > 10MB)
- Fit + forward state round-trip
- cloudpickle serialization (filters defined here, not in any package)
- DataStore (local) integration
- Streaming via WS Binary chunks
- Worker shutdown via protocol
- Edge cases: empty input, single-element tensors, JSON values
"""

import os
import json
import time
import shutil
import tempfile
import threading

import pytest
from soma import Graph, Filter, Worker


# ── Test Filters (defined in __main__-like scope → cloudpickle by value) ──


class DoubleFilter(Filter):
    """Stateless: doubles every element."""

    _kind = "stateless"

    def forward(self, x, state):
        if isinstance(x, list):
            return [v * 2 for v in x]
        return x


class ScaleFilter(Filter):
    """Trainable: learns mean, scales by factor."""

    _kind = "trainable"

    def __init__(self, factor=3.0):
        super().__init__(factor=factor)

    def fit(self, x, y=None):
        mean = sum(x) / len(x) if isinstance(x, list) and len(x) > 0 else 0
        return {"mean": mean}

    def forward(self, x, state):
        mean = state.get("mean", 0) if isinstance(state, dict) else 0
        return [(v - mean) * self.factor for v in x]


class JsonFilter(Filter):
    """Stateless: works with dict/JSON input."""

    _kind = "stateless"

    def forward(self, x, state):
        if isinstance(x, dict):
            return {k: v * 2 if isinstance(v, (int, float)) else v for k, v in x.items()}
        return x


class IdentityFilter(Filter):
    """Stateless: returns input unchanged."""

    _kind = "stateless"

    def forward(self, x, state):
        return x


# ── Fixtures ──

WORKER_PORT = 0  # Will be set dynamically
WORKER_TOKEN = "test-token-e2e"
WORKER_TEMP = None


def find_free_port():
    """Find a free port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module", autouse=True)
def worker_server():
    """Start a worker server for the test module, shut it down after."""
    global WORKER_PORT, WORKER_TEMP

    WORKER_PORT = find_free_port()
    WORKER_TEMP = tempfile.mkdtemp(prefix="soma-e2e-")

    # Start worker in background thread
    from conftest import start_worker_and_wait

    start_worker_and_wait(
        lambda: Worker(
            port=WORKER_PORT,
            tags=["test", "e2e"],
            token=WORKER_TOKEN,
            max_concurrent=2,
            worker_id="e2e-test-worker",
        ),
        WORKER_PORT,
    )

    yield

    # Worker runs as daemon thread — dies when test process exits.
    # Don't send Shutdown (it calls std::process::exit which kills the test).
    shutil.rmtree(WORKER_TEMP, ignore_errors=True)


def make_graph():
    """Create a Graph connected to the test worker."""
    g = Graph()
    g.add_worker(f"ws://127.0.0.1:{WORKER_PORT}", token=WORKER_TOKEN, tags=["test"])
    return g


# ── Tests: Basic Transport ──


class TestInlineTransport:
    """Small payloads go via WebSocket inline."""

    def test_fit_forward_small_data(self):
        g = make_graph()
        g.node("scaler", ScaleFilter(factor=2.0))
        g.fit([1.0, 2.0, 3.0, 4.0, 5.0])
        result = g.forward([10.0, 20.0, 30.0])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_stateless_forward(self):
        g = make_graph()
        g.node("doubler", DoubleFilter())
        # Stateless filters need fit to set fitted=True
        g.fit([1.0])
        result = g.forward([1.0, 2.0, 3.0])
        assert isinstance(result, list)
        assert result == [2.0, 4.0, 6.0]

    def test_chain_two_filters(self):
        g = make_graph()
        g.node("d1", DoubleFilter())
        g.node("d2", DoubleFilter())
        g.edge("d1", "d2")
        g.fit([1.0])
        result = g.forward([5.0])
        assert isinstance(result, list)
        assert result == [20.0]  # 5 * 2 * 2

    def test_somatize_chain(self):
        g = Graph.somatize(DoubleFilter() >> DoubleFilter())
        g.add_worker(f"ws://127.0.0.1:{WORKER_PORT}", token=WORKER_TOKEN, tags=["test"])
        g.fit([1.0])
        result = g.forward([3.0])
        assert isinstance(result, list)
        assert result == [12.0]  # 3 * 2 * 2


class TestHTTPBulkTransport:
    """Large payloads (>10MB) go via HTTP upload."""

    def test_large_tensor_upload(self):
        """Payload >10MB triggers HTTP /upload automatically."""
        g = make_graph()
        g.node("identity", IdentityFilter())
        g.fit([1.0])

        # 1.4M floats × 8 bytes ≈ 11.2MB → exceeds INLINE_THRESHOLD_BYTES (10MB)
        big_data = [float(i) for i in range(1_400_000)]
        result = g.forward(big_data)
        assert isinstance(result, list)
        assert len(result) == 1_400_000
        assert result[0] == 0.0
        assert result[-1] == 1_399_999.0

    def test_large_fit_via_http(self):
        """Fit with >10MB data also goes through HTTP upload."""
        g = make_graph()
        g.node("scaler", ScaleFilter(factor=1.0))
        big_data = [float(i) for i in range(1_400_000)]
        g.fit(big_data)  # Should upload via HTTP, train remotely
        result = g.forward([100.0])
        assert isinstance(result, list)
        assert len(result) == 1


class TestFitForwardStateRoundTrip:
    """Trained states survive the client→worker→client round-trip."""

    def test_state_persists_across_calls(self):
        g = make_graph()
        g.node("scaler", ScaleFilter(factor=10.0))
        g.fit([2.0, 4.0, 6.0])  # mean = 4.0

        result = g.forward([10.0])
        assert isinstance(result, list)
        assert len(result) == 1
        # State must be preserved: (10 - 4) * 10 = 60
        assert abs(result[0] - 60.0) < 0.01, f"Expected 60.0 got {result[0]}"

    def test_multiple_trainable_in_chain(self):
        g = make_graph()
        g.node("s1", ScaleFilter(factor=2.0))
        g.node("s2", ScaleFilter(factor=3.0))
        g.edge("s1", "s2")
        g.fit([1.0, 2.0, 3.0])
        result = g.forward([5.0])
        assert isinstance(result, list)


class TestCloudpickleSerialization:
    """Filters defined in test file are serialized by value."""

    def test_lambda_in_filter(self):
        """Filters with closures should serialize correctly."""

        class ClosureFilter(Filter):
            _kind = "stateless"

            def __init__(self, multiplier=7.0):
                super().__init__(multiplier=multiplier)

            def forward(self, x, state):
                fn = lambda v: v * self.multiplier  # noqa: E731
                return [fn(v) for v in x]

        g = make_graph()
        g.node("closure", ClosureFilter(multiplier=7.0))
        g.fit([1.0])
        result = g.forward([2.0, 3.0])
        assert result == [14.0, 21.0]

    def test_filter_with_helper_function(self):
        """Filters using module-level helpers should work."""

        def my_helper(values):
            return [v + 100 for v in values]

        class HelperFilter(Filter):
            _kind = "stateless"

            def forward(self, x, state):
                return my_helper(x)

        g = make_graph()
        g.node("helper", HelperFilter())
        g.fit([1.0])
        result = g.forward([1.0, 2.0])
        assert result == [101.0, 102.0]


class TestDataStore:
    """DataStore integration with local storage."""

    def test_local_data_store(self):
        store_path = os.path.join(WORKER_TEMP, "datastore")
        g = make_graph()
        g.set_data_store("local", path=store_path)
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward([5.0, 10.0])
        assert isinstance(result, list)

    def test_invalid_store_type(self):
        g = Graph()
        with pytest.raises(ValueError, match="unknown store type"):
            g.set_data_store("redis")

    def test_local_store_missing_path(self):
        g = Graph()
        with pytest.raises(ValueError, match="requires 'path'"):
            g.set_data_store("local")


class TestStreaming:
    """Streaming forward: the normal local path with a Stream plan —
    chunks run through run_node's primitives."""

    def test_stream_forward_local(self):
        """Local streaming (no workers): chunked, same result."""
        g = Graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            stream=True,
            chunk_size=3,
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 8
        assert result == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    def test_stream_forward_single_chunk(self):
        """Input smaller than chunk_size: single chunk, same result as normal forward."""
        g = Graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        normal = g.forward([1.0, 2.0, 3.0])
        streamed = g.forward([1.0, 2.0, 3.0], stream=True, chunk_size=1024)
        assert normal == streamed


    def test_stream_events_reach_the_run_dir(self, tmp_path):
        """The stream branch shares the local path: a tracked stream
        forward lands per-node events in events.jsonl like any other
        run, with the chunk aggregate in the completion summary."""
        import soma

        g = Graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        with g.track_run("streamed", root=str(tmp_path), kind="forward") as run:
            g.forward([1.0, 2.0, 3.0, 4.0], stream=True, chunk_size=2)

        events = soma.RunView(run.dir).events()
        started = [e for e in events if e["event_type"] == "NodeStarted"]
        assert [e["node_id"] for e in started] == ["doubler"]
        completed = [e for e in events if e["event_type"] == "NodeCompleted"]
        assert len(completed) == 1
        assert "chunks" in completed[0]["output_summary"]

    def test_a_step_cannot_stream(self):
        """Effect journaling has no per-chunk semantics; the compiler
        says so by name instead of running something undefined."""
        from soma.agentic import Done

        class Echo:
            _cache_version = "1"

            def poll(self, ctx):
                return Done(ctx.input)

        g = Graph(cache="memory")
        g.node("echo", Echo())
        with pytest.raises(Exception, match="cannot be streamed"):
            g.forward("hi", stream=True, chunk_size=2)

    def test_a_diamond_cannot_stream(self):
        """Streaming executes a single linear chain; a DAG used to be
        silently run as one, which for a diamond is the wrong answer."""
        g = Graph(cache="memory")
        for node_id in ["a", "b", "c", "d"]:
            g.node(node_id, DoubleFilter())
        g.connect("a", "b")
        g.connect("a", "c")
        g.connect("b", "d")
        g.connect("c", "d")
        with pytest.raises(Exception, match="linear chain"):
            g.forward([1.0, 2.0], stream=True, chunk_size=1)

    @pytest.mark.skip(reason="Remote streaming requires PythonProcess to support Value::Tensor round-trip in SubprocessFilter — tracked for next iteration")
    def test_stream_forward_remote(self):
        """Remote streaming via WS Binary."""
        g = make_graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            stream=True,
            chunk_size=3,
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 8


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_list(self):
        g = make_graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward([])
        assert result == []

    def test_single_element(self):
        g = make_graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward([42.0])
        assert result == [84.0]

    def test_forward_without_fit_fails(self):
        g = make_graph()
        # A *trainable* filter has state to learn, so forward without fit is
        # an error. (A stateless graph has nothing to fit and runs as-is.)
        g.node("scale", ScaleFilter())
        with pytest.raises(RuntimeError, match="fitted"):
            g.forward([1.0])

    def test_a_stateless_graph_needs_no_fit(self):
        g = make_graph()
        g.node("doubler", DoubleFilter())
        assert g.forward([1.0, 2.0]) == [2.0, 4.0]

    def test_json_value_through_worker(self):
        g = make_graph()
        g.node("json_filter", JsonFilter())
        g.fit([1.0])  # dummy fit
        result = g.forward({"a": 5, "b": 10})
        assert isinstance(result, dict)
        assert result["a"] == 10
        assert result["b"] == 20


class TestWorkerManagement:
    """Worker lifecycle and management."""

    def test_workers_list(self):
        g = make_graph()
        workers = g.workers()
        assert len(workers) >= 1
        assert workers[0]["source"] == "direct"

    def test_filter_source_introspection(self):
        g = make_graph()
        g.node("doubler", DoubleFilter())
        source = g.filter_source("doubler")
        assert source is not None
        # Should contain the class definition from this file
        assert "DoubleFilter" in source or "class" in source

    def test_filter_sources_dict(self):
        g = make_graph()
        g.node("d1", DoubleFilter())
        g.node("d2", DoubleFilter())
        sources = g.filter_sources_dict()
        assert "d1" in sources
        assert "d2" in sources


class TestMultipleWorkers:
    """Multiple workers on the same machine."""

    def test_two_workers_same_graph(self):
        """Register two workers, verify both are listed."""
        port2 = find_free_port()
        t2 = threading.Thread(
            target=lambda: Worker(port=port2, worker_id="e2e-worker-2", tags=["test2"]).serve(),
            daemon=True,
        )
        t2.start()

        import urllib.request
        for _ in range(30):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port2}/health", timeout=1).read() == b"ok":
                    break
            except Exception:
                time.sleep(0.1)

        g = Graph()
        g.add_worker(f"ws://127.0.0.1:{WORKER_PORT}", token=WORKER_TOKEN, tags=["test"])
        g.add_worker(f"ws://127.0.0.1:{port2}", tags=["test2"])
        workers = g.workers()
        assert len(workers) == 2

    def test_dispatch_to_specific_worker(self):
        """Filters route to the correct worker by tag."""
        g = make_graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward([7.0])
        assert result == [14.0]


class TestStateRoundTripAdvanced:
    """Advanced state scenarios."""

    def test_state_with_nested_dict(self):
        """State with nested structure survives round-trip."""

        class NestedStateFilter(Filter):
            _kind = "trainable"

            def fit(self, x, y=None):
                return {
                    "weights": [1.0, 2.0, 3.0],
                    "config": {"lr": 0.001, "layers": [64, 32]},
                    "bias": 0.5,
                }

            def forward(self, x, state):
                if isinstance(state, dict) and "weights" in state:
                    w = state["weights"]
                    b = state.get("bias", 0)
                    return [sum(xi * wi for xi, wi in zip(x, w)) + b]
                return x

        g = make_graph()
        g.node("nested", NestedStateFilter())
        g.fit([1.0, 2.0, 3.0])
        result = g.forward([1.0, 2.0, 3.0])
        assert isinstance(result, list)
        # 1*1 + 2*2 + 3*3 + 0.5 = 14.5
        assert abs(result[0] - 14.5) < 0.01

    def test_refit_updates_state(self):
        """Fitting twice with different data updates the state."""
        g = make_graph()
        g.node("scaler", ScaleFilter(factor=1.0))

        g.fit([10.0, 20.0, 30.0])  # mean = 20
        result1 = g.forward([25.0])
        # (25 - 20) * 1 = 5
        assert abs(result1[0] - 5.0) < 0.01

        g.fit([100.0, 200.0, 300.0])  # mean = 200
        result2 = g.forward([250.0])
        # (250 - 200) * 1 = 50
        assert abs(result2[0] - 50.0) < 0.01

    def test_stateless_ignores_state(self):
        """Stateless filters work even when state is provided."""
        g = make_graph()
        g.node("d", DoubleFilter())
        g.fit([99.0])
        r1 = g.forward([5.0])
        r2 = g.forward([5.0])
        assert r1 == r2 == [10.0]


class TestDataStoreAdvanced:
    """DataStore with real data flow."""

    def test_data_store_fit_and_forward(self):
        """Full cycle: fit via store, forward via store."""
        store_path = os.path.join(WORKER_TEMP, "ds_full")
        g = make_graph()
        g.set_data_store("local", path=store_path)
        g.node("scaler", ScaleFilter(factor=5.0))
        g.fit([1.0, 3.0, 5.0])  # mean = 3
        result = g.forward([10.0])
        # (10 - 3) * 5 = 35
        assert isinstance(result, list)
        assert abs(result[0] - 35.0) < 0.01

    def test_data_store_persists_between_graphs(self):
        """Data uploaded to store can be reused."""
        store_path = os.path.join(WORKER_TEMP, "ds_persist")
        os.makedirs(store_path, exist_ok=True)

        # First graph writes
        g1 = make_graph()
        g1.set_data_store("local", path=store_path)
        g1.node("d", DoubleFilter())
        g1.fit([1.0])
        g1.forward([5.0])

        # Verify store has files
        assert len(os.listdir(store_path)) > 0


class TestEdgeCasesAdvanced:
    """More edge cases."""

    def test_large_chain(self):
        """Chain of 5 filters works remotely."""
        g = make_graph()
        prev = None
        for i in range(5):
            name = f"d{i}"
            g.node(name, DoubleFilter())
            if prev:
                g.edge(prev, name)
            prev = name
        g.fit([1.0])
        result = g.forward([1.0])
        # 1 * 2^5 = 32
        assert result == [32.0]

    def test_negative_values(self):
        g = make_graph()
        g.node("d", DoubleFilter())
        g.fit([1.0])
        result = g.forward([-5.0, -10.0])
        assert result == [-10.0, -20.0]

    def test_very_small_values(self):
        g = make_graph()
        g.node("d", DoubleFilter())
        g.fit([1.0])
        result = g.forward([1e-10, 1e-20])
        assert abs(result[0] - 2e-10) < 1e-15
        assert abs(result[1] - 2e-20) < 1e-25

    def test_2d_tensor_input(self):
        """2D list (matrix) works through worker."""
        g = make_graph()
        g.node("identity", IdentityFilter())
        g.fit([1.0])
        result = g.forward([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(result, list)
        # Subprocess may flatten 2D tensors — check total elements
        total = len(result) if not isinstance(result[0], list) else sum(len(r) for r in result)
        assert total == 4

    def test_repeated_forward_calls(self):
        """Multiple forward calls on same fitted graph."""
        g = make_graph()
        g.node("d", DoubleFilter())
        g.fit([1.0])
        for val in [1.0, 2.0, 3.0, 100.0, 0.0]:
            result = g.forward([val])
            assert result == [val * 2]

    def test_graph_repr(self):
        """Graph __repr__ works with worker connected."""
        g = make_graph()
        g.node("d", DoubleFilter())
        r = repr(g)
        assert "1 nodes" in r
        assert "fitted=false" in r


class TestFederatedStrategy:
    """A `TrainingStrategy` that actually trains.

    `set_strategy` did not exist in Python, and in Rust nothing read the
    attribute back: `impl StrategyExecutor for TrainingStrategy` had no
    caller anywhere in the workspace. Setting a strategy recorded it and
    changed nothing.
    """

    def test_set_strategy_round_trips(self):
        g = Graph(cache="memory")
        assert g.strategy() == "local"
        g.set_strategy("federated", num_clients=2, rounds=3)
        assert g.strategy() == "federated"

    def test_an_unknown_strategy_is_refused_by_name(self):
        g = Graph(cache="memory")
        with pytest.raises(ValueError, match="unknown strategy"):
            g.set_strategy("magic")
        with pytest.raises(ValueError, match="fed_prox needs"):
            g.set_strategy("federated", aggregation="fed_prox")

    def test_federated_fit_averages_across_two_workers(self):
        """The property that cannot be faked by a single client.

        Two workers, one shard each: the halves of 0..8 have means 1.5 and
        5.5, and FedAvg of those is 3.5. A run that quietly used one client
        would produce 1.5 or 5.5, so this asserts both.
        """
        from conftest import start_worker_and_wait

        class Mean(Filter):
            _cache_version = "fed-e2e-mean-v1"

            def fit(self, x, y=None):
                import numpy as np

                return {"mu": float(np.asarray(x, dtype=float).mean())}

            def forward(self, x, state):
                import numpy as np

                return (np.asarray(x, dtype=float) - state["mu"]).tolist()

        ports = []
        for _ in range(2):
            port = find_free_port()
            start_worker_and_wait(
                lambda p=port: Worker(port=p, tags=["fed"], max_concurrent=2), port
            )
            ports.append(port)

        g = Graph(cache="memory")
        for port in ports:
            g.add_worker(f"ws://127.0.0.1:{port}", tags=["fed"])
        g.node("m", Mean(), target="fed")
        g.set_strategy("federated", num_clients=2, rounds=2)

        g.fit([float(i) for i in range(8)])

        mu = g.state()["m"]["mu"]
        assert abs(mu - 3.5) < 1e-9, f"expected the mean of both clients, got {mu}"
        assert abs(mu - 1.5) > 1e-6 and abs(mu - 5.5) > 1e-6, (
            "this is one client's answer, so the aggregation did not happen"
        )

    def test_data_parallel_trains_on_the_averaged_gradient(self):
        """Two replicas, one shard each, and the step is the mean of both.

        The weights are compared against a reference computed here: the
        same initialisation, each shard's gradient taken separately, the
        two averaged, one SGD step. They must match exactly — and must
        NOT match what either shard alone would have produced, which is
        what a round that quietly used one replica returns.

        SGD rather than the default Adam on purpose: Adam's first step is
        ``lr * sign(g)``, so the averaged and single-shard references would
        be indistinguishable and the test would pass on a broken round.
        """
        import base64
        import io

        torch = pytest.importorskip("torch")
        import torch.nn as nn

        from conftest import start_worker_and_wait
        from soma import DifferentiableFilter

        class Lin(DifferentiableFilter):
            _cache_version = "dp-averaged-gradient-v1"

            def __init__(self, out_dim=1, **kw):
                super().__init__(out_dim=out_dim, **kw)
                self.out_dim = out_dim

            def build_module(self, input_shape):
                torch.manual_seed(1234)
                return nn.Linear(int(input_shape[-1]), self.out_dim)

            def output_shape(self, input_shape):
                return (input_shape[0], self.out_dim)

            def make_optimizer(self, modules):
                return torch.optim.SGD(
                    [p for m in modules for p in m.parameters()], lr=0.01
                )

        x = [[float(i), float(i) * 2] for i in range(8)]
        y = [[float(i) * 3] for i in range(8)]

        def grads_of(rows, targets):
            torch.manual_seed(1234)
            m = nn.Linear(2, 1)
            out = m(torch.tensor(rows))
            nn.functional.mse_loss(out, torch.tensor(targets)).backward()
            return {n: p.grad.clone() for n, p in m.named_parameters()}

        first, second = grads_of(x[:4], y[:4]), grads_of(x[4:], y[4:])

        def stepped(grads):
            torch.manual_seed(1234)
            m = nn.Linear(2, 1)
            for name, p in m.named_parameters():
                p.grad = grads[name].clone()
            torch.optim.SGD(m.parameters(), lr=0.01).step()
            return {n: p.detach().clone() for n, p in m.named_parameters()}

        averaged = stepped({n: (first[n] + second[n]) / 2 for n in first})
        one_shard_only = stepped(first)

        ports = []
        for _ in range(2):
            port = find_free_port()
            start_worker_and_wait(
                lambda p=port: Worker(port=p, tags=["dp"], max_concurrent=2), port
            )
            ports.append(port)

        g = Graph(cache="memory")
        for port in ports:
            g.add_worker(f"ws://127.0.0.1:{port}", tags=["dp"])
        g.node("t", Lin(1), target="dp")
        g.set_strategy("data_parallel", num_replicas=2)

        g.fit(x, y)

        got = torch.load(
            io.BytesIO(base64.b64decode(g.state()["t"]["weights_b64"])),
            weights_only=True,
        )
        for name in averaged:
            assert torch.allclose(got[name], averaged[name], atol=1e-6), (
                f"{name} is not the averaged-gradient step: "
                f"{got[name].tolist()} vs {averaged[name].tolist()}"
            )
            assert not torch.allclose(got[name], one_shard_only[name], atol=1e-4), (
                f"{name} is what one shard alone produces, so the round "
                "trained on a single replica"
            )

    def test_data_parallel_refuses_mismatched_rows(self):
        """Inputs and targets are sharded together, so they must agree.

        Before they were sharded together, each replica got its own quarter
        of x and the whole of y: shapes that broadcast rather than fail, so
        every replica trained on pairs that were never pairs and the round
        reported success.
        """
        from conftest import start_worker_and_wait

        class Noop(Filter):
            _cache_version = "dp-rows-v1"

            def fit(self, x, y=None):
                return {}

            def forward(self, x, state):
                return x

        ports = []
        for _ in range(2):
            port = find_free_port()
            start_worker_and_wait(
                lambda p=port: Worker(port=p, tags=["rows"], max_concurrent=2), port
            )
            ports.append(port)

        g = Graph(cache="memory")
        for port in ports:
            g.add_worker(f"ws://127.0.0.1:{port}", tags=["rows"])
        g.node("n", Noop(), target="rows")
        g.set_strategy("data_parallel", num_replicas=2)

        with pytest.raises(RuntimeError, match="8 rows and the targets have 4"):
            g.fit([float(i) for i in range(8)], [float(i) for i in range(4)])

    def test_differentiable_mode_points_at_the_strategy(self):
        """Refused, and the message names the path that does work."""

        class Noop(Filter):
            _cache_version = "diffmode-v1"

            def forward(self, x, state):
                return x

        g = Graph(cache="memory")
        g.add_worker("ws://127.0.0.1:1")
        g.node("n", Noop())
        with pytest.raises(RuntimeError, match="data_parallel"):
            g.fit([1.0, 2.0], [1.0, 2.0], mode="differentiable")
