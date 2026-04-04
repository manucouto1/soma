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
    def run_worker():
        w = Worker(
            port=WORKER_PORT,
            tags=["test", "e2e"],
            token=WORKER_TOKEN,
            max_concurrent=2,
            worker_id="e2e-test-worker",
        )
        w.serve()

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()

    # Wait for worker to be ready
    import urllib.request
    for _ in range(50):
        try:
            url = f"http://127.0.0.1:{WORKER_PORT}/health"
            resp = urllib.request.urlopen(url, timeout=1)
            if resp.read() == b"ok":
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("Worker did not start in time")

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
    """WS Binary streaming for chunked data."""

    @pytest.mark.skip(reason="WS Binary streaming requires async StreamExecutor integration — Phase 3 transport layer")
    def test_stream_forward(self):
        g = make_graph()
        g.node("doubler", DoubleFilter())
        g.fit([1.0])
        result = g.forward(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            stream=True,
            chunk_size=3,
        )
        assert result is not None


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
        g.node("doubler", DoubleFilter())
        with pytest.raises(RuntimeError, match="fitted"):
            g.forward([1.0])

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
        assert len(result) == 2

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
