"""Exhaustive integration tests for the Soma Python API.

Covers: caching, graph topologies, somatize edge cases, events,
compilation modes, visualization, and worker management.
"""
import time

from soma import Graph, Filter, search


# ── Test Filters ──

class Scaler(Filter):
    _kind = "trainable"
    scale: float = search(0.1, 10.0, scale="log")

    def __init__(self, scale=2.0):
        super().__init__(scale=scale)

    def fit(self, x, y=None):
        return {"mean": sum(x) / len(x)}

    def forward(self, x, state):
        mean = state.get("mean", 0) if isinstance(state, dict) else 0
        return [(v - mean) * self.scale for v in x]


class Doubler(Filter):
    _kind = "stateless"

    def forward(self, x, state):
        return [v * 2 for v in x]


class Adder(Filter):
    _kind = "stateless"

    def __init__(self, amount=10.0):
        super().__init__(amount=amount)

    def forward(self, x, state):
        return [v + self.amount for v in x]


class Identity(Filter):
    _kind = "stateless"

    def forward(self, x, state):
        return x


class Merge(Filter):
    """Receives JSON dict from multiple predecessors, sums all values."""
    _kind = "stateless"

    def forward(self, x, state):
        if isinstance(x, dict):
            values = []
            for v in x.values():
                if isinstance(v, list):
                    values.extend(v)
                elif isinstance(v, (int, float)):
                    values.append(v)
            return values if values else x
        return x


# ── Graph Topology Tests ──

class TestGraphTopologies:
    def test_linear_chain(self):
        g = Graph.somatize(Doubler() >> Adder(amount=5.0))
        g.fit([1.0, 2.0, 3.0])
        result = g.forward([10.0])
        # doubler: [10] → [20], adder: [20+5] = [25]
        assert result == [25.0]

    def test_three_step_chain(self):
        g = Graph.somatize(Doubler() >> Adder(amount=1.0) >> Doubler())
        g.fit([1.0])
        result = g.forward([5.0])
        # doubler: [5] → [10], adder: [10+1] = [11], doubler: [11] → [22]
        assert result == [22.0]

    def test_fork_with_collect(self):
        g = Graph.somatize(
            Doubler() >> (Adder(amount=1.0) | Adder(amount=2.0)) >> Merge()
        )
        assert len(g) == 4

    def test_multi_input_fork(self):
        g = Graph.somatize(
            (Doubler() >> Adder(amount=1.0) | Doubler() >> Adder(amount=2.0))
            >> Merge()
        )
        assert len(g) == 5

    def test_deep_nested_topology(self):
        g = Graph.somatize(
            Doubler()
            >> (
                Adder(amount=1.0) >> Doubler()
                | Adder(amount=2.0) >> Doubler()
                | Adder(amount=3.0) >> Doubler()
            )
            >> Merge()
        )
        # 1 doubler + 3*(adder+doubler) + 1 merge = 8
        assert len(g) == 8

    def test_manual_construction(self):
        g = Graph()
        g.node(Doubler())
        g.node(Adder())
        g.edge("doubler", "adder")
        g.fit([1.0, 2.0])
        result = g.forward([5.0])
        assert result == [20.0]  # doubler: [5]→[10], adder: [10+10]=[20]

    def test_mixed_somatize_and_manual(self):
        g = Graph.somatize(Doubler() >> Adder(amount=0.0))
        g.node(Identity())
        g.edge("adder", "identity")
        assert len(g) == 3


# ── Caching Tests ──

class TestCaching:
    def test_fit_caches_state(self):
        """Second fit with same data should use cached state."""
        g = Graph.somatize(Scaler(scale=2.0))
        g.fit([10.0, 20.0, 30.0])
        # Fit again — should hit cache (no error, same result)
        g.fit([10.0, 20.0, 30.0])
        result = g.forward([20.0])
        # mean=20, (20-20)*2 = 0
        assert abs(result[0]) < 0.01

    def test_different_data_different_state(self):
        g1 = Graph.somatize(Scaler(scale=1.0))
        g1.fit([10.0, 20.0, 30.0])
        r1 = g1.forward([20.0])

        g2 = Graph.somatize(Scaler(scale=1.0))
        g2.fit([100.0, 200.0, 300.0])
        r2 = g2.forward([200.0])

        # Both should normalize their mean to ~0
        assert abs(r1[0]) < 0.01
        assert abs(r2[0]) < 0.01


# ── Compilation Tests ──

class TestCompilation:
    def test_compile_inference_mode(self):
        g = Graph.somatize(Doubler() >> Adder())
        info = g.compile("inference")
        assert info["total_nodes"] == 2
        assert "plan_text" in info
        assert "plan_mermaid" in info

    def test_compile_no_cache_mode(self):
        g = Graph.somatize(Doubler() >> Adder())
        info = g.compile("no_cache")
        assert info["cached_nodes"] == 0

    def test_compile_differentiable_mode(self):
        g = Graph.somatize(Scaler() >> Doubler())
        info = g.compile("differentiable")
        assert info["total_nodes"] == 2


# ── Visualization Tests ──

class TestVisualization:
    def test_mermaid_output(self):
        g = Graph.somatize(Doubler() >> Adder())
        m = g.to_mermaid()
        assert "graph LR" in m
        assert "doubler" in m
        assert "adder" in m
        assert "-->" in m

    def test_text_output(self):
        g = Graph.somatize(Doubler() >> Adder())
        text = g.to_text()
        assert "2 nodes" in text
        assert "doubler" in text

    def test_str_representation(self):
        g = Graph.somatize(Doubler() >> Adder())
        s = str(g)
        assert "2 nodes" in s

    def test_repr(self):
        g = Graph.somatize(Doubler())
        r = repr(g)
        assert "1 nodes" in r
        assert "fitted=false" in r

    def test_fork_mermaid(self):
        g = Graph.somatize(
            Doubler() >> (Adder() | Adder()) >> Merge()
        )
        m = g.to_mermaid()
        assert "adder" in m
        assert "adder_2" in m
        assert "merge" in m

    def test_compile_plan_mermaid(self):
        g = Graph.somatize(Doubler() >> Adder())
        info = g.compile("no_cache")
        assert "graph TD" in info["plan_mermaid"]


# ── Event Tests ──

class TestEvents:
    def test_fit_emits_events(self, tmp_path, monkeypatch):
        # A cold cache, on purpose. Fitting now goes through the same
        # execution site as running, so it consults the output cache like
        # everything else: a fit whose result is already cached reports a
        # NodeCacheHit and never starts the node. Sharing the session
        # cache dir made this test depend on whether some earlier test
        # had already fitted this graph.
        monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))

        events = []
        g = Graph.somatize(Doubler() >> Adder())
        g.on_event(lambda e: events.append(e))
        g.fit([1.0, 2.0])
        time.sleep(0.2)  # wait for background thread

        event_types = [e["event_type"] for e in events]
        assert "NodeStarted" in event_types
        assert "NodeCompleted" in event_types

    def test_a_second_identical_fit_is_served_from_cache(self, tmp_path, monkeypatch):
        """Fitting caches like running does.

        The old fit path had its own walk that never consulted the output
        cache, so refitting recomputed every node in silence. Now the
        second fit reports hits — which is also what makes a resumed
        study skip work it has already done.
        """
        monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))

        def fit_once():
            events = []
            g = Graph.somatize(Doubler() >> Adder())
            g.on_event(lambda e: events.append(e))
            g.fit([1.0, 2.0])
            time.sleep(0.2)
            return [e["event_type"] for e in events]

        assert "NodeCacheMiss" in fit_once()
        assert "NodeCacheHit" in fit_once()

    def test_forward_emits_events(self):
        events = []
        g = Graph.somatize(Doubler())
        g.on_event(lambda e: events.append(e))
        g.fit([1.0])
        g.forward([1.0])
        time.sleep(0.2)

        node_ids = [e.get("node_id", "") for e in events]
        assert "doubler" in node_ids

    def test_multiple_callbacks(self):
        events_a = []
        events_b = []
        g = Graph.somatize(Doubler())
        g.on_event(lambda e: events_a.append(e))
        g.on_event(lambda e: events_b.append(e))
        g.fit([1.0])
        time.sleep(0.2)

        assert len(events_a) > 0
        assert len(events_b) > 0


# ── Worker Management Tests ──

class TestWorkerManagement:
    def test_add_worker(self):
        g = Graph()
        g.add_worker("ws://host1:8080", token="sk-xxx", tags=["gpu"])
        g.add_worker("ws://host2:8080", tags=["cpu"])
        workers = g.workers()
        assert len(workers) == 2
        assert workers[0]["address"] == "ws://host1:8080"
        assert workers[0]["tags"] == ["gpu"]
        assert workers[1]["source"] == "direct"

    def test_set_coordinator(self):
        g = Graph()
        g.set_coordinator("http://coord:9090", token="sk-secret")
        # No coordinator running, so workers() should just return direct workers
        workers = g.workers()
        assert len(workers) == 0  # no direct workers added

    def test_mixed_workers_and_coordinator(self):
        g = Graph()
        g.add_worker("ws://host1:8080")
        g.set_coordinator("http://coord:9090")
        workers = g.workers()
        # At least the direct worker should be there
        assert len(workers) >= 1


# ── Somatize Edge Cases ──

class TestSomatizeEdgeCases:
    def test_single_filter(self):
        g = Graph.somatize(Doubler())
        assert len(g) == 1

    def test_long_chain(self):
        chain = Doubler()
        for _ in range(10):
            chain = chain >> Adder(amount=1.0)
        g = Graph.somatize(chain)
        assert len(g) == 11

    def test_wide_fork(self):
        g = Graph.somatize(
            Doubler() >> (
                Adder(amount=1.0)
                | Adder(amount=2.0)
                | Adder(amount=3.0)
                | Adder(amount=4.0)
            ) >> Merge()
        )
        assert len(g) == 6  # 1 doubler + 4 adders + 1 merge

    def test_to_with_list_and_collect(self):
        g = Graph.somatize(
            Doubler().to([Adder(), Adder()]).collect(Merge())
        )
        assert len(g) == 4

    def test_duplicate_filter_classes(self):
        """Multiple instances of the same class get unique IDs."""
        g = Graph.somatize(Doubler() >> Doubler() >> Doubler())
        assert len(g) == 3
        text = g.to_text()
        assert "doubler" in text
        assert "doubler_2" in text
        assert "doubler_3" in text

    def test_auto_naming(self):
        g = Graph()
        id1 = g.node(Doubler())
        id2 = g.node(Adder())
        assert id1 == "doubler"
        assert id2 == "adder"

    def test_explicit_naming(self):
        g = Graph()
        id1 = g.node("my_scaler", Doubler())
        assert id1 == "my_scaler"
