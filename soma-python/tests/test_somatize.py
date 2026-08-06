"""Tests for the fluent Graph builder API: >>, |, .to(), .collect(), Graph.somatize()."""
from soma import Graph, Filter


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


class Merge(Filter):
    _kind = "stateless"
    def forward(self, x, state):
        return x


class TestOperatorChain:
    def test_linear_rshift(self):
        g = Graph.somatize(Doubler() >> Adder())
        assert len(g) == 2
        text = g.to_text()
        assert "doubler" in text
        assert "adder" in text
        assert "← doubler" in text

    def test_three_step_chain(self):
        g = Graph.somatize(Doubler() >> Adder() >> Merge())
        assert len(g) == 3

    def test_fork_with_pipe(self):
        g = Graph.somatize(Doubler() | Adder())
        assert len(g) == 2

    def test_fork_then_collect(self):
        g = Graph.somatize(
            Doubler() >> (Adder() | Adder()) >> Merge()
        )
        assert len(g) == 4
        text = g.to_text()
        assert "merge" in text.lower()
        # Merge should have 2 predecessors
        assert "adder, adder_2" in text or "adder_2, adder" in text

    def test_nested_branches(self):
        g = Graph.somatize(
            (Doubler() >> Adder() | Doubler() >> Adder())
            >> Merge()
        )
        assert len(g) == 5
        text = g.to_text()
        assert "merge" in text.lower()


class TestMethodSyntax:
    def test_to_linear(self):
        g = Graph.somatize(Doubler().to(Adder()).to(Merge()))
        assert len(g) == 3

    def test_to_fork_collect(self):
        g = Graph.somatize(
            Doubler().to([Adder(), Adder()]).collect(Merge())
        )
        assert len(g) == 4

    def test_to_with_chains_in_fork(self):
        g = Graph.somatize(
            Doubler().to([
                Adder() >> Merge(),
                Adder() >> Merge(),
            ]).collect(Doubler())
        )
        assert len(g) == 6


class TestMixedAPI:
    def test_somatize_then_manual(self):
        """somatize creates initial topology, then add nodes manually."""
        g = Graph.somatize(Doubler() >> Adder())
        g.node(Merge())
        g.edge("adder", "merge")
        assert len(g) == 3

    def test_single_filter(self):
        """A single filter (no chain) should work too."""
        g = Graph.somatize(Doubler())
        assert len(g) == 1


class Collector(Filter):
    """Fan-in head: a multi-predecessor node receives a dict keyed by
    upstream node id (CONTRACT — pinned by the tests below)."""

    _kind = "stateless"

    def forward(self, x, state):
        assert isinstance(x, dict), f"collector expected dict, got {type(x)}"
        return [sum(sum(branch) for branch in x.values())]


class TestExecution:
    def test_linear_fit_forward(self):
        g = Graph.somatize(Doubler() >> Adder(amount=5.0))
        g.fit([1.0, 2.0, 3.0])
        result = g.forward([10.0])
        assert len(result) == 1
        # doubler: [10] → [20], adder: [20] → [25]
        assert result[0] == 25.0

    def test_fork_executes_both_branches(self):
        """Parallel branches execute concurrently on executor threads —
        this used to deadlock on the GIL and no test caught it because
        forks were only ever checked structurally."""
        g = Graph.somatize(
            Doubler() >> (Adder(amount=1.0) | Adder(amount=100.0)) >> Collector()
        )
        g.fit([1.0, 2.0])
        result = g.forward([10.0])
        # doubler: [20]; branch A: [21]; branch B: [120]; collector: 141
        assert result == [141.0]

    def test_collect_receives_all_branches_keyed_by_node(self):
        seen = {}

        class Probe(Filter):
            _kind = "stateless"

            def forward(self, x, state):
                seen.update(x)
                return [0.0]

        g = Graph.somatize(Doubler() >> (Adder(amount=1.0) | Adder(amount=2.0)) >> Probe())
        g.fit([1.0])
        g.forward([5.0])
        assert sorted(seen.keys()) == ["adder", "adder_2"]
        assert seen["adder"] == [11.0]  # 5*2 + 1
        assert seen["adder_2"] == [12.0]  # 5*2 + 2

    def test_dsl_graphs_share_the_cache(self, tmp_path, monkeypatch):
        """Two graphs built with the fluent DSL and identical filters
        hit the same persistent cache entries."""
        monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path))
        calls = {"n": 0}

        class CountingFit(Filter):
            _cache_version = "dsl-cache-v1"

            def fit(self, x, y=None):
                calls["n"] += 1
                return {"m": sum(x)}

            def forward(self, x, state):
                return [v + state["m"] for v in x]

        for _ in range(2):
            g = Graph.somatize(CountingFit() >> Doubler())
            g.fit([1.0, 2.0])
        assert calls["n"] == 1, "identical DSL pipelines must share fit work"

    def test_mermaid_output(self):
        g = Graph.somatize(
            Doubler() >> (Adder() | Adder()) >> Merge()
        )
        m = g.to_mermaid()
        assert "graph LR" in m
        assert "-->" in m
