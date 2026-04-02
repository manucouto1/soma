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
        g.connect("adder", "merge")
        assert len(g) == 3

    def test_single_filter(self):
        """A single filter (no chain) should work too."""
        g = Graph.somatize(Doubler())
        assert len(g) == 1


class TestExecution:
    def test_linear_fit_forward(self):
        g = Graph.somatize(Doubler() >> Adder(amount=5.0))
        g.fit([1.0, 2.0, 3.0])
        result = g.forward([10.0])
        assert len(result) == 1
        # doubler: [10] → [20], adder: [20] → [25]
        assert result[0] == 25.0

    def test_mermaid_output(self):
        g = Graph.somatize(
            Doubler() >> (Adder() | Adder()) >> Merge()
        )
        m = g.to_mermaid()
        assert "graph LR" in m
        assert "-->" in m
