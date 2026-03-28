"""Tests for the Soma Python Pipeline API."""
import pytest
from soma import Pipeline, Filter, search


class Doubler(Filter):
    """Doubles every value."""

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return [v * 2 for v in x]


class Adder(Filter):
    """Adds a fixed amount."""
    amount: float = search(0.0, 100.0)

    def __init__(self, amount=10.0):
        super().__init__(amount=amount)

    def fit(self, x, y=None):
        return {"mean": sum(x) / len(x)}

    def forward(self, x, state):
        mean = state.get("mean", 0) if isinstance(state, dict) else 0
        return [v + self.amount - mean for v in x]


class TestPipeline:
    def test_create_pipeline(self):
        pipeline = Pipeline([Doubler(), Adder(amount=5.0)])
        assert not pipeline.is_fitted()
        assert pipeline.filter_names() == ["Doubler", "Adder"]

    def test_fit_and_predict(self):
        pipeline = Pipeline([Doubler()])
        pipeline.fit([1.0, 2.0, 3.0])
        assert pipeline.is_fitted()

        result = pipeline.predict([4.0, 5.0])
        assert result == [8.0, 10.0]

    def test_predict_before_fit_raises(self):
        pipeline = Pipeline([Doubler()])
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.predict([1.0])

    def test_multi_filter_pipeline(self):
        pipeline = Pipeline([Doubler(), Doubler()])
        pipeline.fit([1.0])
        result = pipeline.predict([3.0])
        assert result == [12.0]  # 3 * 2 * 2


class TestFilter:
    def test_search_descriptor_collected(self):
        adder = Adder(amount=5.0)
        assert hasattr(Adder, "_soma_search_space")
        dims = Adder._soma_search_space
        assert len(dims) == 1
        assert dims[0]["name"] == "amount"
        assert dims[0]["type"] == "float"

    def test_filter_default_methods(self):
        f = Filter()
        assert f.fit([1, 2, 3]) == {}
        assert f.forward([1, 2, 3], {}) == [1, 2, 3]


class TestSearch:
    def test_float_search(self):
        class F(Filter):
            lr: float = search(0.001, 0.1, scale="log")

        dims = F._soma_search_space
        assert len(dims) == 1
        assert dims[0]["type"] == "float"
        assert dims[0]["low"] == 0.001
        assert dims[0]["high"] == 0.1
        assert dims[0]["scale"] == "log"

    def test_categorical_search(self):
        class F(Filter):
            method: str = search(choices=["a", "b", "c"])

        dims = F._soma_search_space
        assert dims[0]["type"] == "categorical"
        assert dims[0]["choices"] == ["a", "b", "c"]

    def test_int_search(self):
        class F(Filter):
            epochs: int = search(10, 100)

        dims = F._soma_search_space
        assert dims[0]["type"] == "int"
        assert dims[0]["low"] == 10
        assert dims[0]["high"] == 100

    def test_auto_bool_search(self):
        class F(Filter):
            enabled: bool = search()

        dims = F._soma_search_space
        assert dims[0]["type"] == "categorical"
        assert dims[0]["choices"] == [True, False]
