"""Tests for the Soma Python Study API."""
import pytest
from soma import Study


def simple_executor(params):
    """f1 = 1.0 - |x - 0.5| * 2"""
    x = params.get("x", 0.5)
    f1 = max(0.0, 1.0 - abs(x - 0.5) * 2)
    return {"f1": f1}


class TestStudy:
    def test_create_study(self):
        study = Study(
            name="test",
            search_space=[
                {"type": "float", "name": "x", "low": 0.0, "high": 1.0, "scale": "linear"},
            ],
            strategy="random",
            n_trials=10,
            objectives=[("f1", "maximize")],
            seed=42,
        )
        assert study.n_trials == 0  # no trials run yet

    def test_run_random_study(self):
        study = Study(
            name="random_test",
            search_space=[
                {"type": "float", "name": "x", "low": 0.0, "high": 1.0, "scale": "linear"},
            ],
            strategy="random",
            n_trials=20,
            objectives=[("f1", "maximize")],
            seed=42,
        )
        study.run(simple_executor)

        assert study.n_trials == 20
        assert study.progress == 1.0

        best = study.best_trial
        assert best is not None
        assert "f1" in best["metrics"]
        assert best["metrics"]["f1"] > 0.5

    def test_run_grid_study(self):
        study = Study(
            name="grid_test",
            search_space=[
                {"type": "float", "name": "x", "low": 0.0, "high": 1.0, "scale": "linear"},
                {"type": "categorical", "name": "method", "choices": ["a", "b"]},
            ],
            strategy="grid",
            n_trials=5,  # 5 points per dim
            objectives=[("f1", "maximize")],
        )
        study.run(simple_executor)

        assert study.n_trials == 10  # 5 * 2

    def test_study_with_categorical(self):
        def executor(params):
            method = params.get("method", "a")
            return {"score": 0.9 if method == "best" else 0.1}

        study = Study(
            name="cat_test",
            search_space=[
                {"type": "categorical", "name": "method", "choices": ["bad", "best", "worst"]},
            ],
            strategy="grid",
            n_trials=3,
            objectives=[("score", "maximize")],
        )
        study.run(executor)

        best = study.best_trial
        assert best["params"]["method"] == "best"
        assert best["metrics"]["score"] == 0.9
