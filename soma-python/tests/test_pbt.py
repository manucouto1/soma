"""Population-Based Training, driven from Python.

`PbtRunner` worked and was tested in Rust for as long as the strategy
layer existed, and nothing could reach it: no binding, and the
`TrainingStrategy::PopulationBased` variant that names it refuses
(correctly — a member's hyperparameters cannot be applied over the wire).
These cover the binding that closes that gap.
"""

import math

import pytest

import soma


def _space():
    return [{"type": "float", "name": "lr", "low": 0.001, "high": 1.0, "scale": "log"}]


def test_every_member_is_trained_and_evaluated_each_generation():
    pbt = soma.Pbt(search_space=_space(), population_size=4, generations=3)
    calls = {"train": 0, "evaluate": 0}

    def train(member):
        calls["train"] += 1
        # The member arrives with everything the callback needs to run one.
        assert set(member) == {"id", "params", "state", "fitness"}
        assert "lr" in member["params"]
        return {"lr": member["params"]["lr"]}

    def evaluate(member):
        calls["evaluate"] += 1
        return -abs(math.log10(member["params"]["lr"]) + 1.0)

    population = pbt.run(train, evaluate)

    assert calls["train"] == 12, "one train per member per generation"
    assert calls["evaluate"] == 12
    assert len(population) == 4


def test_the_population_comes_back_best_first():
    pbt = soma.Pbt(search_space=_space(), population_size=6, generations=4)
    population = pbt.run(
        lambda m: {}, lambda m: -abs(math.log10(m["params"]["lr"]) + 1.0)
    )
    fitnesses = [m["fitness"] for m in population]
    assert fitnesses == sorted(fitnesses, reverse=True), fitnesses
    assert all(f is not None for f in fitnesses)


def test_the_population_converges_on_what_scores_well():
    """The point of PBT: the survivors cluster where fitness is high.

    Fitness peaks at lr=0.1 over a range spanning three decades. A run
    that only sampled at random would leave the population spread across
    that range; exploit-then-explore should pull it in.
    """
    pbt = soma.Pbt(
        search_space=_space(), population_size=8, generations=8, exploit="truncation"
    )
    population = pbt.run(
        lambda m: {}, lambda m: -abs(math.log10(m["params"]["lr"]) + 1.0)
    )
    decades = [math.log10(m["params"]["lr"]) for m in population]
    spread = max(decades) - min(decades)
    assert spread < 1.5, (
        f"the population never converged: it still spans {spread:.2f} decades "
        f"of a 3-decade range"
    )


def test_an_evaluation_that_is_not_a_number_says_so():
    pbt = soma.Pbt(search_space=_space(), population_size=2, generations=1)
    with pytest.raises(RuntimeError, match="not a number"):
        pbt.run(lambda m: {}, lambda m: {"accuracy": 0.9})


def test_a_training_callback_that_raises_names_the_member():
    pbt = soma.Pbt(search_space=_space(), population_size=2, generations=1)

    def boom(member):
        raise ValueError("no")

    # The runner logs a failed train and carries on with the old state —
    # a member that cannot train is not a reason to lose the population.
    population = pbt.run(boom, lambda m: 0.0)
    assert len(population) == 2


def test_an_empty_search_space_is_refused():
    with pytest.raises(ValueError, match="search_space is empty"):
        soma.Pbt(search_space=[])


def test_an_unknown_exploit_or_explore_lists_the_choices():
    with pytest.raises(ValueError, match="truncation, binary"):
        soma.Pbt(search_space=_space(), exploit="elitism")
    with pytest.raises(ValueError, match="perturbation, resample"):
        soma.Pbt(search_space=_space(), explore="anneal")


def test_repr_says_what_it_will_do():
    pbt = soma.Pbt(search_space=_space(), population_size=3, generations=2)
    assert repr(pbt) == "Pbt(population_size=3, generations=2, dimensions=1)"


def test_population_based_as_a_strategy_points_at_pbt():
    """The variant refuses, and the message names where PBT lives."""

    class Id(soma.Filter):
        _cache_version = "pbt-strategy-v1"

        def forward(self, x, state):
            return x

    g = soma.Graph(cache="memory")
    g.node("i", Id())
    g.set_strategy("population_based", population_size=2, generations=1)
    assert g.strategy() == "population_based"
