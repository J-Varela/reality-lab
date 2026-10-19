import numpy as np

from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.random_search import SearchBounds, random_search
from reality_core.optimization.surrogate_search import surrogate_search


def make_bounds() -> SearchBounds:
    return SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )


def test_surrogate_search_accumulates_observations() -> None:
    oracle = DiffusionOracle(
        grid_size=17,
        steps=20,
    )

    result = surrogate_search(
        oracle,
        make_bounds(),
        initial_candidates=4,
        iterations=3,
        seed=7,
        grid_size=10,
    )

    assert len(result.observations) == 7

    assert all(
        np.isfinite(observation.objective)
        for observation in result.observations
    )


def test_surrogate_search_preserves_best_observation() -> None:
    oracle = DiffusionOracle(
        grid_size=17,
        steps=20,
    )

    result = surrogate_search(
        oracle,
        make_bounds(),
        initial_candidates=4,
        iterations=3,
        seed=7,
        grid_size=10,
    )

    expected_best = max(
        result.observations,
        key=lambda observation: observation.objective,
    )


    assert result.best_observation == expected_best


def test_surrogate_search_does_not_repeat_designs() -> None:
    oracle = DiffusionOracle(
        grid_size=17,
        steps=20,
    )

    result = surrogate_search(
        oracle,
        make_bounds(),
        initial_candidates=4,
        iterations=5,
        seed=7,
        grid_size=10,
    )

    designs = [
        (observation.design.alpha, observation.design.sigma)
        for observation in result.observations
    ]

    assert len(designs) == len(set(designs))