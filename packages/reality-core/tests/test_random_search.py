import pytest

from reality_core.discovery.oracle import DiffusionOracle

from reality_core.optimization.random_search import (
    SearchBounds,
    random_search,
)

def test_random_search_evaluates_requested_number_of_candidates() -> None:
    oracle = DiffusionOracle()

    result = random_search(
        oracle, 
        SearchBounds(
            alpha_min=0.01,
            alpha_max=0.10,
            sigma_min=0.04,
            sigma_max=0.15,
        ),
        num_candidates=8,
        seed=42,
    )

    assert len(result.observations) == 8

def test_random_search_returns_best_observation() -> None:
    oracle = DiffusionOracle()

    result = random_search(
        oracle,
        SearchBounds(
            alpha_min=0.01,
            alpha_max=0.10,
            sigma_min=0.04,
            sigma_max=0.15,
        ),
        num_candidates=12,
        seed=42,
    )

    assert result.best_observation.objective == max(
        observation.objective 
        for observation in result.observations
    )

def test_random_search_is_reproducible_with_seed() -> None:
    oracle = DiffusionOracle()

    bounds = SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.15,
    )

    first = random_search(
        oracle,
        bounds,
        num_candidates=5,
        seed=123,
    )

    second = random_search(
        oracle,
        bounds,
        num_candidates=5,
        seed=123,
    )

    assert first.observations == second.observations

def test_search_bounds_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        SearchBounds(
            alpha_min=0.10,
            alpha_max=0.01,
            sigma_min=0.04,
            sigma_max=0.15,
        )