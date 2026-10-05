from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.adaptive_search import (
    adaptive_random_search,
    refine_bounds,
)
from reality_core.optimization.random_search import SearchBounds

def test_refine_bounds_shrinks_search_region() -> None:
    oracle = DiffusionOracle()

    bounds = SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )

    result = adaptive_random_search(
        oracle,
        bounds,
        rounds=1,
        candidates_per_round=5,
        seed=42,
    )

    refined = refine_bounds(
        bounds,
        result.best_observation,
    )

    assert refined.alpha_max - refined.alpha_min < (
        bounds.alpha_max - bounds.alpha_min
    )

    assert refined.sigma_max - refined.sigma_min < (
        bounds.sigma_max - bounds.sigma_min
    )

def test_adaptive_search_runs_multiple_rounds() -> None:
    oracle = DiffusionOracle()

    result = adaptive_random_search(
        oracle,
        SearchBounds(
            alpha_min=0.01,
            alpha_max=0.10,
            sigma_min=0.04,
            sigma_max=0.17,
        ),
        rounds=3,
        candidates_per_round=5,
        seed=42,
    )

    assert len(result.rounds) == 3

def test_adaptive_search_returns_global_best() -> None:
    oracle = DiffusionOracle()

    result = adaptive_random_search(
        oracle,
        SearchBounds(
            alpha_min=0.01,
            alpha_max=0.10,
            sigma_min=0.04,
            sigma_max=0.17,
        ),
        rounds=3,
        candidates_per_round=10,
        seed=42,
    )

    expected = max(
        (
            round_result.best_observation
            for round_result in result.rounds
        ),
        key=lambda observation: observation.objective,
    )

    assert result.best_observation == expected
