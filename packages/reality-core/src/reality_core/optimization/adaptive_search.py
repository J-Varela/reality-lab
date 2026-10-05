from dataclasses import dataclass

from reality_core.discovery.oracle import DiffusionOracle, Observation
from reality_core.optimization.random_search import (
    SearchBounds,
    SearchResult,
    random_search,
)


@dataclass(frozen=True)
class AdaptiveSearchResult:
    best_observation: Observation
    rounds: tuple[SearchResult, ...]


def refine_bounds(
    bounds: SearchBounds,
    best: Observation,
    *,
    alpha_fraction: float = 0.25,
    sigma_fraction: float = 0.25,
) -> SearchBounds:
    alpha_width = bounds.alpha_max - bounds.alpha_min
    sigma_width = bounds.sigma_max - bounds.sigma_min

    alpha_radius = alpha_width * alpha_fraction
    sigma_radius = sigma_width * sigma_fraction

    return SearchBounds(
        alpha_min=max(
            bounds.alpha_min,
            best.design.alpha - alpha_radius,
        ),
        alpha_max=min(
            bounds.alpha_max,
            best.design.alpha + alpha_radius,
        ),
        sigma_min=max(
            bounds.sigma_min,
            best.design.sigma - sigma_radius,
        ),
        sigma_max=min(
            bounds.sigma_max,
            best.design.sigma + sigma_radius,
        ),
    )


def adaptive_random_search(
    oracle: DiffusionOracle,
    initial_bounds: SearchBounds,
    *,
    rounds: int,
    candidates_per_round: int,
    seed: int = 42,
) -> AdaptiveSearchResult:
    if rounds < 1:
        raise ValueError("rounds must be at least 1.")

    bounds = initial_bounds
    round_results: list[SearchResult] = []

    for round_index in range(rounds):
        result = random_search(
            oracle,
            bounds,
            num_candidates=candidates_per_round,
            seed=seed + round_index,
        )

        round_results.append(result)

        bounds = refine_bounds(
            bounds,
            result.best_observation,
        )

    best_observation = max(
        (
            result.best_observation
            for result in round_results
        ),
        key=lambda observation: observation.objective,
    )

    return AdaptiveSearchResult(
        best_observation=best_observation,
        rounds=tuple(round_results),
    )