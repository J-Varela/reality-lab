from dataclasses import dataclass

import numpy as np 

from reality_core.discovery.oracle import DesignPoint, DiffusionOracle, Observation

@dataclass(frozen=True)
class SearchBounds:
    alpha_min: float
    alpha_max: float
    sigma_min: float
    sigma_max: float

    def __post_init__(self) -> None:
        if self.alpha_min <= 0:
            raise ValueError("alpha_min must be positive.")
        if self.sigma_min <= 0:
            raise ValueError("sigma_min must be positive.")
        if self.alpha_min >= self.alpha_max:
            raise ValueError("alpha_min must be less than alpha_max.")
        if self.sigma_min >= self.sigma_max:
            raise ValueError("sigma_min must be less than sigma_max.")

@dataclass(frozen=True)
class SearchResult:
    best_observation: Observation
    observations: tuple[Observation, ...]

def random_search(
        oracle: DiffusionOracle,
        bounds: SearchBounds,
        *,
        num_candidates: int,
        seed: int | None = None,
) -> SearchResult:
    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    rng = np.random.default_rng(seed)

    observations: list[Observation] = []

    for _ in range(num_candidates):
        design = DesignPoint(
            alpha=float(rng.uniform(bounds.alpha_min, bounds.alpha_max)),
            sigma=float(rng.uniform(bounds.sigma_min, bounds.sigma_max)),
        )

        observations.append(oracle.evaluate(design))

    best_observation = max(
        observations,
        key=lambda observation: observation.objective,
    )

    return SearchResult(
        best_observation=best_observation,
        observations=tuple(observations),
    )