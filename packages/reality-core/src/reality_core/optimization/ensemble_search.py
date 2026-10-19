from reality_core.discovery.oracle import DiffusionOracle
from reality_core.ml.ensemble import RBFEnsembleSurrogate
from reality_core.optimization.sampling import latin_hypercube_designs
from reality_core.optimization.random_search import (
    SearchBounds,
    SearchResult,
    random_search,
)



def ensemble_search(
    oracle: DiffusionOracle,
    bounds: SearchBounds,
    *,
    initial_candidates: int = 4,
    iterations: int = 8,
    seed: int = 42,
    grid_size: int = 25,
    exploration_weight: float = 1.0,
) -> SearchResult:
    """Run uncertainty-aware surrogate-assisted search."""

    if initial_candidates < 4:
        raise ValueError("initial_candidates must be at least 4.")

    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    initial_designs = latin_hypercube_designs(
        bounds,
        num_candidates=initial_candidates,
        seed=seed,
    )

    observations = [
        oracle.evaluate(design)
        for design in initial_designs
    ]
    ensemble = RBFEnsembleSurrogate()

    for _ in range(iterations):
        ensemble.fit(observations)

        proposed_design = ensemble.propose(
            bounds,
            target_spread=oracle.target_spread,
            exploration_weight=exploration_weight,
            grid_size=grid_size,
            excluded=[
                observation.design
                for observation in observations
            ],
        )

        observation = oracle.evaluate(proposed_design)
        observations.append(observation)

    best_observation = max(
        observations,
        key=lambda observation: observation.objective,
    )

    return SearchResult(
        best_observation=best_observation,
        observations=tuple(observations),
    )
