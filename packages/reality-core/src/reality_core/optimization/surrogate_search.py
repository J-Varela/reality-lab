from reality_core.discovery.oracle import DiffusionOracle
from reality_core.ml.surrogate import RBFSurrogate
from reality_core.optimization.sampling import latin_hypercube_designs
from reality_core.optimization.random_search import (
    SearchBounds,
    SearchResult,
)


def surrogate_search(
    oracle: DiffusionOracle,
    bounds: SearchBounds,
    *,
    initial_candidates: int = 4,
    iterations: int = 5,
    seed: int = 42,
    grid_size: int = 25,
) -> SearchResult:
    """Run a simple surrogate-assisted discovery loop."""

    if initial_candidates < 3:
        raise ValueError("initial_candidates must be at least 3.")

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
    surrogate = RBFSurrogate()

    for _ in range(iterations):
        surrogate.fit(observations)

        proposed_design = surrogate.propose(
            bounds,
            target_spread=oracle.target_spread,
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
