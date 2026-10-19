from reality_core.discovery.oracle import DiffusionOracle
from reality_core.ml.ensemble import RBFEnsembleSurrogate
from reality_core.ml.surrogate import RBFSurrogate
from reality_core.optimization.sampling import latin_hypercube_designs
from reality_core.optimization.random_search import SearchBounds, random_search


SEED = 72
INITIAL_CANDIDATES = 4
ITERATIONS = 8
GRID_SIZE = 25


def print_observation(prefix, observation, target):
    error = abs(observation.spread - target)

    print(
        f"{prefix} | "
        f"alpha={observation.design.alpha:.6f} | "
        f"sigma={observation.design.sigma:.6f} | "
        f"spread={observation.spread:.6f} | "
        f"error={error:.8f}"
    )


def main() -> None:
    oracle = DiffusionOracle(
        grid_size=33,
        steps=50,
    )

    bounds = SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )

    target = oracle.target_spread

    initial_designs = latin_hypercube_designs(
        bounds,
        num_candidates=INITIAL_CANDIDATES,
        seed=SEED
    )

    initial_observations = tuple(
        oracle.evaluate(design)
        for design in initial_designs
    )

    print("=" * 88)
    print(f"FAILURE DIAGNOSTIC — SEED {SEED}")
    print("=" * 88)

    print()
    print("INITIAL OBSERVATIONS")
    print("-" * 88)

    for index, observation in enumerate(
        initial_observations,
        start=1,
    ):
        print_observation(
            f"{index:02d}",
            observation,
            target,
        )

    #
    # Plain surrogate
    #
    surrogate_observations = list(initial_observations)
    surrogate = RBFSurrogate()

    print()
    print("PLAIN SURROGATE")
    print("-" * 88)

    for iteration in range(1, ITERATIONS + 1):
        surrogate.fit(surrogate_observations)

        proposal = surrogate.propose(
            bounds,
            target_spread=target,
            grid_size=GRID_SIZE,
            excluded=[
                observation.design
                for observation in surrogate_observations
            ],
        )

        predicted_spread = surrogate.predict_spread(proposal)
        observation = oracle.evaluate(proposal)

        surrogate_observations.append(observation)

        print(
            f"{iteration:02d} | "
            f"alpha={proposal.alpha:.6f} | "
            f"sigma={proposal.sigma:.6f} | "
            f"pred={predicted_spread:.6f} | "
            f"actual={observation.spread:.6f} | "
            f"pred_error={abs(predicted_spread - target):.8f} | "
            f"actual_error={abs(observation.spread - target):.8f}"
        )

    #
    # Ensemble
    #
    ensemble_observations = list(initial_observations)
    ensemble = RBFEnsembleSurrogate()

    print()
    print("ENSEMBLE — WEIGHT 2.0")
    print("-" * 88)

    for iteration in range(1, ITERATIONS + 1):
        ensemble.fit(ensemble_observations)

        proposal = ensemble.propose(
            bounds,
            target_spread=target,
            exploration_weight=2.0,
            grid_size=GRID_SIZE,
            excluded=[
                observation.design
                for observation in ensemble_observations
            ],
        )

        predicted_spread, uncertainty = ensemble.predict_spread(
            proposal
        )

        observation = oracle.evaluate(proposal)
        ensemble_observations.append(observation)

        print(
            f"{iteration:02d} | "
            f"alpha={proposal.alpha:.6f} | "
            f"sigma={proposal.sigma:.6f} | "
            f"pred={predicted_spread:.6f} | "
            f"uncertainty={uncertainty:.6f} | "
            f"actual={observation.spread:.6f} | "
            f"pred_error={abs(predicted_spread - target):.8f} | "
            f"actual_error={abs(observation.spread - target):.8f}"
        )

    #
    # Pure random baseline with same total budget
    #
    random_result = random_search(
        oracle,
        bounds,
        num_candidates=12,
        seed=SEED,
    )

    print()
    print("FINAL COMPARISON")
    print("-" * 88)

    groups = {
        "random": random_result.observations,
        "surrogate": surrogate_observations,
        "ensemble": ensemble_observations,
    }

    for name, observations in groups.items():
        best = min(
            observations,
            key=lambda observation: abs(
                observation.spread - target
            ),
        )

        print(
            f"{name:10s} | "
            f"alpha={best.design.alpha:.6f} | "
            f"sigma={best.design.sigma:.6f} | "
            f"spread={best.spread:.6f} | "
            f"error={abs(best.spread - target):.8f}"
        )


if __name__ == "__main__":
    main()
