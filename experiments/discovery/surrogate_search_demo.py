from reality_core.discovery.oracle import DiffusionOracle
from reality_core.ml.surrogate import RBFSurrogate
from reality_core.optimization.random_search import SearchBounds, random_search


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

    initial_candidates = 4
    iterations = 8
    seed = 7

    initial = random_search(
        oracle,
        bounds,
        num_candidates=initial_candidates,
        seed=seed,
    )

    observations = list(initial.observations)

    print("Initial real experiments")
    print("-" * 72)

    for index, observation in enumerate(observations, start=1):
        print(
            f"{index:02d} | "
            f"alpha={observation.design.alpha:.6f} | "
            f"sigma={observation.design.sigma:.6f} | "
            f"spread={observation.spread:.6f} | "
            f"objective={observation.objective:.10f}"
        )

    surrogate = RBFSurrogate()

    print()
    print("Surrogate-guided experiments")
    print("-" * 72)

    for iteration in range(1, iterations + 1):
        surrogate.fit(observations)

        proposal = surrogate.propose(
            bounds,
            target_spread=oracle.target_spread,
            grid_size=25,
            excluded=[
                observation.design
                for observation in observations
            ],
        )

        predicted_spread = surrogate.predict_spread(proposal)

        predicted_objective = surrogate.predict_objective(
            proposal,
            target_spread=oracle.target_spread,
        )

        observation = oracle.evaluate(proposal)
        observations.append(observation)

        best = max(
            observations,
            key=lambda item: item.objective,
        )

        print(
            f"{iteration:02d} | "
            f"alpha={proposal.alpha:.6f} | "
            f"sigma={proposal.sigma:.6f} | "
            f"pred_spread={predicted_spread:.6f} | "
            f"actual_spread={observation.spread:.6f} | "
            f"pred_obj={predicted_objective:.10f} | "
            f"actual_obj={observation.objective:.10f} | "
            f"best={best.objective:.10f}"
        )

    best = max(
        observations,
        key=lambda item: item.objective,
    )

    print()
    print("Best discovery")
    print("-" * 72)
    print(f"alpha:     {best.design.alpha:.6f}")
    print(f"sigma:     {best.design.sigma:.6f}")
    print(f"spread:    {best.spread:.6f}")
    print(f"objective: {best.objective:.12f}")
    print(f"experiments run: {len(observations)}")


if __name__ == "__main__":
    main()
