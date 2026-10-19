from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.ensemble_search import ensemble_search
from reality_core.optimization.random_search import SearchBounds
from reality_core.optimization.surrogate_search import surrogate_search


def error(result, target: float) -> float:
    return abs(
        result.best_observation.spread - target
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

    rows = []

    for seed in range(100):
        surrogate_result = surrogate_search(
            oracle,
            bounds,
            initial_candidates=4,
            iterations=8,
            seed=seed,
            grid_size=25,
        )

        ensemble_result = ensemble_search(
            oracle,
            bounds,
            initial_candidates=4,
            iterations=8,
            seed=seed,
            grid_size=25,
            exploration_weight=2.0,
        )

        rows.append(
            (
                seed,
                error(
                    surrogate_result,
                    oracle.target_spread,
                ),
                error(
                    ensemble_result,
                    oracle.target_spread,
                ),
            )
        )

    rows.sort(
        key=lambda row: max(row[1], row[2]),
        reverse=True,
    )

    print("Worst learned-search seeds")
    print("-" * 72)

    for seed, surrogate_error, ensemble_error in rows[:10]:
        print(
            f"seed={seed:02d} | "
            f"surrogate={surrogate_error:.8f} | "
            f"ensemble={ensemble_error:.8f}"
        )


if __name__ == "__main__":
    main()
