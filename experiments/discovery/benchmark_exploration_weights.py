from statistics import mean, median

from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.ensemble_search import ensemble_search
from reality_core.optimization.random_search import SearchBounds


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

    weights = (
        0.0,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
    )

    seeds = range(20)

    results: dict[float, list[float]] = {
        weight: []
        for weight in weights
    }

    for weight in weights:
        for seed in seeds:
            result = ensemble_search(
                oracle,
                bounds,
                initial_candidates=4,
                iterations=8,
                seed=seed,
                grid_size=25,
                exploration_weight=weight,
            )

            error = abs(
                result.best_observation.spread
                - oracle.target_spread
            )

            results[weight].append(error)

        print(
            f"weight={weight:>4} | "
            f"mean={mean(results[weight]):.8f} | "
            f"median={median(results[weight]):.8f} | "
            f"worst={max(results[weight]):.8f}"
        )

    best_mean = min(
        weights,
        key=lambda weight: mean(results[weight]),
    )

    best_median = min(
        weights,
        key=lambda weight: median(results[weight]),
    )

    print()
    print("=" * 72)
    print(f"BEST MEAN WEIGHT:   {best_mean}")
    print(f"BEST MEDIAN WEIGHT: {best_median}")
    print("=" * 72)


if __name__ == "__main__":
    main()
