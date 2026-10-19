from statistics import mean, median

import numpy as np

from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.ensemble_search import ensemble_search
from reality_core.optimization.random_search import SearchBounds, random_search
from reality_core.optimization.surrogate_search import surrogate_search


def error(result, target: float) -> float:
    return abs(
        result.best_observation.spread - target
    )


def summarize(name: str, values: list[float]) -> None:
    array = np.array(values)

    print()
    print(name)
    print("-" * 72)
    print(f"mean:      {mean(values):.8f}")
    print(f"median:    {median(values):.8f}")
    print(f"p75:       {np.percentile(array, 75):.8f}")
    print(f"p90:       {np.percentile(array, 90):.8f}")
    print(f"best:      {min(values):.8f}")
    print(f"worst:     {max(values):.8f}")


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

    seeds = range(100)

    results = {
        "random": [],
        "surrogate": [],
        "ensemble_0.5": [],
        "ensemble_2.0": [],
    }

    pairwise_05_vs_surrogate = {
        "ensemble": 0,
        "surrogate": 0,
        "tie": 0,
    }

    for seed in seeds:
        random_result = random_search(
            oracle,
            bounds,
            num_candidates=12,
            seed=seed,
        )

        surrogate_result = surrogate_search(
            oracle,
            bounds,
            initial_candidates=4,
            iterations=8,
            seed=seed,
            grid_size=25,
        )

        ensemble_05 = ensemble_search(
            oracle,
            bounds,
            initial_candidates=4,
            iterations=8,
            seed=seed,
            grid_size=25,
            exploration_weight=0.5,
        )

        ensemble_20 = ensemble_search(
            oracle,
            bounds,
            initial_candidates=4,
            iterations=8,
            seed=seed,
            grid_size=25,
            exploration_weight=2.0,
        )

        seed_errors = {
            "random": error(
                random_result,
                oracle.target_spread,
            ),
            "surrogate": error(
                surrogate_result,
                oracle.target_spread,
            ),
            "ensemble_0.5": error(
                ensemble_05,
                oracle.target_spread,
            ),
            "ensemble_2.0": error(
                ensemble_20,
                oracle.target_spread,
            ),
        }

        for name, value in seed_errors.items():
            results[name].append(value)

        surrogate_error = seed_errors["surrogate"]
        ensemble_error = seed_errors["ensemble_0.5"]

        if np.isclose(
            surrogate_error,
            ensemble_error,
            rtol=0.0,
            atol=1e-12,
        ):
            pairwise_05_vs_surrogate["tie"] += 1
        elif ensemble_error < surrogate_error:
            pairwise_05_vs_surrogate["ensemble"] += 1
        else:
            pairwise_05_vs_surrogate["surrogate"] += 1

    print("=" * 72)
    print("REALITY LAB — 100-SEED FINALIST BENCHMARK")
    print("=" * 72)

    for name, values in results.items():
        summarize(name.upper(), values)

    print()
    print("ENSEMBLE 0.5 VS PLAIN SURROGATE")
    print("-" * 72)
    print(
        f"ensemble wins:  "
        f"{pairwise_05_vs_surrogate['ensemble']}"
    )
    print(
        f"surrogate wins: "
        f"{pairwise_05_vs_surrogate['surrogate']}"
    )
    print(
        f"ties:           "
        f"{pairwise_05_vs_surrogate['tie']}"
    )


if __name__ == "__main__":
    main()
