import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.random_search import SearchBounds, random_search
from reality_core.optimization.surrogate_search import surrogate_search
from reality_core.optimization.ensemble_search import ensemble_search


def learning_curve(result, target):
    errors = np.array([
        abs(obs.spread - target)
        for obs in result.observations
    ])
    return np.minimum.accumulate(errors)


def main():
    oracle = DiffusionOracle(grid_size=33, steps=50)

    bounds = SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )

    curves = {
        "Random": [],
        "Surrogate": [],
        "Ensemble": [],
    }

    for seed in range(20):
        results = {
            "Random": random_search(
                oracle, bounds, num_candidates=12, seed=seed
            ),
            "Surrogate": surrogate_search(
                oracle, bounds,
                initial_candidates=4,
                iterations=8,
                seed=seed,
            ),
            "Ensemble": ensemble_search(
                oracle, bounds,
                initial_candidates=4,
                iterations=8,
                seed=seed,
                exploration_weight=0.5,
            ),
        }

        for name, result in results.items():
            curve = learning_curve(result, oracle.target_spread)
            assert len(curve) == 12
            assert np.all(np.diff(curve) <= 1e-12)
            curves[name].append(curve)

    evaluations = np.arange(1, 13)
    means = {
        name: np.mean(values, axis=0)
        for name, values in curves.items()
    }

    output = Path("experiments/discovery/results")
    output.mkdir(parents=True, exist_ok=True)

    with (output / "learning_curves.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["evaluations", *means.keys()])
        for i in range(12):
            writer.writerow([
                i + 1,
                *(means[name][i] for name in means),
            ])

    for name, values in means.items():
        plt.plot(evaluations, values, marker=".", label=name)

    plt.xlabel("Simulator evaluations")
    plt.ylabel("Mean best absolute spread error")
    plt.title("Reality Lab — Scientific Discovery Efficiency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "learning_curves.png", dpi=160)

    print("\nMEAN BEST ERROR")
    print("-" * 45)
    for budget in (4, 8, 12):
        print(f"\nAfter {budget} evaluations:")
        for name, values in means.items():
            print(f"  {name:10s}: {values[budget - 1]:.8f}")

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()