import csv
from pathlib import Path

from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.adaptive_search import adaptive_random_search
from reality_core.optimization.random_search import SearchBounds

output_path = Path("data/generated/discovery/adaptive_search.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

oracle = DiffusionOracle()

result = adaptive_random_search(
    oracle,
    SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    ),
    rounds=4,
    candidates_per_round=20,
    seed=42,
)

for round_number, round_result in enumerate(result.rounds, start=1):
    best = round_result.best_observation

    print(
        f"Round {round_number}: "
        f"alpha={best.design.alpha:.6f} "
        f"sigma={best.design.sigma:.6f} "
        f"spread={best.spread:.6f} "
        f"objective={best.objective:.10f}"
    )

best = result.best_observation

print()
print("GLOBAL BEST")
print(f"alpha:     {best.design.alpha:.6f}")
print(f"sigma:     {best.design.sigma:.6f}")
print(f"spread:    {best.spread:.6f}")
print(f"objective: {best.objective:.10f}")

with output_path.open("w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "round",
            "alpha",
            "sigma",
            "spread",
            "objective",
        ]
    )

    for round_number, round_result in enumerate(result.rounds, start=1):
        for observation in round_result.observations:
            writer.writerow(
                [
                    round_number,
                    observation.design.alpha,
                    observation.design.sigma,
                    observation.spread,
                    observation.objective,
                ]
            )

print()
print(f"Saved search history to {output_path}")