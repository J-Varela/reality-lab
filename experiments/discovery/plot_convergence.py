import csv
from pathlib import Path

import matplotlib.pyplot as plt


input_path = Path("data/generated/discovery/adaptive_search.csv")
output_path = Path("data/generated/discovery/convergence.png")

best_by_round: dict[int, float] = {}

with input_path.open() as file:
    reader = csv.DictReader(file)

    for row in reader:
        round_number = int(row["round"])
        objective = float(row["objective"])

        if round_number not in best_by_round:
            best_by_round[round_number] = objective
        else:
            best_by_round[round_number] = max(
                best_by_round[round_number],
                objective,
            )

rounds = sorted(best_by_round)

best_objectives = []
best_so_far = float("-inf")

for round_number in rounds:
    best_so_far = max(
        best_so_far,
        best_by_round[round_number],
    )

    best_objectives.append(best_so_far)

plt.figure(figsize=(8, 5))

plt.plot(
    rounds,
    best_objectives,
    marker="o",
)

plt.xlabel("round")
plt.ylabel("best objective")
plt.title("Adaptive Search Convergence")

plt.xticks(rounds)

plt.tight_layout()
plt.savefig(output_path, dpi=150)

print(f"Saved convergence plot to {output_path}")