import csv
from pathlib import Path

import matplotlib.pyplot as plt


input_path = Path("data/generated/discovery/adaptive_search.csv")
output_path = Path("data/generated/discovery/adaptive_search.png")

alpha_values = []
sigma_values = []
round_numbers = []

with input_path.open() as file:
    reader = csv.DictReader(file)

    for row in reader:
        alpha_values.append(float(row["alpha"]))
        sigma_values.append(float(row["sigma"]))
        round_numbers.append(int(row["round"]))

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    alpha_values,
    sigma_values,
    c=round_numbers,
)

plt.xlabel("alpha")
plt.ylabel("sigma")
plt.title("Adaptive Search Trajectory")

plt.colorbar(scatter, label="round")

plt.tight_layout()
plt.savefig(output_path, dpi=150)

print(f"Saved plot to {output_path}")