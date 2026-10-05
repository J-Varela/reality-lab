from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.random_search import (
    SearchBounds,
    random_search,
)

oracle = DiffusionOracle()

result = random_search(
    oracle,
    SearchBounds(
        alpha_min=0.015,
        alpha_max=0.10,
        sigma_min=0.135,
        sigma_max=0.16,
    ),
    num_candidates=30,
    seed=7,
)

ranked = sorted(
    result.observations,
    key=lambda observation: observation.objective,
    reverse=True,   
)

for rank, observation in enumerate(ranked, start=1):
    print(
        f"{rank:02d} "
        f"alpha={observation.design.alpha:.5f} "
        f"sigma={observation.design.sigma:.5f} "
        f"spread={observation.spread:.5f} "
        f"objective={observation.objective:.8f}"
    )

best = result.best_observation

print()
print("BEST DESIGN")
print(f"alpha:     {best.design.alpha:.5f}")
print(f"sigma:     {best.design.sigma:.5f}")
print(f"spread:    {best.spread:.5f}")
print(f"objective: {best.objective:.8f}")