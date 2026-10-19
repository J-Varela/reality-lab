from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.ensemble_search import ensemble_search
from reality_core.optimization.random_search import SearchBounds, random_search
from reality_core.optimization.surrogate_search import surrogate_search


def print_result(name: str, result, target_spread: float) -> None:
    best = result.best_observation
    error = abs(best.spread - target_spread)

    print(name)
    print("-" * 72)
    print(f"alpha:       {best.design.alpha:.6f}")
    print(f"sigma:       {best.design.sigma:.6f}")
    print(f"spread:      {best.spread:.6f}")
    print(f"target:      {target_spread:.6f}")
    print(f"abs error:   {error:.8f}")
    print(f"objective:   {best.objective:.12f}")
    print(f"experiments: {len(result.observations)}")
    print()


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

    seed = 7
    total_budget = 12
    initial_candidates = 4
    guided_iterations = total_budget - initial_candidates

    random_result = random_search(
        oracle,
        bounds,
        num_candidates=total_budget,
        seed=seed,
    )

    surrogate_result = surrogate_search(
        oracle,
        bounds,
        initial_candidates=initial_candidates,
        iterations=guided_iterations,
        seed=seed,
        grid_size=25,
    )

    ensemble_result = ensemble_search(
        oracle,
        bounds,
        initial_candidates=initial_candidates,
        iterations=guided_iterations,
        seed=seed,
        grid_size=25,
        exploration_weight=1.0,
    )

    print()
    print("=" * 72)
    print("REALITY LAB — SEARCH STRATEGY COMPARISON")
    print("=" * 72)
    print()

    print_result(
        "RANDOM SEARCH",
        random_result,
        oracle.target_spread,
    )

    print_result(
        "PLAIN SURROGATE",
        surrogate_result,
        oracle.target_spread,
    )

    print_result(
        "UNCERTAINTY-AWARE ENSEMBLE",
        ensemble_result,
        oracle.target_spread,
    )

    results = {
        "random": random_result,
        "surrogate": surrogate_result,
        "ensemble": ensemble_result,
    }

    winner_name, winner_result = min(
        results.items(),
        key=lambda item: abs(
            item[1].best_observation.spread
            - oracle.target_spread
        ),
    )

    winner = winner_result.best_observation

    print("=" * 72)
    print(f"WINNER: {winner_name.upper()}")
    print(
        "target error: "
        f"{abs(winner.spread - oracle.target_spread):.8f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
