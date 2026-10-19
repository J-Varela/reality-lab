from statistics import mean, median

from reality_core.discovery.oracle import DiffusionOracle
from reality_core.optimization.ensemble_search import ensemble_search
from reality_core.optimization.random_search import SearchBounds, random_search
from reality_core.optimization.surrogate_search import surrogate_search


def target_error(result, target: float) -> float:
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

    seeds = range(20)

    total_budget = 12
    initial_candidates = 4
    guided_iterations = total_budget - initial_candidates

    errors = {
        "random": [],
        "surrogate": [],
        "ensemble": [],
    }

    wins = {
        "random": 0,
        "surrogate": 0,
        "ensemble": 0,
    }

    for seed in seeds:
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

        seed_errors = {
            "random": target_error(
                random_result,
                oracle.target_spread,
            ),
            "surrogate": target_error(
                surrogate_result,
                oracle.target_spread,
            ),
            "ensemble": target_error(
                ensemble_result,
                oracle.target_spread,
            ),
        }

        for name, error in seed_errors.items():
            errors[name].append(error)

        winner = min(
            seed_errors,
            key=seed_errors.get,
        )

        wins[winner] += 1

        print(
            f"seed={seed:02d} | "
            f"random={seed_errors['random']:.8f} | "
            f"surrogate={seed_errors['surrogate']:.8f} | "
            f"ensemble={seed_errors['ensemble']:.8f} | "
            f"winner={winner}"
        )

    print()
    print("=" * 72)
    print("20-SEED BENCHMARK")
    print("=" * 72)

    for name in (
        "random",
        "surrogate",
        "ensemble",
    ):
        print()
        print(name.upper())
        print(f"mean error:   {mean(errors[name]):.8f}")
        print(f"median error: {median(errors[name]):.8f}")
        print(f"best error:   {min(errors[name]):.8f}")
        print(f"worst error:  {max(errors[name]):.8f}")
        print(f"wins:         {wins[name]}/{len(errors[name])}")


if __name__ == "__main__":
    main()
