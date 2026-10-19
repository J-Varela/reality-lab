import numpy as np
from scipy.stats import qmc

from reality_core.discovery.oracle import DesignPoint
from reality_core.optimization.random_search import SearchBounds


def latin_hypercube_designs(
    bounds: SearchBounds,
    *,
    num_candidates: int,
    seed: int,
) -> tuple[DesignPoint, ...]:
    """Generate space-filling designs using Latin hypercube sampling."""

    if num_candidates < 1:
        raise ValueError("num_candidates must be at least 1.")

    sampler = qmc.LatinHypercube(
        d=2,
        seed=seed,
    )

    unit_samples = sampler.random(
        n=num_candidates,
    )

    samples = qmc.scale(
        unit_samples,
        [
            bounds.alpha_min,
            bounds.sigma_min,
        ],
        [
            bounds.alpha_max,
            bounds.sigma_max,
        ],
    )

    return tuple(
        DesignPoint(
            alpha=float(alpha),
            sigma=float(sigma),
        )
        for alpha, sigma in samples
    )
