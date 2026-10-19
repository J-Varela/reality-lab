import numpy as np

from reality_core.discovery.oracle import DesignPoint, Observation
from reality_core.ml.ensemble import RBFEnsembleSurrogate
from reality_core.optimization.random_search import SearchBounds


def make_observations() -> tuple[Observation, ...]:
    return (
        Observation(
            design=DesignPoint(alpha=0.01, sigma=0.04),
            spread=0.06,
            objective=-0.0256,
        ),
        Observation(
            design=DesignPoint(alpha=0.03, sigma=0.10),
            spread=0.14,
            objective=-0.0064,
        ),
        Observation(
            design=DesignPoint(alpha=0.06, sigma=0.14),
            spread=0.19,
            objective=-0.0009,
        ),
        Observation(
            design=DesignPoint(alpha=0.08, sigma=0.16),
            spread=0.225,
            objective=-0.000025,
        ),
        Observation(
            design=DesignPoint(alpha=0.10, sigma=0.17),
            spread=0.24,
            objective=-0.0004,
        ),
    )


def make_bounds() -> SearchBounds:
    return SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )


def test_ensemble_predicts_mean_and_uncertainty() -> None:
    ensemble = RBFEnsembleSurrogate()
    ensemble.fit(make_observations())

    mean, uncertainty = ensemble.predict_spread(
        DesignPoint(alpha=0.07, sigma=0.15)
    )

    assert np.isfinite(mean)
    assert np.isfinite(uncertainty)
    assert uncertainty >= 0.0


def test_ensemble_proposes_design_within_bounds() -> None:
    ensemble = RBFEnsembleSurrogate()
    ensemble.fit(make_observations())

    bounds = make_bounds()

    proposal = ensemble.propose(
        bounds,
        target_spread=0.22,
        exploration_weight=1.0,
        grid_size=10,
    )

    assert bounds.alpha_min <= proposal.alpha <= bounds.alpha_max
    assert bounds.sigma_min <= proposal.sigma <= bounds.sigma_max


def test_ensemble_does_not_propose_excluded_design() -> None:
    ensemble = RBFEnsembleSurrogate()
    observations = make_observations()
    ensemble.fit(observations)

    first_proposal = ensemble.propose(
        make_bounds(),
        target_spread=0.22,
        exploration_weight=1.0,
        grid_size=10,
    )

    second_proposal = ensemble.propose(
        make_bounds(),
        target_spread=0.22,
        exploration_weight=1.0,
        grid_size=10,
        excluded=[first_proposal],
    )

    assert second_proposal != first_proposal
