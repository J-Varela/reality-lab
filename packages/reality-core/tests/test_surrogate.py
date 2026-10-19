import numpy as np

from reality_core.discovery.oracle import DesignPoint, Observation
from reality_core.ml.surrogate import RBFSurrogate
from reality_core.optimization.random_search import SearchBounds


def make_observations() -> tuple[Observation, ...]:
    return (
        Observation(
            design=DesignPoint(alpha=0.01, sigma=0.04),
            objective=-0.04,
            spread=0.02,
        ),
        Observation(
            design=DesignPoint(alpha=0.03, sigma=0.12),
            objective=-0.01,
            spread=0.12,
        ),
        Observation(
            design=DesignPoint(alpha=0.08, sigma=0.06),
            objective=-0.02,
            spread=0.08,
        ),
        Observation(
            design=DesignPoint(alpha=0.10, sigma=0.17),
            objective=-0.08,
            spread=0.18,
        ),
    )


def test_surrogate_predicts_finite_spread() -> None:
    surrogate = RBFSurrogate()
    surrogate.fit(make_observations())

    prediction = surrogate.predict_spread(
        DesignPoint(
            alpha=0.05,
            sigma=0.10,
        )
    )

    assert np.isfinite(prediction)


def test_surrogate_reproduces_training_spread() -> None:
    observations = make_observations()

    surrogate = RBFSurrogate()
    surrogate.fit(observations)

    prediction = surrogate.predict_spread(
        observations[1].design
    )

    assert np.isclose(
        prediction,
        observations[1].spread,
    )


def test_predicted_objective_cannot_exceed_zero() -> None:
    surrogate = RBFSurrogate()
    surrogate.fit(make_observations())

    objective = surrogate.predict_objective(
        DesignPoint(alpha=0.05, sigma=0.10),
        target_spread=0.22,
    )

    assert objective <= 0.0


def test_surrogate_proposes_design_within_bounds() -> None:
    surrogate = RBFSurrogate()
    surrogate.fit(make_observations())

    bounds = SearchBounds(
        alpha_min=0.01,
        alpha_max=0.10,
        sigma_min=0.04,
        sigma_max=0.17,
    )

    proposal = surrogate.propose(
        bounds,
        target_spread=0.22,
    )

    assert bounds.alpha_min <= proposal.alpha <= bounds.alpha_max
    assert bounds.sigma_min <= proposal.sigma <= bounds.sigma_max
