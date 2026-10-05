from reality_core.discovery.oracle import DesignPoint, DiffusionOracle


def test_diffusion_oracle_returns_finite_observation() -> None:
    oracle = DiffusionOracle()

    observation = oracle.evaluate(
        DesignPoint(
            alpha=0.05,
            sigma=0.08,
        )
    )

    assert observation.spread > 0.0
    assert observation.objective <= 0.0


def test_design_changes_physical_response() -> None:
    oracle = DiffusionOracle()

    first = oracle.evaluate(
        DesignPoint(
            alpha=0.02,
            sigma=0.05,
        )
    )

    second = oracle.evaluate(
        DesignPoint(
            alpha=0.10,
            sigma=0.12,
        )
    )

    assert first.spread != second.spread