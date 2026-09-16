import numpy as np
import pytest
from reality_core.numerics.grid import Grid2D
from reality_core.simulation.diffusion import (
    DiffusionConfig,
    explicit_stability_limit,
    simulate_diffusion,
)


def test_grid_spacing() -> None:
    grid = Grid2D(
        nx=5,
        ny=5,
        length_x=1.0,
        length_y=1.0,
    )

    assert grid.dx == pytest.approx(0.25)
    assert grid.dy == pytest.approx(0.25)
    assert grid.shape == (5, 5)


def test_explicit_stability_limit() -> None:
    grid = Grid2D(
        nx=5,
        ny=5,
        length_x=1.0,
        length_y=1.0,
    )

    limit = explicit_stability_limit(grid, alpha=1.0)

    assert limit == pytest.approx(1.0 / 64.0)


def test_center_impulse_diffuses_to_neighbors() -> None:
    grid = Grid2D(
        nx=5,
        ny=5,
        length_x=1.0,
        length_y=1.0,
    )

    initial = np.zeros(grid.shape, dtype=np.float64)
    initial[2, 2] = 1.0

    dt = explicit_stability_limit(grid, alpha=1.0) * 0.5

    result = simulate_diffusion(
        initial,
        grid,
        DiffusionConfig(
            alpha=1.0,
            dt=dt,
            steps=1,
        ),
    )

    assert 0.0 < result.final_state[2, 2] < 1.0

    assert result.final_state[1, 2] > 0.0
    assert result.final_state[3, 2] > 0.0
    assert result.final_state[2, 1] > 0.0
    assert result.final_state[2, 3] > 0.0


def test_fixed_boundaries_are_preserved() -> None:
    grid = Grid2D(
        nx=7,
        ny=7,
        length_x=1.0,
        length_y=1.0,
    )

    initial = np.zeros(grid.shape, dtype=np.float64)

    initial[0, :] = 10.0
    initial[-1, :] = 20.0
    initial[:, 0] = 30.0
    initial[:, -1] = 40.0

    original = initial.copy()

    dt = explicit_stability_limit(grid, alpha=0.5) * 0.5

    result = simulate_diffusion(
        initial,
        grid,
        DiffusionConfig(
            alpha=0.5,
            dt=dt,
            steps=10,
        ),
    )

    np.testing.assert_array_equal(
        result.final_state[0, 1:-1],
        original[0, 1:-1],
    )

    np.testing.assert_array_equal(
        result.final_state[-1, 1:-1],
        original[-1, 1:-1],
    )

    np.testing.assert_array_equal(
        result.final_state[1:-1, 0],
        original[1:-1, 0],
    )

    np.testing.assert_array_equal(
        result.final_state[1:-1, -1],
        original[1:-1, -1],
    )


def test_uniform_field_remains_uniform() -> None:
    grid = Grid2D(
        nx=9,
        ny=9,
        length_x=2.0,
        length_y=2.0,
    )

    initial = np.full(grid.shape, 7.0, dtype=np.float64)

    dt = explicit_stability_limit(grid, alpha=0.2) * 0.8

    result = simulate_diffusion(
        initial,
        grid,
        DiffusionConfig(
            alpha=0.2,
            dt=dt,
            steps=25,
        ),
    )

    np.testing.assert_allclose(result.final_state, initial)


def test_unstable_time_step_is_rejected() -> None:
    grid = Grid2D(
        nx=10,
        ny=10,
        length_x=1.0,
        length_y=1.0,
    )

    initial = np.zeros(grid.shape, dtype=np.float64)

    limit = explicit_stability_limit(grid, alpha=1.0)

    with pytest.raises(
        ValueError,
        match="stability condition",
    ):
        simulate_diffusion(
            initial,
            grid,
            DiffusionConfig(
                alpha=1.0,
                dt=limit * 1.01,
                steps=1,
            ),
        )
