import numpy as np
import pytest
from reality_core.numerics.grid import Grid2D
from reality_core.simulation.initial_conditions import gaussian_field


def test_gaussian_field_has_expected_shape() -> None:
    grid = Grid2D(
        nx=21,
        ny=31,
        length_x=1.0,
        length_y=2.0,
    )

    field = gaussian_field(
        grid,
        center_x=0.5,
        center_y=1.0,
        sigma=0.1,
    )

    assert field.shape == grid.shape


def test_gaussian_field_peaks_at_center() -> None:
    grid = Grid2D(
        nx=11,
        ny=11,
        length_x=1.0,
        length_y=1.0,
    )

    field = gaussian_field(
        grid,
        center_x=0.5,
        center_y=0.5,
        sigma=0.1,
        amplitude=2.0,
    )

    assert field[5, 5] == pytest.approx(2.0)


def test_gaussian_field_has_fixed_zero_boundaries() -> None:
    grid = Grid2D(
        nx=11,
        ny=11,
        length_x=1.0,
        length_y=1.0,
    )

    field = gaussian_field(
        grid,
        center_x=0.5,
        center_y=0.5,
        sigma=0.1,
    )

    np.testing.assert_array_equal(field[0, :], 0.0)
    np.testing.assert_array_equal(field[-1, :], 0.0)
    np.testing.assert_array_equal(field[:, 0], 0.0)
    np.testing.assert_array_equal(field[:, -1], 0.0)


def test_gaussian_field_rejects_invalid_sigma() -> None:
    grid = Grid2D(
        nx=11,
        ny=11,
        length_x=1.0,
        length_y=1.0,
    )

    with pytest.raises(ValueError, match="sigma"):
        gaussian_field(
            grid,
            center_x=0.5,
            center_y=0.5,
            sigma=0.0,
        )
