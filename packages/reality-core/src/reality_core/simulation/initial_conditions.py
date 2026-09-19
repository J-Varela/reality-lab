import numpy as np
from numpy.typing import NDArray

from reality_core.numerics.grid import Grid2D

FloatArray = NDArray[np.float64]


def gaussian_field(
    grid: Grid2D,
    *,
    center_x: float,
    center_y: float,
    sigma: float,
    amplitude: float = 1.0,
    fixed_zero_boundaries: bool = True,
) -> FloatArray:
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    if amplitude < 0:
        raise ValueError("Gaussian amplitude must be non-negative.")

    if not 0.0 <= center_x <= grid.length_x:
        raise ValueError("center_x must lie inside the grid domain.")

    if not 0.0 <= center_y <= grid.length_y:
        raise ValueError("center_y must lie inside the grid domain.")

    x = np.linspace(0.0, grid.length_x, grid.nx)
    y = np.linspace(0.0, grid.length_y, grid.ny)

    xx, yy = np.meshgrid(x, y)

    radius_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2

    field = amplitude * np.exp(-radius_squared / (2.0 * sigma**2))

    field = field.astype(np.float64)

    if fixed_zero_boundaries:
        field[0, :] = 0.0
        field[-1, :] = 0.0
        field[:, 0] = 0.0
        field[:, -1] = 0.0

    return field
