from dataclasses import dataclass

import numpy as np

from reality_core.numerics.grid import Grid2D
from reality_core.simulation.diffusion import DiffusionConfig, simulate_diffusion
from reality_core.simulation.initial_conditions import gaussian_field


@dataclass(frozen=True)
class DesignPoint:
    alpha: float
    sigma: float


@dataclass(frozen=True)
class Observation:
    design: DesignPoint
    objective: float
    spread: float


class DiffusionOracle:
    """High-fidelity evaluator for the diffusion inverse-design problem."""

    def __init__(
        self,
        *,
        grid_size: int = 33,
        steps: int = 60,
        dt: float = 1.0e-4,
        target_spread: float = 0.22,
    ) -> None:
        self.grid = Grid2D(
            nx=grid_size,
            ny=grid_size,
            length_x=1.0,
            length_y=1.0,
        )
        self.steps = steps
        self.dt = dt
        self.target_spread = target_spread

    def evaluate(self, design: DesignPoint) -> Observation:
        initial_state = gaussian_field(
            self.grid,
            center_x=0.5,
            center_y=0.5,
            sigma=design.sigma,
            amplitude=1.0,
        )

        result = simulate_diffusion(
            initial_state,
            self.grid,
            DiffusionConfig(
                alpha=design.alpha,
                dt=self.dt,
                steps=self.steps,
            ),
        )

        spread = self._radial_spread(result.final_state)

        objective = -(spread - self.target_spread) ** 2

        return Observation(
            design=design,
            objective=float(objective),
            spread=float(spread),
        )

    def _radial_spread(self, field: np.ndarray) -> float:
        x = np.linspace(0.0, 1.0, self.grid.nx)
        y = np.linspace(0.0, 1.0, self.grid.ny)

        xx, yy = np.meshgrid(x, y)

        radius_squared = (xx - 0.5) ** 2 + (yy - 0.5) ** 2

        total_mass = np.sum(field)

        if total_mass <= 0.0:
            return 0.0

        mean_radius_squared = np.sum(field * radius_squared) / total_mass

        return float(np.sqrt(mean_radius_squared))