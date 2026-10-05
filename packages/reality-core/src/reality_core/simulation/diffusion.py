from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reality_core.numerics.grid import Grid2D
from reality_core.simulation.trajectory import SimulationFrame

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiffusionConfig:
    alpha: float
    dt: float
    steps: int

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("Diffusivity alpha must be positive.")

        if self.dt <= 0:
            raise ValueError("Time step dt must be positive.")

        if self.steps < 1:
            raise ValueError("Simulation must contain at least one step.")


@dataclass(frozen=True)
class DiffusionResult:
    final_state: FloatArray
    steps: int
    dt: float
    simulated_time: float
    stability_limit: float
    frames: tuple[SimulationFrame, ...]


def explicit_stability_limit(grid: Grid2D, alpha: float) -> float:
    if alpha <= 0:
        raise ValueError("Diffusivity alpha must be positive.")

    inverse_spacing_squared = (1.0 / grid.dx**2) + (1.0 / grid.dy**2)

    return 1.0 / (2.0 * alpha * inverse_spacing_squared)


def simulate_diffusion(
    initial_state: FloatArray,
    grid: Grid2D,
    config: DiffusionConfig,
    *,
    capture_steps: tuple[int, ...] | None = None,
) -> DiffusionResult:
    if initial_state.shape != grid.shape:
        raise ValueError(
            f"Initial state shape {initial_state.shape} "
            f"does not match grid shape {grid.shape}."
        )

    stability_limit = explicit_stability_limit(grid, config.alpha)

    if config.dt > stability_limit:
        raise ValueError(
            "Time step violates explicit diffusion stability condition: "
            f"dt={config.dt:.6g}, maximum={stability_limit:.6g}."
        )

    state = np.asarray(initial_state, dtype=np.float64).copy()

    requested_capture_steps = set(capture_steps or ())

    if any(
        step < 0 or step > config.steps
        for step in requested_capture_steps
    ):
        raise ValueError(
            "capture_steps must lie between 0 and config.steps."
        )

    frames: list[SimulationFrame] = []

    if 0 in requested_capture_steps:
        frames.append(
            SimulationFrame(
                step=0,
                time=0.0,
                state=state.copy(),
            )
        )

    dx_squared = grid.dx**2
    dy_squared = grid.dy**2

    for step in range(1, config.steps +1):
        laplacian_x = (
            state[1:-1, 2:] - 2.0 * state[1:-1, 1:-1] + state[1:-1, :-2]
        ) / dx_squared

        laplacian_y = (
            state[2:, 1:-1] - 2.0 * state[1:-1, 1:-1] + state[:-2, 1:-1]
        ) / dy_squared

        next_state = state.copy()

        next_state[1:-1, 1:-1] = state[1:-1, 1:-1] + config.alpha * config.dt * (
            laplacian_x + laplacian_y
        )

        state = next_state

        if step in requested_capture_steps:
            frames.append(
                SimulationFrame(
                    step=step,
                    time=step * config.dt,
                    state=state.copy(),
                )
            )

    return DiffusionResult(
        final_state=state,
        steps=config.steps,
        dt=config.dt,
        simulated_time=config.steps * config.dt,
        stability_limit=stability_limit,
        frames=tuple(frames),
    )
