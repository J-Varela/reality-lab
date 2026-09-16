from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reality_core.numerics.grid import Grid2D
from reality_core.simulation.diffusion import (
    DiffusionConfig,
    explicit_stability_limit,
    simulate_diffusion,
)

OUTPUT_DIR = Path("data/generated/diffusion")


def create_hot_spot(grid: Grid2D) -> np.ndarray:
    x = np.linspace(0.0, grid.length_x, grid.nx)
    y = np.linspace(0.0, grid.length_y, grid.ny)

    xx, yy = np.meshgrid(x, y)

    center_x = grid.length_x / 2.0
    center_y = grid.length_y / 2.0
    sigma = 0.04

    radius_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2

    state = np.exp(-radius_squared / (2.0 * sigma**2)).astype(np.float64)

    # Fixed zero-value boundaries.
    state[0, :] = 0.0
    state[-1, :] = 0.0
    state[:, 0] = 0.0
    state[:, -1] = 0.0

    return state


def main() -> None:
    grid = Grid2D(
        nx=101,
        ny=101,
        length_x=1.0,
        length_y=1.0,
    )

    alpha = 0.01

    stability_limit = explicit_stability_limit(
        grid,
        alpha=alpha,
    )

    dt = stability_limit * 0.9

    initial_state = create_hot_spot(grid)

    config = DiffusionConfig(
        alpha=alpha,
        dt=dt,
        steps=500,
    )

    result = simulate_diffusion(
        initial_state,
        grid,
        config,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "hot_plate.png"

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        constrained_layout=True,
    )

    initial_image = axes[0].imshow(
        initial_state,
        origin="lower",
        extent=(
            0.0,
            grid.length_x,
            0.0,
            grid.length_y,
        ),
        vmin=0.0,
        vmax=1.0,
    )

    axes[0].set_title("Initial field")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    final_image = axes[1].imshow(
        result.final_state,
        origin="lower",
        extent=(
            0.0,
            grid.length_x,
            0.0,
            grid.length_y,
        ),
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].set_title(f"After t = {result.simulated_time:.4f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    figure.colorbar(
        initial_image,
        ax=axes[0],
        label="Field magnitude",
    )

    figure.colorbar(
        final_image,
        ax=axes[1],
        label="Field magnitude",
    )

    figure.suptitle("Hermes Reality Lab — 2D Diffusion")

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)

    print("Hermes Reality Lab")
    print("------------------")
    print("Experiment: 2D diffusion")
    print()
    print(f"Grid: {grid.nx} x {grid.ny}")
    print(f"dx: {grid.dx:.6f}")
    print(f"dy: {grid.dy:.6f}")
    print(f"alpha: {alpha:.6f}")
    print(f"stability limit: {stability_limit:.8f}")
    print(f"dt: {dt:.8f}")
    print(f"steps: {config.steps}")
    print(f"simulated time: {result.simulated_time:.6f}")
    print()
    print(f"initial maximum: {initial_state.max():.6f}")
    print(f"final maximum: {result.final_state.max():.6f}")
    print()
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
