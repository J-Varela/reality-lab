from pathlib import Path

import matplotlib.pyplot as plt
from reality_core.numerics.grid import Grid2D
from reality_core.simulation.diffusion import (
    DiffusionConfig,
    simulate_diffusion,
)
from reality_core.simulation.initial_conditions import gaussian_field
from reality_core.simulation.trajectory import evenly_spaced_capture_steps

grid = Grid2D(
    nx=51,
    ny=51,
    length_x=1.0,
    length_y=1.0,
)

initial = gaussian_field(
    grid,
    center_x=0.5,
    center_y=0.5,
    sigma=0.06,
)

config = DiffusionConfig(
    alpha=0.08,
    dt=1.0e-4,
    steps=200,
)

capture_steps = evenly_spaced_capture_steps(
    total_steps=config.steps,
    frame_count=6,
)

result = simulate_diffusion(
    initial,
    grid,
    config,
    capture_steps=capture_steps,
)

output_dir = Path("data/generated/diffusion/trajectory")
output_dir.mkdir(parents=True, exist_ok=True)

for frame in result.frames:
    output_path = output_dir / f"frame_{frame.step:04d}.png"

    plt.figure(figsize=(6, 5))

    plt.imshow(
        frame.state,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
    )

    plt.colorbar(label="field value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Diffusion at step {frame.step} "
        f"(t={frame.time:.4f})"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved {output_path}")