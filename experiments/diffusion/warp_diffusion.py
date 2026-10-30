# Warp GPU kernels use indexing and thread-ID semantics
# that Pylance cannot fully infer.
# pyright: reportAssignmentType=false, reportGeneralTypeIssues=false, reportIndexIssue=false
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

wp.init()

N = 128
STEPS = 100
RATIO = 0.2


@wp.kernel
def diffusion_step(
    current: wp.array2d[wp.float32],
    next_grid: wp.array2d[wp.float32],
    ratio: float,
):
    i, j = wp.tid()

    # Fixed-temperature boundaries
    if i == 0 or j == 0 or i == N - 1 or j == N - 1:
        next_grid[i, j] = current[i, j]
        return

    center = current[i, j]

    laplacian = (
        current[i - 1, j]
        + current[i + 1, j]
        + current[i, j - 1]
        + current[i, j + 1]
        - 4.0 * center
    )

    next_grid[i, j] = center + ratio * laplacian


def simulate(initial, device):
    a = wp.array(initial, dtype=wp.float32, device=device)
    b = wp.empty_like(a)

    for _ in range(STEPS):
        wp.launch(
            diffusion_step,
            dim=(N, N),
            inputs=[a, b, RATIO],
            device=device,
        )
        a, b = b, a

    return a.numpy()


def main():
    initial = np.zeros((N, N), dtype=np.float32)
    initial[56:72, 56:72] = 1.0

    cpu = simulate(initial, "cpu")
    gpu = simulate(initial, "cuda:0")

    error = np.max(np.abs(cpu - gpu))

    print("Maximum CPU/GPU error:", error)
    print("GPU center temperature:", gpu[N // 2, N // 2])

    np.testing.assert_allclose(cpu, gpu, rtol=1e-5, atol=1e-6)
    print("CPU/GPU validation: PASSED")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, data, title in zip(
        axes,
        [initial, gpu, np.abs(cpu - gpu)],
        ["Initial heat", "GPU diffusion", "Absolute error"],
    ):
        ax.imshow(data, cmap="inferno", origin="lower")
        ax.set_title(title)
        ax.axis("off")

    output = Path("experiments/diffusion/warp_diffusion.png")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print("Saved:", output)


if __name__ == "__main__":
    main()
