# Warp kernels use GPU-specific indexing and thread IDs.
# pyright: reportAssignmentType=false
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import warp as wp

wp.init()

GRID_SIZES = (256, 512, 1024)
STEPS = 100
REPEATS = 10
RATIO = 0.2


@wp.kernel
def diffusion_step(
    current: wp.array2d[wp.float32],
    next_grid: wp.array2d[wp.float32],
    size: int,
    ratio: float,
):
    i, j = wp.tid()

    if i == 0 or j == 0 or i == size - 1 or j == size - 1:
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


def run_steps(a, b, size, device, count):
    for _ in range(count):
        wp.launch(
            diffusion_step,
            dim=(size, size),
            inputs=[a, b, size, RATIO],
            device=device,
        )
        a, b = b, a

    return a, b


def synchronize(device):
    if device != "cpu":
        wp.synchronize_device(device)


def benchmark(initial, device):
    size = initial.shape[0]

    # Compile and warm up outside the timed region.
    a = wp.array(initial, dtype=wp.float32, device=device)
    b = wp.empty_like(a)
    run_steps(a, b, size, device, count=3)
    synchronize(device)

    durations = []

    for _ in range(REPEATS):
        # Reset the initial condition for each trial.
        a = wp.array(initial, dtype=wp.float32, device=device)
        b = wp.empty_like(a)
        synchronize(device)

        start = perf_counter()
        a, b = run_steps(a, b, size, device, STEPS)
        synchronize(device)

        durations.append((perf_counter() - start) * 1000)

    return median(durations), a.numpy()


def main():
    print("REALITY LAB — NVIDIA WARP BENCHMARK")
    print(f"Steps: {STEPS} | Repeats: {REPEATS}")
    print()

    results = []

    for size in GRID_SIZES:
        initial = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        initial[
            center - 8 : center + 8,
            center - 8 : center + 8,
        ] = 1.0

        cpu_ms, cpu_result = benchmark(initial, "cpu")
        gpu_ms, gpu_result = benchmark(initial, "cuda:0")

        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-6)

        error = np.max(np.abs(cpu_result - gpu_result))
        speedup = cpu_ms / gpu_ms

        results.append(
            {
                "grid_size": size,
                "cells": size * size,
                "cpu_ms": cpu_ms,
                "gpu_ms": gpu_ms,
                "speedup": speedup,
                "max_error": float(error),
            }
        )

        print(f"Grid: {size} x {size}")
        print(f"  CPU:     {cpu_ms:.3f} ms")
        print(f"  GPU:     {gpu_ms:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Error:   {error:.3e}")
        print()

    output = Path("experiments/diffusion/warp_benchmark.csv")

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "grid_size",
                "cells",
                "cpu_ms",
                "gpu_ms",
                "speedup",
                "max_error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved benchmark results to {output}")


if __name__ == "__main__":
    main()
