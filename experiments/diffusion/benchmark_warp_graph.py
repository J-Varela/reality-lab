from statistics import median
from time import perf_counter

import numpy as np
import warp as wp
from benchmark_warp import STEPS, benchmark, run_steps


def benchmark_graph(initial, repeats=10):
    device = "cuda:0"
    size = initial.shape[0]

    source = wp.array(initial, dtype=wp.float32, device=device)
    a = wp.empty_like(source)
    b = wp.empty_like(source)

    # Compile the kernel before graph capture.
    run_steps(a, b, size, device, 1)
    wp.synchronize_device(device)

    # Record 100 diffusion steps using fixed GPU buffers.
    with wp.ScopedCapture(device=device) as capture:
        output, _ = run_steps(a, b, size, device, STEPS)

    graph = capture.graph
    assert graph is not None, "CUDA graph capture failed"

    # Warm up graph replay.
    wp.copy(a, source)
    wp.capture_launch(graph)
    wp.synchronize_device(device)

    durations = []

    for _ in range(repeats):
        # Reset initial conditions before every trial.
        wp.copy(a, source)
        wp.synchronize_device(device)

        start = perf_counter()
        wp.capture_launch(graph)
        wp.synchronize_device(device)

        durations.append((perf_counter() - start) * 1000)

    return median(durations), output.numpy()


def main():
    print("REALITY LAB — CUDA GRAPH BENCHMARK")

    for size in (256, 512, 1024):
        initial = np.zeros((size, size), dtype=np.float32)

        center = size // 2
        initial[
            center - 8 : center + 8,
            center - 8 : center + 8,
        ] = 1.0

        direct_ms, direct_result = benchmark(initial, "cuda:0")
        graph_ms, graph_result = benchmark_graph(initial)

        np.testing.assert_allclose(
            direct_result,
            graph_result,
            rtol=1e-5,
            atol=1e-6,
        )

        print(f"\nGrid: {size}x{size}")
        print(f"Direct launches: {direct_ms:.3f} ms")
        print(f"CUDA graph:      {graph_ms:.3f} ms")
        print(f"Graph speedup:   {direct_ms / graph_ms:.2f}x")
        print("Validation: PASSED")


if __name__ == "__main__":
    main()
