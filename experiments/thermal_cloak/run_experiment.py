"""Steady-state 2D thermal cloaking: reference vs exposed core vs cloaked core.

Run from the Reality Lab repository root:
    uv run --with numpy --with scipy --with matplotlib \
        python experiments/thermal_cloak/run_experiment.py

Units are normalized, with T=1 at x=0 and T=0 at x=1.
The top and bottom edges are insulated (zero normal heat flux).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

INNER_RADIUS = 0.12
OUTER_RADIUS = 0.24
BACKGROUND_K = 1.0
CORE_K = 1e-5  # Approximates a perfectly insulating inner disk.


def shell_conductivity(inner: float = INNER_RADIUS, outer: float = OUTER_RADIUS) -> float:
    """2D insulating-core cloak for an imposed *uniform* background gradient."""
    if not (0 < inner < outer):
        raise ValueError("Expected 0 < inner radius < outer radius")
    return BACKGROUND_K * (outer**2 + inner**2) / (outer**2 - inner**2)


def conductivity_map(
    n: int = 101,
    mode: str = "reference",
    inner: float = INNER_RADIUS,
    outer: float = OUTER_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (k[y,x], radius[y,x]) on a square unit domain."""
    if n < 5:
        raise ValueError("Use n >= 5")
    if mode not in {"reference", "exposed", "cloaked"}:
        raise ValueError(f"Unknown mode: {mode}")
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    radius = np.hypot(x - 0.5, y - 0.5)
    k = np.full((n, n), BACKGROUND_K)
    if mode == "cloaked":
        k[(radius >= inner) & (radius < outer)] = shell_conductivity(inner, outer)
    if mode != "reference":
        k[radius < inner] = CORE_K
    return k, radius


def _harmonic_mean(a: float, b: float) -> float:
    return 2.0 * a * b / (a + b)


def steady_temperature(k: np.ndarray) -> np.ndarray:
    """Finite-volume discretization of div(k grad(T)) = 0.

    Left and right boundaries have fixed temperatures 1 and 0;
    no top/bottom neighbor means zero normal flux at those boundaries.
    Harmonic face conductivities conserve flux across material interfaces.
    """
    if k.ndim != 2 or k.shape[0] != k.shape[1] or np.any(k <= 0):
        raise ValueError("k must be a square 2D array with positive values")
    n = k.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros(n * n)

    for j in range(n):
        for i in range(n):
            p = j * n + i
            if i == 0 or i == n - 1:
                rows.append(p)
                cols.append(p)
                values.append(1.0)
                rhs[p] = 1.0 if i == 0 else 0.0
                continue

            diag = 0.0
            for dj, di in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nj, ni = j + dj, i + di
                if not (0 <= nj < n and 0 <= ni < n):
                    continue  # Insulated top/bottom boundaries.
                q = nj * n + ni
                face_k = _harmonic_mean(float(k[j, i]), float(k[nj, ni]))
                diag += face_k
                rows.append(p)
                cols.append(q)
                values.append(-face_k)
            rows.append(p)
            cols.append(p)
            values.append(diag)

    operator = coo_matrix((values, (rows, cols)), shape=(n * n, n * n)).tocsr()
    return np.asarray(spsolve(operator, rhs)).reshape(n, n)


def exterior_rmse(field: np.ndarray, reference: np.ndarray, radius: np.ndarray,
                  outer: float = OUTER_RADIUS) -> float:
    """How detectable is the object outside the cloak's footprint?"""
    outside = radius > outer + 2 / (len(radius) - 1)
    return float(np.sqrt(np.mean((field[outside] - reference[outside]) ** 2)))


def run(n: int = 101, output_dir: Path | None = None) -> dict[str, float]:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {}
    for mode in ("reference", "exposed", "cloaked"):
        k, radius = conductivity_map(n=n, mode=mode)
        cases[mode] = steady_temperature(k)

    reference = cases["reference"]
    exposed_error = exterior_rmse(cases["exposed"], reference, radius)
    cloaked_error = exterior_rmse(cases["cloaked"], reference, radius)
    reduction = 100 * (1 - cloaked_error / exposed_error) if exposed_error else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for ax, (mode, field) in zip(axes, cases.items()):
        picture = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1)
        if mode != "reference":
            ax.add_patch(Circle((0.5, 0.5), INNER_RADIUS, fill=False,
                                edgecolor="white", linestyle="--", linewidth=1.3))
        if mode == "cloaked":
            ax.add_patch(Circle((0.5, 0.5), OUTER_RADIUS, fill=False,
                                edgecolor="white", linestyle=":", linewidth=1.3))
        ax.set(title={"reference": "1. Uniform material",
                      "exposed": "2. Insulating object",
                      "cloaked": "3. Object + cloak"}[mode], xlabel="x", ylabel="y")
        ax.set_aspect("equal")
    fig.colorbar(picture, ax=axes, label="Normalized temperature", shrink=0.8)
    filename = output_dir / "thermal_cloak_comparison.png"
    fig.savefig(filename, dpi=160)
    plt.close(fig)

    print(f"Analytical ring conductivity: {shell_conductivity():.6f} (background = 1)")
    print(f"Exposed-object exterior RMSE: {exposed_error:.8f}")
    print(f"Cloaked-object exterior RMSE: {cloaked_error:.8f}")
    print(f"Exterior temperature-disturbance reduction: {reduction:.2f}%")
    print(f"Figure: {filename}")
    return {"exposed_rmse": exposed_error,
            "cloaked_rmse": cloaked_error, "disturbance_reduction_percent": reduction}


if __name__ == "__main__":
    run()
