import numpy as np
import pytest
from reality_core.numerics.grid import Grid2D
from reality_core.simulation.diffusion import (
    DiffusionConfig,
    explicit_stability_limit,
    simulate_diffusion,
)
from reality_core.simulation.trajectory import evenly_spaced_capture_steps


def test_evenly_spaced_capture_steps_includes_start_and_end() -> None:
    steps = evenly_spaced_capture_steps(
        total_steps=10,
        frame_count=4,
    )

    assert steps[0] == 0
    assert steps[-1] == 10


def test_evenly_spaced_capture_steps_limits_frame_count() -> None:
    steps = evenly_spaced_capture_steps(
        total_steps=3,
        frame_count=10,
    )

    assert steps == (0, 1, 2, 3)


def test_evenly_spaced_capture_steps_returns_requested_count() -> None:
    steps = evenly_spaced_capture_steps(
        total_steps=12,
        frame_count=5,
    )

    assert len(steps) == 5


def test_evenly_spaced_capture_steps_rejects_invalid_total_steps() -> None:
    with pytest.raises(
        ValueError,
        match="total_steps",
    ):
        evenly_spaced_capture_steps(
            total_steps=0,
            frame_count=3,
        )


def test_evenly_spaced_capture_steps_rejects_invalid_frame_count() -> None:
    with pytest.raises(
        ValueError,
        match="frame_count",
    ):
        evenly_spaced_capture_steps(
            total_steps=10,
            frame_count=1,
        )


def test_simulation_captures_requested_frames() -> None:
    grid = Grid2D(
        nx=5,
        ny=5,
        length_x=1.0,
        length_y=1.0,
    )

    initial = np.zeros(grid.shape, dtype=np.float64)
    initial[2, 2] = 1.0

    dt = explicit_stability_limit(
        grid,
        alpha=1.0,
    ) * 0.5

    result = simulate_diffusion(
        initial,
        grid,
        DiffusionConfig(
            alpha=1.0,
            dt=dt,
            steps=4,
        ),
        capture_steps=(0, 2, 4),
    )

    assert tuple(
        frame.step
        for frame in result.frames
    ) == (0, 2, 4)

    assert tuple(
        frame.time
        for frame in result.frames
    ) == (
        0.0,
        2 * dt,
        4 * dt,
    )


def test_simulation_without_capture_steps_has_no_frames() -> None:
    grid = Grid2D(
        nx=5,
        ny=5,
        length_x=1.0,
        length_y=1.0,
    )

    initial = np.zeros(grid.shape, dtype=np.float64)

    dt = explicit_stability_limit(
        grid,
        alpha=1.0,
    ) * 0.5

    result = simulate_diffusion(
        initial,
        grid,
        DiffusionConfig(
            alpha=1.0,
            dt=dt,
            steps=2,
        ),
    )

    assert result.frames == ()


def test_simulation_rejects_capture_step_outside_run() -> None:
    grid = Grid2D(
        nx=5,
        ny=5,
        length_x=1.0,
        length_y=1.0,
    )

    initial = np.zeros(grid.shape, dtype=np.float64)

    dt = explicit_stability_limit(
        grid,
        alpha=1.0,
    ) * 0.5

    with pytest.raises(
        ValueError,
        match="capture_steps",
    ):
        simulate_diffusion(
            initial,
            grid,
            DiffusionConfig(
                alpha=1.0,
                dt=dt,
                steps=4,
            ),
            capture_steps=(0, 5),
        )