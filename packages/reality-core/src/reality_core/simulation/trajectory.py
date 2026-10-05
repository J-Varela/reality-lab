from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SimulationFrame:
    step: int
    time: float
    state: FloatArray


def evenly_spaced_capture_steps(
    total_steps: int,
    frame_count: int,
) -> tuple[int, ...]:
    if total_steps < 1:
        raise ValueError("total_steps must be at least 1.")

    if frame_count < 2:
        raise ValueError("frame_count must be at least 2.")

    actual_frame_count = min(
        frame_count,
        total_steps + 1,
    )

    if actual_frame_count == 2:
        return (0, total_steps)

    steps = {
        round(
            index * total_steps / (actual_frame_count - 1)
        )
        for index in range(actual_frame_count)
    }

    steps.add(0)
    steps.add(total_steps)

    return tuple(sorted(steps))