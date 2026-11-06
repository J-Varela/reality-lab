"""Ideal battery and prescribed sunlight schedule for a small power experiment."""

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PowerConfig:
    """Illustrative values, not specifications for a flight spacecraft."""

    capacity_wh: float = 20.0
    initial_energy_wh: float = 16.0
    solar_power_w: float = 25.0
    bus_power_w: float = 5.0
    payload_power_w: float = 8.0
    orbit_period_s: float = 5400.0
    eclipse_duration_s: float = 1800.0

    def __post_init__(self) -> None:
        for field in fields(self):
            if not isfinite(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be finite.")
        if self.capacity_wh <= 0 or self.orbit_period_s <= 0:
            raise ValueError("Battery capacity and orbit period must be positive.")
        if not 0 <= self.initial_energy_wh <= self.capacity_wh:
            raise ValueError("Initial battery energy must lie within its capacity.")
        if min(self.solar_power_w, self.bus_power_w, self.payload_power_w) < 0:
            raise ValueError("Power values must be nonnegative.")
        if not 0 <= self.eclipse_duration_s <= self.orbit_period_s:
            raise ValueError("Eclipse duration must lie within the orbit period.")


@dataclass(frozen=True)
class SolarGenerationFault:
    """A persistent reduction of available solar power after a chosen time."""

    starts_at_s: float
    remaining_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not isfinite(self.starts_at_s) or self.starts_at_s < 0:
            raise ValueError("Fault start time must be finite and nonnegative.")
        if (
            not isfinite(self.remaining_fraction)
            or not 0 <= self.remaining_fraction <= 1
        ):
            raise ValueError(
                "Remaining solar fraction must be finite and within [0, 1]."
            )


@dataclass(frozen=True)
class PowerTrace:
    """Energy at time points; power and energy accounting over each interval."""

    time_s: FloatArray
    battery_energy_wh: FloatArray
    solar_power_w: FloatArray
    load_power_w: FloatArray
    curtailed_energy_wh: FloatArray
    unserved_energy_wh: FloatArray


def simulate_power(
    config: PowerConfig,
    *,
    orbits: int = 3,
    dt_s: float = 1.0,
    solar_fault: SolarGenerationFault | None = None,
) -> PowerTrace:
    """Run an always-enabled payload, optionally with a solar-generation fault.

    Each orbit starts in sunlight. Environmental and fault transitions split time steps,
    including when dt_s does not divide an orbit. The ideal battery has no
    efficiency losses or rate limits. Excess generation is curtailed; demand
    that cannot be supplied by an empty battery is explicitly recorded.
    """
    if isinstance(orbits, bool) or not isinstance(orbits, int) or orbits < 1:
        raise ValueError("orbits must be a positive integer.")
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive.")

    duration_s = orbits * config.orbit_period_s
    sunlight_s = config.orbit_period_s - config.eclipse_duration_s
    orbit_starts = np.arange(orbits, dtype=np.float64) * config.orbit_period_s
    fault_edges = (
        [solar_fault.starts_at_s]
        if solar_fault is not None and solar_fault.starts_at_s < duration_s
        else []
    )
    time_s = np.unique(
        np.concatenate(
            (
                np.arange(0.0, duration_s, dt_s),
                orbit_starts,
                orbit_starts + sunlight_s,
                fault_edges,
                [duration_s],
            )
        )
    )
    # Midpoints avoid assigning an interval the illumination of its right edge.
    midpoints = (time_s[:-1] + time_s[1:]) / 2.0
    sunlit = np.remainder(midpoints, config.orbit_period_s) < sunlight_s
    solar_power_w = np.where(sunlit, config.solar_power_w, 0.0)
    if solar_fault is not None:
        solar_power_w[midpoints >= solar_fault.starts_at_s] *= (
            solar_fault.remaining_fraction
        )
    load_power_w = np.full_like(
        solar_power_w, config.bus_power_w + config.payload_power_w
    )
    net_energy_wh = (solar_power_w - load_power_w) * np.diff(time_s) / 3600.0

    energy = np.empty_like(time_s)
    energy[0] = config.initial_energy_wh
    curtailed = np.zeros_like(net_energy_wh)
    unserved = np.zeros_like(net_energy_wh)
    for index, change in enumerate(net_energy_wh):
        available = energy[index] + change
        curtailed[index] = max(available - config.capacity_wh, 0.0)
        unserved[index] = max(-available, 0.0)
        energy[index + 1] = np.clip(available, 0.0, config.capacity_wh)

    return PowerTrace(time_s, energy, solar_power_w, load_power_w, curtailed, unserved)
