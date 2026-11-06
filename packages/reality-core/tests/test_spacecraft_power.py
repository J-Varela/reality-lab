from dataclasses import replace

import numpy as np
import pytest
from reality_core.simulation.spacecraft_power import PowerConfig, simulate_power


@pytest.mark.parametrize("dt_s", [1.0, 7.0, 65.0, 700.0, 10000.0])
def test_healthy_orbits_match_analytical_energy_budget(dt_s: float) -> None:
    config = PowerConfig()
    trace = simulate_power(config, dt_s=dt_s)

    # An eclipse uses 13 W * 0.5 h = 6.5 Wh from a full 20 Wh battery.
    assert trace.battery_energy_wh[-1] == pytest.approx(13.5)
    assert trace.battery_energy_wh.min() == pytest.approx(13.5)
    assert trace.battery_energy_wh.max() == pytest.approx(20.0)
    assert trace.curtailed_energy_wh.sum() == pytest.approx(19.0)
    assert trace.unserved_energy_wh.sum() == 0.0
    for eclipse_end in (5400.0, 10800.0, 16200.0):
        index = np.flatnonzero(trace.time_s == eclipse_end)[0]
        assert trace.battery_energy_wh[index] == pytest.approx(13.5)


@pytest.mark.parametrize("initial_energy_wh", [0.0, 1.0, 20.0])
def test_complete_energy_accounting_including_battery_limits(
    initial_energy_wh: float,
) -> None:
    config = replace(PowerConfig(), initial_energy_wh=initial_energy_wh)
    trace = simulate_power(config, dt_s=7.0)
    hours = np.diff(trace.time_s) / 3600.0
    expected_change = (
        np.sum((trace.solar_power_w - trace.load_power_w) * hours)
        - trace.curtailed_energy_wh.sum()
        + trace.unserved_energy_wh.sum()
    )
    assert trace.battery_energy_wh[-1] - initial_energy_wh == pytest.approx(
        expected_change, abs=1e-9
    )


def test_empty_battery_reports_unsupplied_demand() -> None:
    config = PowerConfig(
        initial_energy_wh=1,
        bus_power_w=5,
        payload_power_w=0,
        orbit_period_s=3600,
        eclipse_duration_s=3600,
    )
    trace = simulate_power(config)
    assert trace.solar_power_w.sum() == 0
    assert trace.battery_energy_wh[-1] == 0
    assert trace.unserved_energy_wh.sum() == pytest.approx(14.0)


def test_continuous_sunlight_recharges_without_eclipse() -> None:
    config = PowerConfig(initial_energy_wh=0, orbit_period_s=3600, eclipse_duration_s=0)
    trace = simulate_power(config, orbits=1)
    assert trace.battery_energy_wh[-1] == pytest.approx(12.0)
    assert np.all(trace.solar_power_w == 25.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"capacity_wh": 0},
        {"initial_energy_wh": -1},
        {"initial_energy_wh": 21},
        {"solar_power_w": -1},
        {"eclipse_duration_s": 5401},
        {"bus_power_w": float("nan")},
    ],
)
def test_invalid_physical_configuration_is_rejected(changes: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        replace(PowerConfig(), **changes)


@pytest.mark.parametrize("dt_s", [0.0, -1.0, float("inf")])
def test_invalid_time_step_is_rejected(dt_s: float) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        simulate_power(PowerConfig(), dt_s=dt_s)
