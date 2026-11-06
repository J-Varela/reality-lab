from dataclasses import replace

import numpy as np
import pytest
from reality_core.simulation.spacecraft_power import (
    PowerConfig,
    SolarGenerationFault,
    simulate_power,
)


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


def test_midrun_solar_fault_preserves_the_healthy_past() -> None:
    healthy = simulate_power(PowerConfig())
    damaged = simulate_power(PowerConfig(), solar_fault=SolarGenerationFault(6000))
    before_fault = healthy.time_s <= 6000
    np.testing.assert_array_equal(
        healthy.battery_energy_wh[before_fault], damaged.battery_energy_wh[before_fault]
    )
    # At minute 100: 15.5 Wh. Another 110 minutes at -0.5 W in sunlight
    # and two half-hour eclipses at -13 W leave 19/12 Wh at minute 270.
    assert damaged.battery_energy_wh[-1] == pytest.approx(19 / 12)
    assert damaged.unserved_energy_wh.sum() == 0


@pytest.mark.parametrize("dt_s", [1.0, 65.0, 700.0, 10000.0])
def test_fault_onset_splits_steps_without_changing_energy_budget(dt_s: float) -> None:
    fault = SolarGenerationFault(6000.5)
    damaged = simulate_power(PowerConfig(), dt_s=dt_s, solar_fault=fault)
    assert fault.starts_at_s in damaged.time_s
    expected = 19 / 12 + 0.5 * 12.5 / 3600
    assert damaged.battery_energy_wh[-1] == pytest.approx(expected)


def test_fault_during_eclipse_becomes_observable_at_sunrise() -> None:
    healthy = simulate_power(PowerConfig())
    damaged = simulate_power(PowerConfig(), solar_fault=SolarGenerationFault(4200))
    changed_intervals = np.flatnonzero(damaged.solar_power_w != healthy.solar_power_w)
    assert damaged.time_s[changed_intervals[0]] == 5400
    np.testing.assert_array_equal(
        healthy.battery_energy_wh[:5401], damaged.battery_energy_wh[:5401]
    )


def test_unrecovered_solar_fault_exhausts_battery_and_preserves_accounting() -> None:
    config = PowerConfig()
    damaged = simulate_power(config, orbits=4, solar_fault=SolarGenerationFault(6000))
    assert damaged.battery_energy_wh[-1] == 0
    assert damaged.curtailed_energy_wh.sum() == pytest.approx(8)
    assert damaged.unserved_energy_wh.sum() == pytest.approx(65 / 12)
    hours = np.diff(damaged.time_s) / 3600
    accounted_change = (
        np.sum((damaged.solar_power_w - damaged.load_power_w) * hours)
        - damaged.curtailed_energy_wh.sum()
        + damaged.unserved_energy_wh.sum()
    )
    assert damaged.battery_energy_wh[-1] - config.initial_energy_wh == pytest.approx(
        accounted_change, abs=1e-9
    )


@pytest.mark.parametrize("start_time", [16200.0, 20000.0])
def test_fault_at_or_after_end_does_not_extend_the_run(start_time: float) -> None:
    healthy = simulate_power(PowerConfig())
    damaged = simulate_power(
        PowerConfig(), solar_fault=SolarGenerationFault(start_time)
    )
    np.testing.assert_array_equal(damaged.time_s, healthy.time_s)
    np.testing.assert_array_equal(damaged.battery_energy_wh, healthy.battery_energy_wh)


@pytest.mark.parametrize(
    "start, fraction",
    [
        (-1, 0.5),
        (float("nan"), 0.5),
        (0, -0.1),
        (0, 1.1),
        (0, float("inf")),
    ],
)
def test_invalid_solar_fault_is_rejected(start: float, fraction: float) -> None:
    with pytest.raises(ValueError):
        SolarGenerationFault(start, fraction)
