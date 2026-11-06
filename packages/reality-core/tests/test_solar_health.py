import numpy as np
import pytest
from reality_core.simulation.spacecraft_power import (
    PowerConfig,
    SolarGenerationFault,
    simulate_power,
)
from reality_core.solar_health import SolarAlarm, SolarPowerMonitor


def feed(
    monitor: SolarPowerMonitor,
    start: float,
    end: float,
    observed: float = 12.5,
    expected: float = 25.0,
) -> SolarAlarm | None:
    return monitor.update(
        start_s=start, end_s=end, observed_w=observed, expected_w=expected
    )


def test_ten_seconds_of_evidence_are_required_and_alarm_is_latched() -> None:
    monitor = SolarPowerMonitor()
    assert feed(monitor, 100, 109) is None
    event = feed(monitor, 109, 110)
    assert event is not None
    assert event.time_s == 110
    assert event.residual_w == -12.5
    assert event.evidence_duration_s == 10
    assert feed(monitor, 110, 120, observed=25) is None
    assert feed(monitor, 120, 130, observed=0, expected=0) is None
    assert feed(monitor, 130, 140) is None
    assert monitor.alarm is event


def test_transient_and_gap_cannot_combine_into_a_confirmed_fault() -> None:
    monitor = SolarPowerMonitor()
    assert feed(monitor, 0, 9) is None
    assert feed(monitor, 9, 10, observed=25) is None
    assert feed(monitor, 10, 19) is None
    assert feed(monitor, 20, 29) is None  # Missing telemetry breaks continuity.
    assert feed(monitor, 29, 30) is not None


def test_confirmation_uses_elapsed_time_and_never_backdates_alarm() -> None:
    monitor = SolarPowerMonitor()
    assert feed(monitor, 0, 3) is None
    assert feed(monitor, 3, 7) is None
    event = feed(monitor, 7, 12)
    assert event is not None and event.time_s == 12


def test_eclipse_breaks_pending_evidence_even_with_negative_sensor_noise() -> None:
    monitor = SolarPowerMonitor()
    assert feed(monitor, 0, 9) is None
    assert feed(monitor, 9, 100, observed=-0.5, expected=0) is None
    assert feed(monitor, 100, 109) is None
    assert feed(monitor, 109, 110) is not None


def test_invalid_sample_resets_pending_evidence() -> None:
    monitor = SolarPowerMonitor()
    feed(monitor, 0, 9)
    with pytest.raises(ValueError, match="finite"):
        feed(monitor, 9, 10, observed=float("nan"))
    assert feed(monitor, 10, 19) is None
    assert feed(monitor, 19, 20) is not None


def test_overlapping_and_out_of_order_readings_are_rejected() -> None:
    monitor = SolarPowerMonitor()
    feed(monitor, 10, 15)
    with pytest.raises(ValueError, match="overlap"):
        feed(monitor, 14, 16)


def test_threshold_boundary_does_not_count_as_a_shortfall() -> None:
    monitor = SolarPowerMonitor()
    assert feed(monitor, 0, 100, observed=20) is None


@pytest.mark.parametrize("fault_start, alarm_time", [(6000, 6010), (4200, 5410)])
def test_physical_fault_is_detected_only_after_observable_sunlight(
    fault_start: float, alarm_time: float
) -> None:
    config = PowerConfig()
    trace = simulate_power(config, solar_fault=SolarGenerationFault(fault_start))
    monitor = SolarPowerMonitor()
    for start, end, observed in zip(
        trace.time_s[:-1], trace.time_s[1:], trace.solar_power_w, strict=True
    ):
        # Prediction comes from the nominal schedule, not from the fault model.
        phase = ((start + end) / 2) % config.orbit_period_s
        expected = config.solar_power_w if phase < 3600 else 0.0
        feed(monitor, start, end, observed, expected)
    assert monitor.alarm is not None
    assert monitor.alarm.time_s == alarm_time


def test_healthy_noisy_telemetry_has_no_alarm_across_four_eclipses() -> None:
    trace = simulate_power(PowerConfig(), orbits=4)
    noise = np.random.default_rng(7).normal(0, 0.5, len(trace.solar_power_w))
    monitor = SolarPowerMonitor()
    for start, end, expected, error in zip(
        trace.time_s[:-1], trace.time_s[1:], trace.solar_power_w, noise, strict=True
    ):
        assert feed(monitor, start, end, expected + error, expected) is None
    assert monitor.alarm is None


@pytest.mark.parametrize(
    "options",
    [
        {"threshold_w": 0},
        {"threshold_w": float("nan")},
        {"confirmation_s": -1},
        {"confirmation_s": float("inf")},
    ],
)
def test_invalid_monitor_settings_are_rejected(options: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SolarPowerMonitor(**options)
