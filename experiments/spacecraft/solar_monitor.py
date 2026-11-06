"""Detect solar underproduction from noisy telemetry and an independent schedule."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reality_core.simulation.spacecraft_power import (
    PowerConfig,
    SolarGenerationFault,
    simulate_power,
)
from reality_core.solar_health import SolarPowerMonitor


def main() -> None:
    config = PowerConfig()
    fault = SolarGenerationFault(starts_at_s=6000)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for label, injection, color in (
        ("Healthy", None, "#26775b"),
        ("Solar fault", fault, "#c44e36"),
    ):
        trace = simulate_power(config, orbits=4, solar_fault=injection)
        # Paired runs use the same seeded sensor noise to isolate the fault's effect.
        noise = np.random.default_rng(7).normal(0, 0.5, len(trace.solar_power_w))
        observed = trace.solar_power_w + noise
        midpoints = (trace.time_s[:-1] + trace.time_s[1:]) / 2
        phase = midpoints % config.orbit_period_s
        expected = np.where(
            phase < config.orbit_period_s - config.eclipse_duration_s,
            config.solar_power_w,
            0.0,
        )
        monitor = SolarPowerMonitor(threshold_w=5, confirmation_s=10)
        for start, end, reading, prediction in zip(
            trace.time_s[:-1], trace.time_s[1:], observed, expected, strict=True
        ):
            monitor.update(
                start_s=start, end_s=end, observed_w=reading, expected_w=prediction
            )

        sample_times = trace.time_s[1:]  # Interval telemetry arrives at its end.
        residual = observed - expected
        axes[0].plot(
            sample_times / 60,
            residual,
            color=color,
            alpha=0.7,
            linewidth=0.6,
            label=label,
        )
        relative = sample_times - fault.starts_at_s
        zoom = (relative >= -20) & (relative <= 30)
        axes[1].step(
            relative[zoom], residual[zoom], where="post", color=color, label=label
        )
        if monitor.alarm is None:
            print(f"{label} alarm: none")
        else:
            alarm = monitor.alarm
            print(f"{label} alarm: {alarm.time_s:.2f} s ({alarm.time_s / 60:.3f} min)")
            print(f"Alarm residual: {alarm.residual_w:.2f} W")
            # Fault metadata is used only here, for scoring and plot annotations.
            print(f"Detection delay: {alarm.time_s - fault.starts_at_s:.2f} s")
            axes[0].axvline(alarm.time_s / 60, color="#444444", linestyle="--")
            axes[1].axvline(
                alarm.time_s - fault.starts_at_s,
                color="#444444",
                linestyle="--",
                label="First alarm",
            )

    for start in np.arange(4) * config.orbit_period_s:
        eclipse = start + config.orbit_period_s - config.eclipse_duration_s
        axes[0].axvspan(
            eclipse / 60,
            (start + config.orbit_period_s) / 60,
            color="#758195",
            alpha=0.15,
        )
    axes[0].set_xlim(0, 4 * config.orbit_period_s / 60)
    axes[0].set_xlabel("Simulated time (minutes); shaded intervals are eclipse")
    axes[1].axvline(0, color="#888888", linestyle=":", label="Injected fault")
    axes[1].set_xlim(-20, 30)
    axes[1].set_xlabel("Time relative to injected fault (seconds)")
    for ax in axes:
        ax.axhline(-5, color="#345985", linestyle="--", label="Shortfall threshold")
        ax.set_ylabel("Observed - expected power (W)")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left")
    fig.suptitle(
        "Solar-power anomaly detection: 5 W shortfall confirmed for 10 seconds"
    )
    fig.tight_layout()
    output = Path("data/generated/spacecraft/solar_monitor/solar_monitor.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
