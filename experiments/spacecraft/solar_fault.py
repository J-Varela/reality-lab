"""Compare a healthy spacecraft with a midrun reduction in solar generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reality_core.simulation.spacecraft_power import (
    PowerConfig,
    SolarGenerationFault,
    simulate_power,
)


def main() -> None:
    config = PowerConfig()
    orbits = 4
    fault = SolarGenerationFault(starts_at_s=100 * 60, remaining_fraction=0.5)
    healthy = simulate_power(config, orbits=orbits)
    damaged = simulate_power(config, orbits=orbits, solar_fault=fault)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for trace, label, color, style in (
        (healthy, "Healthy", "#26775b", "--"),
        (damaged, "Solar fault", "#c44e36", "-"),
    ):
        minutes = trace.time_s / 60
        axes[0].step(
            minutes,
            np.r_[trace.solar_power_w, trace.solar_power_w[-1]],
            where="post",
            label=label,
            color=color,
            linestyle=style,
        )
        axes[1].plot(
            minutes,
            trace.battery_energy_wh,
            label=label,
            color=color,
            linestyle=style,
            linewidth=2,
        )
        print(
            f"{label}: final battery {trace.battery_energy_wh[-1]:.2f} Wh; "
            f"unserved demand {trace.unserved_energy_wh.sum():.2f} Wh"
        )

    axes[0].axhline(
        config.bus_power_w + config.payload_power_w,
        color="#345985",
        linewidth=1,
        label="13 W demand",
    )
    axes[0].set_ylabel("Available solar power (W)")
    axes[1].set_ylabel("Battery energy (Wh)")
    axes[1].set_xlabel("Simulated time (minutes)")
    axes[1].set_ylim(0, config.capacity_wh * 1.12)
    for ax in axes:
        for start in np.arange(orbits) * config.orbit_period_s:
            eclipse = start + config.orbit_period_s - config.eclipse_duration_s
            ax.axvspan(
                eclipse / 60,
                (start + config.orbit_period_s) / 60,
                color="#758195",
                alpha=0.15,
            )
        ax.axvline(
            fault.starts_at_s / 60, color="#444444", linestyle=":", linewidth=1.5
        )
        ax.set_xlim(0, healthy.time_s[-1] / 60)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper right")
    axes[0].set_title("Solar output halves at minute 100; shaded intervals are eclipse")
    fig.tight_layout()
    output = Path("data/generated/spacecraft/solar_fault/solar_fault.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Fault begins at {fault.starts_at_s / 60:.0f} minutes")
    for orbit in range(1, orbits + 1):
        index = np.flatnonzero(damaged.time_s == orbit * config.orbit_period_s)[0]
        print(f"Battery after orbit {orbit}: {damaged.battery_energy_wh[index]:.2f} Wh")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
