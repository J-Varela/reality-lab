"""Run from the repository root with the reality-core source on PYTHONPATH."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reality_core.simulation.spacecraft_power import PowerConfig, simulate_power


def main() -> None:
    config = PowerConfig()
    trace = simulate_power(config)
    output_dir = Path("data/generated/spacecraft/power_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "power_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "start_time_s",
                "end_time_s",
                "solar_power_w",
                "requested_load_power_w",
                "battery_start_wh",
                "battery_end_wh",
                "curtailed_energy_wh",
                "unserved_energy_wh",
            )
        )
        writer.writerows(
            zip(
                trace.time_s[:-1],
                trace.time_s[1:],
                trace.solar_power_w,
                trace.load_power_w,
                trace.battery_energy_wh[:-1],
                trace.battery_energy_wh[1:],
                trace.curtailed_energy_wh,
                trace.unserved_energy_wh,
                strict=True,
            )
        )

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    minutes = trace.time_s / 60.0
    for values, label, color in (
        (trace.solar_power_w, "Available solar power", "#d89b16"),
        (trace.load_power_w, "Bus + payload demand", "#345985"),
    ):
        axes[0].step(
            minutes, np.r_[values, values[-1]], where="post", label=label, color=color
        )
    axes[0].set_ylabel("Power (W)")
    axes[0].set_ylim(-1, 30)
    axes[0].legend(loc="upper right")
    axes[1].plot(minutes, trace.battery_energy_wh, color="#26775b", linewidth=2)
    axes[1].axhline(config.capacity_wh, color="#888888", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Battery energy (Wh)")
    axes[1].set_xlabel("Simulated time (minutes)")
    axes[1].set_ylim(0, config.capacity_wh * 1.12)
    for orbit_start in np.arange(3) * config.orbit_period_s:
        eclipse_start = orbit_start + config.orbit_period_s - config.eclipse_duration_s
        for ax in axes:
            ax.axvspan(
                eclipse_start / 60.0,
                (orbit_start + config.orbit_period_s) / 60.0,
                color="#758195",
                alpha=0.15,
            )
    for ax in axes:
        ax.set_xlim(0, minutes[-1])
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Healthy spacecraft power baseline — shaded intervals are eclipse")
    fig.tight_layout()
    plot_path = output_dir / "power_baseline.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    hours = np.diff(trace.time_s) / 3600.0
    generated_wh = float(np.sum(trace.solar_power_w * hours))
    requested_wh = float(np.sum(trace.load_power_w * hours))
    curtailed_wh = float(np.sum(trace.curtailed_energy_wh))
    unserved_wh = float(np.sum(trace.unserved_energy_wh))
    balance_error = (
        trace.battery_energy_wh[-1]
        - trace.battery_energy_wh[0]
        - generated_wh
        + requested_wh
        + curtailed_wh
        - unserved_wh
    )
    print(f"Simulated duration: {minutes[-1]:.0f} minutes")
    print(
        f"Minimum / final battery: {trace.battery_energy_wh.min():.2f} / {trace.battery_energy_wh[-1]:.2f} Wh"
    )
    print(f"Generated / requested energy: {generated_wh:.2f} / {requested_wh:.2f} Wh")
    print(f"Curtailed solar energy: {curtailed_wh:.2f} Wh")
    print(f"Unserved load: {unserved_wh:.2f} Wh")
    print(f"Energy balance error: {balance_error:.3e} Wh")
    print(f"Saved {plot_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
