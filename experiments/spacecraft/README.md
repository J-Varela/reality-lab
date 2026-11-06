# Spacecraft power experiments

The first step toward an Autonomous Spacecraft Health Monitor is a healthy
reference trace whose behavior we can explain from conservation of energy.

From the repository root, using the existing uv environment:

```bash
PYTHONPATH=packages/reality-core/src uv run python experiments/spacecraft/power_baseline.py
```

The illustrative spacecraft has a 20 Wh battery initially holding 16 Wh,
25 W solar generation, a 5 W essential bus, and an always-enabled 8 W payload.
Each prescribed 90-minute cycle starts with 60 minutes of sunlight, then
30 minutes of eclipse. The run covers three cycles.

- Sunlight: the 12 W surplus charges the battery until it reaches 20 Wh.
- Eclipse: the 13 W load consumes 6.5 Wh, leaving 13.5 Wh (67.5%).
- Across the run: 75 Wh is available from solar, 58.5 Wh is requested by loads,
  and 19 Wh is curtailed because the battery is full. All load is supplied.

Accounting uses watt-hours and converts each interval from seconds to hours:

`final - initial = solar - requested load - curtailed energy + unserved energy`

Environmental transitions split integration intervals, so coarse time steps do
not smear sunlight across an eclipse boundary. Focused tests compare multiple
time steps against analytical results and check both full and empty batteries:

```bash
PYTHONPATH=packages/reality-core/src uv run --package reality-core --group dev pytest packages/reality-core/tests/test_spacecraft_power.py
```

The runner writes a plot and an interval-based CSV under
`data/generated/spacecraft/power_baseline/`. The CSV distinguishes interval
start/end times and battery energy from the powers applied during that interval.

This is an ideal energy-storage model with a prescribed illumination schedule.
It omits battery losses, charge/discharge rate limits, temperature dependence,
attitude effects, sensor noise, and autonomous fault handling. The parameters
are teaching examples, not measurements of an actual spacecraft.

## One solar-generation fault

```bash
PYTHONPATH=packages/reality-core/src uv run python experiments/spacecraft/solar_fault.py
```

This comparison runs four cycles. At minute 100, available solar generation
falls to 50% of its healthy value. The payload remains commanded on and no
recovery controller intervenes. Unmet demand is explicitly recorded. Healthy
and faulty runs share the same configuration.

Before the fault their histories match. Afterward, the faulty spacecraft has
12.5 W generation against 13 W demand in sunlight, so its battery cannot
recharge. Energy at the end of each orbit is approximately 13.50, 8.58, 1.58,
and 0 Wh. It accumulates 5.42 Wh of unserved demand by minute 360.

The plot is saved to `data/generated/spacecraft/solar_fault/solar_fault.png`.
Fault onset splits integration intervals, including for coarse time steps.
Tests also show that a fault injected during eclipse only changes solar
telemetry after sunlight returns. This illustrates a limit on observability:
the injection time is simulator ground truth, not a diagnosis available to a
health monitor.

## Detecting a persistent solar shortfall

```bash
PYTHONPATH=packages/reality-core/src uv run python experiments/spacecraft/solar_monitor.py
```

The detector receives observed and expected solar power for each telemetry
interval, available at the interval's end. Its interface has no fault label,
injection time, battery truth, or simulator configuration. The experiment
computes expected power from the known healthy sunlight schedule and adds
seeded Gaussian sensor noise (0.5 W standard deviation) to the observed power.

A residual below -5 W during expected sunlight must persist across consecutive
intervals totaling at least 10 seconds. Healthy readings, eclipse, invalid
readings, and telemetry gaps reset pending evidence. The first alarm is
latched; eclipse does not erase an alarm already raised. Confirmation uses
elapsed time, and an alarm is never backdated before telemetry arrives.

For this demonstration the healthy run has no alarms; the faulty run alarms
at 6010 seconds (100.167 minutes), 10 seconds after the injection. The runner
uses the injection time only for scoring and plot annotations. The plot,
including a zoom around the event, is saved under
`data/generated/spacecraft/solar_monitor/`.

These are teaching thresholds for a simple model with an exact illumination
schedule, not calibrated operational settings. The alarm identifies solar
underproduction, not its root cause. A sensor bias or an inaccurate prediction
could also create a residual. Recovery and richer diagnosis are later steps.
