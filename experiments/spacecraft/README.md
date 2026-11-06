# Healthy spacecraft power baseline

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
are teaching examples, not measurements of an actual spacecraft. The next
experiment can introduce one solar-generation fault against this reference.
