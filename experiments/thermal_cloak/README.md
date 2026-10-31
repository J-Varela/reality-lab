# Reality Lab — Thermal Cloak MVP

A small, reproducible model of steady-state 2D thermal conduction. This is a teaching experiment based on the established analytic two-dimensional insulating-core cloak, **not** a replication of the 2026 free-form 3D anisotropic-lattice experiment.

## Install and run (from the Reality Lab repository root)

Copy `experiments/thermal_cloak/` into your Reality Lab repo, then run:

```bash
uv run --with numpy --with scipy --with matplotlib \
  python experiments/thermal_cloak/run_experiment.py
```

Figure: `experiments/thermal_cloak/results/thermal_cloak_comparison.png`

## Test

```bash
uv run --with numpy --with scipy --with matplotlib --with pytest \
  pytest experiments/thermal_cloak/test_thermal_cloak.py -q
```

## Model

Solve `div(k grad(T))=0`, with `T(left)=1`, `T(right)=0`, and insulated top/bottom edges. A circular insulating core with radius 0.12 is surrounded by a ring of radius 0.24. The ring conductivity is `5/3` when the background conductivity is 1, per the 2D analytic formula.

Metric: RMS temperature disturbance *outside* the outer ring relative to a uniform-material reference. A smaller error means the object is less detectable from the external temperature field. This does not show infrared invisibility for self-heating objects or transient heat flow.

## Sources

- Li et al., *Free-form thermal cloaks in three dimensions*, Nature Communications, 2026, DOI: 10.1038/s41467-026-73167-0
- Yang et al., *Design, implementation, and extension of thermal invisibility cloaks*, AIP Advances, 2015 (analytic 2D shell formula).