"""Regression tests for the thermal cloak teaching model."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import (conductivity_map, exterior_rmse,
                            shell_conductivity, steady_temperature)


def test_uniform_material_has_linear_temperature_profile():
    k, _ = conductivity_map(n=41, mode="reference")
    temperature = steady_temperature(k)
    target = np.linspace(1.0, 0.0, 41)[None, :]
    assert np.max(np.abs(temperature - target)) < 1e-9


def test_cloak_reduces_exterior_temperature_disturbance():
    n = 61
    fields = {}
    for mode in ("reference", "exposed", "cloaked"):
        k, radius = conductivity_map(n=n, mode=mode)
        fields[mode] = steady_temperature(k)
    exposed = exterior_rmse(fields["exposed"], fields["reference"], radius)
    cloaked = exterior_rmse(fields["cloaked"], fields["reference"], radius)
    assert exposed > 0.001
    assert cloaked < exposed * 0.40  # Discrete stair-stepped geometry is imperfect.


def test_ring_conductivity_is_analytic_value():
    assert np.isclose(shell_conductivity(0.12, 0.24), 5 / 3)
