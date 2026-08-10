import numpy as np

from src.monitoring import drift_status, psi


def test_identical_distributions_are_stable():
    values = np.linspace(0, 1, 100)
    assert psi(values, values) == 0
    assert drift_status(0.02) == "stable"


def test_drift_thresholds_are_explicit():
    assert drift_status(0.12) == "warning"
    assert drift_status(0.3) == "critical"

