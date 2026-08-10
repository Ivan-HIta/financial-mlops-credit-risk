from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with shared quantile bins."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_share = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_share = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def drift_status(value: float, warning: float = 0.1, critical: float = 0.25) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "stable"

