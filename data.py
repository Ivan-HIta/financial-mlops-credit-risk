from __future__ import annotations

import numpy as np
import pandas as pd


def make_dataset(rows: int = 10_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(10.5, 0.45, rows)
    debt = income * rng.uniform(0.05, 0.75, rows)
    utilization = rng.beta(2.3, 4.0, rows)
    delinquencies = rng.poisson(0.35, rows)
    tenure = rng.integers(2, 180, rows)
    logit = -3 + 2.2 * utilization + 0.18 * delinquencies + 0.00002 * debt - 0.001 * tenure
    probability = 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "income": income.round(2),
            "debt": debt.round(2),
            "utilization": utilization.round(4),
            "delinquencies_12m": delinquencies,
            "tenure_months": tenure,
            "default": rng.binomial(1, probability),
        }
    )

