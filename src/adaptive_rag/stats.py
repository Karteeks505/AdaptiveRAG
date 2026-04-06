from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci_proportions(
    values: list[float],
    n_resamples: int,
    random_seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap mean of binary/continuous [0,1] values at query level."""
    rng = np.random.default_rng(random_seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = float(sample.mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def clopper_pearson_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 1.0)
    r = binomtest(successes, trials, p=0.5, alternative="two-sided")
    ci = r.proportion_ci(confidence_level=1 - alpha, method="exact")
    return (float(ci.low), float(ci.high))
