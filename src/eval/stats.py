"""Significance and effect-size statistics.

Wilcoxon signed-rank compares per-target metric vectors of two methods;
Holm-Bonferroni adjusts a family of p-values; Cliff's delta reports the
dominance effect size with the usual thresholds (negligible < 0.147 <=
small < 0.330 <= medium < 0.474 <= large).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x - y
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) < 2:
        return 1.0
    try:
        return float(wilcoxon(diff, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm_bonferroni(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float("nan")
    greater = sum(np.searchsorted(y, v, side="left") for v in x)  # y_j < x_i
    less = sum(n_y - np.searchsorted(y, v, side="right") for v in x)  # y_j > x_i
    return float((greater - less) / (n_x * n_y))


def cliffs_magnitude(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.330:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"
