"""Uncertainty estimation and adaptive pseudo-label generation.

Round-k teacher probabilities q give, per sample, the normalized binary
entropy H~ in [0, 1] and the boundary-distance confidence c = 2|q - 0.5|.
The confidence threshold decays linearly over rounds, tau_k = max(tau0 -
0.05 k, tau_min), applied in halved form (tau_k / 2) as the lower bound on c
for both teacher-positive and teacher-negative candidates.

Class-balance constraint (labels are never used):
  * positives are topped up to N_min+ = max(10, floor(N_t rho / 2)) first
    from high-confidence teacher-positive candidates, then from the most
    positive-leaning (highest q) still-unselected samples;
  * negatives are symmetrically topped up to N_min- =
    max(10, floor(N_t (1 - rho) / 2)) from the still-unselected samples,
    taking the most negative-leaning (lowest q) first, so that forced
    negatives stay as close to the decision boundary as possible and carry
    only a small confidence weight;
  * when one side cannot reach its minimum bound because every sample is
    already admitted on the other side (a collapsed teacher), the needed
    number of boundary-nearest members is moved across from the other
    class, so that a training round never degenerates to a single class;
  * when one class exceeds majority_ratio times the other, the majority
    side is truncated by descending confidence;
  * sample weight w_j = c_j * min(r- / r+, weight_cap) for positives and
    w_j = c_j for negatives, with r+ / r- the admitted class proportions.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalized_entropy(q: np.ndarray) -> np.ndarray:
    q = np.clip(q, EPS, 1.0 - EPS)
    return (-(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))) / np.log(2.0)


def confidence(q: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(q - 0.5)


def round_threshold(tau0: float, k: int, decay: float, tau_min: float) -> float:
    return max(tau0 - k * decay, tau_min)


def generate_pseudo_labels(
    q: np.ndarray,
    tau_k: float,
    rho: float = 0.2,
    weight_cap: float = 3.0,
    majority_ratio: float = 3.0,
):
    """Returns (indices, y_tilde, weights) of the admitted pseudo-label set."""
    n = len(q)
    c = confidence(q)
    bound = tau_k / 2.0

    pos = [j for j in range(n) if q[j] > 0.5 and c[j] > bound]
    neg = [j for j in range(n) if q[j] <= 0.5 and c[j] > bound]
    in_pos = np.zeros(n, dtype=bool)
    in_neg = np.zeros(n, dtype=bool)
    in_pos[pos] = True
    in_neg[neg] = True

    n_min_pos = max(10, int(np.floor(n * rho / 2.0)))
    n_min_neg = max(10, int(np.floor(n * (1.0 - rho) / 2.0)))

    if len(pos) < n_min_pos:
        cand = [j for j in np.argsort(-c) if not in_pos[j] and q[j] > 0.5]
        for j in cand:
            if len(pos) >= n_min_pos:
                break
            pos.append(int(j))
            in_pos[j] = True
        if len(pos) < n_min_pos:
            cand = [j for j in np.argsort(-q) if not in_pos[j] and not in_neg[j]]
            for j in cand:
                if len(pos) >= n_min_pos:
                    break
                pos.append(int(j))
                in_pos[j] = True

    if len(neg) < n_min_neg:
        cand = [j for j in np.argsort(q) if not in_pos[j] and not in_neg[j]]
        for j in cand:
            if len(neg) >= n_min_neg:
                break
            neg.append(int(j))
            in_neg[j] = True

    # Collapsed-teacher rescue: if one class cannot reach its bound because
    # all samples were already admitted on the other side, move the needed
    # number of boundary-nearest members across.
    if len(neg) < n_min_neg and len(pos) > n_min_pos:
        need = min(n_min_neg - len(neg), len(pos) - n_min_pos)
        for j in sorted(pos, key=lambda j: c[j])[:need]:
            pos.remove(j)
            in_pos[j] = False
            neg.append(j)
            in_neg[j] = True
    if len(pos) < n_min_pos and len(neg) > n_min_neg:
        need = min(n_min_pos - len(pos), len(neg) - n_min_neg)
        for j in sorted(neg, key=lambda j: c[j])[:need]:
            neg.remove(j)
            in_neg[j] = False
            pos.append(j)
            in_pos[j] = True

    pos = sorted(pos, key=lambda j: -c[j])
    neg = sorted(neg, key=lambda j: -c[j])
    if len(pos) > majority_ratio * max(len(neg), 1):
        pos = pos[: int(majority_ratio * max(len(neg), 1))]
    elif len(neg) > majority_ratio * max(len(pos), 1):
        neg = neg[: int(majority_ratio * max(len(pos), 1))]

    n_pos, n_neg = len(pos), len(neg)
    total = max(n_pos + n_neg, 1)
    r_pos, r_neg = n_pos / total, n_neg / total
    pos_factor = min(r_neg / max(r_pos, EPS), weight_cap)

    idx = np.asarray(pos + neg, dtype=np.int64)
    y_tilde = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)
    w = np.concatenate(
        [c[pos] * pos_factor, c[neg] * 1.0]
    ).astype(np.float32)
    return idx, y_tilde, w
