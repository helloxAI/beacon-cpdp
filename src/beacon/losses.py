"""Loss components: Focal Loss (hard pseudo-label term) and CGAD (KD term).

Focal Loss uses alpha_FL = 0.25, gamma_FL = 2.0 on the raw logit.

CGAD forms the teacher-soft probability p_t = sigmoid(logit / T) and applies
an asymmetric two-term distillation: the teacher-predicted-class term is
amplified by (1 + conf_s) ** theta while the other class term carries
(1 - conf_s), with conf_s = max(q, 1 - q); the result is scaled by T ** 2.

EGRW enters as per-sample weights w_egrw in {1, epsilon} multiplying both
loss terms; it is computed from the adaptive entropy percentile threshold.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12


def focal_loss(logit: torch.Tensor, y: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logit)
    p_t = p * y + (1.0 - p) * (1.0 - y)
    alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
    return -alpha_t * (1.0 - p_t) ** gamma * torch.log(p_t.clamp_min(EPS))


def cgad_loss(
    logit: torch.Tensor,
    q_teacher: torch.Tensor,
    theta: float = 1.2,
    temperature: float = 1.5,
) -> torch.Tensor:
    p_t = torch.sigmoid(logit / temperature)
    c_hat = (q_teacher > 0.5).float()
    p_pred = c_hat * p_t + (1.0 - c_hat) * (1.0 - p_t)
    p_other = (1.0 - c_hat) * p_t + c_hat * (1.0 - p_t)
    conf = torch.maximum(q_teacher, 1.0 - q_teacher)
    loss = -(
        (1.0 + conf) ** theta * torch.log(p_pred.clamp_min(EPS))
        + (1.0 - conf) * torch.log(p_other.clamp_min(EPS))
    )
    return loss * temperature**2


def egrw_threshold(entropies: np.ndarray, percentile: float = 60.0, low: float = 0.4, high: float = 0.8) -> float:
    if len(entropies) == 0:
        return low
    omega = float(np.percentile(entropies, percentile))
    return float(np.clip(omega, low, high))


def egrw_weights(entropies: np.ndarray, omega: float, epsilon: float = 0.05) -> np.ndarray:
    return np.where(entropies < omega, 1.0, epsilon).astype(np.float32)
