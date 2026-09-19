"""Self-training loop over K rounds.

The teacher probabilities start from the black-box outputs and are updated
by an exponential moving average after each round (gamma = 0.6). Each round
regenerates the pseudo-label set with the decayed confidence threshold,
derives the EGRW weights from the entropy distribution of the admitted set,
and fine-tunes the same network on that set with Adam (lr 1e-3, weight
decay 1e-5), gradient clipping at 1.0, batch size 32, at most 200 epochs,
ReduceLROnPlateau(factor 0.5, patience 5) and early stopping (patience 20)
on the epoch-mean total loss; the best weights are restored when early
stopping fires. The total loss is 0.85 * Focal-CE + 0.15 * KD, both terms
reweighted per sample by the class-balance weight (CE only) and the EGRW
weight (both terms).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.beacon.losses import cgad_loss, egrw_threshold, egrw_weights, focal_loss
from src.beacon.network import BeaconNet
from src.beacon.pseudolabel import (
    generate_pseudo_labels,
    normalized_entropy,
    round_threshold,
)


@dataclass
class BeaconConfig:
    k_rounds: int = 5
    tau0: float = 0.6
    tau_decay: float = 0.05
    tau_min: float = 0.5
    rho: float = 0.2
    weight_cap: float = 3.0
    majority_ratio: float = 3.0
    theta: float = 1.2
    temperature: float = 1.5
    gamma_ema: float = 0.6
    epsilon: float = 0.05
    egrw_enabled: bool = True
    kd_mode: str = "cgad"  # "cgad" | "soft"
    lambda_ce: float = 0.85
    lambda_kd: float = 0.15
    alpha_fl: float = 0.25
    gamma_fl: float = 2.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 200
    es_patience: int = 20
    sched_factor: float = 0.5
    sched_patience: int = 5
    percentile: float = 60.0
    omega_low: float = 0.4
    omega_high: float = 0.8
    hidden: int = 128
    calib: int = 64
    dropout: float = 0.3
    seed: int = 42
    extra: dict = field(default_factory=dict)


def soft_kd_loss(logit: torch.Tensor, q_teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logit / temperature, q_teacher, reduction="none"
    ) * temperature**2


class BeaconTrainer:
    def __init__(self, config: BeaconConfig | None = None):
        self.cfg = config or BeaconConfig()
        self.net_: BeaconNet | None = None
        self.pema_: np.ndarray | None = None
        self.history_: list[dict] = []

    def fit(self, X: np.ndarray, p_black: np.ndarray) -> "BeaconTrainer":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        X = np.asarray(X, dtype=np.float32)
        p_black = np.asarray(p_black, dtype=np.float64).reshape(-1)
        d = X.shape[1]
        self.net_ = BeaconNet(d, h=cfg.hidden, c=cfg.calib, dropout=cfg.dropout)
        self.history_ = []

        pema = p_black.copy()
        X_t = torch.as_tensor(X)
        for k in range(cfg.k_rounds):
            q = pema
            H = normalized_entropy(q)
            tau_k = round_threshold(cfg.tau0, k, cfg.tau_decay, cfg.tau_min)
            idx, y_tilde, w = generate_pseudo_labels(
                q, tau_k, rho=cfg.rho, weight_cap=cfg.weight_cap, majority_ratio=cfg.majority_ratio
            )
            if len(idx) < 2:  # BatchNorm needs at least two samples
                break
            if cfg.egrw_enabled:
                omega = egrw_threshold(H[idx], cfg.percentile, cfg.omega_low, cfg.omega_high)
                w_e = egrw_weights(H[idx], omega, cfg.epsilon)
            else:
                omega = float("nan")
                w_e = np.ones(len(idx), dtype=np.float32)

            bs = min(cfg.batch_size, len(idx))
            stats = self._train_round(X_t[idx], y_tilde, w, w_e, q[idx].astype(np.float32), rng, bs)
            stats.update(round=k, tau=tau_k, omega=omega, n_pseudo=len(idx))
            self.history_.append(stats)

            p_model = self.predict_proba(X)
            pema = cfg.gamma_ema * pema + (1.0 - cfg.gamma_ema) * p_model

        self.pema_ = pema
        return self

    def _train_round(
        self,
        X: torch.Tensor,
        y: np.ndarray,
        w: np.ndarray,
        w_e: np.ndarray,
        q: np.ndarray,
        rng: np.random.Generator,
        bs: int,
    ) -> dict:
        cfg = self.cfg
        net = self.net_
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=cfg.sched_factor, patience=cfg.sched_patience
        )
        y_t = torch.as_tensor(y)
        w_t = torch.as_tensor(w)
        we_t = torch.as_tensor(w_e)
        q_t = torch.as_tensor(q)

        n = len(X)
        best_loss = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        bad_epochs = 0
        epochs_run = 0
        for _ in range(cfg.max_epochs):
            net.train()
            perm = rng.permutation(n)
            total, nb = 0.0, 0
            for start in range(0, n, bs):
                b = perm[start : start + bs]
                if len(b) < 2:  # BatchNorm needs at least two samples
                    continue
                xb = X[b]
                logit = net(xb)
                fl = focal_loss(logit, y_t[b], cfg.alpha_fl, cfg.gamma_fl)
                l_ce = (fl * w_t[b] * we_t[b]).mean()
                if cfg.kd_mode == "cgad":
                    kd_vec = cgad_loss(logit, q_t[b], cfg.theta, cfg.temperature)
                else:
                    kd_vec = soft_kd_loss(logit, q_t[b], cfg.temperature)
                l_kd = (kd_vec * we_t[b]).mean()
                loss = cfg.lambda_ce * l_ce + cfg.lambda_kd * l_kd
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += float(loss.item())
                nb += 1
            epochs_run += 1
            mean_loss = total / max(nb, 1)
            sched.step(mean_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_state = {k2: v.detach().clone() for k2, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.es_patience:
                    net.load_state_dict(best_state)
                    break
        return {"epochs": epochs_run, "best_loss": best_loss}

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return self.net_.predict_proba(X_t, batch_size=batch_size).cpu().numpy().astype(np.float64)
