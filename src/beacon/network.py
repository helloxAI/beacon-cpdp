"""Adapter network: Feature Calibrator followed by Lightweight Adapter.

FeatureCalibrator: residual three-layer bottleneck d -> 2c -> 2c -> d with
BatchNorm + Tanh on the intermediate layers, c = 64; output is x + C(x).

LightweightAdapter: four-layer network d -> 2h -> h -> h -> 1, h = 128, each
hidden layer applies Linear -> BatchNorm1d -> ReLU -> Dropout(0.3); the
output is a single logit.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureCalibrator(nn.Module):
    def __init__(self, dim: int, c: int = 64):
        super().__init__()
        w = 2 * c
        self.net = nn.Sequential(
            nn.Linear(dim, w), nn.BatchNorm1d(w), nn.Tanh(),
            nn.Linear(w, w), nn.BatchNorm1d(w), nn.Tanh(),
            nn.Linear(w, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LightweightAdapter(nn.Module):
    def __init__(self, dim: int, h: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * h), nn.BatchNorm1d(2 * h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BeaconNet(nn.Module):
    """Feature Calibrator + Lightweight Adapter in series; outputs a logit."""

    def __init__(self, dim: int, h: int = 128, c: int = 64, dropout: float = 0.3):
        super().__init__()
        self.calibrator = FeatureCalibrator(dim, c=c)
        self.adapter = LightweightAdapter(dim, h=h, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.calibrator(x)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        self.eval()
        out = []
        for i in range(0, len(X), batch_size):
            out.append(torch.sigmoid(self.forward(X[i : i + batch_size])))
        return torch.cat(out)
