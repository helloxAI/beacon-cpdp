"""Target-domain preprocessing for adaptation and evaluation.

Protocol: stratified 80/20 split of the target project (random_state=42);
StandardScaler fitted on the 80% part only and applied to both parts;
adaptation consumes the concatenated unlabeled feature matrix (transductive);
metrics are computed on the 20% held-out part. The labels are used solely
for stratification and final evaluation, never for adapter training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TargetSplit:
    X_all: np.ndarray        # scaled features of all target samples
    y_all: np.ndarray        # labels of all target samples (evaluation only)
    test_idx: np.ndarray     # indices of the held-out 20%
    scaler: StandardScaler

    @property
    def X_test(self) -> np.ndarray:
        return self.X_all[self.test_idx]

    @property
    def y_test(self) -> np.ndarray:
        return self.y_all[self.test_idx]


def prepare_target(
    X_spec: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
) -> TargetSplit:
    X = np.nan_to_num(np.asarray(X_spec, dtype=np.float64), nan=0.0)
    y = np.asarray(y).astype(np.int64)
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X[idx_tr])
    X_all = scaler.transform(X)
    return TargetSplit(X_all=X_all, y_all=y, test_idx=np.sort(idx_te), scaler=scaler)
