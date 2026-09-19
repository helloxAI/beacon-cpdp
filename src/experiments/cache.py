"""Disk caches shared by all experiment runs.

Two caches keep the expensive, deterministic stages out of the parallel
adaptation stage:

  * target feature cache  -- the target project's native features scaled by
    the train-fitted scaler (adapter input, d_t-dimensional), the
    spec-mapped features (black-box query input, 20-dimensional), labels
    and the held-out indices of one target project;
  * black-box output cache -- the frozen probabilities of one black box on
    all samples of one target project.

Both are keyed by content-defining names only, so repeated runs, ablation
variants and parameter sweeps reuse them.
"""

from __future__ import annotations

import os

import numpy as np

from src.blackbox.api import get_blackbox, spec_features
from src.data.arff_data import load_project
from src.data.preprocessing import prepare_target

CACHE_DIR = os.path.join("results", "cache")


def _path(*parts: str) -> str:
    path = os.path.join(CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def feature_cache_path(suite: str, project: str) -> str:
    return _path("features", f"{suite}__{project}.npz")


def pblack_cache_path(pair_key: str, model_name: str, suite: str, project: str) -> str:
    return _path("pblack", f"{pair_key}__{model_name}__{suite}__{project}.npy")


def ensure_features(suite: str, project: str, data_root: str = "Datasets", seed: int = 42) -> str:
    path = feature_cache_path(suite, project)
    if os.path.isfile(path):
        return path
    data = load_project(suite, project, data_root)
    X_spec = spec_features(data)
    split = prepare_target(data.X, data.y, seed=seed)
    np.savez_compressed(
        path,
        X_all=split.X_all,
        y_all=split.y_all,
        test_idx=split.test_idx,
        X_spec=X_spec,
    )
    return path


def load_features(suite: str, project: str):
    path = feature_cache_path(suite, project)
    z = np.load(path)
    return z["X_all"], z["y_all"], z["test_idx"], z["X_spec"]


def ensure_pblack(
    pair_key: str, model_name: str, suite: str, project: str, data_root: str = "Datasets"
) -> str:
    path = pblack_cache_path(pair_key, model_name, suite, project)
    if os.path.isfile(path):
        return path
    _, _, _, X_spec = load_features(suite, project)
    box = get_blackbox(pair_key, model_name, data_root)
    p = box.predict_proba(X_spec)
    np.save(path, p.astype(np.float64))
    return path


def load_pblack(pair_key: str, model_name: str, suite: str, project: str) -> np.ndarray:
    return np.load(pblack_cache_path(pair_key, model_name, suite, project))
