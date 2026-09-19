"""Resolution layer: (black-box pair, model name) -> probability predictor.

Two pre-trained black-box groups are available, one per evaluation regime:
  rq1: source ant-1.7,     alignment target ivy-2.0   (within PROMISE)
  rq2: source velocity-1.6, alignment target jruby-1.1 (PROMISE -> JIRA)

Every recovered model consumes 20-dim PROMISE-spec inputs. The scaler stored
in an artifact is used when its dimension matches; otherwise a
source-fitted scaler is reconstructed from the source ARFF (the stored one
then belongs to a different feature spec and cannot be applied).

TCA artifacts contain no out-of-sample projection state, so TCA is refitted
with LinearTCA on the scaled source and the scaled, spec-mapped alignment
target; everything else reuses the recovered fitted state.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data.arff_data import load_project
from src.data.feature_map import PROMISE_FEATURES
from src.blackbox import compat
from src.blackbox.tca import LinearTCA

PAIR_REGISTRY = {
    "rq1": {
        "dir": "promise-ant_1_7_promise-ivy_2_0",
        "src": "ant-1.7",
        "src_suite": "PROMISE",
        "align": "ivy-2.0",
        "align_suite": "PROMISE",
    },
    "rq2": {
        "dir": "promise-velocity_1_6_jira-jruby_1_1",
        "src": "velocity-1.6",
        "src_suite": "PROMISE",
        "align": "jruby-1.1",
        "align_suite": "JIRA",
    },
}

MODEL_NAMES = list(compat.MODEL_NAMES)

_cache: dict[tuple[str, str, str], object] = {}


def spec_features(project, spec_names: list[str] | None = None) -> np.ndarray:
    """Express a project's features in the PROMISE spec; unmapped slots NaN."""
    from src.data.feature_map import get_mapping

    spec_names = spec_names or PROMISE_FEATURES
    mapping = get_mapping(project.suite, "PROMISE")
    pos = {n: i for i, n in enumerate(project.feature_names)}
    out = np.full((project.n_samples, len(spec_names)), np.nan)
    for j, name in enumerate(spec_names):
        if project.suite == "PROMISE":
            if name in pos:
                out[:, j] = project.X[:, pos[name]]
        else:
            for tgt_feat, src_feat in mapping.items():
                if src_feat == name and tgt_feat in pos:
                    out[:, j] = project.X[:, pos[tgt_feat]]
                    break
    return out


def _source_scaler(pair: dict, data_root: str) -> StandardScaler:
    src = load_project(pair["src_suite"], pair["src"], data_root)
    scaler = StandardScaler()
    scaler.fit(np.nan_to_num(spec_features(src), nan=0.0))
    return scaler


def _fit_tca(pair: dict, scaler: StandardScaler, data_root: str) -> LinearTCA:
    src = load_project(pair["src_suite"], pair["src"], data_root)
    align = load_project(pair["align_suite"], pair["align"], data_root)
    Xs = scaler.transform(np.nan_to_num(spec_features(src), nan=0.0))
    Xt = scaler.transform(np.nan_to_num(spec_features(align), nan=0.0))
    return LinearTCA(dim=30).fit(Xs, src.y, Xt)


class _TCAPredictor:
    def __init__(self, tca: LinearTCA, scaler: StandardScaler):
        self._tca = tca
        self._scaler = scaler

    def predict_proba(self, X_spec: np.ndarray) -> np.ndarray:
        Xp = self._scaler.transform(np.nan_to_num(X_spec, nan=0.0))
        return self._tca.predict_proba(Xp)


def get_blackbox(pair_key: str, model_name: str, data_root: str = "Datasets"):
    key = (pair_key, model_name, os.path.abspath(data_root))
    if key in _cache:
        return _cache[key]

    pair = PAIR_REGISTRY[pair_key]
    path = os.path.join(
        "black_model", "baseline", "models", pair["dir"],
        f"{pair['src']}_to_{pair['align']}_{model_name}.joblib",
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    compat.install_compat()
    import joblib

    artifact = joblib.load(path)[0]
    stored = artifact._tgt_scaler
    n_spec = len(PROMISE_FEATURES)
    if stored is not None and getattr(stored, "n_features_in_", None) == n_spec:
        scaler = stored
    else:
        scaler = _source_scaler(pair, data_root)

    if model_name == "TCA":
        predictor = _TCAPredictor(_fit_tca(pair, scaler, data_root), scaler)
    else:
        predictor = compat.LoadedBlackBox(artifact, scaler)

    _cache[key] = predictor
    return predictor
