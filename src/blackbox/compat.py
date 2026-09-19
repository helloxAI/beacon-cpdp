"""Compatibility loading of the pre-trained black-box joblib artifacts.

The artifacts were pickled with training-time modules that are not part of
this repository. A meta-path finder fabricates placeholder classes for those
missing modules so the object graph (fitted sklearn models, torch
sub-modules, numpy arrays, scalers) can be reconstructed. On top of the
recovered state, clean inference wrappers with an explicit forward
definition are built; no training-time code is executed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

import joblib
import numpy as np
import torch

_REAL_PRELOADS = (
    "numpy",
    "scipy.optimize",
    "scipy.sparse",
    "scipy.stats",
    "sklearn",
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.naive_bayes",
    "sklearn.svm",
    "sklearn.neural_network",
    "sklearn.cross_decomposition",
    "sklearn.feature_selection",
    "sklearn.tree",
    "sklearn.impute",
    "torch",
)


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _FakeModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (_Placeholder,), {})
        setattr(self, name, cls)
        return cls


_FAKE_PREFIXES = ("baseline_predictor", "baselines_sota")


class _FakeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        if top not in _FAKE_PREFIXES:
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        return _FakeModule(spec.name)

    def exec_module(self, module):
        pass


_installed = False


def install_compat() -> None:
    global _installed
    if _installed:
        return
    import importlib

    for name in _REAL_PRELOADS:
        importlib.import_module(name)
    sys.meta_path.append(_FakeFinder())
    _installed = True


# ---------------------------------------------------------------------------
# torch inference wrappers built from recovered sub-modules
# ---------------------------------------------------------------------------


def _submods(obj):
    return obj.__dict__.get("_modules", {})


def _params(obj):
    return obj.__dict__.get("_parameters", {})


class _TorchWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._out_dim = 1

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).view(-1)


class MLPDropoutNet(_TorchWrapper):
    def __init__(self, seq):
        super().__init__()
        self.net = seq

    def forward(self, x):
        return self.net(x)


class DAENet(_TorchWrapper):
    def __init__(self, encoder, decoder, clf):
        super().__init__()
        self.encoder, self.decoder, self.clf = encoder, decoder, clf

    def forward(self, x):
        return self.clf(self.encoder(x))


class DBNNet(_TorchWrapper):
    def __init__(self, layer1, layer2, output):
        super().__init__()
        self.layer1, self.layer2, self.output = layer1, layer2, output

    def forward(self, x):
        return self.output(self.layer2(self.layer1(x)))


class CNN1DNet(_TorchWrapper):
    def __init__(self, conv1, relu, pool, fc):
        super().__init__()
        self.conv1, self.relu, self.pool, self.fc = conv1, relu, pool, fc

    def forward(self, x):
        h = x.unsqueeze(1)
        h = self.pool(self.relu(self.conv1(h)))
        h = h.flatten(1)
        return self.fc(h)


class FTTransformerNet(_TorchWrapper):
    def __init__(self, feature_embeddings, cls_token, transformer, head):
        super().__init__()
        self.feature_embeddings = feature_embeddings
        self.cls_token = cls_token
        self.transformer = transformer
        self.head = head

    def forward(self, x):
        tokens = torch.stack(
            [emb(x[:, i : i + 1]) for i, emb in enumerate(self.feature_embeddings)],
            dim=1,
        )
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        return self.head(out[:, 0])


class SupConNet(_TorchWrapper):
    def __init__(self, encoder, head, classifier):
        super().__init__()
        self.encoder, self.head, self.classifier = encoder, head, classifier

    def forward(self, x):
        return self.classifier(self.encoder(x))


class DANNNet(_TorchWrapper):
    def __init__(self, feature, class_classifier, domain_classifier):
        super().__init__()
        self.feature = feature
        self.class_classifier = class_classifier
        self.domain_classifier = domain_classifier

    def forward(self, x):
        return self.class_classifier(self.feature(x))


class DeepOTNet(_TorchWrapper):
    def __init__(self, feature, classifier):
        super().__init__()
        self.feature, self.classifier = feature, classifier

    def forward(self, x):
        return self.classifier(self.feature(x))


def _build_torch_net(model_name: str, m) -> _TorchWrapper:
    if model_name == "DeepOT":
        net = DeepOTNet(m.feature, m.classifier)
        net.eval()
        return net
    net_obj = m.model
    mods = _submods(net_obj)
    if model_name == "MLP_Dropout":
        net = MLPDropoutNet(net_obj)
    elif model_name == "DAE":
        net = DAENet(mods["encoder"], mods["decoder"], mods["clf"])
    elif model_name == "DBN":
        net = DBNNet(mods["layer1"], mods["layer2"], mods["output"])
    elif model_name == "CNN_SDP":
        net = CNN1DNet(mods["conv1"], mods["relu"], mods["pool"], mods["fc"])
    elif model_name == "FT_Transformer":
        net = FTTransformerNet(
            mods["feature_embeddings"],
            _params(net_obj)["cls_token"],
            mods["transformer"],
            mods["head"],
        )
    elif model_name == "SupCon_DP":
        net = SupConNet(mods["encoder"], mods["head"], mods["classifier"])
    elif model_name == "DANN":
        net = DANNNet(mods["feature"], mods["class_classifier"], mods["domain_classifier"])
    else:
        raise ValueError(f"no torch wrapper for {model_name}")
    net.eval()
    return net


# ---------------------------------------------------------------------------
# sklearn-side predictors
# ---------------------------------------------------------------------------


def _positive_proba(clf, X: np.ndarray) -> np.ndarray:
    p = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))
    pos = classes.index(1) if 1 in classes else len(classes) - 1
    return p[:, pos]


def _tradaboost_proba(models, betas, X: np.ndarray) -> np.ndarray:
    n = len(models)
    half = n // 2
    used = range(half, n)
    weights, probs = [], []
    for i in used:
        beta = float(np.clip(betas[i], 1e-6, 1.0))
        weights.append(np.log(1.0 / beta))
        probs.append(_positive_proba(models[i], X))
    weights = np.asarray(weights)
    probs = np.asarray(probs)
    return np.average(probs, axis=0, weights=weights)


# ---------------------------------------------------------------------------
# public loader
# ---------------------------------------------------------------------------

TORCH_MODELS = {
    "MLP_Dropout", "DAE", "DBN", "CNN_SDP", "FT_Transformer",
    "SupCon_DP", "DANN", "DeepOT",
}

MODEL_NAMES = [
    "TNB", "HDP_KS", "CPDP_IFS", "CCA_Plus", "TCA", "CORAL",
    "SubspaceAlignment", "TrAdaBoost", "MLP_Zero", "MLP_Mean",
    "MLP_Dropout", "DAE", "DBN", "FT_Transformer", "CNN_SDP",
    "SupCon_DP", "DANN", "DeepOT",
]


class LoadedBlackBox:
    """Recovered black-box state with a uniform probability API."""

    def __init__(self, artifact, source_scaler):
        self.model_name = artifact.model_name
        self.src_name = artifact.src_name
        self.tgt_name = artifact.tgt_name
        self._m = artifact._model
        self._scaler = source_scaler
        self._torch_net = None
        if self.model_name in TORCH_MODELS:
            self._torch_net = _build_torch_net(self.model_name, self._m)

    # -- input preparation -------------------------------------------------
    def prepare(self, X_spec: np.ndarray) -> np.ndarray:
        """X_spec: raw features already expressed in the source feature spec;
        unmapped slots are NaN."""
        X = np.asarray(X_spec, dtype=np.float64)
        if self.model_name == "MLP_Mean":
            Xs = self._scaler.transform(X)
            Xs = self._m.imputer.transform(Xs)
            return np.nan_to_num(Xs, nan=0.0)
        X = np.nan_to_num(X, nan=0.0)
        return self._scaler.transform(X)

    # -- inference ----------------------------------------------------------
    def predict_proba_prepared(self, Xp: np.ndarray) -> np.ndarray:
        name, m = self.model_name, self._m
        if self._torch_net is not None:
            x = torch.as_tensor(Xp, dtype=torch.float32)
            return self._torch_net.prob(x).cpu().numpy()
        if name == "HDP_KS":
            return _positive_proba(m.clf, Xp[:, list(m.selected_)])
        if name == "CPDP_IFS":
            return _positive_proba(m.clf, m.selector.transform(Xp))
        if name == "CCA_Plus":
            return _positive_proba(m.clf, m.cca.transform(Xp))
        if name == "CORAL":
            return _positive_proba(m.clf, Xp @ m.A)
        if name == "SubspaceAlignment":
            return _positive_proba(m.clf, Xp @ m._X_t_pca)
        if name == "TrAdaBoost":
            return _tradaboost_proba(m.models, m.betas, Xp)
        return _positive_proba(m.clf, Xp)

    def predict_proba(self, X_spec: np.ndarray) -> np.ndarray:
        return self.predict_proba_prepared(self.prepare(X_spec))


def load_blackbox(path: str, source_scaler) -> LoadedBlackBox:
    install_compat()
    artifact = joblib.load(path)[0]
    return LoadedBlackBox(artifact, source_scaler)
