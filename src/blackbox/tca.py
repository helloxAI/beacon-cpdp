"""Linear-kernel TCA with an explicit projection matrix.

The stored TCA artifacts keep only (dim, kernel, gamma) and the final
classifier, which does not allow out-of-sample projection; this clean
implementation learns an explicit linear map W so that arbitrary new target
samples can be embedded. W comes from the generalized eigenproblem of the
MMD objective on the concatenated source/target matrix; a logistic
regression is then fitted on the projected labeled source samples.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression


class LinearTCA:
    def __init__(self, dim: int = 30, mu: float = 1.0):
        self.dim = dim
        self.mu = mu
        self.W_: np.ndarray | None = None
        self.clf_: LogisticRegression | None = None

    def fit(self, Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray) -> "LinearTCA":
        ns, nt = Xs.shape[0], Xt.shape[0]
        d = Xs.shape[1]
        k = min(self.dim, d)
        X = np.vstack([Xs, Xt])

        # L and H are structured (block-constant / centering), so the d x d
        # scatter matrices below are formed from column sums in O(n d^2)
        # without materializing n x n matrices.
        S_s = Xs.sum(axis=0)
        S_t = Xt.sum(axis=0)
        G = X.sum(axis=0)
        SL = (
            np.outer(S_s, S_s) / (ns * ns)
            + np.outer(S_t, S_t) / (nt * nt)
            - (np.outer(S_s, S_t) + np.outer(S_t, S_s)) / (ns * nt)
        )
        SH = X.T @ X - np.outer(G, G) / (ns + nt)

        # The n x n kernel eigenproblem with a linear kernel reduces exactly
        # to a d x d pencil sandwiched by M^{1/2}, M = X^T X: the embedding
        # K @ alpha equals X @ W with W = M^{1/2} V.
        mq, ml = np.linalg.eigh(X.T @ X)
        mq = np.clip(mq, 1e-12, None)
        Mh = (np.sqrt(mq) * ml) @ ml.T
        A = Mh @ SL @ Mh + self.mu * np.eye(d)
        B = Mh @ SH @ Mh
        B = (B + B.T) / 2.0
        A = (A + A.T) / 2.0
        vals, vecs = eigh(B, A)
        order = np.argsort(vals)[::-1][:k]
        self.W_ = Mh @ vecs[:, order]

        Zs = Xs @ self.W_
        self.clf_ = LogisticRegression(solver="liblinear", random_state=42)
        self.clf_.fit(Zs, ys)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = X @ self.W_
        p = self.clf_.predict_proba(Z)
        classes = list(self.clf_.classes_)
        pos = classes.index(1) if 1 in classes else len(classes) - 1
        return p[:, pos]
