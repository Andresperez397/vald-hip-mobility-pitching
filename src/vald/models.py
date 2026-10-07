"""Cross-validated ridge models, out-of-fold R^2, a pitcher bootstrap, and a permutation null.

All preprocessing (median imputation, scaling) and the ridge penalty are fit on the training fold only.
There is one row per pitcher, so ordinary K-fold already holds out whole pitchers.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 4, 25)
WINSOR = (2.5, 97.5)  # percentiles, estimated on the training fold only


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its training-fold 2.5th and 97.5th percentiles. Needed because a few static tests are
    percentage asymmetries that blow up when the denominator is near zero (one value of 619% against a rest of
    the sample at 43% or less), and one such point dominates a standardized ridge fit."""

    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanpercentile(X, WINSOR[0], axis=0), np.nanpercentile(X, WINSOR[1], axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


def _ridge():
    return make_pipeline(
        Winsorizer(), SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS, alpha_per_target=True)
    )


def oof_predictions(X: np.ndarray, Y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], ridge: bool = True):
    """Out-of-fold predictions for one or many targets (columns of Y). With `ridge=False` the prediction is the
    training-fold mean of each target (the no-information benchmark). Targets with missing values are fit on the
    training rows where they are observed. Returns (predictions, baseline mean predictions)."""
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    pred = np.full(Y.shape, np.nan)
    mean_pred = np.full(Y.shape, np.nan)
    for tr, te in folds:
        ok = ~np.isnan(Y[tr]).any(axis=1)  # rows with every target observed
        tr_ok = tr[ok]
        mean_pred[te] = np.nanmean(Y[tr], axis=0)
        if ridge:
            pred[te] = _ridge().fit(X[tr_ok], Y[tr_ok]).predict(X[te]).reshape(len(te), -1)
        else:
            pred[te] = mean_pred[te]
    return pred, mean_pred


def make_folds(n: int, k: int, repeats: int, seed: int = 2026):
    return [list(KFold(k, shuffle=True, random_state=seed + r).split(np.arange(n))) for r in range(repeats)]


def r2(y: np.ndarray, pred: np.ndarray, mean_pred: np.ndarray) -> np.ndarray:
    """Out-of-fold R^2 against the training-fold mean (negative when the model is worse than guessing the mean).
    Works column by column; rows with a missing target are ignored."""
    ok = ~np.isnan(y)
    if y.ndim == 1:
        return 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum((y[ok] - mean_pred[ok]) ** 2)
    sse = np.nansum((y - pred) ** 2, axis=0)
    sst = np.nansum((y - mean_pred) ** 2, axis=0)
    return 1 - sse / sst


def cv_r2(X: np.ndarray, y: np.ndarray, folds_per_repeat) -> tuple[float, np.ndarray]:
    """Mean out-of-fold R^2 over repeats, and the first repeat's predictions (kept for the bootstrap)."""
    vals, first = [], None
    for folds in folds_per_repeat:
        p, m = oof_predictions(X, y[:, None], folds)
        vals.append(float(r2(y[:, None], p, m)[0]))
        if first is None:
            first = (p[:, 0], m[:, 0])
    return float(np.mean(vals)), first


def bootstrap_gain(y, p_full, p_base, m, reps: int = 2000, seed: int = 2026):
    """95% interval for the R^2 gain of the full model over the baseline model, resampling pitchers."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ok = ~np.isnan(y)
    idx_all = np.flatnonzero(ok)
    gains = np.empty(reps)
    for b in range(reps):
        i = rng.choice(idx_all, len(idx_all), replace=True)
        yb = y[i]
        sst = np.sum((yb - m[i]) ** 2)
        gains[b] = (np.sum((yb - p_base[i]) ** 2) - np.sum((yb - p_full[i]) ** 2)) / sst
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5)), n


def target_gain(Xb: np.ndarray, Xf: np.ndarray, Y: np.ndarray, folds_per_repeat) -> np.ndarray:
    """Out-of-fold R^2 gain of (baseline + static tests) over (baseline), for every target column, averaged over
    repeats."""
    gains = []
    for folds in folds_per_repeat:
        pb, m = oof_predictions(Xb, Y, folds)
        pf, _ = oof_predictions(Xf, Y, folds)
        gains.append(r2(Y, pf, m) - r2(Y, pb, m))
    return np.mean(gains, axis=0)


def permutation_null(Xb, V, Y, folds_per_repeat, perms: int = 200, seed: int = 2026) -> np.ndarray:
    """Largest gain over all targets when the static tests are shuffled across pitchers (baseline columns stay put).
    The observed top gain must beat this null, not zero: with hundreds of targets the best one is always positive."""
    rng = np.random.default_rng(seed)
    best = np.empty(perms)
    for b in range(perms):
        Vp = V[rng.permutation(len(V))]
        best[b] = target_gain(Xb, np.hstack([Xb, Vp]), Y, folds_per_repeat).max()
    return best
