"""Fixed-kernel HSIC and held-out directional residual comparisons.

This module supplies descriptive estimates, not p-values or causal conclusions.
HSIC is already a squared operator norm; do not square it again.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.kernel_ridge import KernelRidge


def _matrix(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty vector/matrix with at least two rows")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return a


def _vector(a, name):
    a = _matrix(a, name)
    if a.shape[1] != 1:
        raise ValueError(f"{name} must contain one scalar variable")
    return a[:, 0]


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _rbf(a, bandwidth):
    scaled = a / _positive(bandwidth, "bandwidth")
    sq = np.sum(scaled * scaled, axis=1)
    distance = np.maximum(sq[:, None] + sq[None, :] - 2 * scaled @ scaled.T, 0)
    return np.exp(-distance / 2)


def _center(k):
    return k - k.mean(axis=0)[None, :] - k.mean(axis=1)[:, None] + k.mean()


def hsic(a, b, *, bandwidth_a=1.0, bandwidth_b=1.0, max_rows=4000):
    """Biased Gaussian-kernel V-statistic: sum(K_centered * L_centered) / n**2.

    Bandwidths are explicit and never estimated from the evaluation sample.
    Dense kernels need O(n^2) memory/time. max_rows is an explicit memory guard.
    No independence test or finite-sample significance threshold is implied.
    """
    a, b = _matrix(a, "a"), _matrix(b, "b")
    if len(a) != len(b):
        raise ValueError("a and b must have matching rows")
    if len(a) > max_rows:
        raise ValueError("Dense HSIC exceeds max_rows; deliberately choose a larger budget or a smaller evaluation set")
    k, l = _center(_rbf(a, bandwidth_a)), _center(_rbf(b, bandwidth_b))
    # A PSD-kernel V-statistic is nonnegative; clip only round-off below zero.
    return float(max(0.0, np.sum(k * l) / len(a) ** 2))


@dataclass(frozen=True)
class DIIResult:
    forward_hsic: float
    reverse_hsic: float
    dii: float
    n: int


def dii_from_residuals(forward_residual, forward_regressors,
                       reverse_residual, reverse_regressors, *,
                       x_bandwidth=1.0, y_bandwidth=1.0,
                       regressor_bandwidth=1.0, max_rows=4000):
    """Compute H_b - H_f using both COMPLETE regressor vectors.

    Forward residual is in Y units; reverse residual is in X units. Caller must
    supply genuinely held-out residuals and fixed, justified transformations.
    With history present, regressors include (X, C) and (Y, C), respectively.
    Residual bandwidths are variable-specific; the full regressor bandwidth is
    isotropic in the supplied transformed coordinates.
    """
    uf = _vector(forward_residual, "forward_residual")
    ub = _vector(reverse_residual, "reverse_residual")
    zf, zb = _matrix(forward_regressors, "forward_regressors"), _matrix(reverse_regressors, "reverse_regressors")
    if len({len(uf), len(ub), len(zf), len(zb)}) != 1:
        raise ValueError("All four inputs must have matching rows")
    hf = hsic(uf, zf, bandwidth_a=y_bandwidth, bandwidth_b=regressor_bandwidth, max_rows=max_rows)
    hb = hsic(ub, zb, bandwidth_a=x_bandwidth, bandwidth_b=regressor_bandwidth, max_rows=max_rows)
    return DIIResult(hf, hb, hb - hf, len(uf))


class DII:
    """Fit two conditional-mean learners, then score a separate evaluation set.

    The default is an RBF kernel-ridge learner. A regressor with sklearn's
    fit/predict interface can be supplied. Finite-sample learner error can
    create or erase asymmetry. This is not an automatic causal-discovery API.

    Standardization is learned from TRAINING variables (not residuals). For
    horizon comparisons, reuse common positive x_scale/y_scale and the same
    history/training calendar, rather than scaling each horizon separately.
    """
    def __init__(self, regressor=None, *, bandwidth=1.0, x_scale=None,
                 y_scale=None, max_rows=4000):
        self.regressor = regressor
        self.bandwidth = _positive(bandwidth, "bandwidth")
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.max_rows = max_rows

    def _history(self, c, n, fitting=False):
        if c is None:
            c = np.empty((n, 0))
        else:
            c = _matrix(c, "history")
            if len(c) != n:
                raise ValueError("history must match the number of rows")
        if fitting:
            self.c_dim_ = c.shape[1]
            self.c_mean_ = c.mean(axis=0)
            self.c_scale_ = c.std(axis=0)
            self.c_scale_[self.c_scale_ == 0] = 1.0
        elif c.shape[1] != self.c_dim_:
            raise ValueError("history must have the same columns as during fit")
        return (c - self.c_mean_) / self.c_scale_

    def fit(self, x, y, history=None):
        x, y = _vector(x, "x"), _vector(y, "y")
        if len(x) != len(y):
            raise ValueError("x and y must have matching rows")
        if len(x) > self.max_rows:
            raise ValueError("Training exceeds max_rows; dense default learner requires an explicit memory budget")
        self.x_mean_, self.y_mean_ = x.mean(), y.mean()
        self.x_scale_ = _positive(x.std() if self.x_scale is None else self.x_scale, "x_scale")
        self.y_scale_ = _positive(y.std() if self.y_scale is None else self.y_scale, "y_scale")
        c = self._history(history, len(x), fitting=True)
        xs, ys = (x - self.x_mean_) / self.x_scale_, (y - self.y_mean_) / self.y_scale_
        zf, zb = np.column_stack((xs, c)), np.column_stack((ys, c))
        learner = self.regressor if self.regressor is not None else KernelRidge(alpha=1.0, kernel="rbf", gamma=0.5)
        self.forward_model_ = clone(learner).fit(zf, ys)
        self.reverse_model_ = clone(learner).fit(zb, xs)
        return self

    def score(self, x, y, history=None):
        """Return descriptive HSIC components and signed DII on held-out rows."""
        if not hasattr(self, "forward_model_"):
            raise ValueError("Call fit before score")
        x, y = _vector(x, "x"), _vector(y, "y")
        if len(x) != len(y):
            raise ValueError("x and y must have matching rows")
        c = self._history(history, len(x))
        xs, ys = (x - self.x_mean_) / self.x_scale_, (y - self.y_mean_) / self.y_scale_
        zf, zb = np.column_stack((xs, c)), np.column_stack((ys, c))
        uf = ys - self.forward_model_.predict(zf)
        ub = xs - self.reverse_model_.predict(zb)
        return dii_from_residuals(uf, zf, ub, zb, x_bandwidth=self.bandwidth,
                                  y_bandwidth=self.bandwidth,
                                  regressor_bandwidth=self.bandwidth,
                                  max_rows=self.max_rows)
