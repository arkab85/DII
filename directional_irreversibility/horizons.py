"""Descriptive threshold crossings for a prespecified signed DII profile."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HalfLifeResult:
    status: str
    half_life: float | None
    bracket: tuple[float, float] | None
    threshold: float
    nonmonotone: bool
    sign_reversal: bool
    observed_through: float


def half_life(horizons, values, *, floor=0.0, mode="sustained", minimum_signal=0.0):
    """Half-decay time of positive signed excess DII on the OBSERVED grid.

    The first horizon is the prespecified reference, not an estimated peak.
    Returns elapsed time from that reference and a bracketing interval; does
    not interpolate, extrapolate, fit an exponential, or provide inference.

    mode='sustained' requires all later observed values to stay <= threshold.
    mode='first' returns the first crossing. A reversal is flagged explicitly.
    floor and minimum_signal must be chosen externally, not tuned to results.
    'not_reached' means not reached on this grid, not an infinite half-life.
    """
    h, d = np.asarray(horizons, dtype=float), np.asarray(values, dtype=float)
    if h.ndim != 1 or d.ndim != 1 or len(h) != len(d) or len(h) < 2:
        raise ValueError("horizons and values must be equal-length vectors with at least two points")
    if not np.isfinite(h).all() or not np.isfinite(d).all() or not np.isfinite(floor):
        raise ValueError("All inputs must be finite")
    if np.any(np.diff(h) <= 0):
        raise ValueError("horizons must be strictly increasing")
    if mode not in ("first", "sustained"):
        raise ValueError("mode must be 'first' or 'sustained'")
    if not np.isfinite(minimum_signal) or minimum_signal < 0:
        raise ValueError("minimum_signal must be nonnegative and finite")
    excess = d - floor
    threshold = float(floor + excess[0] / 2)
    nonmonotone = bool(np.any(np.diff(d) > 0))
    reversal = bool(np.any(d[1:] < 0) and d[0] > 0)
    end = float(h[-1] - h[0])
    if excess[0] <= minimum_signal:
        return HalfLifeResult("no_positive_reference", None, None, threshold, nonmonotone, reversal, end)
    below = d <= threshold
    for j in range(1, len(h)):
        if below[j] and (mode == "first" or below[j:].all()):
            return HalfLifeResult("crossed", float(h[j] - h[0]),
                                  (float(h[j-1] - h[0]), float(h[j] - h[0])),
                                  threshold, nonmonotone, reversal, end)
    return HalfLifeResult("not_reached", None, None, threshold, nonmonotone, reversal, end)
