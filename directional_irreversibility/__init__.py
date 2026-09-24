"""Directional Irreversibility Index: exploratory reference implementation."""
from .core import DII, DIIResult, dii_from_residuals, hsic
from .horizons import HalfLifeResult, half_life

__version__ = "0.1.0"
__all__ = ["DII", "DIIResult", "dii_from_residuals", "hsic", "half_life", "HalfLifeResult"]
