"""Synthetic nonlinear example. No real-world or causal performance claim."""
from dataclasses import asdict
import json
import numpy as np
from directional_irreversibility import DII

rng = np.random.default_rng(42)
x = rng.normal(size=1000)
y = x**2 + 0.5 * rng.normal(size=1000)
model = DII().fit(x[:600], y[:600])
result = model.score(x[600:], y[600:])
print("SYNTHETIC example: Y = X^2 + independent Gaussian noise")
print(json.dumps(asdict(result), indent=2))
print("Descriptive held-out estimates; no p-values or empirical claim.")
