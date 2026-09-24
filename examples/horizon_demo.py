"""Synthetic horizon illustration; this is not a financial event study."""
from dataclasses import asdict
import json
import numpy as np
from directional_irreversibility import DII, half_life

rng = np.random.default_rng(18)
horizons = np.arange(7)
x = rng.normal(size=1200)
signal = x**2 - 1.0
y = np.column_stack([np.exp(-h / 2.0) * signal + 0.8 * rng.normal(size=len(x)) for h in horizons])
split = 800
# One outcome scale is fixed for ALL horizons, using training data only.
common_x_scale = x[:split].std()
common_y_scale = y[:split].std()
profile = []
for j, horizon in enumerate(horizons):
    model = DII(x_scale=common_x_scale, y_scale=common_y_scale)
    model.fit(x[:split], y[:split, j])
    result = model.score(x[split:], y[split:, j])
    profile.append(result.dii)
    print(f"h={horizon}, Hf={result.forward_hsic:.6f}, Hb={result.reverse_hsic:.6f}, DII={result.dii:.6f}")
print("SYNTHETIC descriptive half-crossing; no confidence interval:")
print(json.dumps(asdict(half_life(horizons, profile)), indent=2))
print("DII half-decay is not the known mean-loading half-life of 2*log(2).")
