# Directional Irreversibility Index (DII)

**A Python diagnostic for directional structure in nonlinear relationships.**

Correlation is symmetric. Models of a relationship need not be.

DII asks: **after fitting the relationship in each direction, which representation leaves less dependence between its residual and its full set of inputs?** Across a prespecified set of horizons, it can describe how that asymmetry persists or fades.

Created by **Arka Prava Bandyopadhyay, PhD**.

This initial release provides a compact, inspectable implementation, synthetic examples, and numerical tests. It returns descriptive estimates. It does not implement hypothesis tests, confidence intervals, or automatic causal identification.

## Definition

For a driver $X_t$, outcome $Y_{t+h}$, and the same prespecified history $C_t$:

$$u^f_{t,h}=Y_{t+h}-E[Y_{t+h}\mid X_t,C_t],\qquad
u^b_{t,h}=X_t-E[X_t\mid Y_{t+h},C_t].$$

$$\operatorname{DII}(h)=\underbrace{\operatorname{HSIC}(u^b_{t,h},(Y_{t+h},C_t))}_{H_b(h)}-
\underbrace{\operatorname{HSIC}(u^f_{t,h},(X_t,C_t))}_{H_f(h)}.$$

HSIC is the Hilbert–Schmidt Independence Criterion, a kernel dependence measure. It is already a squared operator norm. DII does **not** square the HSIC statistic again.

| Result | Descriptive interpretation |
|---|---|
| Positive DII | Reverse representation has greater residual dependence at the chosen scales. |
| Negative DII | Forward representation has greater residual dependence at the chosen scales. |
| DII near zero | Little measured separation; both components may be small **or both may be large**. |

The sign is about model ordering, **not the sign of an economic effect**. Always inspect both components. A positive DII does not establish that the forward residual is independent.

## Install and run

From a downloaded or cloned repository:

```bash
python -m pip install .
python examples/quickstart.py
python examples/horizon_demo.py
python -m unittest discover -s tests -v
```

The package is installed from this repository; no PyPI publication is claimed.

```python
import numpy as np
from directional_irreversibility import DII

rng = np.random.default_rng(42)
x = rng.normal(size=1000)
y = x**2 + 0.5 * rng.normal(size=1000)  # synthetic data

model = DII().fit(x[:600], y[:600])
result = model.score(x[600:], y[600:])
print(result.forward_hsic, result.reverse_hsic, result.dii)
```

You can supply a scikit-learn-compatible conditional-mean learner, or use `dii_from_residuals` with your own held-out residuals and complete regressor vectors. The default RBF kernel-ridge model is a starting specification, not a universal estimator of the conditional mean.

## Where to explore DII

These are **candidate applications**, not validated results from this release.

| Area | Example question |
|---|---|
| Finance and economics | How does directional residual asymmetry vary across shock-response horizons? |
| Industrial systems | Does a sensor–output relationship leave different residual structure in opposing representations? |
| Operations | Does a demand–capacity relationship change across prespecified operating states? |
| Machine learning | Which nonlinear relationships are poorly summarized by correlation alone? |

## Horizon profiles and half-decay

```python
from directional_irreversibility import half_life

# A constructed profile used ONLY to show the interface.
summary = half_life([0, 1, 2, 3, 4], [0.020, 0.016, 0.010, 0.007, 0.005])
print(summary)
```

This computes a descriptive half-crossing of a positive DII profile relative to a prespecified reference and floor. It reports a grid bracket and flags nonmonotonicity and sign reversal. It does not estimate a half-life from one observation, turn an insignificant estimate into zero, or equate DII decay with profit decay.

For comparisons, keep outcome-window length, units, bandwidths, history, event sample, and preprocessing consistent. Fix scales on training data. Impact-inclusive cumulative returns can retain the original impact even when all later returns contain no response. See [methodology](docs/methodology.md).

## Interpretation boundaries

- **Not a causal certificate.** Causal ordering requires additional assumptions, including an appropriate independent-noise model, no relevant omitted common causes, and nonreversibility.
- **Zero does not mean no effect.** Linear Gaussian relationships can have nonzero effects with zero population DII. Pure scale dependence can produce equal, positive components.
- **Model and scale dependent.** Learner error, measurement error, omitted state variables, and transformations affect the result. The index is not a percentage, a probability, or a scale-free strength measure.
- **Separate training from evaluation.** Chronological financial data require justified gaps and purging of overlapping outcome windows. Separation alone does not justify inference.
- **No automatic significance claims.** Generated residuals and dependence across observations/horizons require appropriate joint inference. This release deliberately returns estimates only.
- **Dense kernels.** Exact calculation needs quadratic memory and computation; the default 4,000-row guard makes this explicit.

## Contribute

Useful contributions include independent benchmark replications, alternative conditional-mean learners, computational improvements, and carefully specified applications. See [CONTRIBUTING.md](CONTRIBUTING.md). Report limitations and null results along with successful examples.

## Research and papers

Read the public working paper on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7374320) and explore the [author's research website](https://arka-bandyopadhyay-research.arkab85.chatgpt.site). The software is an exploratory reference implementation, not the full inference implementation from the paper.

## Attribution and foundations

Please cite this software using [CITATION.cff](CITATION.cff), including the version/commit used. This implementation combines the DII contrast with established residual-independence and kernel-dependence ideas; it does not claim to have invented HSIC or additive-noise identification.

- Gretton et al., *A Kernel Statistical Test of Independence*, NeurIPS 20: [paper](https://proceedings.neurips.cc/paper_files/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html).
- Peters et al. (2014), *Causal Discovery with Continuous Additive Noise Models*, JMLR 15: [paper](https://jmlr.org/papers/volume15/peters14a/peters14a.pdf).

MIT licensed. Version 0.1.0.
