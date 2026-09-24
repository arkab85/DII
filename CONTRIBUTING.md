# Contributing

Open an issue describing the question, data-generating process, expected behavior and a minimal reproducible example. Use synthetic or shareable public data; do not include confidential data, credentials or unpublished third-party material.

For a pull request:

1. State what changes and why.
2. Add a meaningful numerical or statistical check where appropriate.
3. Run `python -m unittest discover -s tests -v` and the example scripts.
4. Document changes to the estimand, normalization, kernel or inference assumptions.

Especially useful: failure cases, reproducible null examples, benchmarks against simpler diagnostics, and computational improvements that preserve the target. Do not attach empirical claims to a synthetic example or claim causality solely from the sign of DII.
