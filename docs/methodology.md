# Measurement notes

## Population target versus fitted estimate

The population target uses the true conditional means. The software substitutes fitted conditional-mean models and calculates a Gaussian-kernel HSIC V-statistic on a separate evaluation sample. Finite-sample regression error may change either dependence component. Numerical tests establish software behavior, not statistical coverage or empirical validity.

The complete forward input is `(X, C)` and the complete reverse input is `(Y, C)`. Removing a conditional mean does not turn general conditional independence into unconditional residual independence. The latter asks for an invariant residual distribution across the full inputs.

## Kernel and normalization convention

For transformed rows `a_i`, use `k(a_i,a_j) = exp(-||a_i-a_j||^2 / (2 l^2))`. Center both Gram matrices and compute `sum(Kc * Lc) / n^2`. This is the biased HSIC V-statistic. Do not interpret a positive sample HSIC as a rejection of independence.

The high-level `DII` class learns X, Y and history standardization on training observations only. It uses fixed bandwidth one by default and never standardizes residuals by their own variance. Choosing another bandwidth changes the target. A common scalar bandwidth is used for residual and full-regressor kernels in the high-level API; the residual API permits variable-specific residual bandwidths.

When comparing horizons, pass the same training-derived `x_scale` and `y_scale` at every horizon and keep the history sample identical. Different centering constants do not change Gaussian-kernel distances; outcome rescaling does. Scale choices should be specified before inspecting the DII profile.

## Time-series implementation

1. Define the driver, history, outcome window and horizon grid before estimation.
2. Split training and evaluation chronologically. Purge any observations whose lead/lag windows overlap the other sample, and use a further gap justified by temporal dependence.
3. Fit/tune conditional-mean models and preprocessing inside training data.
4. Calculate both residuals and full input vectors on aligned evaluation observations.
5. Report both components and their signed difference.
6. Use an externally justified inference procedure if making inferential claims. Jointly account for both directions, training uncertainty, overlapping horizons and shared event/calendar shocks. Naive IID permutations or an ordinary bootstrap are not generically justified near the degenerate independence null.

The example scripts use independent synthetic rows, so a chronological dependence gap is not needed there.

## Half-decay convention

Choose a fixed reference horizon `h0` and floor `d_inf`. With positive excess `A = D(h0)-d_inf`, the half threshold is `d_inf + A/2`. The helper reports the first observed horizon below that threshold and, by default, requires all later observed horizons also to remain below it.

The result is elapsed time `h-h0`; its bracket reflects grid resolution. Staying below through the end of the grid does not establish permanent decay. A negative DII is a reversal of ordering, not a larger positive signal; never take absolute values silently. If the reference signal is absent, half-life is undefined. If no crossing is observed, report that it was not reached within the measured grid.

For an externally justified exponential model `D(h)=d_inf+A exp(-lambda*(h-h0))`, the excess-DII half-life is `log(2)/lambda`. This release does not assume or fit that model. Because HSIC is a squared dependence functional, DII half-life need not match a response-amplitude half-life.

For inference, use joint uncertainty for the threshold contrast `D(h)-0.5*D(h0)-0.5*d_inf`, including uncertainty in any estimated floor. A non-significant DII is not proof that it is zero. Reporting confidence sets for a crossing may be more honest than a forced finite point estimate.

## Benchmarks

With matched variable-specific kernels, a jointly Gaussian `(X,Y,C)` has zero population DII even with a nonzero regression coefficient. In `Y=X^2+epsilon`, symmetric X and independent additive noise can yield a positive DII because the reverse conditional mean does not remove dependence. A symmetric pure scale model can have zero contrast with equal, positive components. These examples show why both components matter.

These are mathematical illustrations, not evidence that DII outperforms correlation, local projections, Granger prediction tests or existing causal-discovery methods on a particular application.
