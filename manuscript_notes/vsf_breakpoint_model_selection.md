# VSF Breakpoint Model Selection

Goal: fit VSF curves in log-log space and decide whether the data support:

- one line: no turnover/break
- two lines: one breakpoint
- several lines: multiple breakpoints

Use this for the next cleaner fitting implementation, separate from the current manuscript pipeline.

## Data Space

For each region and tracer, fit:

```text
x_i = log10(r_i)
y_i = log10(VSF_i)
sigma_i = VSF_err_i / (VSF_i ln 10)
```

Only use bins with finite positive `VSF`, finite positive uncertainty, and enough pair counts.

## Candidate Models

Let `m` be the number of line segments.

- `m = 1`: one straight line
- `m = 2`: one breakpoint
- `m = 3`: two breakpoints
- etc.

Prefer continuous segmented fits:

```text
y = a + b1 x
y = a + b1 x + c1 max(0, x - tau1)
y = a + b1 x + c1 max(0, x - tau1) + c2 max(0, x - tau2)
...
```

This gives a continuous broken power law. Slopes are:

```text
slope_1 = b1
slope_2 = b1 + c1
slope_3 = b1 + c1 + c2
```

Breakpoints `tau_j` should be searched over observed radial bins, with a minimum number of points per segment.

## Primary Criterion

Use Gaussian-error BIC as the main selector:

```text
chi2 = sum(((y_i - model_i) / sigma_i)^2)
BIC = chi2 + k ln(n)
```

where `n` is the number of fitted bins.

For a continuous segmented fit with `m` segments:

```text
k = 2m
```

because the model has:

- one intercept
- `m` slope-equivalent parameters
- `m - 1` breakpoint locations

So:

```text
m = 1: k = 2
m = 2: k = 4
m = 3: k = 6
```

Choose the model with the lowest BIC. A more complex model should only be accepted if it improves BIC by a meaningful amount:

```text
Delta BIC = BIC simpler - BIC complex
```

Suggested thresholds:

- `Delta BIC < 2`: no meaningful evidence
- `2 <= Delta BIC < 6`: weak evidence
- `6 <= Delta BIC < 10`: moderate evidence
- `Delta BIC >= 10`: strong evidence

For manuscript-grade break claims, use `Delta BIC >= 10` unless visual/physical context justifies a softer threshold.

## KS Role

Do not use KS as the primary selector between one-line and multi-line models. KS does not naturally penalize extra breakpoints and is not designed to compare nested regression models.

Use KS as a diagnostic or supporting criterion:

1. Standardized residual goodness:

```text
z_i = (y_i - model_i) / sigma_i
```

For each candidate model, test whether `z_i` is consistent with `N(0, 1)`.

2. Residual structure:

Compare residuals in each segment. A good breakpoint model should reduce systematic residual differences between segments.

3. Local-slope break evidence:

Use Monte Carlo perturbations of `y_i` by `sigma_i`, compute local slopes, and test whether adjacent groups of local slopes differ across a proposed breakpoint.

This is useful for saying "there is a slope change here", while BIC answers "the added breakpoint is worth the model complexity".

## Recommended Acceptance Rule

For each curve:

1. Fit models with `m = 1, 2, ..., m_max`.
2. Enforce at least `min_points_per_segment` in every segment.
3. Compute BIC for every valid candidate.
4. Select the lowest-BIC model.
5. Accept extra breakpoint(s) only if:

```text
Delta BIC >= 10
```

relative to the best simpler model, and all fitted segment lengths are acceptable.

6. Report KS diagnostics but do not let KS alone choose the number of segments.

## Practical Defaults

```text
m_max = 3
min_points_per_segment = 5
min_delta_bic = 10
continuous = True
```

If the data have fewer than about 12 usable bins, restrict to `m_max = 2`.

## Why This Is Cleaner

The current scripts mix several ideas:

- one-line weighted fitting
- two-line weighted fitting
- KS tests on local slopes
- BIC model improvement
- plotting and table generation

The cleaner version should separate these into:

1. fit candidate segmented models
2. score candidates with BIC
3. compute residual/local-slope diagnostics
4. produce plots/tables
