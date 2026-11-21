# Grid-free Kolmogorov turbulence: test of the finite-λ kurtosis model without an inner cutoff

Prototype on branch `claude/gridfree-turbulence-prototype`. Working note.

- Generator: `manuscript_pipeline/04_flame_kurtosis_3d/gridfree_kolmogorov.py`
- Test: `manuscript_pipeline/04_flame_kurtosis_3d/test_gridfree_kolmogorov_model.py` (`dcheck`, `kurtosis`)
- Output (gitignored): `Figures/manuscript/04_kurtosis_3d_all_regions/gridfree_turbulence_test/`
  (`gridfree_D_check.csv`, `gridfree_kurtosis_curves.csv`, `gridfree_geometric_terms.csv`, `gridfree_kurtosis_summary.csv`)

## Why

The grid generator (`kolmogorov_velocity_grid`, 160³, box 2.4) keeps modes only up to
$k_{\max} = 0.35\,k_{\rm Nyq} = 73.3$, an inner scale of $0.086\,r_{\rm HII}$, larger than every thin-shell
wall. A real H II region has an inner (dissipation) scale far below any wall thickness, so the relevant
limit is $k_{\max}\to\infty$. A grid cannot reach it: $k_{\max}\propto N$ and memory $\propto N^3$.

## Generator

The line-of-sight velocity is a sum of random Fourier modes evaluated directly at the point positions:

$$g(\mathbf r) = \sum_m \sqrt{w_m}\,\big[A_m\cos(\mathbf k_m\cdot\mathbf r) + B_m\sin(\mathbf k_m\cdot\mathbf r)\big],\qquad A_m, B_m \sim N(0,1).$$

- 300 log-spaced shells from $k_{\min} = 5.24$ (same outer scale as the grid runs) to $k_{\max} = 10^5$
  (inner scale $6\times10^{-5}$), 32 random directions per shell, 9,600 modes.
- Within each shell $|\mathbf k|$ is drawn from $p(k)\propto k^{-5/3}$ and the shell carries its share of the
  variance, $w \propto \int_{\rm shell} k^{-5/3}dk$, so the ensemble correlation is exactly the continuum
  $\xi(s) = \int k^{-5/3}\,{\rm sinc}(ks)\,dk / \int k^{-5/3}dk$.
- Given the wavevectors the field is exactly Gaussian with unit variance. No grid, no interpolation.

The analytic structure function is $D(s) = \frac{2 s^{2/3}}{Z}[G(k_{\max}s) - G(k_{\min}s)]$, with
$G(x) = \int_0^x t^{-5/3}(1-\sin t/t)\,dt$ (power series for $x<10$, asymptotic expansion above), and for
$k_{\max}\to\infty$, $D(s) = \frac43 x^{2/3}[I - G(x)]$ with $x = k_{\min}s$, $I = \frac{9}{20}\Gamma(\frac13)$.

## Test 1: D(s) of the generator

6 realisations, 3,000 random pairs per separation:

| $s$ | measured | analytic ($k_{\max}=10^5$) | $k_{\max}=\infty$ | grid ($k_{\max}=73$) |
|---|---|---|---|---|
| 0.001 | 0.046 ± 0.001 | 0.046 | 0.048 | 0.000 |
| 0.0043 | 0.125 ± 0.002 | 0.126 | 0.128 | 0.003 |
| 0.0186 | 0.336 ± 0.002 | 0.337 | 0.339 | 0.060 |
| 0.0805 | 0.864 ± 0.008 | 0.872 | 0.874 | 0.645 |
| 0.347 | 1.888 ± 0.051 | 1.878 | 1.879 | 1.854 |
| 0.566 | 2.087 ± 0.033 | 2.079 | 2.079 | 2.095 |
| 1.5 | 1.965 ± 0.047 | 1.994 | 1.994 | 1.993 |

The generator reproduces the analytic $D$ within the realisation scatter from $s = 0.001$ to 1.5.
$k_{\max} = 10^5$ and $k_{\max}=\infty$ differ by < 4% at $s = 0.001$ and are identical above.

## Test 2: K(ℓ; λ)

$N = 30{,}000$ uniform shell points (point seed 1234), all pairs with $0.002 \le \ell \le 0.6$, 8 field seeds,
bins with at least 200 pairs. Three estimates on the same points:

- **simulation**: realised fields, odd terms removed (equivalent to pooling each seed with $g\to-g$);
- **pair sums**: the exact Gaussian-ensemble average on the same points ($D$ at each pair's 3D separation,
  $\delta\mu$ from the positions), i.e. the formula of §1 of the λ note with no geometric approximation;
- **model**: §§4–6 of the λ note: Abel projection of $C$ for $\langle D\rangle$, $\langle D^2\rangle$;
  thin-shell $\langle\delta\mu^2\rangle = 2/K_b$, $\langle\delta\mu^4\rangle = 4/K_b$ with exact $K_b$;
  approximate cross term.

### Geometric averages, model vs pair sums

| term | agreement (ℓ ≥ 0.01) |
|---|---|
| $\langle D\rangle$, $K_t$ (Abel projection) | ≤ 0.01 at every width |
| $K_b$ (exact sky integrals) | ≤ 0.03; smaller-ℓ bins are noisy (500–2,000 pairs) |
| $\langle\delta\mu^2\rangle$ thin form $2/K_b$ | ≤ 1% for Δ ≤ 0.09; 2% low at Δ = 0.3; 6% low at Δ = 1; 2–9% high at ℓ ≈ 0.5 |
| $\langle D\delta\mu^2\rangle$ approximate | ≤ 2% for Δ ≤ 0.09; 2% low at Δ = 0.3; 6% low at Δ = 1 |
| $\chi = \langle D\delta\mu^2\rangle/(\langle D\rangle\langle\delta\mu^2\rangle)$, thin forms | ≤ 0.01 at every width |

The thin-shell bulk forms and the approximate cross term err by the same factor for thick shells, so
$\chi$ (what enters $K$) is right. This is why the thin-shell combination beats "exact bulk + approximate
cross" (e.g. Δ = 1, λ = 1: rms 0.028 vs 0.066), as already noted in §6 and §9 of the λ note.

### Full K, model vs pair sums vs simulation

rms over the bins. "model − pairs" isolates the geometric approximations; "sim − model" adds the
realisation scatter of 8 seeds.

| Δ | λ | model − pairs (all ℓ) | model − pairs (ℓ ≥ 0.01) | sim − model | plateau sim / pairs / model | seed σ (plateau) |
|---|---|---|---|---|---|---|
| 0.02 | 0 | 0.023 | 0.004 | 0.113 | 4.27 / 4.39 / 4.37 | 0.19 |
| | 1 | 0.037 | 0.006 | 0.067 | 4.93 / 4.89 / 4.86 | 0.33 |
| | 2 | 0.060 | 0.010 | 0.093 | 5.60 / 5.47 / 5.43 | 0.39 |
| | 3 | 0.081 | 0.017 | 0.101 | 5.87 / 5.76 / 5.72 | 0.30 |
| | 10 | 0.116 | 0.030 | 0.116 | 6.09 / 6.07 / 6.03 | 0.05 |
| | ∞ | 0.121 | 0.032 | 0.121 | 6.11 / 6.11 / 6.06 | 0 |
| 0.09 | 0 | 0.009 | 0.003 | 0.039 | 3.71 / 3.72 / 3.72 | 0.18 |
| | 1 | 0.011 | 0.005 | 0.029 | 4.04 / 4.06 / 4.04 | 0.29 |
| | 2 | 0.020 | 0.010 | 0.036 | 4.35 / 4.37 / 4.36 | 0.33 |
| | ∞ | 0.058 | 0.029 | 0.058 | 4.58 / 4.58 / 4.59 | 0 |
| 0.3 | 0 | 0.010 | 0.002 | 0.031 | 3.37 / 3.37 / 3.38 | 0.15 |
| | 1 | 0.013 | 0.003 | 0.021 | 3.56 / 3.56 / 3.55 | 0.20 |
| | 2 | 0.023 | 0.005 | 0.027 | 3.66 / 3.65 / 3.61 | 0.20 |
| | ∞ | 0.054 | 0.012 | 0.054 | 3.56 / 3.56 / 3.45 | 0 |
| 1.0 | 0 | 0.009 | 0.002 | 0.035 | 3.24 / 3.25 / 3.26 | 0.13 |
| | 1 | 0.013 | 0.005 | 0.028 | 3.30 / 3.31 / 3.33 | 0.06 |
| | 2 | 0.024 | 0.008 | 0.035 | 3.15 / 3.16 / 3.19 | 0.05 |
| | ∞ | 0.040 | 0.008 | 0.040 | 2.75 / 2.75 / 2.78 | 0 |

(Plateau = mean of the first five bins. At λ = ∞ the simulation and the pair sums are identical by
construction; the residual is the finite-$N$ estimate of $K_b$ against the integral.)

**Model with $D$ for $k_{\max}=\infty$ instead of $10^5$:** rms difference ≤ 0.004 everywhere.

## Conclusions

1. **The formula holds without an inner cutoff.** Against the exact ensemble pair sums the model is
   within 0.03 for ℓ ≥ 0.01 at every Δ and λ, and within 0.01 for λ ≤ 2. The approximations (thin-shell
   bulk forms, approximate cross term) do not degrade when same-wall pairs are no longer perfectly
   correlated.
2. **$k_{\max}\to\infty$ is a safe assumption.** An inner scale of $6\times10^{-5}\,r_{\rm HII}$ and none at all
   give the same $K$ to 0.004, so the model can use the one-parameter $D(s)$ (outer scale only).
3. **The remaining differences are sampling, not the model.** Below ℓ = 0.01 the bins have 500–2,000
   pairs and the finite-$N$ kurtosis scatters (and is biased low for thin shells, e.g. $K_b$ 5.7–6.0 vs 6.06
   at ℓ ≈ 0.003, Δ = 0.02). At λ ≲ 3 the 8-seed scatter (σ 0.1–0.4 on the plateau) dominates every model
   error.
4. **Thin shells lose kurtosis without an inner cutoff, as predicted.** Plateau at Δ = 0.02: $K_t$ = 4.39
   here vs 5.40 with the grid field; at Δ = 0.09: 3.72 vs 4.22. The bulk limit is unchanged (6.11, 4.58).
   The grid-field simulations therefore overestimate the turbulent and intermediate-λ kurtosis of thin
   shells relative to a real region with the same outer scale.
5. **Odd terms** (single realisations, §10 of the λ note) shift the 8-seed plateau by at most ±0.09 here
   (Δ = 0.3, λ = 1–3), smaller than for the grid field (up to +0.34 at Δ = 0.02, λ = 2–3), probably because
   each grid-free realisation has 32 modes per shell even at $k_{\min}$, more than the grid's 6–24 (not tested).

## Caveats and open items

- The number of modes per shell at the largest scales is a choice (32 directions) and controls the
  realisation variance. To mimic a real region or the grid field, the low-$k$ mode count should follow
  $\propto k^2\Delta k$ for a chosen effective volume.
- The field is a scalar Gaussian field (like the grid generator), not a solenoidal vector field; for real
  incompressible turbulence the line-of-sight correlation depends on the angle of $\mathbf s$ to the line of
  sight.
- Field-measured $\langle\delta g^2\rangle$ on the shell is ~3% below the ensemble $D$ at ℓ ≳ 0.3 (1.96 vs 2.01):
  8-seed scatter of the few largest modes; the expectation is unbiased.
- Unpixelised estimator only, as in the λ note.
