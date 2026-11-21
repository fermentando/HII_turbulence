<!-- Copy of geometric_averages/00_plateau_kurtosis_summary.md: derivation of K(l; lambda) used in the fits. -->

# Summary: analytic plateau kurtosis K(ℓ → 0; λ)

Short summary of how the full finite-λ pair kurtosis reduces, at $\ell\to0$ and for a thin shell, to one closed
formula. Detailed derivations: `geometric_averages/01_turbulent_D.md` (turbulent averages); bulk and cross-term notes
(`geometric_averages/02_…`, `geometric_averages/03_…`) and the mixing form (`geometric_averages/04_…`) are in progress; until then see
`kurtosis_bulk_limit_derivation.md`, `kurtosis_bulk_limit_constants.md` and §§5–8 of
`kurtosis_kolmogorov_lambda_derivation.md`.

Setting: expanding spherical shell, radius 1 and thickness $\Delta$ (units of $r_{\rm HII}$), Gaussian Kolmogorov
turbulence with outer scale $2\pi/k_{\min}$ and **no inner cutoff**, $\lambda = v_{\rm bulk}/v_{\rm turb}$, unpixelised
estimator.

## Result

$$
\boxed{K(0;\lambda) = K_b\,\frac{3\tilde J_2 + 4\lambda^2(\lambda^2+6)}{(\tilde J + 2\lambda^2)^2}}
$$

with

$$
K_b = \tfrac32 + 2\ln2 - \ln(2\Delta), \qquad
\tilde J = J - \tfrac{27}{40}A\,\Delta^{2/3}, \qquad
\tilde J_2 = J_2 - \tfrac{27}{140}A^2\Delta^{4/3}.
$$

Plain text:

```
K(0, λ) = K_b · [ 3 J̃₂ + 4λ²(λ² + 6) ] / ( J̃ + 2λ² )²

K_b = 3/2 + 2 ln 2 − ln(2Δ)           bulk plateau = pair-count factor
J̃   = J  − (27/40)  · A  · Δ^(2/3)    D-weighted pair count (shell-corrected)
J̃₂  = J₂ − (27/140) · A² · Δ^(4/3)    D²-weighted pair count (shell-corrected)
```

Turbulence constants (outer scale only; values for $k_{\min} = 5.24$):

| constant | definition | value |
|---|---|---|
| $A$ | $\frac43 I\,k_{\min}^{2/3}$, $I = \frac{9}{20}\Gamma(\frac13)$; Kolmogorov amplitude $D\simeq As^{2/3}$ | 4.85 |
| $J$ | $\int_0^2 D(s)\,ds/s$ | 6.85 |
| $J_2$ | $\int_0^2 D(s)^2\,ds/s$ | 10.53 |

$D(s) = \frac43 x^{2/3}[I - G(x)]$, $x = k_{\min}s$, $G(x) = \int_0^x t^{-5/3}(1 - \sin t/t)\,dt$ (`geometric_averages/01_turbulent_D.md`).

## How each term of the full K simplifies

Start from the exact (Gaussian-ensemble) formula

$$
K = \frac{3\langle D^2\rangle + 6\lambda^2\langle D\,\delta\mu^2\rangle + \lambda^4\langle\delta\mu^4\rangle}{\left[\langle D\rangle + \lambda^2\langle\delta\mu^2\rangle\right]^2}.
$$

At $\ell\to0$ on a thin shell every average has the **same pair-count denominator**,
$\pi\Delta^2[3 + 2\ln(2/\Delta)] = 2\pi\Delta^2K_b$: same-wall pairs (the 3) plus all pairs beyond $\Delta$, which
follow the shell's $C(s)\propto1/s$ law (the log). Each average is therefore "something" / $K_b$:

| term | plateau form | why |
|---|---|---|
| $\langle\delta\mu^2\rangle$ | $2/K_b$ | bulk: sightline velocity spread $\sim\mu_0^2$ pooled with pair weight $n^2\propto1/\mu_0^2$; the limb pairs ($\delta v\approx0$) dominate the count |
| $\langle\delta\mu^4\rangle$ | $4/K_b$ | same, fourth moment |
| $\langle D\rangle$ | $\tilde J/K_b$ | turbulence: $J$ = D-weighted pair count for an infinitely thin wall; $-\frac{27}{40}A\Delta^{2/3}$ because same-wall pairs are up to $\Delta$ apart (slab count $4\pi(\Delta - s/2)$, $D\simeq As^{2/3}$) |
| $\langle D^2\rangle$ | $\tilde J_2/K_b$ | same with $D^2\simeq A^2s^{4/3}$ |
| $\langle D\,\delta\mu^2\rangle$ | $4/K_b$ | cross term: only opposite-wall pairs carry $\delta\mu\approx\pm2\mu_0$; they are $\approx2\mu_0$ apart, mostly beyond the correlation length, so $D\approx2$: $(4/K_b)\int_0^1\mu_0D(2\mu_0)\,d\mu_0 \approx 4/K_b$ |

## Derivation in three lines

1. Substitute the table into $K$ and multiply numerator and denominator by $K_b^2$:

$$
K = \frac{(3\tilde J_2 + 6\lambda^2\cdot4 + 4\lambda^4)/K_b}{(\tilde J + 2\lambda^2)^2/K_b^2}
= K_b\,\frac{3\tilde J_2 + 4\lambda^2(\lambda^2+6)}{(\tilde J + 2\lambda^2)^2}.
$$

2. $K_b$ factors out: the **geometry sets the level** (growing as $\ln(1/\Delta)$), the **turbulence sets only the
   shape in λ**, through $\tilde J$, $\tilde J_2$, which depend on $\Delta$ only weakly ($\Delta^{2/3}$).

3. Equivalent mixing form, with $\lambda_c^2 = \langle D\rangle/\langle\delta\mu^2\rangle = \tilde J/2$ and
   $K_t = 3\langle D^2\rangle/\langle D\rangle^2 = 3K_b\tilde J_2/\tilde J^2$:

$$
K(0;\lambda) = \frac{K_t\lambda_c^4 + K_b\lambda^2(\lambda^2+6)}{(\lambda^2+\lambda_c^2)^2}
= K_t(1-f)^2 + 6\chi f(1-f) + K_b f^2,\qquad f = \frac{\lambda^2}{\lambda^2+\lambda_c^2},\quad \chi = \frac{K_b}{\lambda_c^2}.
$$

```
K(0, λ) = [ K_t λ_c⁴ + K_b λ²(λ² + 6) ] / (λ² + λ_c²)²,    λ_c² = J̃/2,   K_t = 3 K_b J̃₂ / J̃²
```

## Limits

| limit | value | meaning |
|---|---|---|
| $\lambda\to\infty$ | $K_b$ | expansion only |
| $\lambda = 0$ | $K_t = \frac{3\tilde J_2}{\tilde J^2}K_b \simeq 0.67\,K_b$ | turbulence only: same $\ln(1/\Delta)$ growth, scaled by the turbulence |
| crossover | $\lambda_c = \sqrt{\tilde J/2}\approx1.8$ | where bulk and turbulent variances are equal; nearly independent of $\Delta$ |
| interior maximum | exists if $3\chi > K_t, K_b$ | the overshoot near $\lambda\approx2$ for $\Delta\approx0.3$ |
| scale-free turbulence | $3J_2/J^2 = 1\Rightarrow K_t = K_b$ | $D = As^{2/3}$ without saturation |

## Accuracy

Plateau $K_{\rm plat}$: grid-free simulations (8 seeds, odd terms removed; `gridfree_turbulence_test.md`) /
numerical model (exact Abel projection, exact $K_b$) / this formula:

| λ | Δ = 0.02 | Δ = 0.09 | Δ = 0.3 | Δ = 1 |
|---|---|---|---|---|
| 0 | 4.27 / 4.37 / 4.40 | 3.68 / 3.71 / 3.72 | 3.36 / 3.36 / 3.38 | 3.20 / 3.23 / 3.08 |
| 1 | 4.93 / 4.86 / 4.90 | 4.02 / 4.03 / 4.04 | 3.54 / 3.53 / 3.54 | 3.27 / 3.29 / 3.24 |
| 2 | 5.60 / 5.43 / 5.48 | 4.35 / 4.35 / 4.36 | 3.61 / 3.59 / 3.58 | 3.14 / 3.17 / 2.91 |
| 5 | 6.03 / 5.92 / 5.97 | 4.53 / 4.55 / 4.56 | 3.53 / 3.50 / 3.46 | 2.87 / 2.89 / 2.38 |
| ∞ | 6.11 / 6.06 / 6.11 | 4.56 / 4.58 / 4.60 | 3.47 / 3.44 / 3.40 | 2.77 / 2.77 / 2.19 |

- $\Delta\le0.09$: within 0.05 of the numerical model at every λ, well inside the single-source scatter
  (σ ≈ 0.2–0.8 at λ ≈ 1–3).
- $\Delta = 0.3$: within 0.05 up to λ ≈ 3 (including the hump); 0.04–0.07 low at large λ, where the thin-shell
  $K_b$ starts to fail.
- $\Delta\gtrsim2/3$: fails. Every piece assumes a $1/s$ range between $s = \Delta$ and $2(1-\Delta)$, which vanishes;
  use the numerical model there.

Figure: `Figures/manuscript/04_kurtosis_3d_all_regions/gridfree_turbulence_test/analytic_plateau_kurtosis.png`
(script `manuscript_pipeline/04_flame_kurtosis_3d/plot_analytic_plateau_kurtosis.py`).

## Assumptions

- Thin uniform spherical shell ($\Delta\lesssim0.3$), sharp edges, $\ell\to0$ (in practice $\ell\lesssim\Delta/5$).
- Gaussian isotropic scalar turbulence, $P\propto k^{-11/3}$, sharp outer cutoff, no inner cutoff.
- Cross term from opposite-wall pairs with $D\approx2$ (to be confirmed in `geometric_averages/03_cross_term.md`).
- Ensemble average: a single source scatters about this by up to ±0.8 at λ ≈ 1–3 for thin shells.
- Unpixelised estimator (each 3D point its own sample); IFU sightline averaging changes the bulk terms.
