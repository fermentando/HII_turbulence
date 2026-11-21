# How S2(ℓ) depends on λ and the shell thickness Δ

**What this note gives.** How the second-order structure function $S_2(\ell)$ of an expanding shell with Kolmogorov
turbulence changes with $\lambda = v_{\rm bulk}/v_{\rm turb}$ and the shell thickness $\Delta$. Two cases:
- point samples (each velocity is one point in the shell);
- fibre centroids (each velocity is an emission-weighted average along the sightline, as an LVM fibre measures).

All for turbulence with no inner cutoff ($k_{\max}\to\infty$). Companion to `kurtosis_K_lambda_derivation.md` (the
plateau of $K$) and `kurtosis_K_lambda_fit_formula.md` (the shape of $K$).

**Summary.**
- For point samples $S_2$ is nearly flat in ℓ (power-law index ≤ 0.26) and barely depends on Δ. It cannot match
  the observed indices (0.7–1.6).
- For fibre centroids, sightline averaging makes $S_2$ steep. Its index rises with Δ, through the turbulent part, and
  with λ, through the bulk part.
- So $S_2$ constrains a **combination** of Δ and λ: a thick shell with weak expansion and a thin shell with strong
  expansion can give the same index. Δ and λ must be fitted together, not one after the other.

Script: `manuscript_pipeline/08_shell_inference/study_s2_model.py`. Figure (outside git):
`Figures/manuscript/08_shell_inference/s2_model_study/study_s2_model.png`.

## 1. Setting and symbols

Expanding spherical shell, radius $r_{\rm HII}=1$, thickness $\Delta$ (lengths in units of $r_{\rm HII}$). Gaussian
Kolmogorov turbulence (unit variance, outer scale $L_{\rm out} = 2\pi/k_{\min}$, no inner cutoff) plus radial expansion.
Velocities in units of $v_{\rm turb}$; ℓ is the sky separation of a pair.

| symbol | meaning |
|---|---|
| $D(s)$ | turbulent structure function at 3D separation $s$: $D \simeq A s^{2/3}$ at small $s$, $\to 2$ beyond the outer scale |
| $\mu$ | line-of-sight projection of the radial direction, $\mu = z/r$ |
| $a$ | weight of the far side of the shell (a = 0: only the near side emits or is seen) |
| $\langle\cdot\rangle$ | average over all pairs at sky separation ℓ |

Numbers below: $L_{\rm out} = 1.2$ (the simulated value), $a = 0$, no noise.

## 2. Point samples

For one pair, the velocity difference is $\delta g + \lambda\,\delta\mu$ with $\delta g$ Gaussian of variance
$D(s)$. Averaging over pairs:

$$
S_2(\ell) = \langle D\rangle + \lambda^2\langle\delta\mu^2\rangle
= \frac{N_1(\ell) + 2\,q_2(\ell)\,\lambda^2}{P(\ell)}
$$

($P$, $N_1$, $q_2$ as in `kurtosis_K_lambda_fit_formula.md` §5).

**Why it is nearly flat.** At sky separation ℓ the pairs have 3D separations from ℓ up to ≈ 2: most are across the
cavity, far beyond the correlation length, with $D \approx 2$. Only the few same-wall pairs feel ℓ.
- Turbulent part $\langle D\rangle$: index 0.09–0.16 over ℓ = 0.05–0.5.
- Bulk part $\langle\delta\mu^2\rangle$: index 0.13–0.26. Across-cavity pairs carry $\delta\mu \approx \pm 2\mu_0$ at any ℓ, so it
  also grows only slowly.

**Δ barely matters.** Δ enters only through the same-wall pairs at ℓ < Δ. Beyond the wall $S_2$ does not depend on Δ
at all (the pair counts beyond $s = \Delta$ are the same for any thin shell).

**Check against simulations.** The analytic point-sample $S_2$ agrees with the grid-free simulations (8 seeds) to
1–4% rms for Δ ≤ 0.3 at λ = 0, 1, 3, 10 (largest single-bin difference 8–11%). It fails at Δ = 1, where the
thin-shell forms do not apply.

**Consequence.** The point-sample index cannot exceed ≈ 0.26 for any Δ or λ. The observed indices are 0.72–1.64.
Point samples are not a usable model for $S_2$ of fibre data.

## 3. Fibre centroids

A fibre at sky radius ρ sees the emission-weighted mean velocity along its sightline,
$V = \sum_z w(z)\,[g(z) + \lambda\mu(z)]$, with weights $w(z)$ set by the chord through the shell (see
`fit_centroid_joint_grid.py`). For a pair of fibres:

$$
S_2(\ell) = \langle T\rangle + \lambda^2\langle m^2\rangle ,
$$

- $T$: variance of the turbulent centroid difference (Gaussian turbulence averaged over both sightlines);
- $m = G_1 - G_2$: difference of the bulk centroids, $G(\rho) = \sum_z w\,\mu$ (≈ $\sqrt{1-\rho^2}$ for a thin
  near side).

### 3.1 The turbulent part ⟨T⟩: depth sets the slope

Averaging along a sightline of depth $L$ removes the turbulent fluctuations smaller than $L$. This is the standard
projected-turbulence result:
- for $\ell \ll L$ the projected structure function rises as $\ell^{2/3+1} = \ell^{5/3}$;
- for $L \ll \ell \ll L_{\rm out}$ it rises as $\ell^{2/3}$;
- near $L_{\rm out}$ it flattens towards its saturation value.

The depth of the shell along a sightline is $L(\rho) \approx \Delta/\sqrt{1-\rho^2}$ near the centre, growing to
$\approx\sqrt{2\Delta}$ at the limb. So:
- **thicker shell → deeper sightlines → steeper $\langle T\rangle$** over a wider range of ℓ;
- **thicker shell → more averaging → lower amplitude** ($\langle T\rangle$ at ℓ = 1 falls from 1.9 at Δ = 0.005 to 0.66 at Δ = 1).

| Δ | index of ⟨T⟩, ℓ = 0.01–0.03 | index of ⟨T⟩, ℓ = 0.05–0.5 | ⟨T⟩ at ℓ = 1 |
|---|---|---|---|
| 0.005 | 0.74 | 0.47 | 1.91 |
| 0.016 | 1.01 | 0.52 | 1.80 |
| 0.053 | 1.27 | 0.63 | 1.56 |
| 0.17 | 1.34 | 0.88 | 1.14 |
| 0.31 | 1.32 | 0.94 | 0.89 |
| 1.0 | 1.31 | 0.93 | 0.66 |

- Thin shells: over ℓ = 0.05–0.5 the sightlines are much shallower than ℓ, so $\langle T\rangle$ follows the $\ell^{2/3}$
  regime, flattened by the outer scale to an index of ≈ 0.5.
- Thick shells (Δ ≳ 0.15): the depth is comparable to ℓ, so part of the window is in the steep regime, and the index
  rises to ≈ 0.94.
- Above Δ ≈ 0.3 the index stops growing: the sightlines are as deep as the region itself.
- At the smallest ℓ (0.01–0.03) the index approaches but does not reach 5/3. The limb sightlines are much deeper
  than the central ones, and mixing them softens the slope.

### 3.2 The bulk part ⟨m²⟩: a smooth map gives a steep slope

The bulk centroid $G(\rho)$ is a smooth, deterministic map across the face of the region. For nearby fibres
$G_1 - G_2 \approx \nabla G\cdot\boldsymbol\ell$, so $\langle m^2\rangle \propto \ell^2$. Near the limb $G$ changes steeply
(it drops to 0 at ρ = 1), and those pairs soften the index below 2.

| Δ | index of ⟨m²⟩, ℓ = 0.05–0.5 | ⟨m²⟩ at ℓ = 1 |
|---|---|---|
| 0.005 | 1.43 | 0.15 |
| 0.053 | 1.45 | 0.19 |
| 0.17 | 1.73 | 0.24 |
| 0.31 | 1.83 | 0.25 |
| 1.0 | 1.75 | 0.13 |

The bulk part depends on Δ only mildly: a thicker near side averages μ over a wider range of depths, which smooths $G$
at the limb.

### 3.3 Mixing the two: λ steepens S2

$S_2 = \langle T\rangle + \lambda^2\langle m^2\rangle$ is a sum of a shallow part (index 0.5–0.9) and a steep part
(index 1.4–1.8). The bulk part takes over where $\lambda^2\langle m^2\rangle > \langle T\rangle$, i.e. above

$$
\lambda_{\rm eq}(\ell) = \sqrt{\langle T\rangle/\langle m^2\rangle} .
$$

| Δ | λ_eq at ℓ = 0.1 | λ_eq at ℓ = 0.5 |
|---|---|---|
| 0.005 | 11.5 | 4.8 |
| 0.029 | 9.4 | 4.2 |
| 0.095 | 7.4 | 3.5 |
| 0.31 | 6.4 | 2.6 |

- **λ ≲ 2:** turbulence dominates $S_2$ at every observed ℓ. λ hardly changes $S_2$.
- **λ ≈ 3–10:** the bulk part dominates the large ℓ first, so $S_2$ steepens with ℓ (it curves upwards).
- **λ ≳ 10:** the bulk part dominates over most of the observed range, and the index approaches that of $\langle m^2\rangle$.

## 4. The combined index: Δ and λ are degenerate

Power-law index of the centroid $S_2$ over ℓ = 0.05–0.5:

| Δ | λ = 0 | λ = 1 | λ = 3 | λ = 10 |
|---|---|---|---|---|
| 0.005 | 0.47 | 0.49 | 0.59 | 1.03 |
| 0.029 | 0.56 | 0.58 | 0.70 | 1.12 |
| 0.095 | 0.75 | 0.78 | 0.94 | 1.37 |
| 0.31 | 0.94 | 0.99 | 1.22 | 1.67 |

Observed (noise-subtracted, all regions and lines): median 0.98, 16–84% range 0.72–1.64.

- **An index of ≈ 1** comes either from a thick shell with weak expansion (Δ ≳ 0.15, λ ≲ 1) or from a thin shell with
  strong expansion (Δ ≈ 0.01, λ ≈ 10). This is the diagonal degeneracy in the joint (Δ, λ) maps.
- **Indices above ≈ 1.25** need λ ≳ 3–10. Below λ ≈ 3 no shell thickness makes $S_2$ that steep.
- **Turbulence alone tops out at ≈ 0.94.** About half of the observed lines are steeper, so $S_2$ by itself already
  shows that the expansion contributes.
- **The curvature** of $S_2(\ell)$ helps to break the degeneracy: thick shells are steep at small ℓ and flatten at large
  ℓ (turbulent depth effect); strong expansion is shallow at small ℓ and steepens at large ℓ (bulk takes over).

## 5. Things that change S2 but are not in the tables

- **Noise.** White fibre noise adds a constant $N_2$: $S_2^{\rm obs} = v_{\rm turb}^2 S_2 + N_2$. It flattens the smallest ℓ,
  where the signal is weakest. The fits include it; the indices above do not.
- **Outer scale.** A larger $L_{\rm out}$ delays the saturation of $D$, which steepens the turbulent part at large ℓ. Not
  tabulated here (all numbers are for $L_{\rm out} = 1.2$); the fits use each region's field of view.
- **Far side.** With $a > 0$ the two walls partly cancel in the bulk centroid ($G \propto (1-a)/(1+a)$), and the far side
  adds depth to the turbulent average. At $a = 1$ the bulk term vanishes and $S_2$ is turbulence only.
- **Inner scale.** Going from an inner scale of 0.005 $r_{\rm HII}$ to $k_{\max}\to\infty$ changes the centroid
  index by ≤ 0.02. Sightline averaging already removes scales that small.

## 6. What this means for fitting

- Do not fit $S_2$ with the point-sample model: it is too flat by a factor of 3–6 in the index.
- Do not fix λ (e.g. from the slope of $K$) and then read Δ from $S_2$. A λ wrong by a factor of 3 shifts the
  preferred Δ by a factor of ≈ 3–6 (table in §4). Fit Δ and λ together, as in `fit_slope_lambda.py` (slope of ln K + $S_2$) and the
  joint centroid fit.
- The information on Δ comes from the small-ℓ end of $S_2$ (depth effect). It is limited by the 37″ fibre floor
  (ℓ_min ≈ 0.03–0.1 $r_{\rm HII}$) and by the noise floor, both of which sit exactly there.

## 7. Limits

- The centroid numbers are model calculations (Monte Carlo over 1,500 sightline pairs per ℓ, 240 depth cells). They
  have not yet been checked against a direct simulation of fibre centroids from the grid-free field.
- Scalar Gaussian turbulence; uniform emission in the shell; no extinction gradient beyond the far-side weight $a$.
- Point-like fibres (the fibre footprint is an option of the joint fit, `--beam`, not used here).
