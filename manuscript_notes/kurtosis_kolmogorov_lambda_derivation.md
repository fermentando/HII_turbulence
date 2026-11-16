# Pair kurtosis of an expanding shell with correlated Kolmogorov turbulence

Extension of the bulk-limit derivation (`kurtosis_bulk_limit_derivation.md`,
`kurtosis_bulk_limit_constants.md`) to finite $\lambda = v_{\rm bulk}/v_{\rm turb}$, for the
correlated Gaussian Kolmogorov field produced by `kolmogorov_velocity_grid`. Working note; not yet
manuscript text.

Scripts (`manuscript_pipeline/04_flame_kurtosis_3d/`):

| Script | Content |
|---|---|
| `test_kolmogorov_limits_low_lambda.py` | current model (shell autocorrelation $C$, Abel projection, three choices of $D$) vs the 8-seed runs |
| `test_kolmogorov_model_low_lambda.py` | earlier pair-class model vs the 8-seed runs and the $N = 5{,}000$ sweep at $\lambda = 1$ |
| `compare_kolmogorov_model_lambda10.py` | model vs the original 14-width $\lambda = 10$ runs; exact plateau integrals |
| `derive_kurtosis_shape_kolmogorov.py` | earlier pair-class (same-wall / opposite-wall) model; old internal names: `kappa` = $K_b$, `S` = $D$, `mu` = $\mu_0$ |
| `kurtosis_vratio_seed_ensemble.py` | 8-field-seed pair moments ($N = 30{,}000$, all pairs) used as the test data |

## Inputs and what is measured

All lengths are in units of $r_{\rm HII}$; the shell is $1-\Delta \le r \le 1$.

| Input | Meaning |
|---|---|
| $\Delta$ | shell thickness, $\Delta r_{\rm shell}/r_{\rm HII}$ |
| $\lambda$ | $v_{\rm bulk}/v_{\rm turb}$ |
| $k_{\min}$, $k_{\max}$ | turbulence outer and inner scales ($2\pi/k_{\min}$, $2\pi/k_{\max}$), in units of $1/r_{\rm HII}$; they set $D(s)$ |
| $\ell$ | sky separation of a pair, the variable $K$ is evaluated at |

Only sky positions $(y, z)$ and line-of-sight velocities are measured, and pairs are binned by
$\ell$ alone. 3D positions appear below only as integration variables, summed over every position
in the shell, so every result is a function of $\ell$, $\Delta$, $\lambda$ and the turbulence
scales.

## Notation

| Symbol | Meaning |
|---|---|
| $x$; $(y, z)$; $\rho$ | line of sight; sky plane; sky radius |
| $\mathbf r$, $r = \lvert\mathbf r\rvert$ | 3D position of a point (integration variable); distance from the centre |
| $\mathbf 1_{\rm shell}(\mathbf r)$ | 3D shell indicator: 1 if $1-\Delta \le \lvert\mathbf r\rvert \le 1$, else 0 |
| $\mathbf d = (d_x, \mathbf d_\perp)$, $s = \lvert\mathbf d\rvert$ | 3D separation of a pair; $\lvert\mathbf d_\perp\rvert = \ell$, $d_x$ = depth difference |
| $\mu = x/r$ | line-of-sight cosine of a point (bulk velocity in units of $v_{\rm bulk}$) |
| $\mu_0 = \sqrt{1-\rho^2}$ | sightline label: cosine at the outer surface |
| $n(\rho)$, $m_2(\rho)$, $m_4(\rho)$ | column (chord length) and chord moments of $\mu$ (arctan closed forms, derivation note) |
| $g(\mathbf r)$ | turbulent line-of-sight velocity field, unit variance (units of $v_{\rm turb}$) |
| $\xi(s)$, $D(s) = 2[1-\xi(s)]$ | correlation function of $g$; variance of $\delta g$ at 3D separation $s$ |
| $W(\ell)$ | number of pairs at sky separation $\ell$ |
| $\langle\,\cdot\,\rangle_\ell$ | average over all pairs at sky separation $\ell$ (pair sum divided by $W$) |
| $K_b(\ell)$ | bulk-limit kurtosis ($\lambda = \infty$) |
| $K_t(\ell)$ | turbulence-only kurtosis ($\lambda = 0$) |

## 1. The formula

For a pair, $\delta v = \delta g + \lambda\,\delta\mu$ (units of $v_{\rm turb}$), with
$\delta g = g_i - g_j$, $\delta\mu = \mu_i - \mu_j$.

**Condition on the pair's 3D positions.** $\delta\mu$ is then fixed; $\delta g$ is a difference of
two jointly Gaussian unit-variance values, so it is Gaussian with mean 0 and variance
$1 + 1 - 2\xi(s) = D(s)$. With $E[\delta g] = E[\delta g^3] = 0$, $E[\delta g^2] = D$,
$E[\delta g^4] = 3D^2$:

$$E[\delta v^2] = D + \lambda^2\delta\mu^2, \qquad E[\delta v^4] = 3D^2 + 6\lambda^2D\,\delta\mu^2 + \lambda^4\delta\mu^4 .$$

**Average over all pairs at sky separation $\ell$.** $K$ is the ratio of pooled moments, so

$$
\boxed{K(\ell;\lambda) = \frac{3\,\langle D^2\rangle_\ell + 6\lambda^2\,\langle D\,\delta\mu^2\rangle_\ell + \lambda^4\,\langle\delta\mu^4\rangle_\ell}
{\Big[\langle D\rangle_\ell + \lambda^2\,\langle\delta\mu^2\rangle_\ell\Big]^2}}
$$

This is exact for Gaussian turbulence. Everything that remains is a geometric average over pairs.

## 2. The shell's 3D autocorrelation (exact, closed form)

$$C(s;\Delta) = \int_{\text{all }\mathbf r} d^3r\;\mathbf 1_{\rm shell}(\mathbf r)\,\mathbf 1_{\rm shell}(\mathbf r + \mathbf d)$$

is the volume of positions at which a pair with separation $\mathbf d$ has both points in the shell,
i.e. the number of pairs at 3D separation $s$. It depends only on $s$ because the shell is
spherically symmetric, and no individual depth is needed: $\mathbf r$ runs over all of space.

Writing the shell as "ball of radius 1 minus ball of radius $1-\Delta$" and multiplying out:

$$C(s;\Delta) = V(1,1;s) - 2\,V(1,1-\Delta;s) + V(1-\Delta,1-\Delta;s),$$

where $V(R, R'; s)$ is the overlap volume of two balls of radii $R$, $R'$ whose centres are $s$
apart (only the radii 1 and $1-\Delta$ appear):

$$
V(R,R';s) =
\begin{cases}
\tfrac43\pi\min(R,R')^3 & s \le \lvert R-R'\rvert \quad\text{(small ball inside large)}\\[3pt]
\dfrac{\pi(R+R'-s)^2\big[s^2 + 2s(R+R') - 3(R-R')^2\big]}{12\,s} & \lvert R-R'\rvert < s < R+R' \quad\text{(lens)}\\[3pt]
0 & s \ge R+R' \quad\text{(no overlap)}
\end{cases}
$$

For equal radii the lens reduces to $V(R,R;s) = \tfrac{\pi}{12}(2R-s)^2(s+4R)$. For a thin shell
and $s \gg \Delta$, $C(s) \to 2\pi\Delta^2/s$, the sphere autocorrelation used for $C_d$.

## 3. The turbulent structure function

$$D(s) = 2\left[1 - \frac{\int_{k_{\min}}^{k_{\max}} k^{-5/3}\,\dfrac{\sin ks}{ks}\,dk}{\int_{k_{\min}}^{k_{\max}} k^{-5/3}\,dk}\right],$$

with $\sin(ks)/(ks)$ the angular average of $e^{i\mathbf k\cdot\mathbf s}$. For the generator
($k_{\min} = 5.24$, $k_{\max} = 73.3$; outer scale $\approx 1.2$, inner scale $\approx 0.086$):
$D \approx 181\,s^2$ for $s \lesssim 0.02$ and $D = 2$ for $s \gtrsim 0.4$. The inertial range is
too short ($k_{\max}/k_{\min} = 14$) for a clean $s^{2/3}$ law.

## 4. Turbulent averages (exact): line-of-sight projection of $C$

Pairs at sky separation $\ell$ have unknown depth difference $d_x$ and 3D separation
$s = \sqrt{\ell^2 + d_x^2}$. Summing over all $d_x$ (an Abel projection):

$$
\langle D^k\rangle_\ell = \frac{\displaystyle\int_\ell^2 \frac{C(s;\Delta)\,D(s)^k\,s\,ds}{\sqrt{s^2-\ell^2}}}
{\displaystyle\int_\ell^2 \frac{C(s;\Delta)\,s\,ds}{\sqrt{s^2-\ell^2}}}, \qquad k = 1, 2,
\qquad K_t(\ell) = \frac{3\langle D^2\rangle_\ell}{\langle D\rangle_\ell^2}.
$$

The denominator is $W(\ell)$. It equals the 2D autocorrelation of the projected column map, so it
can be computed on the sky; the turbulent sums cannot, because $D$ depends on the 3D separation,
which the column map has lost. $C$ already contains same-wall pairs (the peak at $s \lesssim \Delta$),
opposite-wall pairs (the broad part out to $s = 2$), the limb, and foreshortening, so no further
approximation is involved.

$K_t$ is the kurtosis of a mixture of Gaussians with variances $D(s)$, weighted by how many pairs at
sky separation $\ell$ have 3D separation $s$. It exceeds 3 because the shell supplies both close
pairs (same wall, small $D$) and distant ones (across the cavity, $D \approx 2$).

## 5. Bulk averages (exact): sky-plane integrals of the column

For a pair with sky positions $\boldsymbol\rho$ and $\boldsymbol\rho' = \boldsymbol\rho + \ell\,\hat{\mathbf e}$:

$$
\langle\delta\mu^2\rangle_\ell = \frac{\iint n\,n'\,(m_2 + m_2')}{\iint n\,n'}, \qquad
\langle\delta\mu^4\rangle_\ell = \frac{\iint n\,n'\,(m_4 + m_4' + 6\,m_2m_2')}{\iint n\,n'},
\qquad K_b(\ell) = \frac{\langle\delta\mu^4\rangle_\ell}{\langle\delta\mu^2\rangle_\ell^2},
$$

with $\iint = \int d^2\rho\int d\hat{\mathbf e}$. In the thin-shell limit,
$\langle\delta\mu^2\rangle \approx 2/K_b$ and $\langle\delta\mu^4\rangle \approx 4/K_b$
(checked: $\langle\delta\mu^2\rangle K_b/2 = 0.93$–$1.09$ for $\ell \le 0.2$).

## 6. Cross term (approximate)

Only opposite-wall pairs carry $\delta\mu^2$, with $\delta\mu \approx \pm 2\mu_0$, and their 3D
separation is $\approx\sqrt{4\mu_0^2 + \ell^2}$. With the thin-shell pair weight $d\mu_0/\mu_0$:

$$\langle D\,\delta\mu^2\rangle_\ell \simeq \frac{4}{K_b(\ell)}\int_0^1 \mu_0\,D\!\left(\sqrt{4\mu_0^2 + \ell^2}\right)d\mu_0 .$$

If $D = 2$ everywhere this gives $4/K_b = 2\langle\delta\mu^2\rangle$, as it should.

**Exact form (not yet computed).** $\delta\mu$ depends on where each point is, not only on
$\mathbf d$, so the cross term needs the depth-resolved, direction-dependent correlation

$$\langle D\,\delta\mu^2\rangle_\ell = \frac{\int dd_x\,\langle B_2(\mathbf d)\rangle_\phi\,D(\lvert\mathbf d\rvert)}{W(\ell)},
\qquad B_2(\mathbf d) = \int_{\text{all }\mathbf r} d^3r\;\mathbf 1_{\rm shell}(\mathbf r)\,\mathbf 1_{\rm shell}(\mathbf r + \mathbf d)\,\big[\mu(\mathbf r) - \mu(\mathbf r + \mathbf d)\big]^2,$$

with $\langle\cdot\rangle_\phi$ the average over the sky direction of $\mathbf d_\perp$. $B_2$ is
axisymmetric about the line of sight (it depends on $s$ and on the angle of $\mathbf d$ to the line
of sight), so it has no closed form like $C$, but it is a 2D table over $(d_x, \ell)$.

**Recommended combination.** Until the exact cross term is computed, use the thin-shell forms for
both §5 and §6 ($\langle\delta\mu^2\rangle = 2/K_b$, $\langle\delta\mu^4\rangle = 4/K_b$, with $K_b$
exact). The approximate cross term is built in that normalisation; combining it with the exact §5
averages breaks the consistency of $\chi$ and is slightly worse for thick shells (see §9).

## 7. Mixing form

With $f$ the bulk share of the variance,

$$K = K_t(1-f)^2 + 6\chi\,f(1-f) + K_b\,f^2, \qquad f = \frac{\lambda^2\langle\delta\mu^2\rangle}{\langle D\rangle + \lambda^2\langle\delta\mu^2\rangle}.$$

| Quantity | Definition | Meaning |
|---|---|---|
| $K_t$ | $3\langle D^2\rangle/\langle D\rangle^2$ | turbulence only ($\lambda = 0$); exact |
| $K_b$ | $\langle\delta\mu^4\rangle/\langle\delta\mu^2\rangle^2$ | expansion only ($\lambda = \infty$); exact |
| $\chi$ | $\langle D\delta\mu^2\rangle/(\langle D\rangle\langle\delta\mu^2\rangle)$ | how turbulent the bulk-carrying pairs are relative to the average pair |
| $\lambda_c$ | $\sqrt{\langle D\rangle/\langle\delta\mu^2\rangle}$ | where $f = \tfrac12$; $\approx 1.6$–1.7 for this field |

$\chi > 1$ (the opposite-wall pairs carry both the largest $D$ and all the $\delta\mu$) makes $K$
overshoot both endpoints at $\lambda \approx 2$; the interior maximum of $K(f)$ is at
$f^* = (3\chi - K_t)/(6\chi - K_t - K_b)$, which requires $3\chi > K_t, K_b$. Since $f$ is nearly
independent of $\ell$ and $\Delta$, the transition is logistic in $\ln\lambda$:
$f \approx 1/[1 + (\lambda_c/\lambda)^2]$, spanning about a factor 9 in $\lambda$ from $f = 0.1$ to
$0.9$.

## 8. Closed-form limits

**Bulk limit, thin shell** (derivation and constants notes):

$$K_b \simeq \begin{cases}\tfrac32 + 2\ln 2 - \ln(2\Delta) & \ell \ll \Delta\\ 3\ln 2 - \ln(2\ell) & \Delta \ll \ell \ll 1\end{cases},
\qquad \ell_b = 2e^{-3/2}\Delta \approx 0.45\,\Delta .$$

**Large $\ell$.** Every pair is beyond the correlation length, $D = 2$, so $K_t = 3$, $\chi = 1$ and
$\langle\delta\mu^2\rangle \approx 2/K_b$:

$$K = 3 + (K_b - 3)\left(\frac{\lambda^2}{K_b + \lambda^2}\right)^2 .$$

Curves for all $\lambda$ cross where $K_b = 3$, near $\ell \approx 0.2$.

**Plateau ($\ell \to 0$), sightline integrals.** At $\ell = 0$ both points share a sightline and
every average is a single integral over sightlines with pair weight $n^2\rho\,d\rho$:

$$W = \int n^2\rho\,d\rho,\quad Q_2 = \int n^2(2m_2)\rho\,d\rho,\quad Q_4 = \int n^2(2m_4 + 6m_2^2)\rho\,d\rho,$$

$$G_k = \int n^2\,\tfrac12\big[\langle D^k\rangle_{\rm opp} + \langle D^k\rangle_{\rm same}\big]\rho\,d\rho,\qquad
X = \int n^2(2m_2)\langle D\rangle_{\rm opp}\,\rho\,d\rho,$$

$$K_{\rm plat}(\lambda) = \frac{W\,[3G_2 + 6\lambda^2X + \lambda^4Q_4]}{[G_1 + \lambda^2Q_2]^2}.$$

Here the sightline at $\rho$ crosses a front wall $x \in [x_1, x_2]$ and a back wall
$x \in [-x_2, -x_1]$ ($x_1 = \sqrt{\max(a^2-\rho^2,0)}$, $x_2 = \mu_0$). Same-wall pairs are
$\lvert x_i - x_j\rvert$ apart (triangular on $[0, x_2-x_1]$, a difference of two uniforms);
opposite-wall pairs are $\lvert x_i\rvert + \lvert x_j\rvert$ apart (triangular on $[2x_1, 2x_2]$,
a sum of two uniforms, peaked at $x_1 + x_2$). Grazing sightlines ($\rho > a$) have one chord, split
at $x = 0$. Checked against exact pair sums:

| $\Delta$ | | $\lambda=0$ | 1 | 2 | 3 | 5 | 10 | $\infty$ |
|---|---|---|---|---|---|---|---|---|
| 0.02 | formula | 5.42 | 5.86 | 6.14 | 6.18 | 6.15 | 6.12 | 6.10 |
| | pair sums | 5.40 | 5.85 | 6.17 | 6.24 | 6.24 | 6.21 | 6.20 |
| 0.09 | formula | 4.22 | 4.53 | 4.70 | 4.70 | 4.65 | 4.61 | 4.59 |
| | pair sums | 4.22 | 4.54 | 4.71 | 4.70 | 4.64 | 4.59 | 4.57 |
| 0.3 | formula | 3.61 | 3.76 | 3.74 | 3.65 | 3.54 | 3.48 | 3.45 |
| | pair sums | 3.59 | 3.77 | 3.79 | 3.72 | 3.64 | 3.58 | 3.55 |

(The ~0.1 offsets at $\lambda \ge 5$ for $\Delta = 0.02$ and 0.3 are already present at
$\lambda = \infty$: finite points and binning, not a $\lambda$ error.)

**Thin-shell plateau with this field ($\Delta \to 0$).** Same-wall pairs fully correlated, opposite
walls $2\mu_0$ apart:

$$K_{\rm plat} \simeq K_b\,\frac{3J_2 + 24J_3\lambda^2 + 4\lambda^4}{(J_1 + 2\lambda^2)^2},\qquad
J_1 = \int_0^1\frac{D(2\mu_0)}{\mu_0}d\mu_0 = 5.67,\ \
J_2 = \int_0^1\frac{D(2\mu_0)^2}{\mu_0}d\mu_0 = 9.49,\ \
J_3 = \int_0^1 \mu_0 D(2\mu_0)\,d\mu_0 = 0.995 .$$

So $K_t(0) = 0.88\,K_b$ and $\lambda_c = \sqrt{J_1/2} = 1.68$, independent of $\Delta$. Valid only for
$\Delta \lesssim 0.003$ with this field (inner scale 0.086).

### Idealised turbulence limits

**Scale-free Kolmogorov** (inner scale $\ll \Delta$, outer scale $\gg 2\,r_{\rm HII}$):
$D = A s^{2/3}$, and the turbulence reduces to one velocity, $\sigma_1^2 = v_{\rm turb}^2A$ (the
turbulent velocity difference across one radius). With $\Lambda = v_{\rm bulk}/\sigma_1$ all
averages are geometric moments $\langle s^p\rangle_\ell$ of the Abel-projected $C$, and
$K_t = 3\langle s^{4/3}\rangle/\langle s^{2/3}\rangle^2$ depends on $\Delta$ and $\ell$ only. On a thin
shell, same-wall pairs drop out ($\propto \Delta^{1/3}$) and

$$K_{\rm plat}(\Lambda) = K_b\left[1 + \frac{3\cdot 2^{2/3}\Lambda^2}{\big(\tfrac32\,2^{2/3} + 2\Lambda^2\big)^2}\right],$$

so $K_t = K_b$ (for any power law $D \propto s^p$, $K_t/K_b = 3p/2$, which equals 1 for $p = 2/3$),
with a maximum of $1.25\,K_b$ at $\Lambda^* = (3/4)^{1/2}2^{1/3} \approx 0.97$. Numerically
$K_t/K_b = 1.006, 1.026, 1.055, 1.12, 1.24$ for $\Delta = 0.001, 0.01, 0.03, 0.1, 0.3$.

**Uncorrelated turbulence** (outer scale $\ll \Delta$): $D = 2$, $K_t = 3$, $\chi = 1$, and the
large-$\ell$ form above holds at all $\ell$.

**Neither limit describes the simulations** (§9): the generator's outer scale ($\approx 1.2$) is
comparable to the region and its inner scale ($\approx 0.086$) to thin-shell walls. Kolmogorov
turbulence is therefore described by two ratios, outer and inner scale over $r_{\rm HII}$, not one.
The one-parameter scale-free model would be justified for a real region only if its driving scale
were much larger than the region.

## 9. Validation

**Turbulent averages vs exact pair sums** (Abel projection of $C$, generator $D$):

| $\Delta$ | $\ell$ | $\langle D\rangle$: Abel / pair sums | $K_t$: Abel / pair sums |
|---|---|---|---|
| 0.02 | 0.009 | 0.973 / 0.970 | 5.14 / 5.18 |
| | 0.03 | 1.177 / 1.175 | 4.22 / 4.22 |
| | 0.095 | 1.543 / 1.544 | 3.33 / 3.33 |
| 0.09 | 0.003 | 1.204 / 1.205 | 4.22 / 4.20 |
| | 0.03 | 1.268 / 1.267 | 3.96 / 3.97 |
| 0.3 | 0.03 | 1.499 / 1.503 | 3.53 / 3.53 |
| 1.0 | 0.003 | 1.625 / 1.621 | 3.41 / 3.41 |
| | 0.095 | 1.722 / 1.722 | 3.20 / 3.20 |

Agreement to about 0.005 at every $\ell$ and $\Delta$, including $\Delta = 1$.

**Full formula vs the 8-seed ensemble** ($N = 30{,}000$, all pairs, 8 field seeds pooled):
rms(data $-$ formula) over $\ell \le 0.6$, thin-shell §5/§6 combination:

| $\Delta$ | $\lambda = 0.5$ | 1 | 2 | 10 |
|---|---|---|---|---|
| 0.02 | 0.12 | 0.15 | 0.22 | 0.15 |
| 0.09 | 0.09 | 0.07 | 0.14 | 0.09 |
| 0.3 | 0.05 | 0.05 | 0.05 | 0.06 |
| 1.0 | 0.05 | 0.03 | 0.04 | 0.04 |

With the exact §5 averages instead, the rms is the same for thin shells and slightly larger for thick
ones (e.g. 0.10 vs 0.03 at $\Delta = 1$, $\lambda = 1$), because of the inconsistency with the
approximate cross term. Largest residual: thin shells at $\lambda \approx 2$, plateau under-predicted
by 0.2–0.4 (6.11 vs 6.49 at $\Delta = 0.02$; 4.70 vs 4.87 at $\Delta = 0.09$). This is **not** the
cross-term approximation: it is the odd terms of the 8-seed realisation, which the ensemble formula
correctly sets to zero (§10).

**Idealised limits vs the same data** (rms over $\ell \le 0.6$):

| $\Delta$ | $\lambda$ | generator $D(s)$ | scale-free $A s^{2/3}$ ($A = 5.86$) | uncorrelated $D = 2$ |
|---|---|---|---|---|
| 0.02 | 0.5 / 1 / 2 | 0.12 / 0.15 / 0.22 | 0.87 / 0.97 / 1.17 | 1.62 / 1.83 / 1.85 |
| 0.09 | 0.5 / 1 / 2 | 0.09 / 0.07 / 0.14 | 0.91 / 1.00 / 1.18 | 0.90 / 1.09 / 1.13 |
| 0.3 | 0.5 / 1 / 2 | 0.05 / 0.05 / 0.05 | 0.71 / 0.92 / 1.23 | 0.52 / 0.60 / 0.53 |
| 1.0 | 0.5 / 1 / 2 | 0.05 / 0.03 / 0.04 | 0.56 / 0.88 / 1.32 | 0.38 / 0.38 / 0.25 |

The scale-free model is too high everywhere (no saturation of $D$; too turbulent within walls), and
changing $A$ only rescales $\lambda$, so it cannot fix $K_t$. Uncorrelated turbulence is too low
($K_t = 3$ instead of 3.4–5.4).

**Original 14-width runs at $\lambda = 10$** (field seed 7, pipeline estimator, pairs from a
10,000-point subsample): the model is within 0.02–0.04 of $K_b$ at $\lambda = 10$, so these runs test
the thickness result. Agreement is good for $\Delta \gtrsim 0.1$ (curve rms 0.08–0.17). For
$\Delta \lesssim 0.07$ the data are 0.2–0.5 below the formula on the plateau (5.62 vs 6.12 at
$\Delta = 0.02$). This is from the runs, not the formula: the 10k-point subsample leaves the
thin-shell plateau bins with few pairs (sample kurtosis biased low for heavy tails), and seed 7 is a
low realisation (with all pairs: 6.05 for seed 7, 6.30 pooled over 8 seeds).

**Single realisations.** The formula gives the ensemble mean. At $\lambda \lesssim 2$ one
realisation of this field scatters by about $\pm 0.2$ (thick shells) to $\pm 0.8$ (thin shells) on the
plateau, because the spectrum has only a few independent modes on the scale of the shell
($k_{\min} = 2k_{\rm box}$). Seed 7, used in all the original runs, sits low (e.g. plateau 3.66 vs
4.50 pooled at $\Delta = 0.09$, $\lambda = 1$), which also explains why the original
$N = 5{,}000$ sweep is below the formula for thin shells at $\lambda = 1$ (rms 0.5–0.7).

## 10. Odd terms and single sources

**Where they come from.** §1 conditions on the pair positions and uses $E[\delta g] = E[\delta g^3] = 0$.
That holds for the ensemble, but a single field realisation has nonzero pair sums
$\langle\delta g\,\delta\mu\rangle$, $\langle\delta g^3\delta\mu\rangle$ and $\langle\delta g\,\delta\mu^3\rangle$,
which enter the realised moments with odd powers of $\lambda$:

$$\langle\delta v^2\rangle = \langle\delta g^2\rangle + 2\lambda\langle\delta g\,\delta\mu\rangle + \lambda^2\langle\delta\mu^2\rangle,$$
$$\langle\delta v^4\rangle = \langle\delta g^4\rangle + 4\lambda\langle\delta g^3\delta\mu\rangle + 6\lambda^2\langle\delta g^2\delta\mu^2\rangle + 4\lambda^3\langle\delta g\,\delta\mu^3\rangle + \lambda^4\langle\delta\mu^4\rangle .$$

**Diagnosis of the thin-shell residual** (plateau bins $\ell \le \ell_b/4$, at least five bins;
8 seeds pooled; script `plot_lambda_model_summary.py` for the bins):

| $\Delta = 0.02$ | $\lambda=0$ | 1 | 2 | 3 | 5 | 10 | $\infty$ |
|---|---|---|---|---|---|---|---|
| simulation, all terms | 5.42 | 6.07 | 6.49 | 6.53 | 6.43 | 6.28 | 6.11 |
| simulation, odd terms removed | 5.42 | 5.85 | 6.15 | 6.19 | 6.16 | 6.13 | 6.11 |
| exact Gaussian-ensemble pair sums | 5.41 | 5.86 | 6.15 | 6.19 | 6.16 | 6.13 | 6.11 |
| formula | 5.35 | 5.79 | 6.09 | 6.13 | 6.11 | 6.08 | 6.06 |

- Without the odd terms the simulation matches the exact ensemble sums to $\le 0.04$, and the formula to
  $\approx 0.06$, for $\Delta = 0.02$, 0.09 and 0.3 at every $\lambda$.
- Flipping the sign of $\lambda$ mirrors the excess: at $\Delta = 0.02$, $K(+3) = 6.53$ and $K(-3) = 5.84$,
  symmetric about the even-only 6.19.
- The per-seed excess has a random sign (at $\lambda = 3$, $\Delta = 0.02$: $-0.30, +0.57, -0.09, +1.19,
  +1.07, -0.13, +0.24, -0.06$); two seeds carry most of the pooled excess. The per-seed correlation
  $\langle\delta g\,\delta\mu\rangle/\sqrt{\langle\delta g^2\rangle\langle\delta\mu^2\rangle}$ scatters by $\pm 0.15$
  about zero.
- All widths share the same 8 field seeds and the same points, so the excess has the same sign at
  every width and $\lambda$ and looks systematic although it is not.
- It peaks at $\lambda \approx 2$–5 because the odd terms need both components: they vanish at
  $\lambda = 0$ (no bulk) and are suppressed at $\lambda \gg 1$ (turbulence negligible).

For the ensemble comparison, pool each seed with its mirror field $g \to -g$ (an equally likely
realisation of the Gaussian field): the odd sums cancel exactly and the even sums are unchanged.

**What it means physically.** A real H II region is one realisation, so the odd terms are real for an
individual source. A nonzero $\langle\delta g\,\delta\mu\rangle$ means the turbulent velocity is, on
average, different on the near side of the shell than on the far side. In the simulation this
happens by chance alignment of the largest modes ($k_{\min}$, scale $\approx 1.2\,r_{\rm HII}$) with the
line of sight. In a real nebula it is probably more likely: turbulence driven at the shell scale puts
its power in exactly those modes, and physical asymmetries (expansion into a density gradient, a
champagne flow, a blown-out wall) produce a front–back velocity offset systematically rather than
by chance. Observationally it is close to an asymmetric expansion, with the approaching and receding
walls at different speeds.

**Size for one source.** The single-seed scatter of $K_{\rm plat}$ (standard deviation over the 8 seeds;
even and odd terms together):

| $\Delta$ | $\lambda=0$ | 1 | 2 | 3 | 5 | 10 | $\infty$ |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.40 | 0.76 | 0.82 | 0.68 | 0.45 | 0.23 | 0 |
| 0.09 | 0.20 | 0.48 | 0.51 | 0.42 | 0.28 | 0.14 | 0 |
| 0.3 | 0.14 | 0.20 | 0.25 | 0.23 | 0.17 | 0.08 | 0 |

- Scatter exists at every finite $\lambda$, including $\lambda = 0$ (one field does not sample its own
  structure function perfectly). The odd terms add to it, and the total peaks at $\lambda \approx 1$–3,
  where turbulence and bulk are comparable. Single seeds at $\Delta = 0.02$, $\lambda = 3$ range from 5.7
  to 7.8 about the ensemble 6.2.
- It matters most for thin shells. Since $K_{\rm plat} \simeq C_p - \ln 2\Delta$, an error of 0.7 in
  $K_{\rm plat}$ is a factor $e^{0.7} \approx 2$ in the inferred $\Delta$: a single thin shell with
  $v_{\rm bulk} \sim v_{\rm turb}$ cannot have its thickness pinned down better than about a factor of 2 from
  $K_{\rm plat}$ alone, however good the data.

**Consequences.**

1. The formula is the expected value over realisations, not what one source will show. Inference for
   individual regions should include a realisation-variance term, e.g. $\sigma(\Delta, \lambda)$ from the
   table above, interpolated.
2. The scatter vanishes as $\lambda \to \infty$: sources with clearly $v_{\rm bulk} \gg v_{\rm turb}$ (clean
   line splitting) give a nearly purely geometric, reliable $K_{\rm plat}$; sources near $\lambda \sim 1$–3
   are the uncertain ones.
3. The amplitude depends on the simulated turbulence (Gaussian field, $k_{\min} = 5.24$, outer scale
   $\approx 1.2\,r_{\rm HII}$). Driving on larger scales, or intermittent turbulence, would probably
   increase it, so the table is a guide, not a universal number.

## 11. Status and open items

| Term | Status | Checked against |
|---|---|---|
| $\langle D\rangle$, $\langle D^2\rangle$, so $K_t$ | exact for any $\Delta$ (Abel projection of $C$) | pair sums, to $\approx 0.005$ |
| $\langle\delta\mu^2\rangle$, $\langle\delta\mu^4\rangle$, so $K_b$ | exact (sky-plane integrals); thin-shell forms $2/K_b$, $4/K_b$ recommended for now | pipeline estimator; $C_p$, $C_d$ derived |
| $\langle D\delta\mu^2\rangle$ | thin-shell approximation | exact Gaussian-ensemble sums to $\approx 0.06$; the 0.2–0.4 thin-shell residual at $\lambda \approx 2$ is realisation odd terms, not this term (§10) |
| The whole $K$ | ensemble mean | 8-seed runs: rms 0.03–0.22 for $\lambda = 0.5$–10 |

Still to do:

- the exact cross term from $B_2(\mathbf d)$, which makes every term exact and allows the exact §5
  averages throughout;
- the realisation variance of $K(\ell)$ beyond the 8-seed estimate in §10 (more seeds, other
  $k_{\min}$), needed to compare a single region with the model;
- re-running the thinnest $\lambda = 10$ widths with all pairs (no 10k subsample);
- the pixelised version (one mean velocity per sightline), required before applying any of this to
  LVM maps.
