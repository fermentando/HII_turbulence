<!-- Copy of geometric_averages/06_break_and_slope_tangent.md: analytic formula used to fit K(l; lambda). Derivation: kurtosis_K_lambda_derivation.md. -->

# Break and post-break slope of K(ℓ): a simple analytic prediction

**What this note gives.** Closed formulas for where the pair-kurtosis curve $K(\ell)$ bends and how steeply it
falls afterwards, for any ratio $\lambda = v_{\rm bulk}/v_{\rm turb}$. They are checked against the grid-free
simulations: for thin shells ($\Delta \lesssim 0.1$) they agree within the scatter a single source would show, from
pure turbulence ($\lambda = 0$) to pure expansion ($\lambda = \infty$).

**The idea in one sentence.** $K(\ell)$ looks like three straight pieces: a flat plateau, a falling line and a floor
at 3. The bend is where the falling line, drawn as the tangent at $\ell = \Delta$, meets the plateau.

## 1. Setting and symbols

An expanding spherical shell of radius $r_{\rm HII} = 1$ and thickness $\Delta$ (all lengths in units of $r_{\rm HII}$),
with Gaussian Kolmogorov turbulence. The velocity is measured at random points, with no pixelisation. $K(\ell)$ is the
kurtosis of the velocity differences of all pairs separated by $\ell$ on the sky.

| symbol | meaning | value used here |
|---|---|---|
| $\Delta$ | shell thickness | |
| $\lambda$ | $v_{\rm bulk}/v_{\rm turb}$ | |
| $\ell$ | sky separation of a pair | |
| $A$ | Kolmogorov amplitude: the turbulent velocity difference squared grows as $D(s) \simeq A\,s^{2/3}$ | 4.85 |
| $J$, $J_2$ | "how turbulent the pairs are" on average: $J = \int_0^2 D(s)\,ds/s$, $J_2 = \int_0^2 D(s)^2\,ds/s$ | 6.85, 10.53 |
| $\tilde J$, $\tilde J_2$ | the same with the shell-thickness correction: $\tilde J = J - \tfrac{27}{40}A\Delta^{2/3}$, $\tilde J_2 = J_2 - \tfrac{27}{140}A^2\Delta^{4/3}$ | |
| $F_d$ | how much turbulence dilutes the fall of $K$: $F_d = 3J_2/J^2$ at $\lambda = 0$ | 0.673 |
| $\ell_0$ | separation beyond which every pair is uncorrelated and $K$ sits at 3 | 0.44 |

Values are for the simulated turbulence: outer scale $2\pi/k_{\min} = 1.2\,r_{\rm HII}$, no inner cutoff.

## 2. What the curve looks like, and why

1. **Plateau** ($\ell \ll \Delta$). Pairs on the same sightline. Some are on the same wall and close in 3D; others are
   across the cavity and far apart. Mixing these very different pairs gives a high kurtosis.
2. **Fall** ($\ell \gtrsim \Delta$). As $\ell$ grows, the close same-wall pairs disappear, so the mixture becomes less
   extreme and $K$ drops.
3. **Floor at 3** (turbulence only, $\ell \gtrsim \ell_0$). Once every pair is farther apart than the turbulence
   correlation length, all pairs look alike and the velocity differences are Gaussian, so $K = 3$.

**Where the two regimes differ.**
- With pure expansion the fall is a clean logarithm: $K = \ln(4/\ell)$, a straight line in $\log\ell$ with slope
  $-\ln 10$ per dex.
- With turbulence the fall bends over as it approaches 3, because $\ell_0$ lies inside the range of $\ell$ we fit. So
  the slope depends on where it is measured.

The curve is steepest at $\ell \approx \Delta$, which makes it the natural place to measure the slope:
- inside the wall ($\ell < \Delta$) $K$ is still nearly flat;
- outside ($\ell > \Delta$) the fall keeps flattening towards the floor.

## 3. The three ingredients

### 3.1 Plateau height $K_0$

$$
K_0 = \Big[\tfrac32 + \ln\tfrac{2}{\Delta}\Big]\cdot\frac{3\tilde J_2 + 4\lambda^2(\lambda^2+6)}{(\tilde J + 2\lambda^2)^2}
$$

- The first factor counts the pairs. It grows for thinner shells.
- The second factor says how turbulence and expansion share the velocity spread. It equals 1 for pure expansion.

Derived in `kurtosis_K_lambda_derivation.md`. At $\lambda = 0$ it reduces to $K_0 = [\tfrac32 + \ln(2/\Delta)]\,3\tilde J_2/\tilde J^2$.

### 3.2 The fall at λ = 0

**Step 1: the slope of the fall.** The exact turbulent curve has a local slope (per factor $e$ in $\ell$) that is
well described by

$$
\frac{dK}{d\ln\ell} = -F_d\,(1-z)^2,\qquad z = \Big(\frac{\ell}{\ell_0}\Big)^{2/3}.
$$

- **At small $\ell$:** $z \to 0$, and the slope is $-F_d$, the diluted logarithm.
- **As $\ell \to \ell_0$:** $z \to 1$, and the slope goes to zero smoothly, so $K$ reaches 3 without overshooting.
- **The power 2/3:** it comes from the Kolmogorov law $D \propto s^{2/3}$.
- **The square:** $(1-z)^2$ is the simplest polynomial in $z$ that satisfies all of:
  - it equals 1 at $z = 0$, giving the $-F_d$ limit;
  - it equals 0 at $z = 1$, where the slope vanishes and $K$ reaches 3;
  - its derivative is also 0 at $z = 1$, so the curve meets the floor tangentially, with no kink.

This form is an ansatz. It matches the exact slope to within 0.015 and the exact $K$ to within 0.03 at every $\ell$.

**Step 2: integrate it.** Use $d\ln\ell = \tfrac32\,dz/z$, and integrate from $\ell$ out to $\ell_0$, where $K = 3$:

$$
K_{\rm dec}(\ell) = 3 + F_d\Big[\ln\frac{\ell_0}{\ell} - 3(1-z) + \tfrac34(1-z^2)\Big].
$$

**Step 3: fix $\ell_0$ (nothing is fitted).** At small $\ell$, step 2 gives $K_{\rm dec} \to 3 + F_d[\ln(\ell_0/\ell) - \tfrac94]$.
It must also equal the known small-$\ell$ result $F_d\ln(4/\ell)$. Equating the two:

$$
\ell_0 = 4\,\exp\!\Big(\tfrac94 - \tfrac{3}{F_d}\Big) = 0.44 .
$$

### 3.3 Slope and break from the tangent at ℓ = Δ

**Slope.** Evaluate step 1 at $\ell = \Delta$ and convert from per $e$-fold to per dex (multiply by $\ln 10$):

$$
\boxed{S^* = -\ln 10\;F_d\,(1 - z_\Delta)^2},\qquad z_\Delta = \Big(\frac{\Delta}{\ell_0}\Big)^{2/3}.
$$

**Break.** Draw the tangent at $\ell = \Delta$. It passes through $K_{\rm dec}(\Delta)$ with slope $-F_d(1-z_\Delta)^2$
per $e$-fold:

$$
K_{\rm tan}(\ell) = K_{\rm dec}(\Delta) - F_d(1-z_\Delta)^2\,\ln\frac{\ell}{\Delta}.
$$

The break is where this line reaches the plateau, $K_{\rm tan}(\ell_b) = K_0$:

$$
\boxed{\ln\frac{\ell_b}{\Delta} = -\,\frac{K_0 - K_{\rm dec}(\Delta)}{F_d\,(1-z_\Delta)^2}}
$$

The bigger the gap between the plateau and the curve at $\Delta$, and the shallower the slope there, the further
inward the break moves.

**Floor.** The same line reaches $K = 3$ at $\ln(\ell_3/\Delta) = [K_{\rm dec}(\Delta) - 3]/[F_d(1-z_\Delta)^2]$.

### 3.4 Worked example: Δ = 0.09, λ = 0

| step | value |
|---|---|
| $z_\Delta = (0.09/0.44)^{2/3}$ | 0.348 |
| slope at $\Delta$: $-F_d(1-z_\Delta)^2$ | $-0.286$ per $e$-fold, so $S^* = -0.66$ per dex |
| $K_{\rm dec}(\Delta)$ | 3.19 |
| $K_0$ | 3.72 |
| $\ln(\ell_b/\Delta) = -(3.72 - 3.19)/0.286$ | $-1.85$, so $\ell_b = 0.16\,\Delta$ ($\log_{10} = -0.80$) |

Simulation, measured the same way (§5): $S^* = -0.65 \pm 0.21$, $\log_{10}(\ell_b/\Delta) = -0.74 \pm 0.26$.

## 4. The large-λ limit comes out of the same construction

With pure expansion there is no turbulence. The fall is the plain logarithm $K = \ln(4/\ell)$, with slope $-1$ per
$e$-fold, and the plateau is $K_0 = \tfrac32 + \ln(2/\Delta)$. The gap at $\ell = \Delta$ is

$$
K_0 - \ln\frac{4}{\Delta} = \tfrac32 - \ln 2 ,
$$

so

$$
\ln\frac{\ell_b}{\Delta} = -\big(\tfrac32 - \ln 2\big)\;\Rightarrow\; \ell_b = 2e^{-3/2}\Delta = 0.45\,\Delta,
\qquad S = -\ln 10 = -2.30 \text{ per dex}.
$$

These are the earlier bulk-limit results. There, $\ell_b/\Delta$ and $S$ do not depend on $\Delta$.

Turbulence changes two things, which together move the break inward:
- the fall is shallower at $\Delta$;
- the plateau sits relatively higher.

For thicker shells the bulk slope at $\Delta$ is a little shallower than $-2.30$ (§6).

## 5. Any λ in between: a mix of the two limits

### 5.1 The mixing formula

At any separation $\ell$, the curve for intermediate λ is the two limits, weighted by how the velocity variance
splits between turbulence and expansion, plus a cross term:

$$
\boxed{K(\ell;\lambda) = (1-f)^2\,K_t + f(1-f)\,C_\times + f^2\,K_b}
$$

| piece | what it is | formula |
|---|---|---|
| $K_t$ | pure turbulence (λ = 0) | $3 + F_d\big[\ln(\ell_0/\ell) - 3(1-z) + \tfrac34(1-z^2)\big]$ (§3.2) |
| $K_b$ | pure expansion (λ = ∞) | $\simeq \ln(4/\ell)$ |
| $C_\times$ | cross term | $12P/N_1 = \sqrt{48\,P\,K_t/J_2} \simeq 2.1\sqrt{K_t K_b}$ |
| $f$ | bulk share of the velocity variance | $2\lambda^2/(N_1 + 2\lambda^2)$ |
| $N_1$ | how turbulent the pairs at separation $\ell$ are, recovered from $K_t$ | $\sqrt{3J_2\,P/K_t}$ |

Here $P = \ln(4/\ell)$ is the pair count (§2).

### 5.2 Where it comes from

**Step 1: one pair.** In units of $v_{\rm turb}$, a pair's velocity difference is the turbulent part plus the bulk
part, $\delta v = \delta g + \lambda\,\delta\mu$.
- For fixed positions, $\delta\mu$ is a fixed number.
- $\delta g$ is Gaussian with variance $D(s)$, where $s$ is the pair's 3D separation.

Expanding the binomials, with $\langle\delta g\rangle = \langle\delta g^3\rangle = 0$ and $\langle\delta g^4\rangle = 3D^2$:

$$
E[\delta v^2] = D + \lambda^2\delta\mu^2,\qquad E[\delta v^4] = 3D^2 + 6\lambda^2 D\,\delta\mu^2 + \lambda^4\delta\mu^4 .
$$

The 6 is the binomial coefficient of the $\delta g^2(\lambda\delta\mu)^2$ term.

**Step 2: all pairs at separation $\ell$.** The kurtosis is the pooled fourth moment over the pooled variance squared:

$$
K = \frac{3\langle D^2\rangle + 6\lambda^2\langle D\,\delta\mu^2\rangle + \lambda^4\langle\delta\mu^4\rangle}{\big[\langle D\rangle + \lambda^2\langle\delta\mu^2\rangle\big]^2} .
$$

This is exact for Gaussian turbulence.

**Step 3: split by share of the variance.** Call $\sigma_t^2 = \langle D\rangle$ the turbulent variance and
$\sigma_b^2 = \lambda^2\langle\delta\mu^2\rangle$ the bulk variance, and let $f = \sigma_b^2/(\sigma_t^2 + \sigma_b^2)$. Dividing
each numerator term by $(\sigma_t^2+\sigma_b^2)^2$ gives:

| term | becomes | coefficient |
|---|---|---|
| $3\langle D^2\rangle$ | $\big[3\langle D^2\rangle/\langle D\rangle^2\big](1-f)^2$ | $K_t$ |
| $\lambda^4\langle\delta\mu^4\rangle$ | $\big[\langle\delta\mu^4\rangle/\langle\delta\mu^2\rangle^2\big]f^2$ | $K_b$ |
| $6\lambda^2\langle D\,\delta\mu^2\rangle$ | $6\big[\langle D\,\delta\mu^2\rangle/(\langle D\rangle\langle\delta\mu^2\rangle)\big]f(1-f)$ | $C_\times = 6\chi$ |

So far this is an exact rewriting, with no approximation.

**Step 4: fill in the pair counts.** Every average is a weighted pair count divided by $P$:
- $\langle D\rangle = N_1/P$ and $\langle D^2\rangle = N_2/P$;
- $\langle\delta\mu^2\rangle = 2/P$ and $\langle\delta\mu^4\rangle = 4/P$;
- $\langle D\,\delta\mu^2\rangle = 4/P$, because only opposite-wall pairs carry $\delta\mu$, and they are far enough apart
  that $D \approx 2$.

This gives $f = 2\lambda^2/(N_1+2\lambda^2)$, $K_t = 3PN_2/N_1^2$, $K_b = P$ and $C_\times = 12P/N_1$. Taking
$N_2 \approx J_2$ and inverting $K_t$ gives $N_1 = \sqrt{3J_2P/K_t}$. So the only turbulence input is the closed-form λ = 0
curve itself.

**Where the weights come from.** The fourth moment of (turbulent + bulk) velocity differences expands like a binomial:
- $(1-f)^2$: pairs whose spread is mostly turbulent;
- $f^2$: pairs whose spread is mostly bulk;
- $f(1-f)$: pairs with both. These are the opposite-wall pairs, which carry the full bulk velocity difference and are
  also far enough apart that their turbulence is uncorrelated.

Compare two independent variables, whose summed kurtosis is $K_1(1-f)^2 + 6f(1-f) + K_2 f^2$. Here the cross coefficient
is $6\chi$ instead of 6:
- $\chi = 2/\langle D\rangle > 1$, because the pairs that carry the bulk spread also carry the largest turbulent spread.
- This is why simply interpolating between the two limits is too steep: it leaves the cross term out.

**The transition.** $f = \tfrac12$ at $\lambda_c = \sqrt{N_1/2} \approx 1.8$. The turbulent weight $(1-f)^2$ is:

| λ | 0.1 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|
| $(1-f)^2$ | 0.99 | 0.86 | 0.60 | 0.21 | ≈ 0.01 |

That is why the λ = 0 formulas of §3 hold up to about λ ≈ 0.5.

### 5.3 Plateau, slope and break from the mix

1. **Plateau.** At $\ell \to 0$, with $N_1 = \tilde J$, the mix is algebraically identical to the plateau formula $K_0$
   of §3.1. Nothing new is needed.
2. **Slope at $\ell = \Delta$.** Mix the slopes of the three pieces with the same weights:

   $$
   S = \ln 10\,\big[(1-f)^2 K_t' + f(1-f)\,C_\times' + f^2 K_b'\big],
   $$

   with, per $e$-fold:
   - $K_t' = -F_d(1-z_\Delta)^2$;
   - $K_b' \approx -1$, or the κ-corrected slope for thicker shells (about $-0.94$ at Δ = 0.09);
   - $C_\times' = C_\times\cdot\tfrac12\,(K_t'/K_t - 1/P)$.

   $f$ changes only slowly with $\ell$. Including its derivative changes $S$ by at most 0.03, so it can be left out.
3. **Break.** As before, the tangent at $\Delta$ meets the plateau:

   $$
   \ln\frac{\ell_b}{\Delta} = -\,\frac{K_0 - K(\Delta)}{|K'(\Delta)|} .
   $$

### 5.4 Check against the simulations

Slope per dex and $\log_{10}(\ell_b/\Delta)$. The simulations are measured as in §6. The pulls, in brackets, are
(mix − simulation) in units of the single-source σ.

| λ | Δ | simulation | mix of limits |
|---|---|---|---|
| 0.1 | 0.09 | −0.65 ± 0.19, −0.73 ± 0.25 | −0.66, −0.80 [−0.1, −0.3] |
| 0.5 | 0.09 | −0.76 ± 0.24, −0.67 ± 0.24 | −0.77, −0.72 [0.0, −0.2] |
| 1 | 0.02 | −1.43 ± 0.85, −0.48 ± 0.08 | −1.44, −0.46 [0.0, +0.3] |
| 1 | 0.09 | −1.02 ± 0.31, −0.58 ± 0.22 | −1.04, −0.59 [−0.1, −0.1] |
| 2 | 0.09 | −1.51 ± 0.40, −0.46 ± 0.05 | −1.58, −0.46 [−0.2, +0.1] |
| 5 | 0.02 | −2.23 ± 0.83, −0.37 ± 0.26 | −2.17, −0.36 [+0.1, 0.0] |
| 5 | 0.09 | −1.98 ± 0.28, −0.38 ± 0.03 | −2.12, −0.37 [−0.5, +0.4] |

- Within 0.5σ at every λ.
- The one visible gap, at λ = 5 and Δ = 0.09, comes from using the thin-shell bulk slope −1. The κ-corrected value
  removes it.

By comparison, the pure λ = 0 formulas are 1σ off at λ = 1 and fail from λ = 2 onwards. A blend that leaves out the
cross term is systematically too steep (−1.32 instead of −1.02 at λ = 1).

## 6. How well it works

**How the simulations were measured.** In each simulated curve: fit a straight line in a sliding window of
$\pm 0.15$ dex to get the local slope, take the steepest point, and intersect its tangent with the plateau (the mean
of $K$ at $\ell < 0.1\Delta$). The error bars are the spread between single turbulent realisations, i.e. what one
source would show.

**Slope** (per dex), analytic / simulation:

| λ | Δ = 0.02 | Δ = 0.09 | Δ = 0.3 |
|---|---|---|---|
| 0 | −1.18 / −1.07 ± 0.58 | −0.66 / −0.65 ± 0.21 | fails (see below) |
| 0.1 | −1.18 / −1.08 ± 0.59 | −0.67 / −0.65 ± 0.19 | fails |
| 1 | −1.44 / −1.43 ± 0.85 | −1.04 / −1.02 ± 0.31 | −0.73 / −0.61 ± 0.41 |
| 2 | −1.79 / −1.87 ± 1.08 | −1.53 / −1.51 ± 0.40 | −1.32 / −1.08 ± 0.58 |
| 5 | −2.16 / −2.23 ± 0.83 | −2.01 / −1.98 ± 0.28 | −1.69 / −1.61 ± 0.34 |
| ∞ | −2.29 / −2.32 | −2.16 / −2.13 | −1.77 / −1.77 |

**Break**, $\log_{10}(\ell_b/\Delta)$, analytic / simulation:

| λ | Δ = 0.02 | Δ = 0.09 | Δ = 0.3 |
|---|---|---|---|
| 0 | −0.50 / −0.52 ± 0.16 | −0.80 / −0.74 ± 0.26 | fails |
| 0.1 | −0.50 / −0.52 ± 0.16 | −0.79 / −0.73 ± 0.25 | fails |
| 1 | −0.46 / −0.48 ± 0.08 | −0.60 / −0.58 ± 0.22 | −0.78 / −0.87 ± 0.15 |
| 2 | −0.41 / −0.43 ± 0.22 | −0.46 / −0.46 ± 0.05 | −0.51 / −0.61 ± 0.09 |
| 5 | −0.36 / −0.37 ± 0.26 | −0.38 / −0.38 ± 0.03 | −0.38 / −0.45 ± 0.04 |
| ∞ | −0.35 / −0.36 | −0.35 / −0.36 | −0.34 / −0.47 |

**Plateau** $K_0$, analytic / simulation, at λ = 0: 4.40 / 4.27 ± 0.19 (Δ = 0.02), 3.72 / 3.68 ± 0.16 (Δ = 0.09),
3.38 / 3.36 ± 0.08 (Δ = 0.3). It agrees within the scatter at every λ tested.

**The whole curve.** The three straight pieces (plateau $K_0$, tangent at $\Delta$, floor at 3) follow the simulated
$K(\ell)$ with rms 0.05–0.14 for Δ ≤ 0.09 at λ = 0–1, inside the single-source scatter.

**Summary.**
- **Δ ≤ 0.09:** within 0.3σ at every λ, for both slope and break.
- **Δ = 0.3:**
  - fails at λ ≲ 0.1, because the steepest point moves inside the wall;
  - works at λ = 1;
  - at λ ≥ 2 the slope is right, but the break is 0.07–0.13 dex too high. The same offset appears without
    turbulence, so it comes from the thin-shell geometry, not from turbulence.

## 7. Using it on data

- **Measure the same quantities.** In the observed $K(\ell)$, take the steepest local slope, and the break as where
  its tangent meets the plateau.
  - Do not compare with hinge (broken-line) fits: a hinge returns a window-averaged slope and a different break
    (at Δ = 0.09, λ = 0 the hinge gives $-0.44$ and $-1.00$ instead of $-0.66$ and $-0.80$).
  - Skip the few smallest-$\ell$ bins when looking for the steepest point; they are noisy.
- **What each quantity tells you.**
  - **The slope mainly tells you λ:** it runs from about $-0.7$ to $-1.2$ at λ = 0 up to about $-2.2$ at λ = ∞ for thin
    shells.
  - **The plateau mainly tells you Δ.**
  - **The break** checks the two, and gives Δ directly once λ ≳ 5, where $\ell_b \to 0.45\,\Delta$.

## 8. Limits

- Thin shells: at low λ the steepest point must be at $\ell \approx \Delta$, which needs $\Delta \ll \ell_0 \approx 0.44$,
  i.e. $\Delta \lesssim 0.1$. For λ ≳ 1 thicker shells work too.
- The values of $A$, $J$, $J_2$, $F_d$ and $\ell_0$ depend on the turbulence outer scale (here 1.2 $r_{\rm HII}$).
  Recompute them for another outer scale.
- The decline shape $(1-z)^2$ is an ansatz that matches the exact curve; it has not been derived from first principles.
- These are ensemble means. A single source scatters as in the tables above.
- Random points, not pixelised IFU data, which average the velocity along each sightline.

Figures, kept outside git, in `Figures/manuscript/04_kurtosis_3d_all_regions/gridfree_turbulence_test/`:
`three_line_lambda0.png`, `three_line_lambda0p1_1.png`, `tangent_break_lambda0.png`. The scripts that made them are
not yet in the repository.
