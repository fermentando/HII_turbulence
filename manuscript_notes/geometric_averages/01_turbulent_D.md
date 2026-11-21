# Geometric averages 1: the turbulent averages ⟨D⟩ and ⟨D²⟩

Part of a series deriving each geometric average in the finite-λ pair-kurtosis formula
(`manuscript_notes/kurtosis_kolmogorov_lambda_derivation.md`, §1):

$$
K(\ell;\lambda) = \frac{3\langle D^2\rangle_\ell + 6\lambda^2\langle D\,\delta\mu^2\rangle_\ell + \lambda^4\langle\delta\mu^4\rangle_\ell}
{\left[\langle D\rangle_\ell + \lambda^2\langle\delta\mu^2\rangle_\ell\right]^2}
$$

This note covers $\langle D\rangle_\ell$, $\langle D^2\rangle_\ell$ and $K_t = 3\langle D^2\rangle/\langle D\rangle^2$, simplified for
turbulence with **no inner cutoff** ($k_{\max}\to\infty$), which is the physical case for an H II region and
is tested in `manuscript_notes/gridfree_turbulence_test.md`.

All lengths are in units of $r_{\rm HII}$; the shell is $1-\Delta \le r \le 1$.

## Result

Plateau ($\ell\to0$), thin shell:

$$
\langle D\rangle_0 \simeq \frac{2J - \frac{27}{20}A\Delta^{2/3}}{3 + 2\ln(2/\Delta)}, \qquad
\langle D^2\rangle_0 \simeq \frac{2J_2 - \frac{27}{70}A^2\Delta^{4/3}}{3 + 2\ln(2/\Delta)}, \qquad
K_t(0) = \frac{3\langle D^2\rangle_0}{\langle D\rangle_0^2}
$$

Plain text:

```
⟨D⟩₀  ≈ [ 2J  − (27/20) · A · Δ^(2/3) ] / [ 3 + 2 ln(2/Δ) ]
⟨D²⟩₀ ≈ [ 2J₂ − (27/70) · A² · Δ^(4/3) ] / [ 3 + 2 ln(2/Δ) ]
```

Constants (depend on the turbulence outer scale only; values for $k_{\min} = 5.24$, outer scale
$2\pi/k_{\min} = 1.2\,r_{\rm HII}$, as in the simulations):

| constant | definition | value |
|---|---|---|
| $I$ | $G(\infty) = \frac{9}{20}\Gamma(\frac13)$ | 1.2055 |
| $A$ | $\frac43 I\,k_{\min}^{2/3}$ (Kolmogorov amplitude, $D\simeq As^{2/3}$) | 4.85 |
| $J$ | $\int_0^2 D(s)\,ds/s$ | 6.85 |
| $J_2$ | $\int_0^2 D(s)^2\,ds/s$ | 10.53 |

To leading order in $\Delta$:

$$K_t(0) \simeq \frac{3J_2}{J^2}\,K_b(0) = 0.67\,K_b(0), \qquad K_b(0) \simeq \tfrac32 + \ln(2/\Delta).$$

At $\ell > \Delta$ (thin shell), with $s = \ell\cosh u$:

$$
\langle D^k\rangle_\ell = \frac{1}{\operatorname{arccosh}(2/\ell)}\int_0^{\operatorname{arccosh}(2/\ell)} D(\ell\cosh u)^k\,du .
$$

**Worked example**, $\Delta = 0.02$ ($\Delta^{2/3} = 0.0737$, $\Delta^{4/3} = 0.00543$):

```
denominator = 3 + 2 ln(100)                     = 12.21
⟨D⟩₀  = (13.70 − 1.35·4.85·0.0737)    / 12.21   = 1.083
⟨D²⟩₀ = (21.05 − 0.386·23.5·0.00543)  / 12.21   = 1.720
K_t(0) = 3 · 1.720 / 1.083²                     = 4.40
```

Exact Abel projection for the same case: $\langle D\rangle_0 = 1.081$, $K_t = 4.40$.

## Ingredients

### D(s): turbulent velocity difference at 3D separation s

For a unit-variance Gaussian field with 3D power $P(k)\propto k^{-11/3}$ on $[k_{\min}, k_{\max}]$,

$$
\xi(s) = \frac{\int k^{-5/3}\,\frac{\sin ks}{ks}\,dk}{\int k^{-5/3}\,dk}, \qquad D(s) = 2\,[1-\xi(s)].
$$

($k^{-11/3}$ times the shell volume $4\pi k^2dk$ gives $k^{-5/3}$; $\sin(ks)/(ks)$ is the angular average of
$e^{i\mathbf k\cdot\mathbf s}$.) With $t = ks$ and $Z = \frac32(k_{\min}^{-2/3} - k_{\max}^{-2/3})$:

$$
D(s) = \frac{2s^{2/3}}{Z}\Big[G(k_{\max}s) - G(k_{\min}s)\Big], \qquad
G(x) = \int_0^x t^{-5/3}\left(1 - \frac{\sin t}{t}\right)dt .
$$

$G$ is computed from

- the power series $G(x) = \sum_{n\ge1}\frac{(-1)^{n+1}x^{2n-2/3}}{(2n+1)!\,(2n-\frac23)}$ (exact; use for $x\lesssim10$);
- the asymptotic form $G(x) \simeq I - \frac32x^{-2/3} + \cos x\,x^{-8/3} + \frac83\sin x\,x^{-11/3}$ (use for $x\gtrsim4$),
  with $I = G(\infty) = -\frac12\Gamma(-\frac53) = \frac{9}{20}\Gamma(\frac13)$ from the Mellin integral
  $\int_0^\infty t^{a-1}\sin t\,dt = \Gamma(a)\sin(\pi a/2)$ continued to $a = -\frac53$.

**No inner cutoff.** $Z \to \frac32k_{\min}^{-2/3}$ and $G(k_{\max}s)\to I$, so with $x = k_{\min}s$

$$
D(s) = \tfrac43\,x^{2/3}\,[I - G(x)] \simeq A\,s^{2/3} - \tfrac16(k_{\min}s)^2 + \dots
$$

$D$ now depends on the outer scale only. The Kolmogorov law $As^{2/3}$ is within ~4% for $s\lesssim0.1$ and
saturates at $D = 2$ beyond $s\approx0.4$. (With the grid's $k_{\max} = 73$ there is no $s^{2/3}$ range at all:
the inner-cutoff correction decays only as $(k_{\max}s)^{-2/3}$.)

### C(s): number of pairs in the shell at 3D separation s

$C(s)$ is the number of pairs per unit volume of 3D separation vector. Two pieces, split at $s = \Delta$:

**$\Delta \le s \le 2(1-\Delta)$, exact:**

$$C(s) = \frac{\pi\Delta^2(2-\Delta)^2}{2s} \;\to\; \frac{2\pi\Delta^2}{s}\ \text{(thin shell)}.$$

Fix one point. Its partners at distance $s$ lie on a sphere of radius $s$ around it, which cuts the shell in
a ring of area $2\pi s\,ds$ and thickness $\Delta$. Dividing by the separation-shell volume $4\pi s^2ds$:
$C = (4\pi\Delta)(\Delta\cdot2\pi s)/(4\pi s^2) = 2\pi\Delta^2/s$. (Checked against the ball-overlap formula
$V(1,1) - 2V(1,1-\Delta) + V(1-\Delta,1-\Delta)$ to 6 digits.) This counts pairs along one wall and pairs across
the cavity together.

**$s < \Delta$, slab approximation:**

$$C(s) \simeq 4\pi\left(\Delta - \frac s2\right).$$

On scales below $\Delta$ the wall is a flat slab. A sphere of radius $s$ around a point is mostly inside it;
averaged over the point's depth in the slab, the fraction outside is $s/(2\Delta)$, so
$C = (\text{wall volume } 4\pi\Delta)(1 - s/2\Delta)$. Accurate to 2% at $\Delta = 0.02$, 9% at $\Delta = 0.09$,
27% at $\Delta = 0.3$. The two pieces meet at $s = \Delta$ (both $2\pi\Delta$).

### Averaging over pairs at sky separation ℓ

A pair at sky separation $\ell$ and depth difference $d_x$ has $s = \sqrt{\ell^2 + d_x^2}$. Summing over $d_x$
(Abel projection):

$$
\langle D^k\rangle_\ell = \frac{\int_\ell^2 C(s)\,D(s)^k\,\frac{s\,ds}{\sqrt{s^2-\ell^2}}}{\int_\ell^2 C(s)\,\frac{s\,ds}{\sqrt{s^2-\ell^2}}}.
$$

## Derivation of the plateau formula

### Step 1. Set ℓ = 0: the projection disappears

At $\ell = 0$ both points share a sightline, $s = d_x$, and the weight $s/\sqrt{s^2-\ell^2} = 1$:

$$\langle D\rangle_0 = \frac{\int_0^2 C(s)\,D(s)\,ds}{\int_0^2 C(s)\,ds}.$$

*Why:* the plateau is the $\ell\to0$ limit, where the average becomes a plain 1D integral over $s$.

### Step 2. Split the pairs at s = Δ

Use the two forms of $C(s)$ above.

*Why here:* at $s = \Delta$ the sphere around a point stops fitting inside the wall, so the counting rule
changes from "fraction of the sphere inside a slab" to "ring on a thin shell".

### Step 3. Inside the wall, use D ≈ A s^(2/3)

*Why allowed:* same-wall pairs have $s < \Delta \le 0.09$, well below the outer scale $1/k_{\min}\approx0.19$, and
with no inner cutoff this is the Kolmogorov range.
*Why needed:* a power law makes the same-wall integral exact.

### Step 4. The denominator: total number of pairs

$$
\int_0^\Delta 4\pi\left(\Delta - \frac s2\right)ds + \int_\Delta^2 \frac{2\pi\Delta^2}{s}\,ds
= 3\pi\Delta^2 + 2\pi\Delta^2\ln\frac2\Delta = \pi\Delta^2\left[3 + 2\ln\frac2\Delta\right].
$$

*Why the log:* under $C\propto1/s$ every factor of 2 in $s$ holds the same number of pairs, so the pairs beyond
$\Delta$ grow as $\ln(1/\Delta)$ and outnumber the same-wall pairs (the "3") more and more for thin shells.

### Step 5. Pull out a universal constant J

The numerator beyond $\Delta$ is $2\pi\Delta^2\int_\Delta^2 D\,ds/s$, which depends on $\Delta$ only through the
lower limit:

$$
\int_\Delta^2\frac{D}{s}\,ds = \underbrace{\int_0^2\frac{D}{s}\,ds}_{J} - \int_0^\Delta A\,s^{-1/3}\,ds
= J - \tfrac32A\Delta^{2/3}.
$$

*Why:* $J$ depends only on the turbulence and is computed once; it converges because $D/s\sim s^{-1/3}$. $J$
pretends the $1/s$ law holds down to $s = 0$; the subtracted term removes that part below $\Delta$, where Step 6
puts back the real slab count.

### Step 6. The same-wall numerator

$$\int_0^\Delta 4\pi\left(\Delta - \frac s2\right)A\,s^{2/3}\,ds = 4\pi A\,\Delta^{8/3}\left(\frac35 - \frac3{16}\right) = \frac{33}{20}\pi A\,\Delta^{8/3}.$$

### Step 7. Assemble

$$
\text{numerator} = 2\pi\Delta^2\left[J - \tfrac32A\Delta^{2/3}\right] + \tfrac{33}{20}\pi A\Delta^{8/3}
= \pi\Delta^2\left[2J - \tfrac{27}{20}A\Delta^{2/3}\right],
$$

and dividing by Step 4 ($\pi\Delta^2$ cancels) gives $\langle D\rangle_0$. The $-\frac{27}{20} = -3 + \frac{33}{20}$ is the net
effect of replacing the $1/s$ extrapolation inside the wall by the real slab count: there are fewer real
same-wall pairs than the $1/s$ law would claim.

### Step 8. ⟨D²⟩ and K_t

Repeat with $D^2 \simeq A^2s^{4/3}$ inside the wall:

$$
-2\int_0^\Delta A^2s^{1/3}ds = -\tfrac32A^2\Delta^{4/3}, \qquad
\int_0^\Delta 4\pi\left(\Delta - \frac s2\right)A^2s^{4/3}ds = 4\pi A^2\Delta^{10/3}\left(\frac37 - \frac3{20}\right) = \tfrac{39}{35}\pi A^2\Delta^{10/3},
$$

net $-\frac32 + \frac{39}{35} = -\frac{27}{70}$, which gives $\langle D^2\rangle_0$ above with $J_2 = \int_0^2 D^2\,ds/s$. Then
$K_t(0) = 3\langle D^2\rangle_0/\langle D\rangle_0^2$.

### Step 9. Link to the bulk plateau

The denominator is twice the thin-shell bulk plateau, $K_b(0) = \frac32 + 2\ln2 - \ln(2\Delta) = \frac12[3 + 2\ln(2/\Delta)]$
(`kurtosis_bulk_limit_constants.md`). Dropping the $\Delta^{2/3}$ corrections,

$$K_t(0) \simeq \frac{3J_2}{J^2}\,K_b(0).$$

Turbulence alone gives the same $\ln(1/\Delta)$ growth as the expansion, scaled by a number fixed by the
turbulence: $3J_2/J^2 = 0.67$ here. For a pure power law $D = As^{2/3}$ with no saturation,
$J = \frac32\,2^{2/3}A$ and $J_2 = \frac34\,2^{4/3}A^2$, so $3J_2/J^2 = 1$ and $K_t = K_b$, as in §8 of the λ note.

## Separations beyond the wall (ℓ > Δ)

For $\ell > \Delta$ only the $1/s$ piece contributes. Substituting $s = \ell\cosh u$,
$ds/\sqrt{s^2-\ell^2} = du$, and the $1/s$ in $C$ cancels the $s$ in the projection weight:

$$
\int_\ell^2\frac{2\pi\Delta^2}{s}\,f(s)\,\frac{s\,ds}{\sqrt{s^2-\ell^2}} = 2\pi\Delta^2\int_0^{u_{\max}} f(\ell\cosh u)\,du, \qquad u_{\max} = \operatorname{arccosh}(2/\ell).
$$

So pairs at sky separation $\ell$ are spread **uniformly in $u$**, and $\langle D^k\rangle_\ell$ is the plain average of
$D(\ell\cosh u)^k$ over $u\in[0, u_{\max}]$. $\Delta$ drops out: every thin shell has the same $\langle D\rangle_\ell$
once $\ell > \Delta$. The pair count $\propto u_{\max} = \ln[(2+\sqrt{4-\ell^2})/\ell]$ is the same logarithm as the
bulk decline $K_b \simeq 3\ln2 - \ln(2\ell)$.

For $\ell < \Delta$ add the slab term $\int 4\pi(\Delta - s/2)\,f(s)\,s\,ds/\sqrt{s^2-\ell^2}$ over $\ell \le s \le \Delta$.

## Accuracy

Simplified forms (slab + $1/s$ pieces, $k_{\max} = \infty$) against the exact Abel projection of the
ball-overlap $C(s)$ with the same $D(s)$:

| Δ | ℓ | ⟨D⟩ exact / simplified | K_t exact / simplified | plateau closed form ⟨D⟩, K_t |
|---|---|---|---|---|
| 0.005 | 0 | 0.902 / 0.902 | 5.18 / 5.18 | 0.902, 5.18 |
| 0.02 | 0 | 1.081 / 1.083 | 4.40 / 4.40 | 1.083, 4.40 |
| 0.02 | 0.03 | 1.336 / 1.337 | 3.64 / 3.64 | |
| 0.02 | 0.1 | 1.656 / 1.657 | 3.17 / 3.17 | |
| 0.09 | 0 | 1.340 / 1.347 | 3.72 / 3.72 | 1.346, 3.72 |
| 0.09 | 0.03 | 1.411 / 1.417 | 3.52 / 3.52 | |
| 0.09 | 0.1 | 1.652 / 1.657 | 3.18 / 3.17 | |
| 0.3 | 0 | 1.574 / 1.596 | 3.38 / 3.36 | 1.586, 3.38 |
| 1.0 | 0 | 1.698 / 1.790 | 3.26 / 3.18 | 1.633, 3.08 |

($\ell = 0$ evaluated at $\ell = 10^{-4}$.) Within 0.01 for $\Delta \le 0.09$ and 0.02 at $\Delta = 0.3$. At $\Delta = 1$
the slab picture fails and the exact $C(s)$ is needed.

## Physical reading

| piece | meaning |
|---|---|
| $3 + 2\ln(2/\Delta)$ | pairs on one sightline: same-wall (3) plus everything at $s > \Delta$ (the log) |
| $2J$ | turbulent spread of those pairs if the wall were infinitely thin |
| $-\frac{27}{20}A\Delta^{2/3}$ | same-wall pairs are up to $\Delta$ apart, so not perfectly correlated |
| $K_t \simeq 0.67\,K_b$ | turbulence gives the same $\ln(1/\Delta)$ growth as the expansion, scaled by $3J_2/J^2$ |

$K_t > 3$ because each $\ell$ bin mixes pairs that are truly close in 3D (small $D$, a narrow core) with pairs
that only look close in projection but are far apart along the line of sight ($D\approx2$, the wings):
$K_t = 3\,[1 + {\rm Var}(D)/\langle D\rangle^2]$.

## Assumptions

- Gaussian, isotropic scalar turbulence with $P\propto k^{-11/3}$, sharp outer cutoff at $k_{\min}$, no inner cutoff.
- Uniform spherical shell with sharp edges; thin-shell forms for $\Delta \lesssim 0.3$.
- Unpixelised estimator (each 3D point is its own sample).

Numbers in this note: `manuscript_pipeline/04_flame_kurtosis_3d/check_simplified_turbulent_averages.py`
(exact Abel vs simplified), using `gridfree_kolmogorov.structure_function` and
`test_gridfree_kolmogorov_model.C_shell`.
