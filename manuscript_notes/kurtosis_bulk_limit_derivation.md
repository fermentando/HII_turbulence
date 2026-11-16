# Pair kurtosis of an expanding shell in the bulk-dominated limit

Semi-analytic derivation of $K(\ell)$ for the synthetic random-point shells, in the limit
$v_{\rm bulk} \gg v_{\rm turb}$. The closed-form constants (plateau level, decline, bend scale)
are in the companion note `kurtosis_bulk_limit_constants.md`.

- Script: `manuscript_pipeline/04_flame_kurtosis_3d/derive_bulk_limit_kurtosis.py`
- Figure: `Figures/manuscript/04_kurtosis_3d_all_regions/bulk_limit_kurtosis_derivation.png`
- Table: `Figures/manuscript/04_kurtosis_3d_all_regions/bulk_limit_kurtosis_derivation.csv`

## Coordinates and notation

| Symbol | Meaning |
|---|---|
| $x$ | line-of-sight axis (depth) |
| $(y, z)$ | sky plane; pairs are built from these positions |
| $r = \sqrt{x^2+y^2+z^2}$ | 3D distance from the shell centre |
| $\rho = \sqrt{y^2+z^2}$ | projected (sky) distance from the centre, $r^2 = x^2 + \rho^2$ |
| $\ell = \lvert\boldsymbol\rho_i - \boldsymbol\rho_j\rvert$ | sky separation of a pair (the "$r$" in VSF($r$) and $dK/d\log r$) |
| $R = r_{\rm HII} = 1$ | outer shell radius; all lengths are in units of $r_{\rm HII}$ |
| $a = R - \Delta$ | inner shell radius, $\Delta = \Delta r_{\rm shell}$ |
| $\mu = x/r = \cos\theta$ | cosine of the angle between a point's radius vector and the line of sight |

The pair kurtosis in a separation bin is

$$
K(\ell) = \frac{\langle \delta v^4 \rangle_{\ell}}{\langle \delta v^2 \rangle_{\ell}^2},
\qquad \delta v = v_i - v_j ,
$$

averaged over all pairs whose sky separation falls in the bin.

## Setup

The shell $a \le r \le R$ is filled uniformly in volume. Gas expands radially,
$\mathbf v = v_{\rm bulk}\,\hat{\mathbf r}$, and only the line-of-sight component is observed:

$$
v_{\rm obs} = \mathbf v \cdot \hat x = v_{\rm bulk}\,\frac{x}{r} = v_{\rm bulk}\,\mu .
$$

At the disc centre ($\rho = 0$) the front and back walls move straight towards / away from the
observer, $\mu = \pm 1$. At the limb ($\rho \to R$) the motion is across the sky, $\mu \to 0$.
Kurtosis is dimensionless, so $v_{\rm bulk}$ cancels and $K(\ell)$ depends only on the geometry
$(\Delta, \ell)$.

## Step 1: a single point at fixed sky position

For a point known to lie at sky radius $\rho$, its depth $x$ is uniform along the chord $C(\rho)$,
the part of that sightline inside the shell, $a^2 \le x^2 + \rho^2 \le R^2$:

- $\rho < a$: two walls (front and back), $\lvert x\rvert \in [x_1, x_2]$;
- $a \le \rho < R$: one grazing segment, $\lvert x\rvert \in [0, x_2]$;

$$
x_1 = \sqrt{\max(a^2-\rho^2, 0)}, \qquad x_2 = \sqrt{R^2 - \rho^2}.
$$

Define the column and the chord moments of $\mu$:

$$
n(\rho) = \int_C dx, \qquad
m_k(\rho) = E[\mu^k \mid \rho] = \frac{1}{n(\rho)} \int_C \left(\frac{x}{\sqrt{x^2+\rho^2}}\right)^k dx .
$$

- $n(\rho)$ is the chord length, proportional to the number of points at that sky position.
- $m_2(\rho) = \langle\mu^2\rangle$ and $m_4(\rho) = \langle\mu^4\rangle$ are the mean squared and
  fourth-power projected velocities on that sightline (in units of $v_{\rm bulk}$).
- Odd moments vanish: the chord is symmetric under $x \to -x$ and $\mu$ is odd in $x$, so
  $m_1 = m_3 = 0$.

**Closed forms.** Partial fractions give

$$
\mu^2 = 1 - \frac{\rho^2}{x^2+\rho^2}
\;\Rightarrow\;
F_2(x) = \int \mu^2\,dx = x - \rho \arctan\frac{x}{\rho},
$$

$$
\mu^4 = 1 - \frac{2\rho^2}{x^2+\rho^2} + \frac{\rho^4}{(x^2+\rho^2)^2}
\;\Rightarrow\;
F_4(x) = x - \tfrac32 \rho \arctan\frac{x}{\rho} + \frac{\rho^2 x}{2(x^2+\rho^2)} .
$$

With $F_0 = x$ and the chord symmetry,

$$
n = 2(x_2 - x_1), \qquad m_k = \frac{2\,[F_k(x_2) - F_k(x_1)]}{n}.
$$

No numerical integration along the line of sight is needed.

**Thin-shell behaviour.** Away from the limb a sightline crosses the two walls at
$\mu \approx \pm\mu_0$, $\mu_0 = \sqrt{1-\rho^2}$, so

$$
m_2 \approx \mu_0^2, \qquad m_4 \approx \mu_0^4, \qquad n \approx \frac{2\Delta}{\mu_0}.
$$

The last relation is the path through two slabs of thickness $\Delta$ crossed at angle $\theta$:
$\Delta/\cos\theta$ each. Near the limb the chord is long (limb brightening) but the velocity is
close to zero.

## Step 2: one pair at fixed sky positions

Take points at $\boldsymbol\rho_i$ and $\boldsymbol\rho_j$. They are separate random points, so
$\mu_i$ and $\mu_j$ are independent, each with zero mean. For $\delta\mu = \mu_i - \mu_j$:

$$
E[\delta\mu^2] = E[\mu_i^2] - 2E[\mu_i]E[\mu_j] + E[\mu_j^2] = m_{2,i} + m_{2,j},
$$

and from

$$
(\mu_i-\mu_j)^4 = \mu_i^4 - 4\mu_i^3\mu_j + 6\mu_i^2\mu_j^2 - 4\mu_i\mu_j^3 + \mu_j^4 ,
$$

where every term with an odd power factorises to zero,

$$
E[\delta\mu^4] = m_{4,i} + 6\,m_{2,i}\,m_{2,j} + m_{4,j}.
$$

**Single-sightline check.** For two points on the same sightline of a thin shell,
$E[\delta\mu^2] = 2\mu_0^2$ and $E[\delta\mu^4] = 8\mu_0^4$, so $K = 2$. The distribution is
two-valued: same wall ($\delta\mu \approx 0$) or opposite walls ($\delta\mu = \pm 2\mu_0$). A single
sightline has *low* kurtosis; the high values come from pooling sightlines (Step 3).

## Step 3: pooling all pairs at separation $\ell$

The estimator pools all pairs in a bin and takes the ratio of pooled moments,
$K = \langle\delta v^4\rangle / \langle\delta v^2\rangle^2$; it is not an average of
per-sightline kurtoses. For large $N$ the sum over pairs becomes an integral. The number of pairs
with one point near $\boldsymbol\rho_i$ and the other near $\boldsymbol\rho_j$ is proportional to
$n(\rho_i)\,n(\rho_j)$. Write $\boldsymbol\rho_j = \boldsymbol\rho_i + \ell\,\hat{\mathbf e}$, with
$\hat{\mathbf e}$ a direction on the sky:

| Quantity | Integral | Counts |
|---|---|---|
| $W(\ell)$ | $\int d^2\rho_i \int d\hat{\mathbf e}\; n_i n_j$ | number of pairs |
| $Q_2(\ell)$ | $\int d^2\rho_i \int d\hat{\mathbf e}\; n_i n_j\,(m_{2,i}+m_{2,j})$ | $\sum \delta\mu^2$ |
| $Q_4(\ell)$ | $\int d^2\rho_i \int d\hat{\mathbf e}\; n_i n_j\,(m_{4,i}+m_{4,j}+6\,m_{2,i}m_{2,j})$ | $\sum \delta\mu^4$ |

With $\langle\delta\mu^2\rangle = Q_2/W$ and $\langle\delta\mu^4\rangle = Q_4/W$,

$$
\boxed{K(\ell) = \frac{W\,Q_4}{Q_2^2}}
$$

and $v_{\rm bulk}^4$ cancels between numerator and denominator.

**Plateau limit.** At $\ell \to 0$ the integrals collapse to 1D integrals over the disc (measure
$\rho\,d\rho$):

$$
K(0) = \frac{I_0\, I_4}{2\, I_2^2}, \qquad
I_0 = \int n^2, \quad I_2 = \int n^2 m_2, \quad I_4 = \int n^2 (m_4 + 3 m_2^2).
$$

### Numerical evaluation

The disc is rotationally symmetric, so the position angle of $\boldsymbol\rho_i$ integrates to a
constant ($2\pi$) that cancels in $K$. The remaining 2D integral runs over $\rho = \lvert\boldsymbol\rho_i\rvert$
(measure $\rho\,d\rho$) and over the angle $\psi$ between $\boldsymbol\rho_i$ and $\hat{\mathbf e}$, with

$$
\lvert\boldsymbol\rho_j\rvert = \sqrt{\rho^2 + \ell^2 + 2\rho\ell\cos\psi}.
$$

Both use composite Gauss–Legendre quadrature, with $\rho$ nodes clustered towards $\rho = a$ and
$\rho = R$ where $n(\rho)$ changes sharply.

## Checks

1. **Against the pipeline estimator.** A pure-bulk run ($v_{\rm turb} = 0$, $N = 100{,}000$,
   no pixelisation, same binning and pair estimator as the simulations) follows the semi-analytic
   curves for every width tested ($\Delta = 0.02, 0.09, 0.30, 1.0$), apart from the noisy
   smallest-$\ell$ bins.
2. **Against the $v_{\rm bulk}/v_{\rm turb} = 10$ simulation** ($N = 100{,}000$, joint softplus fits,
   $\ell \le 0.6$):

| | theory, $\lambda\to\infty$ | simulation, $\lambda = 10$ |
|---|---|---|
| $K_{\rm plat}$ at $\Delta = 0.02 / 0.12 / 1.0$ | 6.11 / 4.28 / 2.77 | 5.70 / 4.28 / 2.78 |
| $\ell_b/\Delta$ | 0.38–0.49, falling to 0.30 at $\Delta = 1$ | 0.37–0.51, 0.31 at $\Delta = 1$ |
| softplus joint fit | $A = 1.70$, $S = -2.08$, $w = 0.16$, rms $= 0.03$ | $A = 1.81$, $S = -1.91$, $w = 0.17$ |

The softplus model describes the exact geometric curves with rms 0.03, so it is a good
approximation of the physical shape, not only a convenient fitting function. The simulation falls
below the theory only for the thinnest shells, where turbulence dilutes the near-zero-$\delta v$
limb population (see the open items below).

## Interpretation

**Why the plateau is high.** Each sightline on its own gives a narrow distribution with $K \approx 2$,
but its width $\mu_0$ differs from sightline to sightline. The disc centre contributes broad
$\delta v$. The limb contributes $\delta v \approx 0$ with a large weight, since the pair count goes as
$n^2 \propto 1/\mu_0^2$. Pooling distributions of very different widths gives heavy tails relative
to the core, i.e. high kurtosis. Thinner shells have more extreme limb brightening, hence a higher
plateau. In the thin-shell limit,

$$
K(0) \simeq C_p - \ln(2\Delta),
$$

with $C_p$ derived in the companion note.

**Why $K$ changes with $\ell$.** Increasing $\ell$ forces the two points onto different sightlines,
where $n$, $m_2$ and $m_4$ differ. The limb pairs that dominate at $\ell = 0$ need both points in the
thin limb band. A separation $\ell$ moves the second point off the band unless $\hat{\mathbf e}$ runs
along the limb. The low-$\delta v$ population therefore shrinks once $\ell$ exceeds the band width,
giving the bend at $\ell_b \approx 0.45\,\Delta$ followed by the decline.

## Assumptions and limitations

- **No pixelisation.** The simulations used here (`no_pixelize=True`) keep each 3D point as its
  own sample, so Step 2 treats two points on the same sightline as two independent velocities.
  Real IFU data give one (intensity-weighted) mean velocity per fibre, in which the front and back
  walls largely cancel for a symmetric shell. The plateau level, $\ell_b/\Delta$ and the log
  formulas apply to the unpixelised estimator and must be redone with sightline-averaged
  velocities before being applied to LVM maps.
- **Bulk-dominated limit.** Turbulence is not included. It enters as a dilution, most strongly
  where the bulk $\delta v$ is smallest (the limb pairs).
- **Uniform, spherical shell** with a sharp inner and outer edge, and no measurement noise.

## Open items

- Sightline-averaged (pixelised) version of Steps 1–3, checked against the existing pixelised runs.
- Turbulent dilution: test whether its effect depends on the single parameter
  $\lambda\sqrt{2\Delta}$ (bulk velocity of the limb pairs relative to $v_{\rm turb}$), using the
  $N = 5{,}000$ $v_{\rm bulk}/v_{\rm turb}$ sweep.
