# Closed-form constants of the bulk-limit pair kurtosis

Analytic derivation of the plateau constant $C_p$, the decline constant $C_d$ and the bend scale
$\ell_b/\Delta$ for an expanding shell in the bulk-dominated, thin-shell limit. Notation and the
general formula $K(\ell) = W Q_4 / Q_2^2$ are defined in `kurtosis_bulk_limit_derivation.md`.

Result:

$$
K(\ell) \simeq
\begin{cases}
\tfrac32 + 2\ln 2 - \ln(2\Delta), & \ell \ll \Delta \quad \text{(log cut off by the shell thickness)}\\[4pt]
3\ln 2 - \ln(2\ell), & \Delta \ll \ell \ll r_{\rm HII} \quad \text{(log cut off by the separation)}
\end{cases}
$$

$$
C_p = \tfrac32 + 2\ln 2 = 2.886, \qquad C_d = 3\ln 2 = \ln 8 = 2.079, \qquad
\frac{\ell_b}{\Delta} = e^{C_d - C_p} = 2e^{-3/2} = 0.446 .
$$

## 1. Plateau constant $C_p$

### Numerical convergence

From the exact 1D integrals ($\varepsilon = 1 - a^2 = 2\Delta - \Delta^2$):

| $\Delta$ | $K(0)$ | $K(0) + \ln\varepsilon$ |
|---|---|---|
| 0.1 | 4.488 | 2.827 |
| 0.03 | 5.691 | 2.863 |
| 0.01 | 6.794 | 2.877 |
| 0.003 | 8.001 | 2.883 |
| 0.001 | 9.100 | 2.885 |
| prediction | | $\tfrac32 + 2\ln 2 = 2.886$ |

At $\Delta = 10^{-4}$ the quadrature grid no longer resolves the shell and the numerical value
drops (2.617); this is a numerical limitation, not a physical effect. At the simulated widths the
fitted value $K_{\rm plat} + \ln(2\Delta) \approx 2.877$ includes small finite-thickness corrections.

### Derivation

At $\ell = 0$ the general formula reduces to three 1D integrals over the disc (measure
$\rho\,d\rho$):

$$
K(0) = \frac{I_0\, I_4}{2\, I_2^2}, \qquad
I_0 = \int n^2, \quad I_2 = \int n^2 m_2, \quad I_4 = \int n^2 (m_4 + 3 m_2^2).
$$

Change variables to

- $u = 1 - \rho^2$, the squared line-of-sight cosine at the outer surface ($u = \mu_0^2$), with
  $\rho\,d\rho = -du/2$;
- $\varepsilon = 1 - a^2 = 2\Delta - \Delta^2 \approx 2\Delta$, the value of $u$ at the inner edge
  of the limb band.

The chord length becomes

$$
n =
\begin{cases}
2\left(\sqrt u - \sqrt{u-\varepsilon}\right), & u > \varepsilon \quad \text{(sightline crosses the cavity)}\\
2\sqrt u, & u < \varepsilon \quad \text{(sightline grazes the limb band)}
\end{cases}
$$

and for $u \gg \varepsilon$, $n \approx \varepsilon/\sqrt u$ (the $2\Delta/\mu_0$ of the thin-shell
picture).

**$I_2$ and $I_4$ (disc-dominated).** For $u \gg \varepsilon$, $m_2 \approx u$ and $m_4 \approx u^2$,
so $n^2 m_2 \approx \varepsilon^2$ and $n^2(m_4 + 3m_2^2) \approx 4\varepsilon^2 u$. Both integrands
vanish smoothly at the limb, so to leading order

$$
I_2 \approx \tfrac12 \int_0^1 \varepsilon^2\, du = \frac{\varepsilon^2}{2}, \qquad
I_4 \approx \tfrac12 \int_0^1 4\varepsilon^2 u\, du = \varepsilon^2 .
$$

**$I_0$ (log and constant).** $I_0$ is the pair count with no velocity weight, and
$n^2 \approx \varepsilon^2/u$ diverges towards the limb. Split at $u = \varepsilon$.

1. Limb band, $u < \varepsilon$:

$$
\tfrac12 \int_0^\varepsilon 4u\, du = \varepsilon^2 .
$$

2. Rest of the disc, $u > \varepsilon$. With $u = \varepsilon t$:

$$
2\varepsilon^2 \int_1^{1/\varepsilon} \left(\sqrt t - \sqrt{t-1}\right)^2 dt,
\qquad
\left(\sqrt t - \sqrt{t-1}\right)^2 = 2t - 1 - 2\sqrt{t^2 - t}.
$$

Writing $t^2 - t = (t-\tfrac12)^2 - \tfrac14$,

$$
\int_1^{T} \left(2t - 1 - 2\sqrt{t^2-t}\right) dt
= \Big[\, t^2 - t - (t-\tfrac12)\sqrt{t^2-t} + \tfrac14 \ln\!\big(t - \tfrac12 + \sqrt{t^2-t}\big) \Big]_1^T .
$$

At $t = 1$ the bracket is $\tfrac14\ln\tfrac12$. For large $T$,
$\sqrt{T^2-T} \approx T - \tfrac12 - \tfrac{1}{8T}$, and the bracket tends to
$-\tfrac18 + \tfrac14\ln(2T)$. Hence

$$
\int_1^{T} (\dots)\, dt = \tfrac14 \ln T + \tfrac12 \ln 2 - \tfrac18, \qquad T = 1/\varepsilon .
$$

Adding the two parts:

$$
I_0 = \varepsilon^2 \Big[\,
\underbrace{\tfrac12 \ln\tfrac1\varepsilon}_{\text{disc, } 1/\mu_0^2 \text{ weight}}
+ \underbrace{\ln 2 - \tfrac14}_{\text{edge of the log}}
+ \underbrace{1}_{\text{limb band}}
\Big]
= \varepsilon^2 \Big[\tfrac12 \ln\tfrac1\varepsilon + \ln 2 + \tfrac34\Big].
$$

**Combining.**

$$
K(0) = \frac{\varepsilon^2\left[\tfrac12\ln\tfrac1\varepsilon + \ln 2 + \tfrac34\right]\cdot\varepsilon^2}{2\cdot\varepsilon^4/4}
= \ln\frac1\varepsilon + 2\ln 2 + \frac32,
$$

$$
\boxed{K_{\rm plat} \simeq -\ln(2\Delta) + \tfrac32 + 2\ln 2}
$$

### Meaning of each term

- $-\ln(2\Delta)$: logarithmic pile-up of pairs on long, low-velocity sightlines towards the limb,
  from the $1/\mu_0^2$ weight between the disc centre ($\mu_0 = 1$) and the limb cutoff
  ($\mu_0 \approx \sqrt{2\Delta}$).
- $\tfrac32 = 2 \times \tfrac34$: of the $\tfrac34$, $+1$ is the limb band itself ($u < \varepsilon$,
  sightlines grazing the shell, all with $\mu \approx 0$) and $-\tfrac14$ corrects for $n$ deviating
  from $\varepsilon/\sqrt u$ as $u \to \varepsilon$.
- $2\ln 2$: the exact shape of the chord-length function near the inner edge, i.e. the precise
  lower cutoff of the log.
- The overall factor 2 comes from the $I_4/(2I_2^2)$ structure (disc-averaged moments of $\mu$).

$C_p \approx 3$ is therefore not a Gaussian-like 3. It is a geometric number dominated by pair
counting near the limb; the velocity moments enter only as simple disc averages.

## 2. Decline constant $C_d$

### Step 1: in this regime $K(\ell)$ only counts pairs

For $\Delta \ll \ell \ll 1$:

- $Q_2$ and $Q_4$ carry the velocity weights $m_2 \approx \mu_0^2$, $m_4 \approx \mu_0^4$, which
  vanish at the limb. They come from the disc, where $n$, $m_2$, $m_4$ vary on scales
  $\sim r_{\rm HII}$, so a shift $\ell \ll 1$ leaves them at their $\ell = 0$ values:
  $Q_2 \approx 2I_2 = 4\Delta^2$, $Q_4 \approx 2I_4 = 8\Delta^2$.
- $W(\ell)$, the raw pair count, is dominated by the limb-brightened ring and does change with
  $\ell$.

Therefore

$$
K(\ell) \simeq \frac{W(\ell)\cdot 8\Delta^2}{(4\Delta^2)^2} = \frac{W(\ell)}{2\Delta^2}.
$$

In this regime the kurtosis at separation $\ell$ is the autocorrelation of the shell's projected
column-density (brightness) map at lag $\ell$.

### Step 2: closed form for the autocorrelation

For $\Delta \ll \ell$ the shell is effectively infinitely thin, and $n(\rho) = \Delta\,\Sigma(\rho)$,
where $\Sigma$ is the projection along $x$ of a unit-density spherical surface $\delta(r-1)$.

The 2D autocorrelation of a projected map equals the projection of the 3D autocorrelation.

**3D autocorrelation of a sphere.**

$$
\xi_3(s) = \int \delta(\lvert\mathbf r\rvert - 1)\, \delta(\lvert\mathbf r + \mathbf s\rvert - 1)\, d^3 r = \frac{2\pi}{s},
\qquad s < 2 .
$$

With polar angle $\theta$ on the first sphere, $\lvert\mathbf r+\mathbf s\rvert^2 = 1 + s^2 + 2s\cos\theta$.
Changing variable to $w = \lvert\mathbf r+\mathbf s\rvert$ gives $\sin\theta\,d\theta = w\,dw/s$, and the
delta function selects $w = 1$, leaving $2\pi/s$.

**Projection along the line of sight.** At sky lag $\ell$ the 3D separation is
$s = \sqrt{\ell^2 + x^2}$, non-zero for $x^2 < 4 - \ell^2$:

$$
C_2(\ell) = \int \xi_3\!\left(\sqrt{\ell^2+x^2}\right) dx
= 4\pi \int_0^{\sqrt{4-\ell^2}} \frac{dx}{\sqrt{\ell^2 + x^2}}
= 4\pi \ln\frac{2 + \sqrt{4-\ell^2}}{\ell}.
$$

**Normalisation.** $W$ uses the same $(1/2\pi)\int d^2\rho$ normalisation as $I_0$:

$$
W(\ell) = \frac{\Delta^2}{2\pi}\, C_2(\ell) = 2\Delta^2 \ln\frac{2 + \sqrt{4-\ell^2}}{\ell}.
$$

### Step 3: combine

$$
K(\ell) \simeq \ln\frac{2 + \sqrt{4-\ell^2}}{\ell}
\;\xrightarrow{\;\ell \ll 1\;}\;
\ln\frac{4}{\ell} = -\ln(2\ell) + \ln 8 ,
$$

so $C_d = \ln 8 = 3\ln 2$. The 8 traces back to the "$2+$" term: the longest chord between two points
on the sphere is the diameter, 2.

The sphere's autocorrelation goes as $1/s$, so pairs at short 3D separation dominate; projecting
$1/s$ along the line of sight gives a log cut off at the sky separation $\ell$. At $\ell \to 0$ the
cutoff moves to the shell thickness instead, which is the plateau.

### Check against the exact calculation

| $\ell$ | $K$ exact ($\Delta = 0.02$) | $\ln\frac{2+\sqrt{4-\ell^2}}{\ell}$ | $K + \ln(2\ell)$ |
|---|---|---|---|
| 0.032 | 4.841 | 4.840 | 2.080 |
| 0.057 | 4.262 | 4.248 | 2.093 |
| 0.103 | 3.701 | 3.657 | 2.123 |
| 0.21 | 3.071 | 2.946 | 2.202 |
| 0.38 | 2.611 | 2.348 | 2.333 |

- For $\ell \lesssim 0.1$ the formula agrees to better than 1%. $\Delta = 0.037$ gives the same
  numbers, as expected once $\ell \gg \Delta$.
- For larger $\ell$, $K$ exceeds the formula because $Q_2$ and $Q_4$ start to depend on $\ell$. This
  is the flattening towards $r_{\rm HII}$ that makes fitted slopes $S \approx -1.9$ to $-2.1$ per dex
  rather than $-\ln 10 = -2.303$.
- Near the bend ($\ell \sim 10\Delta$) the two regimes are still merging: at $\Delta = 10^{-3}$,
  $K + \ln(2\ell) = 2.18$ at $\ell = 10\Delta$ and 2.09–2.10 at $\ell = 20$–$50\Delta$. This
  crossover is what the softplus width $w \approx 0.16$ dex describes.

## 3. Bend scale

The bend is where the plateau and decline lines cross:

$$
C_p - \ln(2\Delta) = C_d - \ln(2\ell_b)
\;\Rightarrow\;
\frac{\ell_b}{\Delta} = e^{C_d - C_p} = e^{3\ln 2 - \frac32 - 2\ln 2} = 2e^{-3/2} = 0.446 .
$$

## Summary

| Quantity | Derived | Fitted (simulations) |
|---|---|---|
| $C_p$ | $\tfrac32 + 2\ln 2 = 2.886$ | 2.877 |
| $C_d$ | $3\ln 2 = 2.079$ | 2.08 |
| $S$ (small $\ell$) | $-\ln 10 = -2.303$ per dex | $-2.08$ (fit; flattened by the large-$\ell$ correction) |
| $\ell_b/\Delta$ | $2e^{-3/2} = 0.446$ | 0.45 |

## Not yet derived analytically

- The shape of the crossover at $\ell \sim \Delta$ (the softplus width $w$).
- The $\ell \sim r_{\rm HII}$ corrections from the $\ell$-dependence of $Q_2$ and $Q_4$.
- Finite-thickness corrections, $O(\Delta\ln\Delta)$.
- Turbulent dilution, and the change to sightline-averaged (pixelised) velocities. Both are
  required before applying these results to LVM data; see the limitations in
  `kurtosis_bulk_limit_derivation.md`.
