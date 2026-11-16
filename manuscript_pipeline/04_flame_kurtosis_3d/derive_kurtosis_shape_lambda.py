#!/usr/bin/env python3
"""Closed-form K(l; lambda): the bulk-limit shape extended to finite v_bulk/v_turb.

Baseline: turbulent velocities uncorrelated between any two points (S = 2 for every
pair). Then K_t = 3, C = 1 and
    K = 3 + (K_b - 3) f^2,   f = lambda^2 B / (2 + lambda^2 B).
In the thin-shell regime B = Q2/W = 2/K_b (Q2 = 4 Delta^2, W = 2 Delta^2 K_b), so with
kappa(l) = K_b(l):
    K(l; lambda) = 3 + (kappa - 3) [lambda^2 / (kappa + lambda^2)]^2 .
kappa(l) itself is the bulk-limit result:
    plateau  kappa = C_p - ln(2 Delta),  C_p = 3/2 + 2 ln 2   (l << Delta)
    decline  kappa = C_d - ln(2 l),      C_d = 3 ln 2         (Delta << l << 1)

Checks (per width):
  (i)   exact white-noise K from the geometric pair sums (B, <dmu^4> of the random points)
  (ii)  closed form with kappa = exact K_b(l) from the same pair sums
  (iii) closed form with the asymptotic kappa = max-type formula (smoothed)
and the correlated-turbulence simulation (8 field seeds pooled) for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = sys.argv[:1]
import kurtosis_shape_vs_vratio_fixed_width as kv  # noqa: E402

WIDTHS = [0.02, 0.09, 0.3]
LAMS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
C_P = 1.5 + 2 * np.log(2)
C_D = 3 * np.log(2)
OUT_PNG = kv.BASE_DIR / "kurtosis_shape_lambda_closed_form.png"


def k_closed(kappa, lam):
    if np.isinf(lam):
        return kappa
    f = lam ** 2 / (kappa + lam ** 2)
    return 3 + (kappa - 3) * f ** 2


def kappa_asymptotic(ell, width):
    """Bulk curve from the two asymptotes, joined by a smooth max (w = 0.16 dex)."""
    x, xb = np.log10(ell), np.log10(2 * np.exp(-1.5) * width)
    w = 0.16
    xs = xb + w * np.logaddexp(0, (x - xb) / w)          # smooth max(x, x_b)
    return C_D - np.log(2 * 10 ** xs)


def main():
    ell = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    cmap = plt.get_cmap("viridis")
    col = {l: cmap(i / (len(LAMS) - 1)) for i, l in enumerate(LAMS)}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    print("max |(ii) - (i)| and |(iii) - (i)| over l <= 0.2, per width and lambda")
    for j, w in enumerate(WIDTHS):
        tag = f"w{w:g}".replace(".", "p")
        G = dict(np.load(kv.BASE_DIR / f"_kurtosis_vratio_geom_{tag}_n{kv.N_POINTS}.npz"))
        c = np.load(kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}_seeds8.npz")
        M = {k: c[k].sum(0) for k in kv.KEYS}
        good = G["n"] >= 200
        n = G["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            B, B4 = G["B"] / n, G["B2"] / n
            kb = B4 / B ** 2
        ka = kappa_asymptotic(ell, w)
        sel = good & (ell <= 0.2)
        print(f"  Delta={w}:  B*K_b/2 over l<=0.2 = {np.nanmin((B * kb / 2)[sel]):.3f}-{np.nanmax((B * kb / 2)[sel]):.3f}")
        for lam in LAMS:
            with np.errstate(invalid="ignore", divide="ignore"):
                if np.isinf(lam):
                    k_wn = kb
                else:
                    k_wn = (12 + 12 * lam ** 2 * B + lam ** 4 * B4) / (2 + lam ** 2 * B) ** 2
            k_ii, k_iii = k_closed(kb, lam), k_closed(ka, lam)
            if lam in (1.0, 3.0, 10.0):
                print(f"    lambda={lam:g}: (ii) {np.nanmax(np.abs(k_ii - k_wn)[sel]):.3f}   (iii) {np.nanmax(np.abs(k_iii - k_wn)[sel]):.3f}")
            a = ax[0, j]
            a.plot(ell[good], k_wn[good], "o", ms=2.5, color=col[lam], alpha=0.6)
            a.plot(ell[good], k_iii[good], "-", lw=1.6, color=col[lam],
                   label=rf"$\lambda={lam:g}$" if np.isfinite(lam) else r"$\lambda=\infty$")
            b = ax[1, j]
            b.plot(ell[good], kv.k_of_lambda(M, lam)[good], "o", ms=2.5, color=col[lam], alpha=0.6)
            b.plot(ell[good], k_iii[good], "-", lw=1.6, color=col[lam])
        for a, ttl in ((ax[0, j], "uncorrelated turbulence (points: exact pair sums)"),
                       (ax[1, j], "Kolmogorov field, 8 seeds (points)")):
            a.axvline(2 * np.exp(-1.5) * w, color="0.5", ls=":", lw=1)
            a.axvline(4 * np.exp(-3), color="0.5", ls="--", lw=1)
            a.axhline(3, color="0.5", ls=":", lw=1)
            a.set(xscale="log", xlim=(2e-3, 0.7), ylim=(1.5, 7.5), xlabel=r"$\ell/r_{\rm HII}$",
                  ylabel=r"$K(\ell)$", title=rf"$\Delta={w}$: {ttl}")
            a.grid(True, ls="--", alpha=0.4)
        ax[0, j].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(r"Lines: closed form $K=3+(\kappa-3)\,[\lambda^2/(\kappa+\lambda^2)]^2$ with the analytic bulk curve $\kappa(\ell)$;"
                 r"  dotted: $\ell_b=2e^{-3/2}\Delta$, dashed: pivot $\ell=4e^{-3}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
