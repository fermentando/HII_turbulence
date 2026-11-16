#!/usr/bin/env python3
"""Sharp broken-line ("hinge") version of fit_kurtosis_curve_shape.py.

    K(x) = K_plat + S * max(0, x - x_b),     x = log10(l / r_HII)

No smoothing width. For fixed x_b the model is linear in (K_plat, S), so x_b is
profiled on a fine grid and the linear part solved exactly (no initial guesses).

  free : K_plat, x_b, S per shell width
  joint: shared line A + S x for x > x_b; per-width x_b only; K_plat = A + S x_b.
         Solved by alternating (x_b grid per width | linear solve for A, S).

Same data, cache, fit range and r_min truncation as the softplus version, so the
two can be compared directly.
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

from fit_kurtosis_curve_shape import (  # noqa: E402
    BASE_DIR, CHECK_WIDTH_INDEX, L_MAX_FIT, RMIN_SWEEP, WIDTHS, fit_mask, load_or_compute,
)

OUT_PNG = BASE_DIR / "kurtosis_curve_hinge_fits.png"
OUT_CSV = BASE_DIR / "kurtosis_curve_hinge_fits.csv"
XB_GRID = np.linspace(-3.0, np.log10(L_MAX_FIT), 1201)


def hinge(x, k_plat, x_b, s):
    return k_plat + s * np.maximum(0.0, x - x_b)


def fit_free(x, y):
    """Profile over x_b; exact linear LS for (K_plat, S). Returns params, SSE, x_b 1-sigma range."""
    if len(x) < 4:
        return None
    best = None
    sse_grid = np.full(XB_GRID.size, np.inf)
    for k, xb in enumerate(XB_GRID):
        h = np.maximum(0.0, x - xb)
        if np.count_nonzero(h > 0) < 2 or np.count_nonzero(h == 0) < 1:
            continue  # need both a plateau and a decline inside the data
        X = np.column_stack([np.ones_like(x), h])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ coef - y) ** 2))
        sse_grid[k] = sse
        if best is None or sse < best[1]:
            best = (np.array([coef[0], xb, coef[1]]), sse)
    if best is None:
        return None
    p, sse = best
    sigma2 = sse / max(len(x) - 3, 1)
    ok = sse_grid <= sse + sigma2  # delta chi^2 = 1 profile interval for x_b
    return p, sse, (XB_GRID[ok].min(), XB_GRID[ok].max())


def fit_joint(data, n_iter=60):
    nw = len(data)
    xb = np.linspace(-1.9, -0.5, nw)
    a, s = 1.8, -1.9
    for _ in range(n_iter):
        # linear solve for A, S given x_b: K = A + S * max(x, x_b)
        rows_x, rows_y = [], []
        for (x, y), b in zip(data, xb):
            rows_x.append(np.maximum(x, b))
            rows_y.append(y)
        mx, yy = np.concatenate(rows_x), np.concatenate(rows_y)
        X = np.column_stack([np.ones_like(mx), mx])
        (a, s), *_ = np.linalg.lstsq(X, yy, rcond=None)
        # per-width grid for x_b given A, S
        new = xb.copy()
        for i, (x, y) in enumerate(data):
            if len(x) == 0:
                continue
            pred = a + s * np.maximum(x[None, :], XB_GRID[:, None])
            new[i] = XB_GRID[np.argmin(np.sum((pred - y[None, :]) ** 2, axis=1))]
        if np.allclose(new, xb):
            break
        xb = new
    resid = np.concatenate([a + s * np.maximum(x, b) - y for (x, y), b in zip(data, xb)])
    return a, s, xb, float(np.sqrt(np.mean(resid ** 2)))


def masked(curves, r_min):
    return [(lc[fit_mask(lc, kk, r_min)], kk[fit_mask(lc, kk, r_min)]) for lc, kk in curves]


def main():
    curves, check = load_or_compute()
    free = {r: [fit_free(x, y) for x, y in masked(curves, r)] for r in RMIN_SWEEP}
    joint = {r: fit_joint(masked(curves, r)) for r in RMIN_SWEEP}

    with OUT_CSV.open("w") as f:
        f.write("fit,r_min,width,K_plat,l_b,l_b_lo,l_b_hi,S,rms\n")
        for r in RMIN_SWEEP:
            for wd, res, (x, y) in zip(WIDTHS, free[r], masked(curves, r)):
                if res is None:
                    continue
                p, sse, (lo, hi) = res
                f.write(f"free,{r:.5f},{wd:.5f},{p[0]:.4f},{10**p[1]:.5f},{10**lo:.5f},{10**hi:.5f},"
                        f"{p[2]:.4f},{np.sqrt(sse / len(x)):.4f}\n")
            a, s, xb, rms = joint[r]
            for wd, b in zip(WIDTHS, xb):
                f.write(f"joint,{r:.5f},{wd:.5f},{a + s * b:.4f},{10**b:.5f},,,{s:.4f},{rms:.4f}\n")

    print("Joint hinge fit vs r_min:")
    for r in RMIN_SWEEP:
        a, s, xb, rms = joint[r]
        print(f"  r_min={r:.4f}: A={a:.3f} S={s:.3f} rms={rms:.3f}")
    a, s, xb, _ = joint[RMIN_SWEEP[0]]
    print("\nr_min=0.001 joint:  width  l_b  l_b/width  K_plat")
    for wd, b in zip(WIDTHS, xb):
        print(f"   {wd:.3f}  {10**b:.4f}  {10**b / wd:.2f}  {a + s * b:.2f}")
    print("\nr_min=0.001 free:   width  K_plat  l_b [1sig]  S")
    for wd, res in zip(WIDTHS, free[RMIN_SWEEP[0]]):
        if res is None:
            continue
        p, _, (lo, hi) = res
        print(f"   {wd:.3f}  {p[0]:.2f}  {10**p[1]:.4f} [{10**lo:.4f},{10**hi:.4f}]  {p[2]:.2f}")

    # ---------------- figure ----------------
    cmap_w = plt.get_cmap("viridis")
    cmap_r = plt.get_cmap("magma")
    lw = np.log10(WIDTHS)
    wcol = lambda i: cmap_w((lw[i] - lw.min()) / (lw.max() - lw.min()))
    rcol = lambda j: cmap_r(0.15 + 0.7 * j / (len(RMIN_SWEEP) - 1))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    (a1, a2), (a3, a4) = ax

    a_j, s_j, xb_j, rms_j = joint[RMIN_SWEEP[0]]
    for i, (lc, kk) in enumerate(curves):
        a1.plot(10 ** lc, kk, ".", color=wcol(i), ms=4, alpha=0.7)
        xx = np.linspace(lc.min(), np.log10(L_MAX_FIT), 400)
        a1.plot(10 ** xx, a_j + s_j * np.maximum(xx, xb_j[i]), color=wcol(i), lw=1.6)
    a1.axvline(L_MAX_FIT, color="0.5", ls=":")
    a1.set(xscale="log", ylim=(1.5, 7), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title=rf"(a) joint hinge: $A={a_j:.2f}$, $S={s_j:.2f}$ (rms={rms_j:.2f})")
    sm = plt.cm.ScalarMappable(cmap=cmap_w, norm=matplotlib.colors.LogNorm(WIDTHS.min(), WIDTHS.max()))
    fig.colorbar(sm, ax=a1, label=r"$\Delta r_{\rm shell}/r_{\rm HII}$")

    # (b) free-fit S per width, all r_min -> is the decline really shared?
    for j, r in enumerate(RMIN_SWEEP):
        sv = np.array([res[0][2] if res else np.nan for res in free[r]])
        a2.plot(WIDTHS * (1 + 0.02 * j), sv, "-o", ms=4, lw=1.1, color=rcol(j),
                label=rf"$r_{{\rm min}}={r:.4g}$")
    a2.axhline(s_j, color="k", ls="--", lw=1, label="joint S")
    a2.set(xscale="log", ylim=(-4, 0), xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$",
           ylabel=r"decline slope $S$ (free fit)", title="(b) free-fit decline slope per width")
    a2.legend(fontsize=8)

    for j, r in enumerate(RMIN_SWEEP):
        _, _, xb, _ = joint[r]
        ok = 10 ** xb > 1.05 * r
        a3.plot(WIDTHS[ok], 10 ** xb[ok], "-o", color=rcol(j), ms=5, lw=1.2,
                label=rf"$r_{{\rm min}}={r:.4g}$")
        a3.plot(WIDTHS[~ok], 10 ** xb[~ok], "x", color=rcol(j), ms=6)
        a3.axhline(r, color=rcol(j), ls=":", lw=0.8)
    ww = np.array([0.015, 1.2])
    a3.plot(ww, ww, color="0.4", ls="--", lw=1, label=r"$\ell_b=\Delta r_{\rm shell}$")
    a3.plot(ww, 0.45 * ww, color="0.4", ls=":", lw=1, label=r"$\ell_b=0.45\,\Delta r_{\rm shell}$")
    a3.set(xscale="log", yscale="log", ylim=(5e-4, 2), xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$",
           ylabel=r"bend $\ell_b/r_{\rm HII}$", title=r"(c) joint-fit bend (x: $\ell_b$ below $r_{\rm min}$)")
    a3.legend(fontsize=8)

    for j, r in enumerate(RMIN_SWEEP):
        kp = np.array([res[0][0] if res else np.nan for res in free[r]])
        a4.plot(WIDTHS * (1 + 0.02 * j), kp, "-o", ms=4, lw=1.1, color=rcol(j),
                label=rf"$r_{{\rm min}}={r:.4g}$")
    a4.plot(WIDTHS, a_j + s_j * xb_j, "k--", lw=1, label="joint $A+S\\,x_b$")
    a4.set(xscale="log", ylim=(1.5, 8), xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$",
           ylabel=r"plateau $K_{\rm plat}$ (free fit)", title=r"(d) free-fit plateau vs width")
    a4.legend(fontsize=8)

    for a in ax.flat:
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170)
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
