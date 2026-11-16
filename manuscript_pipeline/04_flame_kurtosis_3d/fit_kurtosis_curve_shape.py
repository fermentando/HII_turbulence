#!/usr/bin/env python3
"""Fit the *shape* of K(l) instead of a straight line, and test r_min independence.

Data: N=100,000 no-pixelized random-point shells (v_bulk/v_turb = 10), the same
runs behind the r_min sweep in random_point_count_rmin_and_vbulk_sweeps.png.

Model (x = log10 l / r_HII). Each K(l) is a plateau that bends onto a declining
line:
    K(x) = K_plat + S * w * softplus((x - x_b) / w)
i.e. K = K_plat for x << x_b and K = K_plat + S (x - x_b) for x >> x_b.

Two versions are fitted:
  * free:  K_plat, x_b, S, w per shell width
  * joint: the declining line is shared by all widths (A + S x), each width
           only gets its bend x_b, so K_plat = A + S x_b.

r_min test: the bins with l < r_min are cut and the fit is repeated for each r_min
in RMIN_SWEEP. One width is also recomputed from scratch with min_separation =
r_min, to check that K(l) itself does not depend on r_min (only truncation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "00_core"))

import plot_fixed_rmin_comparison as fr  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402

BASE_DIR = fr.BASE_DIR
N_POINTS = 100_000
WIDTHS = fr.WIDTHS
RMIN_SWEEP = fr.RMIN_SWEEP
L_MAX_FIT = 0.6          # stay below the l ~ r_HII minimum / upturn
CHECK_WIDTH_INDEX = 5    # width 0.09: recomputed explicitly for each r_min
CACHE = BASE_DIR / "_kurtosis_curve_shape_cache.npz"
OUT_PNG = BASE_DIR / "kurtosis_curve_shape_fits.png"
OUT_CSV = BASE_DIR / "kurtosis_curve_shape_fits.csv"


def k_curve(y, z, v, r_min):
    lc, edges, dvb, _ = compute_bins(y, z, v, bin_resolution=fr.BIN_RESOLUTION, min_separation=r_min)
    with np.errstate(invalid="ignore", divide="ignore"):
        *_, kk = compute_pdf_and_statistics(y, z, v, np.zeros_like(v), lc, edges, dvb, order=1, n_jobs=1)
    return np.asarray(lc, float), np.asarray(kk, float)


def load_or_compute():
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        if np.allclose(c["widths"], WIDTHS) and np.allclose(c["rmin_sweep"], RMIN_SWEEP):
            return list(c["curves"]), list(c["check"])
    points, _ = fr.load_shell_points(N_POINTS, WIDTHS)
    curves = []
    for w, (y, z, v) in zip(WIDTHS, points):
        print(f"K(l) width={w:.3f}")
        curves.append(k_curve(y, z, v, RMIN_SWEEP[0]))
    y, z, v = points[CHECK_WIDTH_INDEX]
    check = []
    for r in RMIN_SWEEP:
        print(f"check width={WIDTHS[CHECK_WIDTH_INDEX]:.3f} r_min={r:.4f}")
        check.append(k_curve(y, z, v, r))
    np.savez(CACHE, widths=WIDTHS, rmin_sweep=RMIN_SWEEP,
             curves=np.array(curves, dtype=object), check=np.array(check, dtype=object))
    return curves, check


def softplus(t):
    return np.logaddexp(0.0, t)


def model(x, k_plat, x_b, s, w):
    return k_plat + s * w * softplus((x - x_b) / w)


def fit_mask(lc, kk, r_min):
    return np.isfinite(kk) & (10 ** lc >= r_min) & (10 ** lc <= L_MAX_FIT)


def fit_free(lc, kk, r_min):
    m = fit_mask(lc, kk, r_min)
    if m.sum() < 6:
        return None
    x, y = lc[m], kk[m]
    p0 = [np.median(y[:3]), np.log10(0.05), -1.7, 0.15]
    lo, hi = [0, -3.5, -6, 0.03], [15, 0.5, 0, 0.6]
    r = least_squares(lambda p: model(x, *p) - y, p0, bounds=(lo, hi))
    if not r.success:
        return None
    jtj = r.jac.T @ r.jac
    dof = max(len(y) - 4, 1)
    try:
        cov = np.linalg.inv(jtj) * (2 * r.cost / dof)
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(4, np.nan)
    return r.x, err, x.min()


def fit_joint(curves, r_min):
    """Shared A, S, w; one x_b per width. K_plat = A + S x_b."""
    data = [(lc[fit_mask(lc, kk, r_min)], kk[fit_mask(lc, kk, r_min)]) for lc, kk in curves]
    nw = len(data)

    def unpack(p):
        return p[0], p[1], p[2], p[3:]

    def resid(p):
        a, s, w, xb = unpack(p)
        out = []
        for (x, y), b in zip(data, xb):
            if len(x):
                out.append(a + s * b + s * w * softplus((x - b) / w) - y)
        return np.concatenate(out)

    p0 = np.concatenate(([1.0, -1.7, 0.15], np.linspace(-1.8, -0.4, nw)))
    lo = np.concatenate(([-10, -6, 0.03], np.full(nw, -4.0)))
    hi = np.concatenate(([10, 0, 0.6], np.full(nw, 0.5)))
    r = least_squares(resid, p0, bounds=(lo, hi))
    a, s, w, xb = unpack(r.x)
    rms = np.sqrt(np.mean(r.fun ** 2))
    return a, s, w, xb, rms


def main():
    curves, check = load_or_compute()

    free = {r: [fit_free(lc, kk, r) for lc, kk in curves] for r in RMIN_SWEEP}
    joint = {r: fit_joint(curves, r) for r in RMIN_SWEEP}

    with OUT_CSV.open("w") as f:
        f.write("fit,r_min,width,K_plat,K_plat_err,l_b,x_b_err,S,S_err,w,first_bin\n")
        for r in RMIN_SWEEP:
            for wd, res in zip(WIDTHS, free[r]):
                if res is None:
                    continue
                p, e, xmin = res
                f.write(f"free,{r:.5f},{wd:.5f},{p[0]:.4f},{e[0]:.4f},{10**p[1]:.5f},{e[1]:.4f},"
                        f"{p[2]:.4f},{e[2]:.4f},{p[3]:.4f},{10**xmin:.5f}\n")
            a, s, w, xb, rms = joint[r]
            for wd, b in zip(WIDTHS, xb):
                f.write(f"joint,{r:.5f},{wd:.5f},{a + s * b:.4f},,{10**b:.5f},,{s:.4f},,{w:.4f},\n")

    print("\nJoint fit (shared decline) vs r_min:")
    for r in RMIN_SWEEP:
        a, s, w, xb, rms = joint[r]
        print(f"  r_min={r:.4f}: A={a:.3f} S={s:.3f} w={w:.3f} rms={rms:.3f}  "
              f"l_b={np.array2string(10**xb, precision=3, max_line_width=200)}")

    # ---------------- figure ----------------
    cmap_w = plt.get_cmap("viridis")
    cmap_r = plt.get_cmap("rocket" if "rocket" in plt.colormaps() else "magma")
    lw = np.log10(WIDTHS)
    wcol = lambda i: cmap_w((lw[i] - lw.min()) / (lw.max() - lw.min()))
    rcol = lambda j: cmap_r(0.15 + 0.7 * j / (len(RMIN_SWEEP) - 1))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    (a1, a2), (a3, a4) = ax

    a_j, s_j, w_j, xb_j, rms_j = joint[RMIN_SWEEP[0]]
    for i, (lc, kk) in enumerate(curves):
        a1.plot(10 ** lc, kk, ".", color=wcol(i), ms=4, alpha=0.7)
        xx = np.linspace(lc.min(), np.log10(L_MAX_FIT), 200)
        a1.plot(10 ** xx, a_j + s_j * xb_j[i] + s_j * w_j * softplus((xx - xb_j[i]) / w_j),
                color=wcol(i), lw=1.6)
    xx = np.linspace(-2.7, np.log10(L_MAX_FIT), 50)
    a1.plot(10 ** xx, a_j + s_j * xx, "k--", lw=1, label=rf"shared decline $S={s_j:.2f}$")
    a1.axvline(L_MAX_FIT, color="0.5", ls=":")
    a1.set(xscale="log", ylim=(1.5, 7), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title=f"(a) joint fit, one bend per width (rms={rms_j:.2f})")
    sm = plt.cm.ScalarMappable(cmap=cmap_w, norm=matplotlib.colors.LogNorm(WIDTHS.min(), WIDTHS.max()))
    fig.colorbar(sm, ax=a1, label=r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    a1.legend(loc="lower left", fontsize=9)

    for j, (lc, kk) in enumerate(check):
        a2.plot(10 ** lc, kk, "-o", ms=3, lw=1.2, color=rcol(j),
                label=rf"$r_{{\rm min}}={RMIN_SWEEP[j]:.4g}$")
    a2.set(xscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title=rf"(b) recomputed with each $r_{{\rm min}}$, $\Delta r_{{\rm shell}}={WIDTHS[CHECK_WIDTH_INDEX]:.2f}$")
    a2.legend(fontsize=8)

    for j, r in enumerate(RMIN_SWEEP):
        _, _, _, xb, _ = joint[r]
        ok = 10 ** xb > 1.05 * r
        a3.plot(WIDTHS[ok], 10 ** xb[ok], "-o", color=rcol(j), ms=5, lw=1.2,
                label=rf"$r_{{\rm min}}={r:.4g}$")
        a3.plot(WIDTHS[~ok], 10 ** xb[~ok], "x", color=rcol(j), ms=6)
        a3.axhline(r, color=rcol(j), ls=":", lw=0.8)
    ww = np.array([0.015, 1.2])
    a3.plot(ww, ww, color="0.4", ls="--", lw=1, label=r"$\ell_b=\Delta r_{\rm shell}$")
    a3.set(xscale="log", yscale="log", xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$",
           ylabel=r"bend $\ell_b/r_{\rm HII}$",
           title=r"(c) joint-fit bend vs width (x: $\ell_b$ below $r_{\rm min}$)")
    a3.legend(fontsize=8)

    for j, r in enumerate(RMIN_SWEEP):
        kp = np.array([res[0][0] if res else np.nan for res in free[r]])
        ke = np.array([res[1][0] if res else np.nan for res in free[r]])
        a4.errorbar(WIDTHS * (1 + 0.02 * j), kp, yerr=ke, fmt="-o", ms=4, lw=1.1, capsize=2,
                    color=rcol(j), label=rf"$r_{{\rm min}}={r:.4g}$")
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
