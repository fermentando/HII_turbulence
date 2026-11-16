#!/usr/bin/env python3
"""How K(l) changes with v_bulk/v_turb at a fixed shell width.

v = v_turb * u + v_bulk * mu   (u: unit-variance turbulent x-velocity, mu = x/r)
With lambda = v_bulk / v_turb and per-pair du = u_i - u_j, dmu = mu_i - mu_j:

    <dv^2> ~ <du^2> + 2 l <du dmu> + l^2 <dmu^2>
    <dv^4> ~ <du^4> + 4 l <du^3 dmu> + 6 l^2 <du^2 dmu^2> + 4 l^3 <du dmu^3> + l^4 <dmu^4>

so one pass over the pairs, accumulating these 8 sums per separation bin, gives
K(l; lambda) exactly for every lambda (same points, same field).

"Independent mixing" approximation: odd terms -> 0 and <du^2 dmu^2> -> <du^2><dmu^2>:
    K_mix = (K_t s_t^2 + 6 s_t s_b + K_b s_b^2) / (s_t + s_b)^2,
    s_t = <du^2>, s_b = lambda^2 <dmu^2>, K_t / K_b the pure-turbulence / pure-bulk kurtosis.

All pairs of N_POINTS points are used (no subsampling), brute force in blocks.
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
sys.path.insert(0, str(HERE.parents[0] / "00_core"))

from derive_bulk_limit_kurtosis import k_theory  # noqa: E402
from fit_kurtosis_curve_shape import fit_free, softplus  # noqa: E402
from generate_random_point_expanding_shell_grid import random_points_in_shell, sample_periodic_grid  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402

BASE_DIR = HERE.parents[1] / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
WIDTH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.09  # usage: script.py [width]
N_POINTS = 30_000
FIELD_SEED, POINT_SEED, N_GRID, BOX = 7, 1234, 160, 2.4
EDGES = 10 ** np.arange(np.log10(2e-3), np.log10(2.0) + 1e-9, 1 / 20)
LAMBDAS = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, np.inf])
TAG = f"w{WIDTH:g}".replace(".", "p")
CACHE = BASE_DIR / f"_kurtosis_vratio_moments_{TAG}_n{N_POINTS}.npz"
OUT_PNG = BASE_DIR / f"kurtosis_shape_vs_vratio_{TAG}.png"
OUT_CSV = BASE_DIR / f"kurtosis_shape_vs_vratio_{TAG}.csv"
KEYS = ["n", "u2", "u4", "m2", "m4", "u1m1", "u3m1", "u2m2", "u1m3"]


def pair_moments():
    if CACHE.exists():
        c = np.load(CACHE)
        return {k: c[k] for k in KEYS}
    grids = kolmogorov_velocity_grid(N_GRID, BOX, np.random.default_rng(FIELD_SEED))
    p = random_points_in_shell(N_POINTS, 1.0 - WIDTH, 1.0, np.random.default_rng(POINT_SEED))
    u = sample_periodic_grid(grids, p, BOX)[:, 0]
    u = (u - u.mean()) / u.std()
    mu = p[:, 0] / np.linalg.norm(p, axis=1)
    y, z = p[:, 1], p[:, 2]
    nb = len(EDGES) - 1
    acc = {k: np.zeros(nb) for k in KEYS}
    e2 = EDGES ** 2
    block = 400
    for s in range(0, N_POINTS - 1, block):
        i = np.arange(s, min(s + block, N_POINTS - 1))
        for ii in i:
            j = slice(ii + 1, N_POINTS)
            d2 = (y[j] - y[ii]) ** 2 + (z[j] - z[ii]) ** 2
            b = np.searchsorted(e2, d2) - 1
            ok = (b >= 0) & (b < nb)
            b = b[ok]
            du = (u[j] - u[ii])[ok]
            dm = (mu[j] - mu[ii])[ok]
            du2, dm2 = du * du, dm * dm
            for k, val in (("n", np.ones_like(du)), ("u2", du2), ("u4", du2 * du2), ("m2", dm2),
                           ("m4", dm2 * dm2), ("u1m1", du * dm), ("u3m1", du2 * du * dm),
                           ("u2m2", du2 * dm2), ("u1m3", du * dm2 * dm)):
                acc[k] += np.bincount(b, weights=val, minlength=nb)
        if (s // block) % 10 == 0:
            print(f"  {s}/{N_POINTS}")
    np.savez(CACHE, edges=EDGES, **acc)
    return acc


def k_of_lambda(M, lam):
    n = M["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        if np.isinf(lam):
            return M["m4"] * n / M["m2"] ** 2
        s2 = M["u2"] + 2 * lam * M["u1m1"] + lam ** 2 * M["m2"]
        s4 = (M["u4"] + 4 * lam * M["u3m1"] + 6 * lam ** 2 * M["u2m2"]
              + 4 * lam ** 3 * M["u1m3"] + lam ** 4 * M["m4"])
        return s4 * n / s2 ** 2


def k_mix(M, lam):
    n = M["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        st, sb = M["u2"] / n, M["m2"] / n
        kt, kb = M["u4"] / n / st ** 2, M["m4"] / n / sb ** 2
        if np.isinf(lam):
            return kb
        sb = lam ** 2 * sb
        return (kt * st ** 2 + 6 * st * sb + kb * sb ** 2) / (st + sb) ** 2


def main():
    M = pair_moments()
    lc = np.log10(np.sqrt(EDGES[:-1] * EDGES[1:]))
    ell = 10 ** lc
    good = M["n"] >= 200
    k_bulk_th = k_theory(ell, 1.0 - WIDTH)

    with np.errstate(invalid="ignore", divide="ignore"):
        st, sb = M["u2"] / M["n"], M["m2"] / M["n"]
        corr = (M["u2m2"] / M["n"]) / (st * sb)          # <du^2 dmu^2>/(<du^2><dmu^2>)

    fits, rows = {}, []
    for lam in LAMBDAS:
        k = np.where(good, k_of_lambda(M, lam), np.nan)
        res = fit_free(lc, k, EDGES[0])
        fits[lam] = res
        if res is not None:
            p, e, _ = res
            rows.append((lam, *p, *e))
    with OUT_CSV.open("w") as f:
        f.write("v_bulk_over_v_turb,K_plat,x_b,S,w,K_plat_err,x_b_err,S_err,w_err\n")
        for r in rows:
            f.write(",".join(f"{v:.5g}" for v in r) + "\n")

    print(f"Delta={WIDTH}, N={N_POINTS:,} (all pairs). Free softplus fits:")
    print(" lambda   K_plat   l_b     l_b/D    S      w")
    for lam in LAMBDAS:
        res = fits[lam]
        if res is None:
            print(f" {lam:>6g}   no fit"); continue
        p, e, _ = res
        print(f" {lam:>6g}   {p[0]:.2f}   {10**p[1]:.4f}  {10**p[1]/WIDTH:.2f}   {p[2]:.2f}  {p[3]:.2f}")
    sel = (ell > 0.005) & (ell < 0.6) & good
    print(f"<du^2 dmu^2>/(<du^2><dmu^2>) over 0.005<l<0.6: min {np.nanmin(corr[sel]):.3f} "
          f"max {np.nanmax(corr[sel]):.3f}")
    for lam in [1.0, 3.0, 10.0]:
        d = np.nanmax(np.abs(k_of_lambda(M, lam) - k_mix(M, lam))[sel])
        print(f"lambda={lam:g}: max |K_exact - K_mix| = {d:.3f}")

    # ---------------- figure ----------------
    fin = LAMBDAS[np.isfinite(LAMBDAS) & (LAMBDAS > 0)]
    cmap = plt.get_cmap("mako" if "mako" in plt.colormaps() else "viridis")
    norm = matplotlib.colors.LogNorm(fin.min(), fin.max())
    lcol = lambda lam: "0.45" if lam == 0 else ("crimson" if np.isinf(lam) else cmap(0.1 + 0.8 * norm(lam)))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    (a1, a2), (a3, a4) = ax

    for lam in LAMBDAS:
        k = np.where(good, k_of_lambda(M, lam), np.nan)
        lab = r"$\lambda=0$ (turb.)" if lam == 0 else (r"$\lambda=\infty$ (bulk)" if np.isinf(lam) else None)
        a1.plot(ell, k, "o", ms=3, color=lcol(lam), alpha=0.75, label=lab)
        res = fits[lam]
        if res is not None:
            p = res[0]
            xx = np.linspace(lc[good].min(), np.log10(0.6), 200)
            a1.plot(10 ** xx, p[0] + p[2] * p[3] * softplus((xx - p[1]) / p[3]), "-", color=lcol(lam), lw=1.3)
    a1.plot(ell, k_bulk_th, "k:", lw=1.2, label="bulk-limit theory")
    a1.axvline(0.6, color="0.6", ls=":")
    a1.set(xscale="log", ylim=(1.5, 6.5), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title=rf"(a) $K(\ell)$ at $\Delta r_{{\rm shell}}={WIDTH}$, lines: softplus fits")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=a1, label=r"$\lambda = v_{\rm bulk}/v_{\rm turb}$")
    a1.legend(fontsize=8, loc="lower left")

    lam_f = np.array([l for l in LAMBDAS if fits[l] is not None and np.isfinite(l) and l > 0])
    par = np.array([fits[l][0] for l in lam_f])
    err = np.array([fits[l][1] for l in lam_f])
    pinf = fits[np.inf][0] if fits[np.inf] is not None else None
    a2.errorbar(lam_f, par[:, 0], yerr=err[:, 0], fmt="-o", color="C0", label=r"$K_{\rm plat}$")
    a2.set(xscale="log", xlabel=r"$\lambda$", ylabel=r"$K_{\rm plat}$", title="(b) fitted parameters vs $\\lambda$")
    a2b = a2.twinx()
    a2b.errorbar(lam_f, 10 ** par[:, 1] / WIDTH, fmt="-s", color="C3", label=r"$\ell_b/\Delta$")
    a2b.errorbar(lam_f, -par[:, 2], fmt="-^", color="C2", label=r"$-S$")
    a2b.set_ylabel(r"$\ell_b/\Delta$,  $-S$")
    if pinf is not None:
        a2.axhline(pinf[0], color="C0", ls=":", lw=1)
        a2b.axhline(10 ** pinf[1] / WIDTH, color="C3", ls=":", lw=1)
        a2b.axhline(-pinf[2], color="C2", ls=":", lw=1)
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    a3.plot(ell[good], st[good], "-", color="0.45", label=r"$\langle\delta u^2\rangle$ (turb., $v_{\rm turb}=1$)")
    a3.plot(ell[good], sb[good], "-", color="crimson", label=r"$\langle\delta\mu^2\rangle$ (bulk, $v_{\rm bulk}=1$)")
    a3.set(xscale="log", yscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel="second moment")
    a3b = a3.twinx()
    a3b.plot(ell[good], corr[good], "--", color="k", lw=1, label=r"$\langle\delta u^2\delta\mu^2\rangle/\langle\delta u^2\rangle\langle\delta\mu^2\rangle$")
    a3b.set_ylabel("cross-term ratio (1 = independent)")
    a3.set_title("(c) the two velocity components, per separation")
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = a3b.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    for lam in [0.5, 1.0, 3.0, 10.0]:
        a4.plot(ell[good], k_of_lambda(M, lam)[good], "o", ms=3, color=lcol(lam), label=rf"exact $\lambda={lam:g}$")
        a4.plot(ell[good], k_mix(M, lam)[good], "-", color=lcol(lam), lw=1.3)
    a4.plot([], [], "k-", label="independent mixing")
    a4.set(xscale="log", ylim=(1.5, 6.5), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
           title="(d) exact vs independent-mixing prediction")
    a4.legend(fontsize=8)

    for a in ax.flat:
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=170)
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
