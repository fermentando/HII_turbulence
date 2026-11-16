#!/usr/bin/env python3
"""Analytic lambda-dependence of K(l) for Gaussian turbulence + shell expansion.

Each turbulent velocity component is an isotropic Gaussian random field with unit
variance and correlation rho(d). For a pair with 3D separation d and bulk
difference dmu (both fixed by the geometry), the turbulent difference du is
Gaussian with variance S(d) = 2 [1 - rho(d)], independent of dmu. Hence, with
lambda = v_bulk / v_turb,

  <dv^2> = <S> + lambda^2 <dmu^2>
  <dv^4> = 3 <S^2> + 6 lambda^2 <S dmu^2> + lambda^4 <dmu^4>

(averages over pairs at sky separation l). Writing T = <S>, B = <dmu^2>,
f = lambda^2 B / (T + lambda^2 B):

  K = K_t (1-f)^2 + 6 C f (1-f) + K_b f^2,
  K_t = 3 <S^2>/T^2,  K_b = <dmu^4>/B^2,  C = <S dmu^2>/(T B).

rho(d) is the ensemble correlation of the generator's spectrum
(amplitude k^(-11/6) for k_min <= k <= k_max), so the prediction uses no
turbulence sample. It is compared with the sampled-field pair moments of
kurtosis_shape_vs_vratio_fixed_width.py (same points, same seeds).
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
from fit_kurtosis_curve_shape import fit_free  # noqa: E402
from generate_random_point_expanding_shell_grid import random_points_in_shell  # noqa: E402

WIDTHS = [0.02, 0.09, 0.3, 1.0]
GKEYS = ["n", "T", "S2", "B", "B2", "SB"]
LAM_FIT = np.array([0.0, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, np.inf])
OUT_PNG = kv.BASE_DIR / "kurtosis_vratio_dependence_derivation.png"
OUT_CSV = kv.BASE_DIR / "kurtosis_vratio_dependence_derivation.csv"


def correlation_function():
    """Ensemble rho(d) of the generator spectrum, on a fine d grid."""
    n, box = kv.N_GRID, kv.BOX
    freq = np.fft.fftfreq(n, d=box / n) * 2 * np.pi
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2).ravel()
    kmin, kmax = 2.0 * 2 * np.pi / box, 0.35 * np.pi * n / box
    k = k[(k >= kmin) & (k <= kmax)]
    p = k ** (-11.0 / 3.0)
    d = np.linspace(0, 2.0, 2001)
    rho = np.array([np.sum(p * np.sinc(k * dd / np.pi)) for dd in d]) / p.sum()
    return d, rho


def geometric_moments(width, d_grid, s_grid):
    tag = f"w{width:g}".replace(".", "p")
    cache = kv.BASE_DIR / f"_kurtosis_vratio_geom_{tag}_n{kv.N_POINTS}.npz"
    if cache.exists():
        c = np.load(cache)
        return {k: c[k] for k in GKEYS}
    p = random_points_in_shell(kv.N_POINTS, 1.0 - width, 1.0, np.random.default_rng(kv.POINT_SEED))
    mu = p[:, 0] / np.linalg.norm(p, axis=1)
    x, y, z = p.T
    nb = len(kv.EDGES) - 1
    e2 = kv.EDGES ** 2
    acc = {k: np.zeros(nb) for k in GKEYS}
    for ii in range(kv.N_POINTS - 1):
        j = slice(ii + 1, None)
        dy, dz = y[j] - y[ii], z[j] - z[ii]
        d2 = dy * dy + dz * dz
        b = np.searchsorted(e2, d2) - 1
        ok = (b >= 0) & (b < nb)
        b = b[ok]
        d3 = np.sqrt(d2[ok] + (x[j][ok] - x[ii]) ** 2)
        S = np.interp(d3, d_grid, s_grid)
        dm2 = (mu[j][ok] - mu[ii]) ** 2
        for k, val in (("n", np.ones_like(S)), ("T", S), ("S2", S * S), ("B", dm2),
                       ("B2", dm2 * dm2), ("SB", S * dm2)):
            acc[k] += np.bincount(b, weights=val, minlength=nb)
    np.savez(cache, **acc)
    return acc


def k_pred(G, lam):
    with np.errstate(invalid="ignore", divide="ignore"):
        if np.isinf(lam):
            return G["B2"] * G["n"] / G["B"] ** 2
        s2 = G["T"] + lam ** 2 * G["B"]
        s4 = 3 * G["S2"] + 6 * lam ** 2 * G["SB"] + lam ** 4 * G["B2"]
        return s4 * G["n"] / s2 ** 2


def components(G):
    n = G["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        T, B = G["T"] / n, G["B"] / n
        return dict(T=T, B=B, Kt=3 * (G["S2"] / n) / T ** 2, Kb=(G["B2"] / n) / B ** 2,
                    C=(G["SB"] / n) / (T * B))


def main():
    d_grid, rho = correlation_function()
    s_grid = 2 * (1 - rho)
    ell = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    lc = np.log10(ell)

    res = {}
    for w in WIDTHS:
        print(f"width {w}")
        G = geometric_moments(w, d_grid, s_grid)
        tag = f"w{w:g}".replace(".", "p")
        # sampled-field moments pooled over 8 field seeds (kurtosis_vratio_seed_ensemble.py)
        c = np.load(kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}_seeds8.npz")
        M = {k: c[k].sum(0) for k in kv.KEYS}
        good = G["n"] >= 200
        fits_p = {l: fit_free(lc, np.where(good, k_pred(G, l), np.nan), kv.EDGES[0]) for l in LAM_FIT}
        fits_s = {l: fit_free(lc, np.where(good, kv.k_of_lambda(M, l), np.nan), kv.EDGES[0]) for l in LAM_FIT}
        res[w] = dict(G=G, M=M, good=good, fp=fits_p, fs=fits_s, comp=components(G))

    # ---- plateau-level closed form: K_plat(lambda) from l->0 components ----
    print("\nPlateau components (first 5 good bins):  K_t   K_b   C    B/T")
    for w in WIDTHS:
        cp, good = res[w]["comp"], res[w]["good"]
        idx = np.flatnonzero(good)[:5]
        print(f"  Delta={w:<5g}  {np.mean(cp['Kt'][idx]):.2f}  {np.mean(cp['Kb'][idx]):.2f}  "
              f"{np.mean(cp['C'][idx]):.2f}  {np.mean(cp['B'][idx] / cp['T'][idx]):.3f}")

    with OUT_CSV.open("w") as f:
        f.write("width,lambda,K_plat_pred,K_plat_sim,lb_pred,lb_sim,S_pred,S_sim\n")
        print("\nSoftplus parameters, prediction vs sampled field:")
        print("  width lambda   K_plat p/s      l_b/D p/s      S p/s")
        for w in WIDTHS:
            for l in LAM_FIT:
                a, b = res[w]["fp"][l], res[w]["fs"][l]
                if a is None or b is None:
                    continue
                pa, pb = a[0], b[0]
                f.write(f"{w},{l},{pa[0]:.4f},{pb[0]:.4f},{10**pa[1]:.5f},{10**pb[1]:.5f},{pa[2]:.4f},{pb[2]:.4f}\n")
                if l in (0.0, 1.0, 3.0, 10.0, np.inf):
                    print(f"  {w:<5g} {l:>6g}   {pa[0]:.2f}/{pb[0]:.2f}     {10**pa[1]/w:.2f}/{10**pb[1]/w:.2f}"
                          f"      {pa[2]:.2f}/{pb[2]:.2f}")

    # ---------------- figure ----------------
    cmap = plt.get_cmap("viridis")
    lam_show = [0.0, 1.0, 2.0, 3.0, 10.0, np.inf]
    lcol = {l: cmap(i / (len(lam_show) - 1)) for i, l in enumerate(lam_show)}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))

    for a, w in zip([ax[0, 0], ax[0, 1]], [0.02, 0.3]):
        r = res[w]
        for l in lam_show:
            a.plot(ell[r["good"]], kv.k_of_lambda(r["M"], l)[r["good"]], "o", ms=2.5, color=lcol[l], alpha=0.6)
            a.plot(ell[r["good"]], k_pred(r["G"], l)[r["good"]], "-", lw=1.6, color=lcol[l],
                   label=rf"$\lambda={l:g}$" if np.isfinite(l) else r"$\lambda=\infty$")
        a.set(xscale="log", ylim=(1.5, 7), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$K(\ell)$",
              title=rf"$\Delta={w}$: points = sampled (8 seeds pooled), lines = prediction")
        a.legend(fontsize=8, ncol=2)

    r = res[0.09]
    cp, good = r["comp"], r["good"]
    a = ax[0, 2]
    a.plot(ell[good], cp["Kt"][good], color="0.4", lw=2, label=r"$K_t=3\langle S^2\rangle/\langle S\rangle^2$")
    a.plot(ell[good], cp["Kb"][good], color="crimson", lw=2, label=r"$K_b=\langle\delta\mu^4\rangle/\langle\delta\mu^2\rangle^2$")
    a.plot(ell[good], 3 * cp["C"][good], color="C0", lw=2, label=r"$3C=3\langle S\delta\mu^2\rangle/\langle S\rangle\langle\delta\mu^2\rangle$")
    a.axhline(3, color="k", ls=":", lw=1)
    a.set(xscale="log", ylim=(1.5, 6), xlabel=r"$\ell/r_{\rm HII}$", ylabel="value",
          title=r"$\Delta=0.09$: the three ingredients")
    a.legend(fontsize=8)

    a = ax[1, 0]
    for l in [0.5, 1.0, 2.0, 3.0, 10.0]:
        f_ = l ** 2 * cp["B"] / (cp["T"] + l ** 2 * cp["B"])
        a.plot(ell[good], f_[good], lw=1.8, label=rf"$\lambda={l:g}$")
    a.set(xscale="log", ylim=(0, 1), xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$f$ (bulk share of $\langle\delta v^2\rangle$)",
          title=r"$\Delta=0.09$: bulk variance fraction $f(\ell;\lambda)$")
    a.legend(fontsize=8)

    lam_d = np.geomspace(0.1, 300, 200)
    for a, key, ylabel in ((ax[1, 1], 0, r"$K_{\rm plat}$"), (ax[1, 2], 2, r"decline slope $S$")):
        for i, w in enumerate(WIDTHS):
            col = cmap(i / (len(WIDTHS) - 1))
            r = res[w]
            lf = np.array([l for l in LAM_FIT if np.isfinite(l) and l > 0 and r["fs"][l] is not None])
            a.plot(lf, [r["fs"][l][0][key] for l in lf], "o", color=col, ms=5)
            lp = np.array([l for l in LAM_FIT if np.isfinite(l) and l > 0 and r["fp"][l] is not None])
            a.plot(lp, [r["fp"][l][0][key] for l in lp], "-", color=col, lw=1.6, label=rf"$\Delta={w:g}$")
            if key == 0:
                idx = np.flatnonzero(r["good"])[:5]
                cp_ = r["comp"]
                Kt, Kb, C = (np.mean(cp_[k][idx]) for k in ("Kt", "Kb", "C"))
                q = np.mean(cp_["B"][idx] / cp_["T"][idx])
                f0 = lam_d ** 2 * q / (1 + lam_d ** 2 * q)
                a.plot(lam_d, Kt * (1 - f0) ** 2 + 6 * C * f0 * (1 - f0) + Kb * f0 ** 2, ":", color=col, lw=1.4)
        a.set(xscale="log", xlabel=r"$\lambda=v_{\rm bulk}/v_{\rm turb}$", ylabel=ylabel)
        a.grid(True, ls="--", alpha=0.4)
    ax[1, 1].set_title(r"$K_{\rm plat}(\lambda)$: points sim, lines pred. fit, dotted closed form")
    ax[1, 2].set_title(r"$S(\lambda)$: points sampled (8 seeds), lines prediction")
    ax[1, 1].legend(fontsize=8)
    for a in ax.flat:
        a.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"wrote {OUT_PNG}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
