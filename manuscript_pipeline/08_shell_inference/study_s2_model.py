"""How the second-order structure function S2(l) depends on Delta and lambda for k_max -> inf turbulence.

S2 / v_turb^2 = <T> + lambda^2 <m^2>   (centroid model, fit_centroid_joint_grid.py; T = turbulent centroid
                                        variance, m = bulk centroid difference)
S2 / v_turb^2 = <D> + lambda^2 <dmu^2> (point samples; analytic (N1 + 2 q2 lambda^2) / P, fit_analytic_point_K.py)

Panels (study_s2_model.png):
  A  point samples: analytic S2(l) vs the grid-free 8-seed simulations (Delta = 0.02, 0.09, 0.3; lambda = 0, 3)
  B  centroid model, lambda = 0: S2(l) for several Delta; k_max -> inf (thick) vs the old inner scale 0.005 (thin)
  C  centroid model, Delta = 0.03: S2(l) for several lambda (shape change from the bulk term)
  D  power-law index d ln S2 / d ln l over l = 0.05-0.5 vs Delta for several lambda: centroid (solid), point
     (dashed); grey band: observed indices (noise-subtracted, all regions/lines)
Outer scale L_out = 1.2 r_HII (as the simulations), back-side weight a = 0, no noise.

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/study_s2_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core"), str(HERE.parent / "04_flame_kurtosis_3d")]
import fit_analytic_point_K as apk  # noqa: E402
import fit_centroid_joint_grid as fcj  # noqa: E402
import test_gridfree_kolmogorov_model as tg  # noqa: E402
from fit_kurtosis_formula import DATA_DIR  # noqa: E402
from fit_kurtosis_s2_joint_grid import load  # noqa: E402

OUT = fcj.OUT_BASE / "s2_model_study"
L_OUT = 2 * np.pi / tg.KMIN  # 1.2
ELL = np.geomspace(0.01, 1.0, 22)
LAMS = (0.0, 1.0, 3.0, 10.0)
WIN = (0.05, 0.5)  # l range of the power-law index (typical observed range)
A_IDX = 0  # a = 0


def index(ell, s2):
    m = (ell >= WIN[0]) & (ell <= WIN[1])
    return np.polyfit(np.log(ell[m]), np.log(s2[..., m].reshape(-1, m.sum()).T), 1)[0].reshape(s2.shape[:-1])


def centroid_s2(l_in):
    cache = OUT / f"_centroid_moments_lin{l_in:g}.npz"
    if cache.exists():
        return np.load(cache)["mom"]
    old = fcj.L_IN
    fcj.L_IN = l_in
    try:
        mom = fcj.centroid_moments(ELL, L_OUT)[..., A_IDX]  # (5, l, D)
    finally:
        fcj.L_IN = old
    np.savez(cache, mom=mom)
    return mom


def observed_indices():
    out = []
    for p in sorted(DATA_DIR.glob("*_formula_data.npz")):
        region, X = p.name.rsplit("_", 3)[0], p.name.rsplit("_", 3)[1]
        d = load(region, X, "overlap")
        if d is None:
            continue
        s = d["S2"] - d["N2"]
        m = (d["ell"] >= WIN[0]) & (d["ell"] <= WIN[1]) & (s > 0)
        if m.sum() >= 3:
            out.append(np.polyfit(np.log(d["ell"][m]), np.log(s[m]), 1)[0])
    return np.array(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    (aA, aB), (aC, aD) = axes
    cm = plt.colormaps["viridis"]

    # A: point model vs simulations
    ellc = np.sqrt(tg.EDGES[:-1] * tg.EDGES[1:])
    for k, w in enumerate((0.02, 0.09, 0.3)):
        c = tg.accumulate(w, 30000)
        good = (c["n"] >= 200) & (ellc <= 0.6)
        e, n = ellc[good], c["n"][good]
        P, q2, q4, N1, N2, chi = apk.terms(e, w, tg.KMIN)
        col = cm(k / 2.5)
        for lam, mk in ((0.0, "o"), (3.0, "s")):
            sim = (c["g2"][:, good].sum(0) + lam ** 2 * 8 * c["m2"][good]) / (8 * n)
            aA.plot(e, sim, mk, ms=3.5, color=col, alpha=0.7)
            aA.plot(e, (N1 + 2 * q2 * lam ** 2) / P, "-" if lam == 0 else "--", color=col, lw=1.6,
                    label=fr"$\Delta$={w:g}, $\lambda$={lam:g}")
    aA.set(xscale="log", yscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$S_2/v_{\rm turb}^2$",
           title="A  point samples: analytic (lines) vs sims (points)")
    aA.legend(fontsize=8, ncol=2)

    mom_new, mom_old = centroid_s2(0.0), centroid_s2(0.005)
    deltas = fcj.D_GRID

    # B: centroid, lambda = 0, several Delta, new vs old inner scale
    show = [0.005, 0.016, 0.053, 0.171, 0.555]
    for k, dv in enumerate(show):
        j = int(np.argmin(np.abs(deltas - dv)))
        col = cm(k / (len(show) - 0.5))
        aB.plot(ELL, mom_new[0, :, j], color=col, lw=2, label=fr"$\Delta$={deltas[j]:.3f}")
        aB.plot(ELL, mom_old[0, :, j], color=col, lw=0.9, ls=":")
    aB.plot(ELL, 0.5 * ELL ** (2 / 3), color="0.5", lw=0.8, ls="--")
    aB.text(0.012, 0.5 * 0.012 ** (2 / 3) * 1.25, r"$\propto\ell^{2/3}$", fontsize=9, color="0.4")
    aB.set(xscale="log", yscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$S_2/v_{\rm turb}^2$ ($\lambda=0$)",
           title=r"B  centroid, $\lambda=0$: $k_{\max}\to\infty$ (solid), $\ell_{\rm in}=0.005$ (dotted)")
    aB.legend(fontsize=8)

    # C: centroid, Delta = 0.03, several lambda
    j = int(np.argmin(np.abs(deltas - 0.03)))
    for k, lam in enumerate(LAMS):
        s2 = mom_new[0, :, j] + lam ** 2 * mom_new[2, :, j]
        aC.plot(ELL, s2 / s2[-1], color=plt.colormaps["mako" if "mako" in plt.colormaps else "plasma"](0.15 + 0.2 * k),
                lw=2, label=fr"$\lambda$={lam:g}")
    aC.set(xscale="log", yscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$S_2(\ell)/S_2(\ell=1)$",
           title=fr"C  centroid, $\Delta$={deltas[j]:.3f}: bulk steepens $S_2$")
    aC.legend(fontsize=9)

    # D: power-law index vs Delta
    obs = observed_indices()
    aD.axhspan(np.percentile(obs, 16), np.percentile(obs, 84), color="0.85", zorder=0)
    aD.axhline(np.median(obs), color="0.6", lw=1, zorder=0)
    pt_d = apk.D_GRID
    pt_terms = [apk.terms(ELL, dv, tg.KMIN) for dv in pt_d]
    for k, lam in enumerate(LAMS):
        col = plt.colormaps["plasma"](0.1 + 0.22 * k)
        s2c = mom_new[0] + lam ** 2 * mom_new[2]  # (l, D)
        aD.plot(deltas, index(ELL, s2c.T), color=col, lw=2, label=fr"$\lambda$={lam:g}")
        s2o = mom_old[0] + lam ** 2 * mom_old[2]
        aD.plot(deltas, index(ELL, s2o.T), color=col, lw=0.8, ls=":")
        s2p = np.array([(t[3] + 2 * t[1] * lam ** 2) / t[0] for t in pt_terms])
        aD.plot(pt_d, index(ELL, s2p), color=col, lw=1.4, ls="--")
    aD.plot([], [], color="0.3", lw=2, label="centroid")
    aD.plot([], [], color="0.3", lw=0.8, ls=":", label=r"centroid, $\ell_{\rm in}=0.005$")
    aD.plot([], [], color="0.3", lw=1.4, ls="--", label="point samples")
    aD.axhline(2 / 3, color="0.4", lw=0.7, ls="-.")
    aD.set(xscale="log", xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$",
           ylabel=fr"$d\ln S_2/d\ln\ell$ over $\ell$={WIN[0]:g}-{WIN[1]:g}",
           title=fr"D  $S_2$ index; grey: observed 16-84\\% (N={len(obs)})")
    aD.legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.title.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(OUT / "study_s2_model.png", dpi=160)
    plt.close(fig)

    # table
    print("observed index: median %.2f, 16-84%% %.2f-%.2f" % (np.median(obs), *np.percentile(obs, [16, 84])))
    print("centroid index (k_max=inf) vs Delta:")
    print("  Delta  " + "  ".join(f"lam={lam:<4g}" for lam in LAMS) + "   | old l_in, lam=0")
    for jj in range(0, len(deltas), 3):
        row = [index(ELL, (mom_new[0, :, jj] + lam ** 2 * mom_new[2, :, jj])[None])[0] for lam in LAMS]
        old0 = index(ELL, mom_old[0, :, jj][None])[0]
        print(f"  {deltas[jj]:.3f}  " + "  ".join(f"{v:8.2f}" for v in row) + f"   | {old0:.2f}")
    print(f"wrote {OUT / 'study_s2_model.png'}")


if __name__ == "__main__":
    main()
