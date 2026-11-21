#!/usr/bin/env python3
"""Fully analytic plateau kurtosis K(l -> 0; lambda) vs the grid-free simulations and the numerical model.

Thin shell, no inner cutoff (k_max = inf), outer scale k_min:

    K_plat(lambda) = K_b * [3 J2t + 4 lambda^2 (lambda^2 + 6)] / (Jt + 2 lambda^2)^2
    K_b = 3/2 + 2 ln 2 - ln(2 Delta)
    Jt  = J  - (27/40)  A   Delta^(2/3)          (= 2 lambda_c^2)
    J2t = J2 - (27/140) A^2 Delta^(4/3)
    J = int_0^2 D/s ds, J2 = int_0^2 D^2/s ds, A = (4/3) I k_min^(2/3)

It follows from <D>_0 = Jt/K_b, <D^2>_0 = J2t/K_b (manuscript_notes/geometric_averages/01_turbulent_D.md),
<dmu^2>_0 = 2/K_b, <dmu^4>_0 = 4/K_b (bulk), and <D dmu^2>_0 = 4/K_b (opposite-wall pairs, D = 2).

Output: analytic_plateau_kurtosis.png (K_plat vs lambda: simulation points with single-source 1 sigma,
numerical model as solid lines, analytic formula as dashed lines) and a table on stdout.
Run (needs matplotlib, seaborn):
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_analytic_plateau_kurtosis.py
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
import plot_gridfree_lambda_model_summary as ps  # noqa: E402
from gridfree_kolmogorov import I_INF, structure_function  # noqa: E402

KMIN = ps.tg.KMIN
A = (4 / 3) * I_INF * KMIN ** (2 / 3)
_s = np.geomspace(1e-9, 2, 200001)
_D = structure_function(_s, KMIN)
J = np.trapezoid(_D / _s, _s)
J2 = np.trapezoid(_D ** 2 / _s, _s)
OUT = ps.tg.OUT_DIR / "analytic_plateau_kurtosis.png"


def k_plat_analytic(lam, d):
    kb = 1.5 + 2 * np.log(2) - np.log(2 * d)
    jt = J - 27 / 40 * A * d ** (2 / 3)
    j2t = J2 - 27 / 140 * A ** 2 * d ** (4 / 3)
    lam = np.asarray(lam, float)
    with np.errstate(invalid="ignore"):
        out = kb * (3 * j2t + 4 * lam ** 2 * (lam ** 2 + 6)) / (jt + 2 * lam ** 2) ** 2
    return np.where(np.isinf(lam), kb, out)


def main():
    try:
        plt.style.use(str(ps.REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass
    print(f"k_min = {KMIN:.3f}: A = {A:.3f}, J = {J:.3f}, J2 = {J2:.3f}")
    data = [ps.load_width(w, 30000) for w in ps.WIDTHS]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    print(f"{'Delta':>6} {'lambda':>6} | {'sim':>5} {'1sig':>5} | {'num. model':>10} | {'analytic':>8}")
    for w, d in zip(ps.WIDTHS, data):
        col = ps.W_CMAP(ps.W_NORM(w))
        pl = d["plat"]
        ax.plot(ps.LAM_DENSE, [np.mean(ps.model_curve(d, l)[pl]) for l in ps.LAM_DENSE], "-", color=col, lw=2, alpha=0.6)
        ax.plot(ps.LAM_DENSE, k_plat_analytic(ps.LAM_DENSE, w), "--", color=col, lw=2.4)
        for lam in (0.0, np.inf):
            ax.plot(ps.lam_plot(lam), k_plat_analytic(lam, w), "_", color=col, ms=18, mew=2.4)
        dodge = 10 ** (0.035 * (ps.WIDTHS.index(w) - 1.5))
        x, y, e = [], [], []
        for lam in ps.SIM_LAMS:
            single = [np.nanmean(ps.tg.K_sim(d["acc"], lam, seeds=slice(s, s + 1), odd=True)[pl]) for s in range(ps.tg.N_SEEDS)]
            x.append(ps.lam_plot(lam) * dodge)
            y.append(np.nanmean(ps.sim_curve(d, lam)[pl]))
            e.append(np.nanstd(single, ddof=1))
            print(f"{w:6g} {lam:6g} | {y[-1]:5.2f} {e[-1]:5.2f} | {np.mean(ps.model_curve(d, lam)[pl]):10.2f} | "
                  f"{float(k_plat_analytic(lam, w)):8.2f}")
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col, elinewidth=1.2, capsize=3, alpha=0.9, zorder=4)
        ax.scatter(x, y, color=col, s=45, edgecolors="black", linewidths=0.8, zorder=5,
                   label=rf"$\Delta r_{{\rm shell}}/r_{{\rm HII}}={w:g}$")
    ax.axhline(3, color="0.3", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 150)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([ps.LAM_ZERO_PLOT, 1, 10, ps.LAM_INF_PLOT]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0", "1", "10", r"$\infty$"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for xb in (np.sqrt(ps.LAM_ZERO_PLOT * 0.2), np.sqrt(50 * ps.LAM_INF_PLOT)):
        ax.axvline(xb, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\lambda=v_{\rm bulk}/v_{\rm turb}$")
    ax.set_ylabel(r"$K_{\rm plat}$")
    ax.set_ylim(2.5, 8)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    h, l = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], color="0.3", lw=2, alpha=0.6), plt.Line2D([], [], color="0.3", lw=2.4, ls="--")]
    l += ["numerical model", "analytic formula"]
    ax.legend(h, l, loc="upper left", fontsize=10, ncol=2)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
