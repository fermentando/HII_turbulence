#!/usr/bin/env python3
"""Manuscript summary: how the thin-shell prediction tracks K(l) across lambda = v_bulk/v_turb,
for Kolmogorov turbulence with NO inner cutoff (k_max -> inf).

kurtosis_lambda_model_summary.png, one column:
  top    -- K(l) at Delta r_shell = SHAPE_WIDTH for several lambda: 8-seed simulation, odd terms removed
           (= mirror pooling, points) and the model (lines); colour = lambda (mako)
  bottom -- plateau K_plat vs lambda for each width (viridis): simulation (points) and model (lines);
           lambda = 0 and lambda = inf at the axis ends. Error bars = scatter of K_plat between single
           (unmirrored, odd terms included) realisations, i.e. the uncertainty for one source.

Simulation: grid-free random-mode fields (gridfree_kolmogorov.py), pair-moment caches of
test_gridfree_kolmogorov_model.py (N = 30,000, all pairs, 8 seeds, k_min = 5.24, k_max = 1e5, i.e. an inner
scale of 6e-5 r_HII). These replace the 160^3 grid fields (k_max = 73, inner scale 0.086 r_HII > thin walls).
Model: thin-shell 2/K_b, 4/K_b with exact K_b, Abel-projected <D>, <D^2>, approximate cross term, with the
continuum D(s) for k_max -> inf (differs from k_max = 1e5 by <= 0.004 in K).
Plateau: bins with l <= PLATEAU_FRAC * l_b (l_b = 2e^{-3/2} Delta), at least the first PLATEAU_MIN_BINS.

Run (needs matplotlib, seaborn):
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_lambda_model_summary.py [--n-points 30000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import test_gridfree_kolmogorov_model as tg  # noqa: E402

WIDTHS = [0.02, 0.09, 0.3, 1.0]
SHAPE_WIDTH = 0.09
SHAPE_LAMS = [0.0, 1.0, 10.0, np.inf]
SIM_LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, np.inf]
MIN_PAIRS = 200
PLATEAU_FRAC = 0.25
PLATEAU_MIN_BINS = 5
BEND_RATIO = 2 * np.exp(-1.5)

LAM_ZERO_PLOT, LAM_INF_PLOT = 0.1, 100.0
LAM_DENSE = np.geomspace(0.2, 50.0, 80)

OUT = tg.OUT_DIR.parent / "kurtosis_lambda_model_summary.png"

_MAKO = sns.color_palette("mako", as_cmap=True)
LAM_COLORS = [_MAKO(x) for x in np.linspace(0.85, 0.12, len(SHAPE_LAMS))]
W_NORM = matplotlib.colors.LogNorm(vmin=0.02, vmax=1.0)
W_CMAP = plt.colormaps["viridis"]


def lam_label(lam):
    return r"$\lambda=\infty$" if np.isinf(lam) else rf"$\lambda={lam:g}$"


def lam_plot(lam):
    if lam == 0:
        return LAM_ZERO_PLOT
    return LAM_INF_PLOT if np.isinf(lam) else lam


def load_width(width, n_points):
    acc = tg.accumulate(width, n_points)
    ell_all = np.sqrt(tg.EDGES[:-1] * tg.EDGES[1:])
    good = (acc["n"] >= MIN_PAIRS) & (ell_all <= tg.L_MAX)
    ell = ell_all[good]
    a = {k: v[..., good] for k, v in acc.items()}
    plat = ell <= PLATEAU_FRAC * BEND_RATIO * width
    plat[:PLATEAU_MIN_BINS] = True
    return dict(ell=ell, acc=a, terms=tg.model_terms(ell, width, inf=True), plat=plat)


def sim_curve(d, lam):
    return tg.K_sim(d["acc"], lam)


def model_curve(d, lam):
    return np.array([tg.K_from(t["D"], t["D2"], t["Dm2"], t["m2_thin"], t["m4_thin"], lam) for t in d["terms"]])


def draw_shape(ax, d):
    for lam, col in zip(SHAPE_LAMS, LAM_COLORS):
        ax.plot(d["ell"], sim_curve(d, lam), "o", ms=5, mfc=col, mec="none", alpha=0.85, zorder=3)
        ax.plot(d["ell"], model_curve(d, lam), "-", color=col, lw=2.2, zorder=4, label=lam_label(lam))
    ax.axhline(3, color="0.3", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell/r_{\rm HII}$")
    ax.set_ylabel(r"$K(\ell)$")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=5),
                plt.Line2D([], [], color="0.3", lw=2.2)]
    labels += ["simulation", "prediction"]
    ax.legend(handles, labels, loc="lower left", fontsize=10, ncol=2)
    ax.text(0.97, 0.95, rf"$\Delta r_{{\rm shell}}/r_{{\rm HII}}={SHAPE_WIDTH:g}$" "\n"
            r"no inner cutoff ($k_{\max}\to\infty$)", transform=ax.transAxes, ha="right", va="top", fontsize=12)


def draw_plateau(ax, data):
    print("width   lambda   K_plat sim (one source, mean err)   prediction")
    for w, d in zip(WIDTHS, data):
        col = W_CMAP(W_NORM(w))
        pl = d["plat"]
        ax.plot(LAM_DENSE, [np.mean(model_curve(d, lam)[pl]) for lam in LAM_DENSE], "-", color=col, lw=2.2, zorder=3)
        for lam in (0.0, np.inf):
            ax.plot(lam_plot(lam), np.mean(model_curve(d, lam)[pl]), "_", color=col, ms=16, mew=2.2, zorder=3)
        x, y, e = [], [], []
        dodge = 10 ** (0.035 * (WIDTHS.index(w) - 1.5))
        for lam in SIM_LAMS:
            single = [np.nanmean(tg.K_sim(d["acc"], lam, seeds=slice(s, s + 1), odd=True)[pl])
                      for s in range(tg.N_SEEDS)]
            mirr = [np.nanmean(tg.K_sim(d["acc"], lam, seeds=slice(s, s + 1))[pl]) for s in range(tg.N_SEEDS)]
            x.append(lam_plot(lam) * dodge)
            y.append(np.nanmean(sim_curve(d, lam)[pl]))
            e.append(np.nanstd(single, ddof=1))
            print(f"{w:<6g}  {lam:>6g}   {y[-1]:.2f} ({e[-1]:.2f}, {np.nanstd(mirr, ddof=1) / np.sqrt(len(mirr)):.2f})"
                  f"   {np.mean(model_curve(d, lam)[pl]):.2f}")
        ax.errorbar(x, y, yerr=e, fmt="none", ecolor=col, elinewidth=1.3, capsize=3, alpha=0.9, zorder=4)
        ax.scatter(x, y, color=col, s=50, edgecolors="black", linewidths=0.8, zorder=5,
                   label=rf"$\Delta r_{{\rm shell}}/r_{{\rm HII}}={w:g}$")
    ax.axhline(3, color="0.3", ls="--", lw=1.1)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 150)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([LAM_ZERO_PLOT, 1, 10, LAM_INF_PLOT]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0", "1", "10", r"$\infty$"]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
        [v * 10 ** p for p in (-1, 0, 1) for v in range(2, 10) if 0.2 <= v * 10 ** p <= 50]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for xb in (np.sqrt(LAM_ZERO_PLOT * 0.2), np.sqrt(50 * LAM_INF_PLOT)):
        ax.axvline(xb, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\lambda=v_{\rm bulk}/v_{\rm turb}$")
    ax.set_ylabel(r"$K_{\rm plat}$")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", lw=2.2),
                plt.Line2D([], [], color="0.3", marker="|", ms=12, mew=1.3, ls="none")]
    labels += ["prediction", r"single-source $1\sigma$"]
    ax.set_ylim(top=8.0)
    ax.legend(handles, labels, loc="upper left", fontsize=10, ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-points", type=int, default=30000)
    args = ap.parse_args()
    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass
    data = [load_width(w, args.n_points) for w in WIDTHS]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.0, 12.0), gridspec_kw={"hspace": 0.22})
    draw_shape(axA, data[WIDTHS.index(SHAPE_WIDTH)])
    draw_plateau(axB, data)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
