#!/usr/bin/env python3
"""Manuscript summary: how the thin-shell prediction tracks K(l) across lambda = v_bulk/v_turb.

kurtosis_lambda_model_summary.png, one column:
  top    -- K(l) at Delta r_shell = SHAPE_WIDTH for several lambda: 8-seed pooled simulation (points)
           and the prediction (lines); colour = lambda (mako)
  bottom -- plateau level K_plat vs lambda for each simulated width (colour = width, viridis as in
           random_point_count_kurtosis_and_N_comparison.png): mirror-pooled simulation (points) and the
           prediction (lines); lambda = 0 and lambda = inf are drawn at the axis ends.
           Error bars = scatter of K_plat between single (unmirrored) field realisations, i.e. the
           uncertainty for one source, not the error of the 8-seed mean.

Mirror pooling: each seed is paired with its mirrored field u -> -u (an equally likely realisation of
the Gaussian, symmetric turbulent field). Summing a seed's moments with its mirror's cancels the odd
terms <du dmu>, <du^3 dmu>, <du dmu^3> exactly (they average to zero in the ensemble but not in one
realisation), leaving the ensemble-average K(l) that the prediction describes.

Simulation: kurtosis_vratio_seed_ensemble.py caches (N = 30,000, all pairs, 8 field seeds); K(l; lambda) is
rebuilt from the pooled pair moments for any lambda (kurtosis_shape_vs_vratio_fixed_width.k_of_lambda).
Prediction: derive_kurtosis_shape_kolmogorov.model_sums / k_model with the bulk curve K_b(l).
Plateau: mean over the bins with l <= PLATEAU_FRAC * l_b (l_b = 2e^{-3/2} Delta), at least the first
PLATEAU_MIN_BINS bins -- the same bins for simulation and prediction.

Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_lambda_model_summary.py
"""

from __future__ import annotations

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
sys.path[:0] = [str(HERE), str(REPO_ROOT / "manuscript_pipeline" / "00_core")]
sys.argv = sys.argv[:1]  # imported modules read optional argv
import kurtosis_shape_vs_vratio_fixed_width as kv  # noqa: E402
from derive_bulk_limit_kurtosis import k_theory  # noqa: E402
from derive_kurtosis_shape_kolmogorov import k_model, model_sums  # noqa: E402

WIDTHS = [0.02, 0.09, 0.3, 1.0]
SHAPE_WIDTH = 0.09
SHAPE_LAMS = [0.0, 1.0, 10.0, np.inf]
SIM_LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, np.inf]
L_MAX = 0.6
MIN_PAIRS = 200
PLATEAU_FRAC = 0.25
PLATEAU_MIN_BINS = 5
BEND_RATIO = 2 * np.exp(-1.5)

# lambda axis of the right panel: lambda = 0 and lambda = inf sit at the ends of a log axis
LAM_ZERO_PLOT, LAM_INF_PLOT = 0.1, 100.0
LAM_DENSE = np.geomspace(0.2, 50.0, 80)

OUT = kv.BASE_DIR / "kurtosis_lambda_model_summary.png"

_MAKO = sns.color_palette("mako", as_cmap=True)
LAM_COLORS = [_MAKO(x) for x in np.linspace(0.85, 0.12, len(SHAPE_LAMS))]
W_NORM = matplotlib.colors.LogNorm(vmin=0.02, vmax=1.0)
W_CMAP = plt.colormaps["viridis"]


def lam_label(lam: float) -> str:
    return r"$\lambda=\infty$" if np.isinf(lam) else rf"$\lambda={lam:g}$"


def lam_plot(lam: float) -> float:
    if lam == 0:
        return LAM_ZERO_PLOT
    return LAM_INF_PLOT if np.isinf(lam) else lam


def load_width(width: float):
    """Bin centres, pooled and per-seed moments (restricted to usable bins), and the model sums."""
    tag = f"w{width:g}".replace(".", "p")
    c = np.load(kv.BASE_DIR / f"_kurtosis_vratio_moments_{tag}_n{kv.N_POINTS}_seeds8.npz")
    pooled = {k: c[k].sum(0) for k in kv.KEYS}
    seeds = [{k: c[k][s] for k in kv.KEYS} for s in range(c["n"].shape[0])]
    ell_all = np.sqrt(kv.EDGES[:-1] * kv.EDGES[1:])
    good = (pooled["n"] >= MIN_PAIRS) & (ell_all <= L_MAX)
    ell = ell_all[good]
    kb = k_theory(ell, 1.0 - width)
    sums = [model_sums(l, width, k) for l, k in zip(ell, kb)]
    plat = ell <= PLATEAU_FRAC * BEND_RATIO * width
    plat[:PLATEAU_MIN_BINS] = True
    return dict(ell=ell, good=good, pooled=pooled, seeds=seeds, sums=sums, plat=plat)


ODD_KEYS = ("u1m1", "u3m1", "u1m3")  # odd in u: flip sign under u -> -u


def mirror_pool(M):
    """Moments of a field plus its mirror u -> -u (odd terms cancel; overall factor 2 drops out of K)."""
    return {k: (np.zeros_like(v) if k in ODD_KEYS else 2 * v) for k, v in M.items()}


def sim_curve(d, lam):
    return kv.k_of_lambda(mirror_pool(d["pooled"]), lam)[d["good"]]


def model_curve(d, lam):
    return np.array([k_model(s, lam) for s in d["sums"]])


def draw_shape(ax, d) -> None:
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
    ax.text(0.97, 0.95, rf"$\Delta r_{{\rm shell}}/r_{{\rm HII}}={SHAPE_WIDTH:g}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=13)


def draw_plateau(ax, data) -> None:
    print("width   lambda   K_plat sim (+-)   prediction")
    for w, d in zip(WIDTHS, data):
        col = W_CMAP(W_NORM(w))
        pl = d["plat"]
        k_dense = [np.mean(model_curve(d, lam)[pl]) for lam in LAM_DENSE]
        ax.plot(LAM_DENSE, k_dense, "-", color=col, lw=2.2, zorder=3)
        for lam in (0.0, np.inf):  # prediction at the axis ends
            ax.plot(lam_plot(lam), np.mean(model_curve(d, lam)[pl]), "_", color=col, ms=16, mew=2.2, zorder=3)
        x, y, e, em = [], [], [], []
        dodge = 10 ** (0.035 * (WIDTHS.index(w) - 1.5))  # small x offset so error bars don't overlap
        for lam in SIM_LAMS:
            # single-source scatter: K_plat of each raw (unmirrored) realisation, odd terms included
            single = [np.nanmean(kv.k_of_lambda(s, lam)[d["good"]][pl]) for s in d["seeds"]]
            mirr = [np.nanmean(kv.k_of_lambda(mirror_pool(s), lam)[d["good"]][pl]) for s in d["seeds"]]
            x.append(lam_plot(lam) * dodge)
            y.append(np.nanmean(sim_curve(d, lam)[pl]))
            e.append(np.nanstd(single, ddof=1))
            em.append(np.nanstd(mirr, ddof=1) / np.sqrt(len(mirr)))
            print(f"{w:<6g}  {lam:>6g}   {y[-1]:.2f} (one source {e[-1]:.2f}, mean {em[-1]:.2f})"
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
    for xb in (np.sqrt(LAM_ZERO_PLOT * 0.2), np.sqrt(50 * LAM_INF_PLOT)):  # axis breaks
        ax.axvline(xb, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\lambda=v_{\rm bulk}/v_{\rm turb}$")
    ax.set_ylabel(r"$K_{\rm plat}$")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", lw=2.2)]
    handles += [plt.Line2D([], [], color="0.3", marker="|", ms=12, mew=1.3, ls="none")]
    labels += ["prediction", r"single-source $1\sigma$"]
    ax.set_ylim(top=8.0)
    ax.legend(handles, labels, loc="upper left", fontsize=10, ncol=2)


def main() -> None:
    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass
    data = [load_width(w) for w in WIDTHS]
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(8.0, 12.0), gridspec_kw={"hspace": 0.22})
    draw_shape(axA, data[WIDTHS.index(SHAPE_WIDTH)])
    draw_plateau(axB, data)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
