"""Selected-region version of slope_lambda_maps_physical.png (slope of ln K + S2, centroid model).

Reads the saved (Delta, lambda) maps of fit_slope_lambda.py (<region>_<LINE>_profiles.npz: dchi2_2d, v_turb_2d)
and draws, for each region, the 1-sigma regions of every line in physical units:
  x: Delta r_shell [pc] = Delta x r_HII (adopted radius, manuscript_notes/adopted_hii_region_radii.csv)
  y: v_bulk [km/s]      = lambda x v_turb(Delta, lambda)   (v_turb profiled from S2 at each grid point)
Star = best fit; dashed vertical = r_HII.

Run from the repo root (after fit_slope_lambda.py):
    python manuscript_pipeline/08_shell_inference/plot_slope_lambda_selected_physical.py [--regions Rosette_nebula RCW79 Sh2-311]
Output: Figures/manuscript/08_shell_inference/slope_lambda/slope_lambda_maps_physical_selected.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from fit_centroid_joint_grid import LINES, LINESTYLES, OUT_BASE  # noqa: E402
from fit_kurtosis_formula import DATA_DIR  # noqa: E402
from plot_centroid_joint_selected_physical import load_adopted_radii  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

RUN = OUT_BASE / "slope_lambda"
REGIONS = ("Rosette_nebula", "RCW79", "Sh2-311")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=list(REGIONS))
    args = ap.parse_args()
    adopted = load_adopted_radii()
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    fig, axes = plt.subplots(1, len(args.regions), figsize=(4.6 * len(args.regions), 4.0), squeeze=False)
    for ax, region in zip(axes[0], args.regions):
        lines = [X for X in LINES if (RUN / f"{region}_{X}_profiles.npz").exists()]
        r_hii = adopted.get(region) or float(np.load(DATA_DIR / f"{region}_{lines[0]}_formula_data.npz")["r_hii_pc"])
        xs, ys = [r_hii], []
        for X in lines:
            p = np.load(RUN / f"{region}_{X}_profiles.npz")
            D, L, m, v = p["delta_grid"], p["lambda_grid"], p["dchi2_2d"], p["v_turb_2d"]
            c = LINE_COLORS[X]
            xx = (D * r_hii)[:, None] * np.ones(len(L))[None]
            yy = L[None, :] * v
            inside = m <= 2.30
            xs += [xx[inside].min(), xx[inside].max()]
            ys += [yy[inside].min(), yy[inside].max()]
            ax.contourf(xx, yy, m, levels=[0, 2.30], colors=[c], alpha=0.25)
            ax.contour(xx, yy, m, levels=[2.30], colors=[c], linestyles=[LINESTYLES[X]], linewidths=[1.8])
            jD, jL = np.unravel_index(np.argmin(m), m.shape)
            ax.plot(D[jD] * r_hii, L[jL] * v[jD, jL], "*", ms=14, color=c, mec="white", mew=0.7)
            ax.plot([], [], color=c, ls=LINESTYLES[X], lw=1.8, label=LINE_LABELS[X])
        ax.axvline(r_hii, color="0.3", ls="--", lw=1.2)
        ax.set(xscale="log", yscale="log", xlabel=r"$\Delta r_{\rm shell}$ [pc]")
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
        ylo, yhi = min(ys) / 1.2, max(ys) * 1.2
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(FixedLocator([t for t in (1, 2, 3, 5, 10, 20, 30, 50, 100) if ylo <= t <= yhi]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(region.replace("_", " "), fontsize=13)
    axes[0, 0].legend(fontsize=11, loc="upper left")
    if len(args.regions) > 1:
        h1, = axes[0, 1].plot([], [], color="0.3", ls="--", lw=1.2)
        axes[0, 1].legend([h1], [r"$r_{\rm HII}$"], fontsize=10, loc="lower right")
    axes[0, 0].set_ylabel(r"$v_{\rm bulk}$ (km/s)")
    fig.tight_layout()
    out = RUN / "slope_lambda_maps_physical_selected.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
