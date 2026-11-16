"""Joint K(l) + S2(l) centroid-model maps for selected regions, in physical units.

Reads the (Delta, lambda_eff) maps of fit_centroid_joint_grid.py and converts
  x: Delta r_shell [pc] = Delta x r_HII  (r_HII = adopted radius from manuscript_notes/adopted_hii_region_radii.csv)
  y: v_bulk [km/s]      = lambda_eff x v_turb  (v_turb = the region's shared best-fit value; its uncertainty
                          is not propagated into the contours)

Run from the repo root (after fit_centroid_joint_grid.py --fix-a 0 --lam-min 1):
    python manuscript_pipeline/08_shell_inference/plot_centroid_joint_selected_physical.py [--run centroid_joint_v_a0_lammin1]
Output: Figures/manuscript/08_shell_inference/centroid_joint_maps_physical_<run suffix>.png
The vertical dashed line marks the adopted HII-region radius (manuscript_notes/adopted_hii_region_radii.csv).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from fit_centroid_joint_grid import LINES, LINESTYLES, OUT_BASE  # noqa: E402
from fit_kurtosis_formula import DATA_DIR  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

REGIONS = ("Sh2-311", "RCW79", "BFS65")
ADOPTED_RADII = Path("manuscript_notes/adopted_hii_region_radii.csv")


def load_adopted_radii() -> dict[str, float]:
    with open(ADOPTED_RADII, encoding="utf-8") as f:
        return {r["object"]: float(r["radius_pc"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="centroid_joint_v_a0_lammin1")
    args = ap.parse_args()
    run = OUT_BASE / args.run
    with open(run / "joint_summary.csv", encoding="utf-8") as f:
        summ = {(r["region"], r["line"]): r for r in csv.DictReader(f)}
    adopted = load_adopted_radii()
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass

    fig, axes = plt.subplots(1, len(REGIONS), figsize=(4.6 * len(REGIONS), 4.0), squeeze=False)
    for ax, region in zip(axes[0], REGIONS):
        lines = [X for X in LINES if (run / f"{region}_{X}_map.npz").exists()]
        r_hii = adopted.get(region)
        if r_hii is None:  # fall back to the radius the fit used
            r_hii = float(np.load(DATA_DIR / f"{region}_{lines[0]}_formula_data.npz")["r_hii_pc"])
        v_turb = float(summ[(region, lines[0])]["v_turb"])
        xs, ys = [r_hii], []  # extent of the 2-sigma regions (+ r_HII) for cropping
        for X in lines:
            m = np.load(run / f"{region}_{X}_map.npz")
            xx, yy = np.meshgrid(m["delta_grid"] * r_hii, m["lambda_grid"] * v_turb, indexing="ij")
            c, s = LINE_COLORS[X], summ[(region, X)]
            inside = m["dm2lnL"] <= 6.18
            xs += [xx[inside].min(), xx[inside].max()]
            ys += [yy[inside].min(), yy[inside].max()]
            ax.contourf(xx, yy, m["dm2lnL"], levels=[0, 2.30], colors=[c], alpha=0.25)
            ax.contour(xx, yy, m["dm2lnL"], levels=[2.30, 6.18], colors=[c], linestyles=[LINESTYLES[X]],
                       linewidths=[1.8, 0.9])
            xb, yb = float(s["best_delta"]) * r_hii, float(s["best_lambda_eff"]) * v_turb
            ax.plot(xb, yb, "*", ms=14, color=c, mec="white", mew=0.7)
            ax.plot([], [], color=c, ls=LINESTYLES[X], lw=1.8, label=LINE_LABELS[X])
        ax.axvline(r_hii, color="0.3", ls="--", lw=1.2)
        ax.set(xscale="log", yscale="log", xlabel=r"$\Delta r_{\rm shell}$ [pc]")
        # crop to the 2-sigma regions; x always extends past r_HII (where the Delta grid ends) so the line shows
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
        ylo, yhi = max(min(ys), yy.min()) / 1.1, min(max(ys), yy.max()) * 1.1
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(FixedLocator([t for t in (2, 3, 5, 10, 20, 30, 50, 100) if ylo <= t <= yhi]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(region.replace("_", " "), fontsize=13)
    axes[0, 0].legend(fontsize=11, loc="upper left")
    if len(REGIONS) > 1:
        (h,) = axes[0, 1].plot([], [], color="0.3", ls="--", lw=1.2)
        axes[0, 1].legend([h], [r"$r_{\rm HII}$"], fontsize=11, loc="lower right")
    axes[0, 0].set_ylabel(r"$v_{\rm bulk}$ (km/s)")
    fig.tight_layout()
    suffix = args.run.removeprefix("centroid_joint_v").lstrip("_") or "default"
    out = OUT_BASE / f"centroid_joint_maps_physical_{suffix}.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
