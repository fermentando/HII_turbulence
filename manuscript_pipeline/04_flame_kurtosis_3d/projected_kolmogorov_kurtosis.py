#!/usr/bin/env python3
"""Project a Kolmogorov-like 3-D sphere and run the 2-D kurtosis analysis.

The mock observation keeps projected coordinates (x, y) and uses v_z as the
line-of-sight velocity, matching the way the region analysis uses sky position
plus one measured velocity per point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from simulate_kolmogorov_sphere import (  # noqa: E402
    kolmogorov_velocity_grid,
    random_points_in_sphere,
    sample_periodic_grid,
)
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_kolmogorov"
)


def finite_kurtosis_norm(kk: np.ndarray, default_min: float = 3.0, default_max: float = 6.0):
    finite = kk[np.isfinite(kk) & (kk > 0)]
    if finite.size == 0:
        return plt.Normalize(default_min, default_max)
    vmin = min(default_min, float(np.nanpercentile(finite, 5)))
    vmax = max(default_max, float(np.nanpercentile(finite, 95)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return plt.Normalize(default_min, default_max)
    return plt.Normalize(vmin, vmax)


def make_projected_map(x: np.ndarray, y: np.ndarray, vlos: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    lim = 1.05 * max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)))
    sc = ax.scatter(x, y, c=vlos, s=5, cmap="RdBu_r", linewidths=0, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("projected x")
    ax.set_ylabel("projected y")
    ax.set_title(r"Projected synthetic sphere, color = $v_{\rm LOS}$")
    fig.colorbar(sc, ax=ax, label=r"$v_{\rm LOS}$")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_kurtosis_plot(
    pdf: np.ndarray,
    dv_bins: np.ndarray,
    log_centers: np.ndarray,
    kk: np.ndarray,
    out: Path,
    x_min: float,
    x_max: float,
) -> None:
    r = 10**log_centers
    dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    mask = (dv_centers >= x_min) & (dv_centers <= x_max)
    if not np.any(mask):
        mask = np.isfinite(dv_centers)

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.03, 1.0], hspace=0.18)
    cax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], projection="3d")

    cmap = cm.viridis
    norm = finite_kurtosis_norm(kk)
    for i in range(len(r)):
        y = pdf[i][mask]
        good = np.isfinite(y) & (y > 0)
        if not np.any(good):
            continue
        ax.plot(
            dv_centers[mask][good],
            np.log10(np.full(np.sum(good), r[i])),
            y[good],
            color=cmap(norm(kk[i])),
            linewidth=1.0,
        )

    ax.set_xlabel(r"$(\Delta v_{\rm LOS})$", fontsize=14, labelpad=10)
    ax.set_ylabel(r"$\log r_{\rm proj}$", fontsize=14, labelpad=10)
    ax.set_zlabel(r"$\mathcal{P}(\Delta v_{\rm LOS})$", fontsize=14, labelpad=10)
    ax.set_xlim(float(np.nanmin(dv_centers[mask])), float(np.nanmax(dv_centers[mask])))
    ax.set_ylim(np.log10(r[0]), np.log10(r[-1]))
    ax.view_init(elev=25, azim=45)
    try:
        ax.dist = 9
    except Exception:
        pass
    positive = pdf[:, mask][np.isfinite(pdf[:, mask]) & (pdf[:, mask] > 0)]
    if positive.size:
        ax.set_zlim(max(float(np.nanpercentile(positive, 1)), 1e-4), min(float(np.nanmax(positive)), 1.0))
    else:
        ax.set_zlim(1e-4, 1.0)
    ax.set_zscale("log")

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=12)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("w")

    ax.set_title("Projected Kolmogorov sphere", pad=18)
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(r"Kurtosis", labelpad=3, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=2500)
    parser.add_argument("--n-grid", type=int, default=96)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=-6.0)
    parser.add_argument("--x-max", type=float, default=6.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    rng = np.random.default_rng(args.seed)
    box_size = 2.4 * args.radius
    points = random_points_in_sphere(args.n_points, args.radius, rng)
    grids = kolmogorov_velocity_grid(args.n_grid, box_size, rng)
    velocities = sample_periodic_grid(grids, points, box_size)

    x = points[:, 0]
    y = points[:, 1]
    vlos = velocities[:, 2]
    vel_err = np.zeros_like(vlos)

    log_centers, bin_edges, dv_bins, _ = compute_bins(
        x, y, vlos, bin_resolution=args.bin_resolution
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk = compute_pdf_and_statistics(
            x,
            y,
            vlos,
            vel_err,
            log_centers,
            bin_edges,
            dv_bins,
            order=1,
            n_jobs=args.n_jobs,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_out = args.output_dir / "Synthetic_projected_Kolmogorov_LOS_order1_pdfkk.npz"
    map_out = args.output_dir / "Synthetic_projected_Kolmogorov_LOS_map.png"
    kurt_out = args.output_dir / "Synthetic_projected_Kolmogorov_LOS_order1_3d_kurtosis.png"

    np.savez_compressed(
        npz_out,
        x=x,
        y=y,
        vlos=vlos,
        points_3d=points,
        velocities_3d=velocities,
        pdf=pdf,
        dv_bins=dv_bins,
        log_centers=log_centers,
        kk=kk,
        c2=c2,
        c4=c4,
        cnt=cnt,
        dv_centers=dv_centers,
    )
    make_projected_map(x, y, vlos, map_out)
    make_kurtosis_plot(pdf, dv_bins, log_centers, kk, kurt_out, args.x_min, args.x_max)

    finite = kk[np.isfinite(kk) & (kk > 0)]
    print(f"Wrote {npz_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {map_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {kurt_out.relative_to(REPO_ROOT)}")
    if finite.size:
        print(
            "Kurtosis range: "
            f"min={np.nanmin(finite):.2f}, median={np.nanmedian(finite):.2f}, "
            f"max={np.nanmax(finite):.2f}"
        )


if __name__ == "__main__":
    main()
