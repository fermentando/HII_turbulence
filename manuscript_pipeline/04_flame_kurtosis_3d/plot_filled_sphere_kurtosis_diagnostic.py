#!/usr/bin/env python3
"""Diagnostic plots for the filled-sphere shell case Delta R / r_out = 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from projected_cube_kurtosis import (  # noqa: E402
    make_kurtosis_plot,
    make_kurtosis_with_gaussian_slices_plot,
)


DEFAULT_NPZ = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_width_to_rout"
    / "Synthetic_x_projected_shell_rin0p0000_rout1p0000_order1_pdfkk.npz"
)


def symmetric_norm(values: np.ndarray, percentile: float = 99.0) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def draw_3d_shell(ax, vx: np.ndarray, shell_mask: np.ndarray, box_size: float) -> None:
    stride = 4
    coords = np.linspace(-box_size / 2.0, box_size / 2.0, vx.shape[0])
    coords_s = coords[::stride]
    vx_s = vx[::stride, ::stride, ::stride]
    mask_s = shell_mask[::stride, ::stride, ::stride]
    x, y, z = np.meshgrid(coords_s, coords_s, coords_s, indexing="ij")
    norm = symmetric_norm(vx_s[mask_s])
    colors = plt.colormaps["RdBu_r"](norm(vx_s[mask_s]))
    colors[:, 3] = 0.10 + 0.45 * np.clip(np.abs(vx_s[mask_s]) / max(abs(norm.vmin), abs(norm.vmax)), 0.0, 1.0)
    ax.scatter(x[mask_s], y[mask_s], z[mask_s], c=colors, s=0.8, linewidths=0, depthshade=False)
    half = box_size / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(r"3D selected gas: filled sphere, colored by $v_x$")
    ax.view_init(elev=22, azim=-48)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False


def make_overview(data: np.lib.npyio.NpzFile, out: Path) -> dict:
    projected_map = np.asarray(data["projected_map"], dtype=float)
    shell_counts = np.asarray(data["shell_counts"], dtype=float)
    shell_mask = np.asarray(data["shell_mask"], dtype=bool)
    vx = np.asarray(data["vx_cube"], dtype=float)
    kk = np.asarray(data["kk"], dtype=float)
    log_centers = np.asarray(data["log_centers"], dtype=float)
    r = 10**log_centers
    r_inner = float(data["r_inner"])
    r_outer = float(data["r_outer"])
    box_size = 2.4
    coords = np.linspace(-box_size / 2.0, box_size / 2.0, projected_map.shape[0])
    finite_proj = projected_map[np.isfinite(projected_map)]
    norm_proj = symmetric_norm(finite_proj)
    finite_k = kk[np.isfinite(kk)]
    above = np.flatnonzero(np.isfinite(kk) & (kk > 3.0))

    fig = plt.figure(figsize=(14.2, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0])
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_proj = fig.add_subplot(grid[0, 1])
    ax_counts = fig.add_subplot(grid[1, 1])
    ax_text = fig.add_subplot(grid[:, 2])

    draw_3d_shell(ax3d, vx, shell_mask, box_size)

    im = ax_proj.imshow(
        projected_map.T,
        origin="lower",
        extent=(coords.min(), coords.max(), coords.min(), coords.max()),
        cmap="RdBu_r",
        norm=norm_proj,
        interpolation="nearest",
        aspect="equal",
    )
    ax_proj.set_title(r"Projection: $\langle v_x\rangle_x$")
    ax_proj.set_xlabel("y")
    ax_proj.set_ylabel("z")
    fig.colorbar(im, ax=ax_proj, label=r"$\langle v_x\rangle_x$")

    im_counts = ax_counts.imshow(
        shell_counts.T,
        origin="lower",
        extent=(coords.min(), coords.max(), coords.min(), coords.max()),
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    ax_counts.set_title("LOS path length through selected gas")
    ax_counts.set_xlabel("y")
    ax_counts.set_ylabel("z")
    fig.colorbar(im_counts, ax=ax_counts, label="cells")

    ax_text.axis("off")
    lines = [
        fr"$r_{{in}}={r_inner:.3f}$, $r_{{out}}={r_outer:.3f}$",
        fr"$(r_{{out}}-r_{{in}})/r_{{out}}={(r_outer-r_inner)/r_outer:.3f}$",
        "",
        fr"K bins: {np.count_nonzero(np.isfinite(kk))}",
        fr"median K = {np.nanmedian(finite_k):.3f}",
        fr"max K = {np.nanmax(finite_k):.3f}",
        fr"bins with K > 3 = {above.size}",
        "",
        "K > 3 bins:",
    ]
    if above.size:
        for idx in above:
            lines.append(fr"  r={r[idx]:.4f}, K={kk[idx]:.3f}")
    else:
        lines.append("  none")
    ax_text.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", family="monospace", fontsize=10)

    fig.suptitle("Filled-sphere projected kurtosis diagnostic", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_bins": int(np.count_nonzero(np.isfinite(kk))),
        "n_above": int(above.size),
        "k_median": float(np.nanmedian(finite_k)),
        "k_max": float(np.nanmax(finite_k)),
        "above_r": r[above],
        "above_k": kk[above],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--n-slices", type=int, default=6)
    args = parser.parse_args()

    out_dir = args.npz.parent
    prefix = args.npz.name.replace("_order1_pdfkk.npz", "")
    overview_out = out_dir / f"{prefix}_filled_sphere_overview_projection_kurtosis.png"
    kurt_out = out_dir / f"{prefix}_filled_sphere_3d_kurtosis.png"
    slices_out = out_dir / f"{prefix}_filled_sphere_kurtosis_gaussian_slices.png"

    with np.load(args.npz) as data:
        stats = make_overview(data, overview_out)
        pdf = np.asarray(data["pdf"], dtype=float)
        dv_bins = np.asarray(data["dv_bins"], dtype=float)
        log_centers = np.asarray(data["log_centers"], dtype=float)
        kk = np.asarray(data["kk"], dtype=float)
        make_kurtosis_plot(pdf, dv_bins, log_centers, kk, kurt_out, None, None)
        make_kurtosis_with_gaussian_slices_plot(
            pdf, dv_bins, log_centers, kk, slices_out, None, None, n_slices=args.n_slices
        )

    print(f"Wrote {overview_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {kurt_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {slices_out.relative_to(REPO_ROOT)}")
    print(f"K median={stats['k_median']:.3f}; K max={stats['k_max']:.3f}; bins above 3={stats['n_above']}/{stats['n_bins']}")
    if stats["n_above"]:
        for rr, kk_val in zip(stats["above_r"], stats["above_k"]):
            print(f"  K>3: r={rr:.6f}, K={kk_val:.6f}")


if __name__ == "__main__":
    main()
