#!/usr/bin/env python3
"""Run the PDF/kurtosis analysis on an x-projected turbulent shell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from projected_cube_kurtosis import (  # noqa: E402
    kurtosis_from_pdf,
    make_kurtosis_plot,
    make_kurtosis_with_gaussian_slices_plot,
)
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)


def shell_projected_velocity(
    vx: np.ndarray,
    box_size: float,
    r_inner: float,
    r_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average v_x along x using only cells inside a spherical shell."""
    n_grid = vx.shape[0]
    half = box_size / 2
    coords = np.linspace(-half, half, n_grid)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    radius = np.sqrt(x**2 + y**2 + z**2)
    shell = (radius >= r_inner) & (radius <= r_outer)

    shell_counts = np.sum(shell, axis=0)
    weighted_sum = np.sum(np.where(shell, vx, 0.0), axis=0)
    projected_v = np.full_like(weighted_sum, np.nan, dtype=float)
    valid = shell_counts > 0
    projected_v[valid] = weighted_sum[valid] / shell_counts[valid]

    return coords, coords, projected_v, shell_counts, shell


def render_shell_3d(
    ax,
    coords: np.ndarray,
    vx: np.ndarray,
    shell_mask: np.ndarray,
    stride: int,
) -> TwoSlopeNorm:
    step = max(stride, 1)
    coords_s = coords[::step]
    vx_s = vx[::step, ::step, ::step]
    shell_s = shell_mask[::step, ::step, ::step]
    x, y, z = np.meshgrid(coords_s, coords_s, coords_s, indexing="ij")

    lo, hi = np.nanpercentile(vx_s[shell_s], [1, 99])
    lim = max(abs(lo), abs(hi))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    colors = plt.colormaps["RdBu_r"](norm(vx_s[shell_s]))
    colors[:, 3] = 0.12 + 0.45 * np.clip(np.abs(vx_s[shell_s]) / lim, 0.0, 1.0)

    ax.scatter(
        x[shell_s],
        y[shell_s],
        z[shell_s],
        c=colors,
        s=max(0.8, 5.0 / step),
        linewidths=0,
        depthshade=False,
    )

    half = max(abs(coords.min()), abs(coords.max()))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Selected 3D shell")
    ax.view_init(elev=22, azim=-48)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
    return norm


def make_shell_map(
    coords: np.ndarray,
    vx: np.ndarray,
    shell_mask: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    projected_v: np.ndarray,
    shell_counts: np.ndarray,
    out: Path,
    r_inner: float,
    r_outer: float,
    render_stride: int,
) -> None:
    finite_v = projected_v[np.isfinite(projected_v)]
    lo, hi = np.nanpercentile(finite_v, [1, 99])
    lim = max(abs(lo), abs(hi))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig = plt.figure(figsize=(15.0, 4.9), constrained_layout=True)
    ax_shell = fig.add_subplot(1, 3, 1, projection="3d")
    ax_proj = fig.add_subplot(1, 3, 2)
    ax_count = fig.add_subplot(1, 3, 3)

    shell_norm = render_shell_3d(ax_shell, coords, vx, shell_mask, render_stride)

    im = ax_proj.imshow(
        projected_v.T,
        origin="lower",
        extent=(y.min(), y.max(), z.min(), z.max()),
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax_proj.set_xlabel("y")
    ax_proj.set_ylabel("z")
    ax_proj.set_title(r"Projection: $\langle v_x\rangle_x$")
    fig.colorbar(im, ax=ax_proj, label=r"$\langle v_x\rangle_x$")

    im_count = ax_count.imshow(
        shell_counts.T,
        origin="lower",
        extent=(y.min(), y.max(), z.min(), z.max()),
        cmap="magma",
        interpolation="nearest",
        aspect="equal",
    )
    ax_count.set_xlabel("y")
    ax_count.set_ylabel("z")
    ax_count.set_title(fr"Path length, {r_inner:.2f} < r < {r_outer:.2f}")
    fig.colorbar(im_count, ax=ax_count, label="shell cells")

    mappable = cm.ScalarMappable(norm=shell_norm, cmap="RdBu_r")
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax_shell, shrink=0.74, pad=0.08, label=r"$v_x$")

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=160)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--r-inner", type=float, default=0.75)
    parser.add_argument("--r-outer", type=float, default=1.05)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--n-slices", type=int, default=4)
    parser.add_argument("--render-stride", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    if args.r_inner <= 0 or args.r_outer <= args.r_inner:
        raise ValueError("--r-outer must be larger than --r-inner, and both must be positive")

    rng = np.random.default_rng(args.seed)
    vx, _, _ = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng)
    y, z, projected_v, shell_counts, shell_mask = shell_projected_velocity(
        vx, args.box_size, args.r_inner, args.r_outer
    )
    coords = y

    yy, zz = np.meshgrid(y, z, indexing="ij")
    valid = np.isfinite(projected_v)
    y_flat = yy[valid].ravel()
    z_flat = zz[valid].ravel()
    v_flat = projected_v[valid].ravel()
    vel_err = np.zeros_like(v_flat)

    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y_flat, z_flat, v_flat, bin_resolution=args.bin_resolution
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y_flat,
            z_flat,
            v_flat,
            vel_err,
            log_centers,
            bin_edges,
            dv_bins,
            order=1,
            n_jobs=args.n_jobs,
        )
    kk_pdf = kurtosis_from_pdf(pdf, dv_centers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shell_tag = f"rin{args.r_inner:.4f}_rout{args.r_outer:.4f}".replace(".", "p")
    npz_out = args.output_dir / f"Synthetic_x_projected_shell_{shell_tag}_order1_pdfkk.npz"
    map_out = args.output_dir / f"Synthetic_x_projected_shell_{shell_tag}_map.png"
    kurt_out = args.output_dir / f"Synthetic_x_projected_shell_{shell_tag}_order1_3d_kurtosis.png"
    slices_out = args.output_dir / f"Synthetic_x_projected_shell_{shell_tag}_order1_3d_kurtosis_gaussian_slices.png"

    np.savez_compressed(
        npz_out,
        y=y_flat,
        z=z_flat,
        projected_v=v_flat,
        projected_map=projected_v,
        shell_counts=shell_counts,
        shell_mask=shell_mask,
        r_inner=args.r_inner,
        r_outer=args.r_outer,
        vx_cube=vx,
        pdf=pdf,
        dv_bins=dv_bins,
        log_centers=log_centers,
        kk=kk_pdf,
        kk_pdf=kk_pdf,
        kk_pairs=kk_pairs,
        c2=c2,
        c4=c4,
        cnt=cnt,
        dv_centers=dv_centers,
    )
    make_shell_map(
        coords,
        vx,
        shell_mask,
        y,
        z,
        projected_v,
        shell_counts,
        map_out,
        args.r_inner,
        args.r_outer,
        args.render_stride,
    )
    make_kurtosis_plot(pdf, dv_bins, log_centers, kk_pdf, kurt_out, args.x_min, args.x_max)
    make_kurtosis_with_gaussian_slices_plot(
        pdf,
        dv_bins,
        log_centers,
        kk_pdf,
        slices_out,
        args.x_min,
        args.x_max,
        args.n_slices,
    )

    finite = kk_pdf[np.isfinite(kk_pdf) & (kk_pdf > 0)]
    print(f"Wrote {npz_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {map_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {kurt_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {slices_out.relative_to(REPO_ROOT)}")
    print(f"Projected pixels with shell gas: {v_flat.size}")
    print(f"Shell filling fraction in cube: {np.mean(shell_mask):.3f}")
    if finite.size:
        print(
            "PDF-derived kurtosis range: "
            f"min={np.nanmin(finite):.2f}, median={np.nanmedian(finite):.2f}, "
            f"max={np.nanmax(finite):.2f}"
        )


if __name__ == "__main__":
    main()
