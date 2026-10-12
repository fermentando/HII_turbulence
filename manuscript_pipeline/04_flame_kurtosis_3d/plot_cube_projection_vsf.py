#!/usr/bin/env python3
"""Plot a Kolmogorov-like cube, its x-projection, and 2-D/3-D VSFs."""

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
sys.path.insert(0, str(HERE))

from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "kolmogorov_cube_projection_vsf.png"
)


def binned_vsf_2d(
    coords: np.ndarray,
    velocity: np.ndarray,
    n_bins: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = coords.shape[0]
    i = rng.integers(0, n_points, size=n_pairs)
    j = rng.integers(0, n_points, size=n_pairs)
    good_pair = i != j
    i = i[good_pair]
    j = j[good_pair]

    dr = coords[i] - coords[j]
    dv = velocity[i] - velocity[j]
    sep = np.linalg.norm(dr, axis=1)
    dv2 = dv**2

    good = sep > 0
    sep = sep[good]
    dv2 = dv2[good]

    edges = np.geomspace(np.nanmin(sep), np.nanpercentile(sep, 85), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    s2 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    which = np.digitize(sep, edges) - 1

    for i in range(n_bins):
        idx = np.flatnonzero(which == i)
        counts[i] = idx.size
        if idx.size:
            s2[i] = np.nanmean(dv2[idx])

    return centers, s2, counts


def binned_vsf_3d_grid(
    velocity_cube: np.ndarray,
    box_size: float,
    n_bins: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_grid = velocity_cube.shape[0]
    coords_1d = np.linspace(-box_size / 2, box_size / 2, n_grid)

    idx1 = rng.integers(0, n_grid, size=(n_pairs, 3))
    idx2 = rng.integers(0, n_grid, size=(n_pairs, 3))
    different = np.any(idx1 != idx2, axis=1)
    idx1 = idx1[different]
    idx2 = idx2[different]

    p1 = coords_1d[idx1]
    p2 = coords_1d[idx2]
    sep = np.linalg.norm(p1 - p2, axis=1)
    dv = velocity_cube[idx1[:, 0], idx1[:, 1], idx1[:, 2]] - velocity_cube[
        idx2[:, 0], idx2[:, 1], idx2[:, 2]
    ]
    dv2 = dv**2

    edges = np.geomspace(np.nanmin(sep), np.nanpercentile(sep, 85), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    s2 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    which = np.digitize(sep, edges) - 1

    for i in range(n_bins):
        idx = np.flatnonzero(which == i)
        counts[i] = idx.size
        if idx.size:
            s2[i] = np.nanmean(dv2[idx])

    return centers, s2, counts


def render_cube_panel(ax, vx: np.ndarray, box_size: float, stride: int, percentile: float) -> TwoSlopeNorm:
    step = max(stride, 1)
    cube = vx[::step, ::step, ::step]
    n_grid = vx.shape[0]
    half = box_size / 2
    coords_1d = np.linspace(-half, half, n_grid)[::step]
    x, y, z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

    threshold = np.nanpercentile(np.abs(cube), percentile)
    mask = np.abs(cube) >= threshold
    lo, hi = np.nanpercentile(cube, [1, 99])
    lim = max(abs(lo), abs(hi))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    colors = plt.colormaps["RdBu_r"](norm(cube[mask]))
    colors[:, 3] = 0.10 + 0.55 * np.clip(np.abs(cube[mask]) / lim, 0, 1)

    ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=colors,
        s=max(1.6, 8.0 / step),
        linewidths=0,
        depthshade=False,
    )

    corners = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="0.2", linewidth=0.6, alpha=0.45)

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(r"3D cube: $v_x$")
    ax.view_init(elev=22, azim=-48)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
    return norm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=96)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--percentile", type=float, default=64.0)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--n-pairs", type=int, default=1_500_000)
    parser.add_argument("--n-pairs-3d", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vx, _, _ = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng)

    projected_v = np.nanmean(vx, axis=0)
    half = args.box_size / 2
    y = np.linspace(-half, half, args.n_grid)
    z = np.linspace(-half, half, args.n_grid)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    coords = np.column_stack([yy.ravel(), zz.ravel()])
    v_flat = projected_v.ravel()

    r, s2, counts = binned_vsf_2d(coords, v_flat, args.bins, args.n_pairs, rng)
    fit_mask = np.isfinite(s2) & (counts > 50) & (r > 0.12 * half) & (r < 0.65 * half)
    slope, intercept = np.polyfit(np.log10(r[fit_mask]), np.log10(s2[fit_mask]), 1)

    r3d, s2_3d, counts_3d = binned_vsf_3d_grid(
        vx, args.box_size, args.bins, args.n_pairs_3d, rng
    )
    fit_mask_3d = (
        np.isfinite(s2_3d)
        & (counts_3d > 50)
        & (r3d > 0.07 * half)
        & (r3d < 0.50 * half)
    )
    slope_3d, intercept_3d = np.polyfit(
        np.log10(r3d[fit_mask_3d]), np.log10(s2_3d[fit_mask_3d]), 1
    )

    fig = plt.figure(figsize=(16.0, 5.4), constrained_layout=True)
    ax_cube = fig.add_subplot(1, 3, 1, projection="3d")
    norm_cube = render_cube_panel(ax_cube, vx, args.box_size, args.stride, args.percentile)

    ax_map = fig.add_subplot(1, 3, 2)
    lo, hi = np.nanpercentile(projected_v, [1, 99])
    lim = max(abs(lo), abs(hi))
    norm_map = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax_map.imshow(
        projected_v.T,
        origin="lower",
        extent=(-half, half, -half, half),
        cmap="RdBu_r",
        norm=norm_map,
        interpolation="nearest",
        aspect="equal",
    )
    ax_map.set_xlabel("y")
    ax_map.set_ylabel("z")
    ax_map.set_title(r"Projection along x: $\langle v_x\rangle_x$")
    fig.colorbar(im, ax=ax_map, shrink=0.78, label=r"$\langle v_x\rangle_x$")

    ax_vsf = fig.add_subplot(1, 3, 3)
    ax_vsf.loglog(r3d, s2_3d, "s", ms=3.8, color="tab:purple", label="3D cube")
    ax_vsf.loglog(
        r3d[fit_mask_3d],
        10**intercept_3d * r3d[fit_mask_3d] ** slope_3d,
        color="tab:purple",
        lw=2,
        alpha=0.85,
        label=fr"3D fit = {slope_3d:.2f}",
    )
    ax_vsf.loglog(r, s2, "o", ms=4, color="tab:blue", label="x-projected map")
    ax_vsf.loglog(
        r[fit_mask],
        10**intercept * r[fit_mask] ** slope,
        color="tab:red",
        lw=2,
        label=fr"projected fit = {slope:.2f}",
    )
    ref = s2[fit_mask][0] * (r / r[fit_mask][0]) ** (2 / 3)
    ax_vsf.loglog(r, ref, "--", color="0.3", label=r"$r^{2/3}$")
    ax_vsf.set_xlabel(r"projected separation $r_{\rm proj}$")
    ax_vsf.set_ylabel(r"$S_2(r)=\langle(\Delta v)^2\rangle$")
    ax_vsf.set_title("Velocity structure functions")
    ax_vsf.legend(frameon=False)

    mappable = cm.ScalarMappable(norm=norm_cube, cmap="RdBu_r")
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax_cube, shrink=0.65, pad=0.08, label=r"$v_x$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        vx=vx,
        projected_v=projected_v,
        r=r,
        s2=s2,
        counts=counts,
        fitted_slope=slope,
        r3d=r3d,
        s2_3d=s2_3d,
        counts_3d=counts_3d,
        fitted_slope_3d=slope_3d,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.npz')}")
    print(f"3D cube VSF slope: {slope_3d:.3f}")
    print(f"Projected VSF slope: {slope:.3f}")


if __name__ == "__main__":
    main()
