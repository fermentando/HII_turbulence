#!/usr/bin/env python3
"""Visualize total and bulk-only LOS velocities for an expanding shell."""

from __future__ import annotations

import argparse
import json
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

from generate_expanding_shell_thickness_grid import add_radial_bulk_vx  # noqa: E402
from projected_shell_kurtosis import shell_projected_velocity  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402


DEFAULT_SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_expanding_vbulk10p0"
)


def shell_mask_and_coords(n_grid: int, box_size: float, r_inner: float, r_outer: float) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(-box_size / 2.0, box_size / 2.0, n_grid)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    radius = np.sqrt(x**2 + y**2 + z**2)
    shell = (radius >= r_inner) & (radius <= r_outer)
    return coords, shell


def project_shell(vx: np.ndarray, box_size: float, r_inner: float, r_outer: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, z, projected_v, counts, _ = shell_projected_velocity(vx, box_size, r_inner, r_outer)
    return y, z, projected_v, counts


def draw_3d(ax, coords_1d: np.ndarray, shell: np.ndarray, values: np.ndarray, title: str, norm: TwoSlopeNorm) -> None:
    stride = 4
    coords_s = coords_1d[::stride]
    shell_s = shell[::stride, ::stride, ::stride]
    values_s = values[::stride, ::stride, ::stride]
    x, y, z = np.meshgrid(coords_s, coords_s, coords_s, indexing="ij")
    colors = plt.colormaps["RdBu_r"](norm(values_s[shell_s]))
    colors[:, 3] = 0.18 + 0.55 * np.clip(np.abs(values_s[shell_s]) / max(abs(norm.vmin), abs(norm.vmax)), 0, 1)
    ax.scatter(x[shell_s], y[shell_s], z[shell_s], c=colors, s=1.0, linewidths=0, depthshade=False)
    half = max(abs(coords_1d.min()), abs(coords_1d.max()))
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(title)
    ax.view_init(elev=22, azim=-48)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False


def finite_symmetric_norm(*arrays: np.ndarray, percentile: float = 99.0) -> TwoSlopeNorm:
    finite = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in arrays])
    lim = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    parser.add_argument("--width", type=float, default=0.1158)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metadata = json.loads((args.shell_dir / "expanding_shell_metadata.json").read_text(encoding="utf-8"))
    n_grid = int(metadata["n_grid"])
    box_size = float(metadata["box_size"])
    seed = int(metadata["seed"])
    v_bulk = float(metadata["v_bulk"])
    r_outer = float(metadata["r_outer"])
    r_inner = r_outer - args.width

    rng = np.random.default_rng(seed)
    vx_turb, _, _ = kolmogorov_velocity_grid(n_grid, box_size, rng)
    vx_total, radial_x = add_radial_bulk_vx(vx_turb, box_size, v_bulk)
    vx_bulk = v_bulk * radial_x

    coords_1d, shell = shell_mask_and_coords(n_grid, box_size, r_inner, r_outer)
    y, z, total_proj, counts = project_shell(vx_total, box_size, r_inner, r_outer)
    _, _, bulk_proj, _ = project_shell(vx_bulk, box_size, r_inner, r_outer)

    norm_3d = finite_symmetric_norm(vx_total[shell], vx_bulk[shell])
    norm_proj = finite_symmetric_norm(total_proj, bulk_proj)

    fig = plt.figure(figsize=(12.8, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.05])

    ax_total_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_bulk_3d = fig.add_subplot(grid[0, 1], projection="3d")
    draw_3d(ax_total_3d, coords_1d, shell, vx_total, r"3D shell: $v_x = v_{x,turb}+v_{bulk}x/r$", norm_3d)
    draw_3d(ax_bulk_3d, coords_1d, shell, vx_bulk, r"3D shell: bulk-only $v_{bulk}x/r$", norm_3d)

    ax_total = fig.add_subplot(grid[1, 0])
    ax_bulk = fig.add_subplot(grid[1, 1])
    im_total = ax_total.imshow(
        total_proj.T,
        origin="lower",
        extent=(y.min(), y.max(), z.min(), z.max()),
        cmap="RdBu_r",
        norm=norm_proj,
        interpolation="nearest",
        aspect="equal",
    )
    im_bulk = ax_bulk.imshow(
        bulk_proj.T,
        origin="lower",
        extent=(y.min(), y.max(), z.min(), z.max()),
        cmap="RdBu_r",
        norm=norm_proj,
        interpolation="nearest",
        aspect="equal",
    )
    for ax, title in [
        (ax_total, r"Projected mean $\langle v_x\rangle_x$: total"),
        (ax_bulk, r"Projected mean $\langle v_x\rangle_x$: bulk-only"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("y")
        ax.set_ylabel("z")

    cax = fig.add_subplot(grid[:, 2])
    fig.colorbar(im_total, cax=cax, label=r"$v_x$")

    valid_bulk = np.isfinite(bulk_proj)
    valid_total = np.isfinite(total_proj)
    fig.suptitle(
        fr"Expanding shell diagnostic: $v_{{bulk}}={v_bulk:g}\sigma_{{v,turb}}$, "
        fr"$\Delta R={args.width:.4f}$; "
        fr"bulk projection RMS={np.sqrt(np.nanmean(bulk_proj[valid_bulk]**2)):.2e}, "
        fr"total projection RMS={np.sqrt(np.nanmean(total_proj[valid_total]**2)):.2f}",
        fontsize=12,
    )

    out = args.output
    if out is None:
        tag = f"width{args.width:.4f}".replace(".", "p")
        out = args.shell_dir / f"expanding_shell_bulk_diagnostic_{tag}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    print(f"Applied bulk velocity as vx_bulk = v_bulk * x / sqrt(x^2 + y^2 + z^2)")
    print(f"v_bulk = {v_bulk:g}; width = {args.width:.4f}; r_inner = {r_inner:.4f}; r_outer = {r_outer:.4f}")
    print(f"bulk-only projected RMS = {np.sqrt(np.nanmean(bulk_proj[valid_bulk]**2)):.6e}")
    print(f"bulk-only projected max abs = {np.nanmax(np.abs(bulk_proj[valid_bulk])):.6e}")
    print(f"total projected RMS = {np.sqrt(np.nanmean(total_proj[valid_total]**2)):.6e}")


if __name__ == "__main__":
    main()
