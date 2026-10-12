#!/usr/bin/env python3
"""Visualize the 3-D geometry and projections for the shell-thickness grid."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
FNAME_RE = re.compile(r"Synthetic_x_projected_shell_rin([\dp]+)_rout([\dp]+)_order1_pdfkk\.npz")


def parse_tag(path: Path) -> tuple[float, float]:
    match = FNAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected shell filename: {path.name}")
    return float(match.group(1).replace("p", ".")), float(match.group(2).replace("p", "."))


def load_shells(shell_dir: Path) -> list[dict]:
    shells = []
    for path in sorted(shell_dir.glob("Synthetic_x_projected_shell_rin*_order1_pdfkk.npz")):
        if FNAME_RE.match(path.name) is None:
            continue
        with np.load(path) as data:
            r_inner, r_outer = parse_tag(path)
            shell_mask = np.asarray(data["shell_mask"], dtype=bool)
            shell_counts = np.asarray(data["shell_counts"], dtype=float)
            projected_map = np.asarray(data["projected_map"], dtype=float)
            shells.append(
                {
                    "path": path,
                    "r_inner": r_inner,
                    "r_outer": r_outer,
                    "width": r_outer - r_inner,
                    "shell_mask": shell_mask,
                    "shell_counts": shell_counts,
                    "projected_map": projected_map,
                }
            )
    return sorted(shells, key=lambda item: item["width"])


def coords_for_mask(mask: np.ndarray, box_size: float = 2.4) -> np.ndarray:
    return np.linspace(-box_size / 2.0, box_size / 2.0, mask.shape[0])


def plot_shell_voxels(ax, shell: dict, stride: int = 4) -> None:
    mask = shell["shell_mask"][::stride, ::stride, ::stride]
    coords = coords_for_mask(shell["shell_mask"])[::stride]
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=np.abs(x[mask]),
        cmap="viridis",
        s=0.65,
        linewidths=0,
        alpha=0.25,
        depthshade=False,
    )
    half = 1.2
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=20, azim=-48)
    ax.set_title(fr"$\Delta R={shell['width']:.3f}$", fontsize=8)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    args = parser.parse_args()
    out_fig = args.shell_dir / "shell_thickness_geometry_grid.png"

    shells = load_shells(args.shell_dir)
    if not shells:
        raise FileNotFoundError(f"No shell NPZ files found in {args.shell_dir}")

    n_show = min(6, len(shells))
    indices = np.linspace(0, len(shells) - 1, n_show).round().astype(int)
    selected = [shells[i] for i in indices]

    fig = plt.figure(figsize=(3.2 * n_show, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(3, n_show, height_ratios=[1.15, 1.0, 1.0])

    vmax_count = max(float(np.nanmax(shell["shell_counts"])) for shell in selected)
    finite_v = np.concatenate([shell["projected_map"][np.isfinite(shell["projected_map"])] for shell in selected])
    vlim = max(abs(np.nanpercentile(finite_v, 1)), abs(np.nanpercentile(finite_v, 99)))

    last_count = None
    last_vel = None
    for col, shell in enumerate(selected):
        ax3d = fig.add_subplot(grid[0, col], projection="3d")
        plot_shell_voxels(ax3d, shell)

        ax_count = fig.add_subplot(grid[1, col])
        last_count = ax_count.imshow(
            shell["shell_counts"].T,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax_count,
            interpolation="nearest",
            aspect="equal",
        )
        ax_count.set_xticks([])
        ax_count.set_yticks([])
        if col == 0:
            ax_count.set_ylabel("path length")

        ax_vel = fig.add_subplot(grid[2, col])
        last_vel = ax_vel.imshow(
            shell["projected_map"].T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
            aspect="equal",
        )
        ax_vel.set_xticks([])
        ax_vel.set_yticks([])
        if col == 0:
            ax_vel.set_ylabel(r"$\langle v_x\rangle_x$")

    fig.colorbar(last_count, ax=[fig.axes[i] for i in range(1, 2 * n_show, 2)], shrink=0.8, label="shell cells")
    fig.colorbar(last_vel, ax=[fig.axes[i] for i in range(2, 3 * n_show, 3)], shrink=0.8, label=r"$\langle v_x\rangle_x$")
    fig.suptitle("Fixed outer radius shell grid: only inner radius changes", fontsize=13)
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    print("Shell radii:")
    for shell in shells:
        print(
            f"  width={shell['width']:.4f}  "
            f"r_inner={shell['r_inner']:.4f}  r_outer={shell['r_outer']:.4f}"
        )


if __name__ == "__main__":
    main()
