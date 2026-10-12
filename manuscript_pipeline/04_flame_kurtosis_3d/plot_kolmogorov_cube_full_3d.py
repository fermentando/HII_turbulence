#!/usr/bin/env python3
"""Render the full Kolmogorov-like turbulent velocity cube in 3-D."""

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
    / "kolmogorov_cube_full_3d.png"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=96)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--percentile", type=float, default=62.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, _, vz = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng)

    step = max(args.stride, 1)
    cube = vz[::step, ::step, ::step]
    n = cube.shape[0]
    half = args.box_size / 2.0
    coords_1d = np.linspace(-half, half, args.n_grid)[::step]
    x, y, z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")

    threshold = np.nanpercentile(np.abs(cube), args.percentile)
    mask = np.abs(cube) >= threshold

    lo, hi = np.nanpercentile(cube, [1, 99])
    lim = max(abs(lo), abs(hi))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    colors = plt.colormaps["RdBu_r"](norm(cube[mask]))
    amplitudes = np.clip(np.abs(cube[mask]) / lim, 0.0, 1.0)
    colors[:, 3] = 0.10 + 0.55 * amplitudes

    fig = plt.figure(figsize=(8.4, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x[mask],
        y[mask],
        z[mask],
        c=colors,
        s=max(2.0, 10.0 / step),
        linewidths=0,
        depthshade=False,
    )

    # Draw a subtle cube outline so the full volume is easy to read.
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
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color="0.2", linewidth=0.7, alpha=0.45)

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(r"Full Kolmogorov-like velocity cube, colored by $v_z$")
    ax.view_init(elev=22, azim=-48)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False

    mappable = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label(r"$v_z / \sigma$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, bbox_inches="tight")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        vz=vz,
        rendered_stride=step,
        rendered_percentile=args.percentile,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.npz')}")
    print(f"Rendered {np.count_nonzero(mask)} of {n**3} downsampled voxels")


if __name__ == "__main__":
    main()
