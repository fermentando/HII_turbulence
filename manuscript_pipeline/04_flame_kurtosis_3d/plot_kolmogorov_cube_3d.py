#!/usr/bin/env python3
"""Plot a 3-D view of a Kolmogorov-like turbulent velocity cube."""

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
    / "kolmogorov_cube_3d_slices.png"
)


def surface_rgba(values: np.ndarray, norm: TwoSlopeNorm, cmap_name: str = "RdBu_r") -> np.ndarray:
    cmap = plt.colormaps[cmap_name]
    rgba = cmap(norm(values))
    rgba[..., 3] = 0.82
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=96)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, _, vz = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng)

    n = args.n_grid
    half = args.box_size / 2.0
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    z = np.linspace(-half, half, n)
    x2, y2 = np.meshgrid(x, y, indexing="ij")
    xz, zz = np.meshgrid(x, z, indexing="ij")
    yz, zy = np.meshgrid(y, z, indexing="ij")

    mid = n // 2
    lo, hi = np.nanpercentile(vz, [1, 99])
    lim = max(abs(lo), abs(hi))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        x2,
        y2,
        np.full_like(x2, z[mid]),
        facecolors=surface_rgba(vz[:, :, mid], norm),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot_surface(
        xz,
        np.full_like(xz, y[mid]),
        zz,
        facecolors=surface_rgba(vz[:, mid, :], norm),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.plot_surface(
        np.full_like(yz, x[mid]),
        yz,
        zy,
        facecolors=surface_rgba(vz[mid, :, :], norm),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(r"Kolmogorov-like velocity cube: slices of $v_z$")
    ax.view_init(elev=23, azim=-55)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False

    mappable = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label(r"$v_z / \sigma$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    np.savez_compressed(args.output.with_suffix(".npz"), vz=vz)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
