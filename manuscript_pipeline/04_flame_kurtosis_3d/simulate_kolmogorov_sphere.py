#!/usr/bin/env python3
"""Generate random points in a sphere with Kolmogorov-like turbulent velocities.

The velocity field is made on a periodic 3-D Fourier grid with

    P_3D(k) proportional to k^(-11/3),

which corresponds to the Kolmogorov energy spectrum E(k) proportional to
k^(-5/3). The sampled points should therefore have a second-order velocity
structure function S2(r) roughly proportional to r^(2/3) across the usable
inertial range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree


def random_points_in_sphere(n_points: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random points inside a sphere."""
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = radius * rng.random(n_points) ** (1.0 / 3.0)
    return directions * radii[:, None]


def kolmogorov_velocity_grid(
    n_grid: int,
    box_size: float,
    rng: np.random.Generator,
    kmin_factor: float = 2.0,
    kmax_fraction: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a Gaussian random 3-D velocity field with Kolmogorov-like power."""
    freq = np.fft.fftfreq(n_grid, d=box_size / n_grid) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(freq, freq, freq, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    k_fund = 2.0 * np.pi / box_size
    k_nyq = np.pi * n_grid / box_size
    kmin = kmin_factor * k_fund
    kmax = kmax_fraction * k_nyq

    amplitude = np.zeros_like(k)
    inertial = (k >= kmin) & (k <= kmax)
    amplitude[inertial] = k[inertial] ** (-11.0 / 6.0)

    grids = []
    for _ in range(3):
        white_noise = rng.normal(size=(n_grid, n_grid, n_grid))
        modes = np.fft.fftn(white_noise)
        field = np.fft.ifftn(modes * amplitude).real
        field -= np.mean(field)
        field /= np.std(field)
        grids.append(field)

    return tuple(grids)


def sample_periodic_grid(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    points: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """Trilinearly interpolate the periodic grid at arbitrary point positions."""
    n_grid = grids[0].shape[0]
    coords = ((points + box_size / 2.0) / box_size * n_grid) % n_grid
    coords_for_scipy = [coords[:, 0], coords[:, 1], coords[:, 2]]

    velocities = np.column_stack(
        [
            map_coordinates(grid, coords_for_scipy, order=1, mode="wrap")
            for grid in grids
        ]
    )
    return velocities


def binned_structure_function(
    points: np.ndarray,
    velocities: np.ndarray,
    n_bins: int,
    max_pairs_per_bin: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate S2(r) = <|delta v|^2> in logarithmic separation bins."""
    radius = np.max(np.linalg.norm(points, axis=1))
    tree = cKDTree(points)
    pairs = np.array(list(tree.query_pairs(r=2.0 * radius, output_type="set")), dtype=int)

    dr = points[pairs[:, 0]] - points[pairs[:, 1]]
    dv = velocities[pairs[:, 0]] - velocities[pairs[:, 1]]
    sep = np.linalg.norm(dr, axis=1)
    dv2 = np.sum(dv**2, axis=1)

    good = sep > 0.0
    sep = sep[good]
    dv2 = dv2[good]

    edges = np.geomspace(np.percentile(sep, 0.02), np.percentile(sep, 85), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    s2 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    which = np.digitize(sep, edges) - 1
    for i in range(n_bins):
        idx = np.flatnonzero(which == i)
        counts[i] = idx.size
        if idx.size > 0:
            if idx.size > max_pairs_per_bin:
                idx = rng.choice(idx, size=max_pairs_per_bin, replace=False)
            s2[i] = np.mean(dv2[idx])

    return centers, s2, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--n-grid", type=int, default=96)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--max-pairs-per-bin", type=int, default=50_000)
    parser.add_argument("--fit-r-min", type=float, default=0.12)
    parser.add_argument("--fit-r-max", type=float, default=0.50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Figures/manuscript/04_kurtosis_3d_all_regions/kolmogorov_sphere_example.png"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    box_size = 2.4 * args.radius

    points = random_points_in_sphere(args.n_points, args.radius, rng)
    grids = kolmogorov_velocity_grid(args.n_grid, box_size, rng)
    velocities = sample_periodic_grid(grids, points, box_size)

    r, s2, counts = binned_structure_function(
        points, velocities, args.bins, args.max_pairs_per_bin, rng
    )
    fit_mask = (
        np.isfinite(s2)
        & (counts > 50)
        & (r >= args.fit_r_min * args.radius)
        & (r <= args.fit_r_max * args.radius)
    )
    slope, intercept = np.polyfit(np.log10(r[fit_mask]), np.log10(s2[fit_mask]), 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)

    sc = axes[0].scatter(
        points[:, 0],
        points[:, 1],
        c=velocities[:, 2],
        s=3,
        cmap="RdBu_r",
        linewidths=0,
        alpha=0.8,
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Projected sphere sample")
    fig.colorbar(sc, ax=axes[0], label=r"$v_z$")

    axes[1].loglog(r, s2, "o", ms=4, label="sampled field")
    axes[1].loglog(
        r[fit_mask],
        10.0 ** intercept * r[fit_mask] ** slope,
        color="tab:red",
        lw=2,
        label=fr"fit slope = {slope:.2f}",
    )
    ref = s2[fit_mask][0] * (r / r[fit_mask][0]) ** (2.0 / 3.0)
    axes[1].loglog(r, ref, "--", color="0.3", label=r"$r^{2/3}$")
    axes[1].set_xlabel("separation r")
    axes[1].set_ylabel(r"$S_2(r)=\langle|\delta v|^2\rangle$")
    axes[1].set_title("Kolmogorov-like velocity structure")
    axes[1].legend(frameon=False)

    fig.savefig(args.output, dpi=220)
    np.savez(
        args.output.with_suffix(".npz"),
        points=points,
        velocities=velocities,
        r=r,
        s2=s2,
        counts=counts,
        fitted_slope=slope,
    )

    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.npz')}")
    print(f"Fitted S2 slope: {slope:.3f}; Kolmogorov target is 0.667")


if __name__ == "__main__":
    main()
