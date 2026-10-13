#!/usr/bin/env python3
"""Generate random-point shell projections with radial expansion.

Unlike the grid-cell shell projection, this samples a finite number of random
points in each shell and bins them in the projected (y, z) plane. The radial
bulk term therefore does not cancel exactly between front and back points in
each sightline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from generate_shell_thickness_grid import R_OUTER_FIXED, shell_output_path, width_grid  # noqa: E402
from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid, sample_periodic_grid  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_random_points_expanding_vbulk10p0"
)

# Reference point: the grid used by every run prior to this fix (50,000 points
# on a 160x160 sky grid). Scaling the grid as sqrt(n_points) around this
# anchor keeps mean points-per-occupied-pixel roughly constant across the
# random-point-count sweep, instead of letting extra points just pile up on a
# fixed pixel grid (which caps spatial resolution and makes r_min/coverage
# drift with N rather than reflecting genuine resolution).
_REFERENCE_N_POINTS = 50_000
_REFERENCE_GRID = 160

# Per-chunk subsample cap inside compute_pdf_and_statistics_chunk (vsf_module.py).
_CHUNK_SAMPLE_CAP = 10_000
_MAX_AUTO_N_JOBS = 8


def grid_size_for_n_points(n_points: int, base_n_points: int = _REFERENCE_N_POINTS, base_grid: int = _REFERENCE_GRID) -> int:
    """Projection grid side length that keeps points-per-pixel ~constant vs N."""
    scale = np.sqrt(n_points / base_n_points)
    grid = int(round(base_grid * scale))
    return int(np.clip(grid, 40, 640))


def n_jobs_for_sample(n_valid: int, cap: int = _CHUNK_SAMPLE_CAP, max_n_jobs: int = _MAX_AUTO_N_JOBS) -> int:
    """Scale the number of pooled subsample draws with the available data.

    compute_pdf_and_statistics_chunk subsamples at most `cap` points per job
    (see vsf_module.py). With n_jobs fixed at 1, increasing n_points beyond
    `cap` never changes the effective sample used for the fit. Drawing and
    pooling multiple independent `cap`-sized subsamples lets more of the data
    actually contribute as N grows, bounded so runtime stays reasonable.
    """
    return int(np.clip(round(n_valid / cap), 1, max_n_jobs))


def random_points_in_shell(
    n_points: int,
    r_inner: float,
    r_outer: float,
    rng: np.random.Generator,
) -> np.ndarray:
    directions = rng.normal(size=(n_points, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    u = rng.random(n_points)
    radii = (r_inner**3 + u * (r_outer**3 - r_inner**3)) ** (1.0 / 3.0)
    return directions * radii[:, None]


def project_random_points(
    points: np.ndarray,
    vx_points: np.ndarray,
    box_size: float,
    n_proj_grid: int,
    r_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half = box_size / 2.0
    coords = np.linspace(-half, half, n_proj_grid)
    step = coords[1] - coords[0]
    y_idx = np.floor((points[:, 1] - coords[0]) / step + 0.5).astype(int)
    z_idx = np.floor((points[:, 2] - coords[0]) / step + 0.5).astype(int)
    valid = (
        (y_idx >= 0)
        & (y_idx < n_proj_grid)
        & (z_idx >= 0)
        & (z_idx < n_proj_grid)
        & (points[:, 1] ** 2 + points[:, 2] ** 2 <= r_outer**2)
    )
    y_idx = y_idx[valid]
    z_idx = z_idx[valid]
    vx_valid = vx_points[valid]

    counts = np.zeros((n_proj_grid, n_proj_grid), dtype=float)
    sums = np.zeros_like(counts)
    np.add.at(counts, (y_idx, z_idx), 1.0)
    np.add.at(sums, (y_idx, z_idx), vx_valid)

    projected = np.full_like(counts, np.nan, dtype=float)
    nonzero = counts > 0
    projected[nonzero] = sums[nonzero] / counts[nonzero]
    return coords, projected, counts, nonzero


def write_random_shell_npz(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    box_size: float,
    r_inner: float,
    r_outer: float,
    v_bulk: float,
    n_points: int,
    n_proj_grid: int,
    out_path: Path,
    rng: np.random.Generator,
    bin_resolution: int,
    n_jobs: int,
    no_pixelize: bool = False,
) -> dict[str, float | int]:
    points = random_points_in_shell(n_points, r_inner, r_outer, rng)
    velocities = sample_periodic_grid(grids, points, box_size)
    radius = np.linalg.norm(points, axis=1)
    radial_x = np.divide(points[:, 0], radius, out=np.zeros_like(radius), where=radius > 0)
    vx_turb = velocities[:, 0]
    vx_bulk = v_bulk * radial_x
    vx_total = vx_turb + vx_bulk

    if no_pixelize:
        # Skip the sky-pixel binning/averaging entirely: feed each random
        # point's own projected (y, z, v) straight into the pair statistics,
        # instead of first averaging multiple points per sightline pixel.
        # This loses the physical near/far-side line-of-sight averaging that
        # project_random_points performs, but is otherwise a valid VSF/
        # kurtosis calculation (compute_bins/compute_pdf_and_statistics work
        # on scattered points via a KD-tree, no grid required).
        r_proj = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
        valid = r_proj <= r_outer
        y_flat = points[valid, 1]
        z_flat = points[valid, 2]
        v_flat = vx_total[valid]
        projected_map = np.ones_like(v_flat)
        point_counts = np.ones_like(v_flat)
    else:
        coords, projected_map, point_counts, valid_map = project_random_points(
            points, vx_total, box_size, n_proj_grid, r_outer
        )
        yy, zz = np.meshgrid(coords, coords, indexing="ij")
        y_flat = yy[valid_map].ravel()
        z_flat = zz[valid_map].ravel()
        v_flat = projected_map[valid_map].ravel()
    vel_err = np.zeros_like(v_flat)

    effective_n_jobs = n_jobs_for_sample(v_flat.size) if n_jobs is None else n_jobs

    log_centers, bin_edges, dv_bins, _ = compute_bins(
        y_flat, z_flat, v_flat, bin_resolution=bin_resolution
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
            n_jobs=effective_n_jobs,
        )
    kk_pdf = kurtosis_from_pdf(pdf, dv_centers)

    np.savez_compressed(
        out_path,
        y=y_flat,
        z=z_flat,
        projected_v=v_flat,
        projected_map=projected_map,
        shell_counts=point_counts,
        point_counts=point_counts,
        valid_projected_mask=np.ones_like(v_flat, dtype=bool) if no_pixelize else valid_map,
        r_inner=r_inner,
        r_outer=r_outer,
        n_random_points=n_points,
        v_bulk=v_bulk,
        point_x=points[:, 0],
        point_y=points[:, 1],
        point_z=points[:, 2],
        point_vx_turb=vx_turb,
        point_vx_bulk=vx_bulk,
        point_vx_total=vx_total,
        pdf=pdf,
        dv_bins=dv_bins,
        log_centers=log_centers,
        kk=kk_pairs,
        kk_pdf=kk_pdf,
        kk_pairs=kk_pairs,
        c2=c2,
        c4=c4,
        cnt=cnt,
        dv_centers=dv_centers,
        n_proj_grid=n_proj_grid,
        n_jobs_used=effective_n_jobs,
        r_min=float(10 ** log_centers[0]),
    )

    finite = kk_pairs[np.isfinite(kk_pairs) & (kk_pairs > 0)]
    valid_counts = point_counts[point_counts > 0]
    return {
        "n_proj_pixels": int(v_flat.size),
        "points_per_pixel_median": float(np.nanmedian(valid_counts)),
        "points_per_pixel_max": int(np.nanmax(valid_counts)),
        "single_point_pixels": int(np.count_nonzero(valid_counts == 1)),
        "k_median": float(np.nanmedian(finite)) if finite.size else np.nan,
        "k_max": float(np.nanmax(finite)) if finite.size else np.nan,
        "projected_rms": float(np.nanstd(v_flat)),
        "bulk_rms_points": float(np.nanstd(vx_bulk)),
        "n_proj_grid": n_proj_grid,
        "n_jobs_used": effective_n_jobs,
        "r_min": float(10 ** log_centers[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=160)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--point-seed", type=int, default=1234)
    parser.add_argument("--v-bulk", type=float, default=10.0)
    parser.add_argument("--r-outer", type=float, default=R_OUTER_FIXED)
    parser.add_argument("--width-min", type=float, default=0.02)
    parser.add_argument("--width-max", type=float, default=1.0)
    parser.add_argument("--n-widths", type=int, default=14)
    parser.add_argument("--n-points", type=int, default=50_000)
    parser.add_argument(
        "--n-proj-grid",
        type=int,
        default=None,
        help="Sky projection grid side length. Default: auto-scaled as "
        "160*sqrt(n_points/50000) so points-per-pixel stays ~constant "
        "across a random-point-count sweep (pass explicitly to override).",
    )
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of pooled 10k-point subsample draws for the pair "
        "statistics. Default: auto-scaled with the number of populated "
        "sky pixels so more random points actually feed the fit (pass "
        "explicitly, e.g. 1, to override).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-pixelize",
        action="store_true",
        help="Skip the sky-pixel binning/averaging step and feed each random "
        "point's own projected (y, z, v) directly into the pair statistics "
        "instead. Loses the near/far-side line-of-sight averaging that a "
        "real sightline pixel performs; --n-proj-grid/--n-jobs auto-scaling "
        "is unused in this mode (n_jobs falls back to n_jobs_for_sample "
        "using the raw point count).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng_field = np.random.default_rng(args.seed)
    grids = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng_field)
    rng_points = np.random.default_rng(args.point_seed)

    n_proj_grid = args.n_proj_grid if args.n_proj_grid is not None else grid_size_for_n_points(args.n_points)

    metadata = {
        "description": "Random-point shell projections with v = v_turb + v_bulk r_hat.",
        "n_grid": args.n_grid,
        "box_size": args.box_size,
        "seed": args.seed,
        "point_seed": args.point_seed,
        "v_bulk": args.v_bulk,
        "r_outer": args.r_outer,
        "width_min": args.width_min,
        "width_max": args.width_max,
        "n_widths": args.n_widths,
        "n_points_per_shell": args.n_points,
        "n_proj_grid": n_proj_grid,
        "n_proj_grid_auto": args.n_proj_grid is None,
        "n_jobs_requested": args.n_jobs,
        "sigma_vx_turb_grid": float(np.std(grids[0])),
        "no_pixelize": args.no_pixelize,
    }
    metadata_path = args.output_dir / "random_point_expanding_shell_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path.relative_to(REPO_ROOT)}")

    for width in width_grid(args.width_min, args.width_max, args.n_widths):
        r_inner = args.r_outer - float(width)
        if r_inner < 0:
            raise ValueError(f"Shell width {width:g} gives negative r_inner")
        out_path = shell_output_path(args.output_dir, r_inner, args.r_outer)
        stats = write_random_shell_npz(
            grids,
            args.box_size,
            r_inner,
            args.r_outer,
            args.v_bulk,
            args.n_points,
            n_proj_grid,
            out_path,
            rng_points,
            args.bin_resolution,
            args.n_jobs,
            no_pixelize=args.no_pixelize,
        )
        print(
            f"Wrote {out_path.relative_to(REPO_ROOT)} width={width:.4f} "
            f"grid={stats['n_proj_grid']} n_jobs={stats['n_jobs_used']} "
            f"pixels={stats['n_proj_pixels']} median_pts={stats['points_per_pixel_median']:.1f} "
            f"single_pix={stats['single_point_pixels']} r_min={stats['r_min']:.5f} "
            f"Kmed={stats['k_median']:.2f} Kmax={stats['k_max']:.2f} proj_rms={stats['projected_rms']:.2f}"
        )


if __name__ == "__main__":
    main()
