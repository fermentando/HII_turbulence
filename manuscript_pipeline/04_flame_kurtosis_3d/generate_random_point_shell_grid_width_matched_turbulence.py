#!/usr/bin/env python3
"""Random-point shell projections where each shell gets its OWN turbulence field,
whose outer/injection scale (maximum eddy size) is tied to that shell's width.

The normal pipeline (generate_random_point_expanding_shell_grid.py) draws every
shell width from one shared Kolmogorov velocity cube, with a fixed outer scale
(kmin_factor=2 -> largest eddy ~ box_size/2) independent of shell width. This
script instead regenerates the field per shell with

    kmin = 2*pi / width   =>   kmin_factor = box_size / width

so turbulent structures larger than the shell width are excluded by
construction, not just averaged over by the line-of-sight projection.

Computational caveat: resolving an outer scale as small as the narrowest shell
widths while still leaving room for an inertial cascade below it would need an
enormous FFT grid (e.g. ~800^3 for width=0.02*r_out). That's not tractable, so
the per-shell grid resolution is scaled with 1/width (finer grid for narrower
shells, same idea as the N-dependent sky-projection grid in the main script)
and clamped to [GRID_FLOOR, GRID_CAP]. For shells narrower than the point
where the cap binds, the achieved outer scale is coarser than the literal
shell width -- this is reported explicitly in the per-shell metadata and the
console log (requested vs. achieved kmin_factor / outer scale).
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

from generate_random_point_expanding_shell_grid import (  # noqa: E402
    grid_size_for_n_points,
    n_jobs_for_sample,
    project_random_points,
    random_points_in_shell,
)
from generate_shell_thickness_grid import R_OUTER_FIXED, shell_output_path, width_grid  # noqa: E402
from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid, sample_periodic_grid  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_random_points_width_matched_turbulence"
)

KMAX_FRACTION = 0.35
GRID_MARGIN = 1.2  # safety factor so the inertial band isn't a single k-shell
GRID_FLOOR = 128
GRID_CAP = 220


def turbulence_grid_size_for_width(width: float, box_size: float, kmax_fraction: float = KMAX_FRACTION) -> int:
    """Field resolution needed so kmax > kmin = 2*pi/width, clamped to a feasible range."""
    kmin_factor = box_size / width
    n_grid_needed = GRID_MARGIN * 2.0 * kmin_factor / kmax_fraction
    return int(np.clip(round(n_grid_needed), GRID_FLOOR, GRID_CAP))


def achieved_kmin_factor(n_grid: int, box_size: float, kmax_fraction: float = KMAX_FRACTION) -> float:
    """The largest kmin_factor this grid can support with the same safety margin."""
    return kmax_fraction * n_grid / (GRID_MARGIN * 2.0)


def write_random_shell_npz_width_matched(
    box_size: float,
    r_inner: float,
    r_outer: float,
    width: float,
    v_bulk: float,
    n_points: int,
    n_proj_grid: int,
    out_path: Path,
    rng_field: np.random.Generator,
    rng_points: np.random.Generator,
    bin_resolution: int,
    n_jobs,
) -> dict:
    requested_kmin_factor = box_size / width
    n_grid = turbulence_grid_size_for_width(width, box_size)
    used_kmin_factor = min(requested_kmin_factor, achieved_kmin_factor(n_grid, box_size))
    clamped = used_kmin_factor < requested_kmin_factor

    grids = kolmogorov_velocity_grid(n_grid, box_size, rng_field, kmin_factor=used_kmin_factor, kmax_fraction=KMAX_FRACTION)

    points = random_points_in_shell(n_points, r_inner, r_outer, rng_points)
    velocities = sample_periodic_grid(grids, points, box_size)
    radius = np.linalg.norm(points, axis=1)
    radial_x = np.divide(points[:, 0], radius, out=np.zeros_like(radius), where=radius > 0)
    vx_turb = velocities[:, 0]
    vx_bulk = v_bulk * radial_x
    vx_total = vx_turb + vx_bulk

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
        valid_projected_mask=valid_map,
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
        kk=kk_pdf,
        kk_pdf=kk_pdf,
        kk_pairs=kk_pairs,
        c2=c2,
        c4=c4,
        cnt=cnt,
        dv_centers=dv_centers,
        n_proj_grid=n_proj_grid,
        n_jobs_used=effective_n_jobs,
        r_min=float(10 ** log_centers[0]),
        turbulence_n_grid=n_grid,
        requested_kmin_factor=requested_kmin_factor,
        used_kmin_factor=used_kmin_factor,
        requested_outer_scale=width,
        achieved_outer_scale=box_size / used_kmin_factor,
        outer_scale_clamped=clamped,
    )

    finite = kk_pdf[np.isfinite(kk_pdf) & (kk_pdf > 0)]
    valid_counts = point_counts[point_counts > 0]
    return {
        "n_proj_pixels": int(v_flat.size),
        "points_per_pixel_median": float(np.nanmedian(valid_counts)) if valid_counts.size else float("nan"),
        "k_median": float(np.nanmedian(finite)) if finite.size else np.nan,
        "k_max": float(np.nanmax(finite)) if finite.size else np.nan,
        "n_proj_grid": n_proj_grid,
        "n_jobs_used": effective_n_jobs,
        "r_min": float(10 ** log_centers[0]),
        "turbulence_n_grid": n_grid,
        "requested_kmin_factor": requested_kmin_factor,
        "used_kmin_factor": used_kmin_factor,
        "achieved_outer_scale": box_size / used_kmin_factor,
        "outer_scale_clamped": clamped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--field-seed", type=int, default=7)
    parser.add_argument("--point-seed", type=int, default=1234)
    parser.add_argument("--v-bulk", type=float, default=0.0)
    parser.add_argument("--r-outer", type=float, default=R_OUTER_FIXED)
    parser.add_argument("--width-min", type=float, default=0.02)
    parser.add_argument("--width-max", type=float, default=1.0)
    parser.add_argument("--n-widths", type=int, default=14)
    parser.add_argument("--n-points", type=int, default=50_000)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng_field = np.random.default_rng(args.field_seed)
    rng_points = np.random.default_rng(args.point_seed)

    n_proj_grid = grid_size_for_n_points(args.n_points)

    metadata = {
        "description": (
            "Random-point shell projections where each shell's turbulence field "
            "has its outer/injection scale (max eddy size) tied to that shell's "
            "width: kmin_factor = box_size/width, clamped by GRID_FLOOR/GRID_CAP."
        ),
        "box_size": args.box_size,
        "field_seed": args.field_seed,
        "point_seed": args.point_seed,
        "v_bulk": args.v_bulk,
        "r_outer": args.r_outer,
        "width_min": args.width_min,
        "width_max": args.width_max,
        "n_widths": args.n_widths,
        "n_points_per_shell": args.n_points,
        "n_proj_grid": n_proj_grid,
        "n_jobs_requested": args.n_jobs,
        "kmax_fraction": KMAX_FRACTION,
        "grid_margin": GRID_MARGIN,
        "grid_floor": GRID_FLOOR,
        "grid_cap": GRID_CAP,
    }
    metadata_path = args.output_dir / "random_point_width_matched_shell_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path.relative_to(REPO_ROOT)}")

    for width in width_grid(args.width_min, args.width_max, args.n_widths):
        r_inner = args.r_outer - float(width)
        if r_inner < 0:
            raise ValueError(f"Shell width {width:g} gives negative r_inner")
        out_path = shell_output_path(args.output_dir, r_inner, args.r_outer)
        stats = write_random_shell_npz_width_matched(
            args.box_size,
            r_inner,
            args.r_outer,
            float(width),
            args.v_bulk,
            args.n_points,
            n_proj_grid,
            out_path,
            rng_field,
            rng_points,
            args.bin_resolution,
            args.n_jobs,
        )
        clamp_note = " [CLAMPED]" if stats["outer_scale_clamped"] else ""
        print(
            f"Wrote {out_path.relative_to(REPO_ROOT)} width={width:.4f} "
            f"turb_grid={stats['turbulence_n_grid']} "
            f"kmin_factor req={stats['requested_kmin_factor']:.1f} used={stats['used_kmin_factor']:.1f} "
            f"outer_scale={stats['achieved_outer_scale']:.4f}{clamp_note} "
            f"proj_grid={stats['n_proj_grid']} n_jobs={stats['n_jobs_used']} "
            f"pixels={stats['n_proj_pixels']} Kmed={stats['k_median']:.2f} Kmax={stats['k_max']:.2f}"
        )


if __name__ == "__main__":
    main()
