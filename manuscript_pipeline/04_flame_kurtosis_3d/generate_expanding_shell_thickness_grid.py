#!/usr/bin/env python3
"""Generate fixed-outer-radius shell projections with radial expansion.

The turbulent cube is generated exactly as in generate_shell_thickness_grid.py,
then a radial bulk flow is added:

    v = v_turb + v_bulk r_hat

Only the x component is needed for the x-projected shell analysis, so the
stored/projection velocity is

    v_x = v_x,turb + v_bulk x / r.

The default v_bulk=1 is in units of the turbulent 1-D component dispersion,
because each turbulent velocity component is normalized to sigma=1.
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
sys.path.insert(0, str(HERE))

from generate_shell_thickness_grid import (  # noqa: E402
    N_WIDTHS,
    R_OUTER_FIXED,
    WIDTH_MAX,
    WIDTH_MIN,
    shell_output_path,
    width_grid,
    write_shell_npz,
)
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_expanding_vbulk1p0"
)


def add_radial_bulk_vx(vx_turb: np.ndarray, box_size: float, v_bulk: float) -> tuple[np.ndarray, np.ndarray]:
    n_grid = vx_turb.shape[0]
    coords = np.linspace(-box_size / 2.0, box_size / 2.0, n_grid)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    radius = np.sqrt(x**2 + y**2 + z**2)
    radial_x = np.zeros_like(radius)
    valid = radius > 0
    radial_x[valid] = x[valid] / radius[valid]
    return vx_turb + v_bulk * radial_x, radial_x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=160)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--v-bulk", type=float, default=1.0)
    parser.add_argument("--r-outer", type=float, default=R_OUTER_FIXED)
    parser.add_argument("--width-min", type=float, default=WIDTH_MIN)
    parser.add_argument("--width-max", type=float, default=WIDTH_MAX)
    parser.add_argument("--n-widths", type=int, default=N_WIDTHS)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vx_turb, _, _ = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng)
    vx_expanding, radial_x = add_radial_bulk_vx(vx_turb, args.box_size, args.v_bulk)

    metadata = {
        "description": "Fixed-outer-radius synthetic shell projections with v = v_turb + v_bulk r_hat.",
        "n_grid": args.n_grid,
        "box_size": args.box_size,
        "seed": args.seed,
        "v_bulk": args.v_bulk,
        "v_bulk_units": "turbulent 1D sigma_v; sigma_v,turb = 1 per component",
        "r_outer": args.r_outer,
        "width_min": args.width_min,
        "width_max": args.width_max,
        "n_widths": args.n_widths,
        "sigma_vx_turb": float(np.std(vx_turb)),
        "sigma_vx_expanding_full_cube": float(np.std(vx_expanding)),
        "sigma_radial_x_full_cube": float(np.std(radial_x)),
    }
    metadata_path = args.output_dir / "expanding_shell_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path.relative_to(REPO_ROOT)}")
    print(
        f"v_bulk={args.v_bulk:.3f}; "
        f"sigma(vx_turb)={metadata['sigma_vx_turb']:.3f}; "
        f"sigma(vx_expanding full cube)={metadata['sigma_vx_expanding_full_cube']:.3f}"
    )

    for width in width_grid(args.width_min, args.width_max, args.n_widths):
        r_inner = args.r_outer - float(width)
        if r_inner <= 0:
            raise ValueError(f"Shell width {width:g} gives non-positive r_inner")
        out_path = shell_output_path(args.output_dir, r_inner, args.r_outer)
        stats = write_shell_npz(
            vx_expanding,
            args.box_size,
            r_inner,
            args.r_outer,
            out_path,
            args.bin_resolution,
            args.n_jobs,
        )
        print(
            f"Wrote {out_path.relative_to(REPO_ROOT)}  "
            f"width={width:.4f} pixels={stats['n_proj_pixels']} "
            f"Kmed={stats['k_median']:.2f} Kmax={stats['k_max']:.2f}"
        )


if __name__ == "__main__":
    main()
