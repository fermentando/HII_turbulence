#!/usr/bin/env python3
"""Generate shell-projection NPZ files for the shell-thickness fit analysis."""

from __future__ import annotations

import argparse
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

from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from projected_shell_kurtosis import shell_projected_velocity  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
R_OUTER_FIXED = 1.0
WIDTH_MIN = 0.02
WIDTH_MAX = 0.5
N_WIDTHS = 12


def width_grid(width_min: float, width_max: float, n_widths: int) -> np.ndarray:
    return np.logspace(np.log10(width_min), np.log10(width_max), n_widths)


def shell_output_path(out_dir: Path, r_inner: float, r_outer: float) -> Path:
    tag = f"rin{r_inner:.4f}_rout{r_outer:.4f}".replace(".", "p")
    return out_dir / f"Synthetic_x_projected_shell_{tag}_order1_pdfkk.npz"


def write_shell_npz(
    vx: np.ndarray,
    box_size: float,
    r_inner: float,
    r_outer: float,
    out_path: Path,
    bin_resolution: int,
    n_jobs: int,
) -> dict[str, float | int]:
    y, z, projected_v, shell_counts, shell_mask = shell_projected_velocity(
        vx, box_size, r_inner, r_outer
    )

    yy, zz = np.meshgrid(y, z, indexing="ij")
    valid = np.isfinite(projected_v)
    y_flat = yy[valid].ravel()
    z_flat = zz[valid].ravel()
    v_flat = projected_v[valid].ravel()
    vel_err = np.zeros_like(v_flat)

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
            n_jobs=n_jobs,
        )
    kk_pdf = kurtosis_from_pdf(pdf, dv_centers)

    np.savez_compressed(
        out_path,
        y=y_flat,
        z=z_flat,
        projected_v=v_flat,
        projected_map=projected_v,
        shell_counts=shell_counts,
        shell_mask=shell_mask,
        r_inner=r_inner,
        r_outer=r_outer,
        vx_cube=vx,
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
    )

    finite = kk_pairs[np.isfinite(kk_pairs) & (kk_pairs > 0)]
    return {
        "n_proj_pixels": int(v_flat.size),
        "fill_fraction": float(np.mean(shell_mask)),
        "k_median": float(np.nanmedian(finite)) if finite.size else np.nan,
        "k_max": float(np.nanmax(finite)) if finite.size else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=160)
    parser.add_argument("--box-size", type=float, default=2.4)
    parser.add_argument("--seed", type=int, default=7)
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
    vx, _, _ = kolmogorov_velocity_grid(args.n_grid, args.box_size, rng)

    for width in width_grid(args.width_min, args.width_max, args.n_widths):
        r_inner = args.r_outer - float(width)
        if r_inner < 0:
            raise ValueError(f"Shell width {width:g} gives negative r_inner")
        out_path = shell_output_path(args.output_dir, r_inner, args.r_outer)
        stats = write_shell_npz(
            vx,
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
