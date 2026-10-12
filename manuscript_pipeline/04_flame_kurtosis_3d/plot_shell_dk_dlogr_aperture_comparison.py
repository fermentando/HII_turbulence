#!/usr/bin/env python3
"""Compare dK/dlog10(r) fits for full disk and central-aperture shell maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from filled_sphere_aperture_kurtosis import aperture_stats  # noqa: E402
from plot_shell_kurtosis_linear_vs_thickness import load_shell  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import (  # noqa: E402
    KURTOSIS_FLOOR,
    MIN_FIT_POINTS,
    RNG_SEED,
    shell_paths_in_dir,
)


DEFAULT_SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_width_to_rout"
)


def fit_k_vs_logr_above_floor(record: dict, floor: float = KURTOSIS_FLOOR) -> dict:
    mask = (
        np.isfinite(record["r"])
        & np.isfinite(record["kk"])
        & (record["r"] > 0)
        & (record["kk"] >= floor)
    )
    n_points = int(np.count_nonzero(mask))
    result = {
        "slope": 0.0,
        "slope_err": 0.0,
        "intercept": np.nan,
        "intercept_err": np.nan,
        "rms_resid": np.nan,
        "n_points": n_points,
        "fit_mask": mask,
        "used_zero_fallback": n_points < MIN_FIT_POINTS,
    }
    if n_points < MIN_FIT_POINTS:
        return result

    x = np.log10(record["r"][mask])
    y = record["kk"][mask]
    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope_err, intercept_err = np.sqrt(np.diag(cov))
    except (ValueError, np.linalg.LinAlgError):
        coeffs = np.polyfit(x, y, 1)
        slope_err = np.nan
        intercept_err = np.nan
    model = coeffs[1] + coeffs[0] * x
    result["slope"] = float(coeffs[0])
    result["intercept"] = float(coeffs[1])
    result["slope_err"] = float(slope_err)
    result["intercept_err"] = float(intercept_err)
    result["rms_resid"] = float(np.sqrt(np.nanmean((y - model) ** 2)))
    result["used_zero_fallback"] = False
    return result


def aperture_record_from_npz(path: Path, aperture_fraction: float, bin_resolution: int, n_jobs: int) -> dict:
    with np.load(path) as data:
        y = np.asarray(data["y"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
        v = np.asarray(data["projected_v"], dtype=float)
        projected_map = np.asarray(data["projected_map"], dtype=float)
        shell_counts = np.asarray(data["shell_counts"], dtype=float)
        valid_map = np.isfinite(projected_map)
        path_flat = shell_counts[valid_map].ravel()
        r_inner = float(data["r_inner"])
        r_outer = float(data["r_outer"])

    stats = aperture_stats(
        y,
        z,
        v,
        path_flat,
        aperture_fraction * r_outer,
        bin_resolution,
        n_jobs,
    )
    record = {
        "path": path,
        "r": stats["r"],
        "kk": stats["kk"],
        "r_inner": r_inner,
        "r_outer": r_outer,
        "width": r_outer - r_inner,
        "n_proj_pixels": stats["n_pixels"],
        "path_min": stats["path_min"],
        "path_max": stats["path_max"],
        "path_median": stats["path_median"],
    }
    return {**record, **fit_k_vs_logr_above_floor(record)}


def plot_comparison(full_records: list[dict], aperture_records: list[dict], aperture_fraction: float, out: Path) -> None:
    full = sorted(full_records, key=lambda record: record["width"] / record["r_outer"])
    ap = sorted(aperture_records, key=lambda record: record["width"] / record["r_outer"])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    full_widths = np.array([record["width"] / record["r_outer"] for record in full])
    full_slopes = np.array([record["slope"] for record in full])
    full_errs = np.array([record["slope_err"] for record in full])
    ap_widths = np.array([record["width"] / record["r_outer"] for record in ap])
    ap_slopes = np.array([record["slope"] for record in ap])
    ap_errs = np.array([record["slope_err"] for record in ap])

    ax.errorbar(
        full_widths,
        full_slopes,
        yerr=full_errs,
        fmt="o-",
        ms=5,
        lw=1.0,
        capsize=3,
        color="0.35",
        alpha=0.3,
        label="full aperture",
        zorder=1,
    )
    ax.errorbar(
        ap_widths,
        ap_slopes,
        yerr=ap_errs,
        fmt="s-",
        ms=6,
        lw=1.5,
        capsize=3,
        color="C3",
        label=fr"central aperture $R_{{proj}}\leq {aperture_fraction:.1f}r_{{out}}$",
        zorder=3,
    )

    zero = [record for record in ap if record["used_zero_fallback"]]
    if zero:
        ax.scatter(
            [record["width"] / record["r_outer"] for record in zero],
            [0.0 for _ in zero],
            marker="x",
            s=45,
            color="C3",
            label=fr"aperture: <{MIN_FIT_POINTS} bins above K=3",
            zorder=4,
        )

    ax.axvline(0.1, color="0.25", ls=":", lw=1.1, label=r"$\Delta R/r_{out}=0.1$")
    ax.axhline(0, color="0.35", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$(r_{out}-r_{in})/r_{out}$")
    ax.set_ylabel(r"Linear slope, $dK/d\log_{10}r$")
    ax.set_title(r"$dK/d\log r$: full aperture vs central-aperture control")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_csv(full_records: list[dict], aperture_records: list[dict], out: Path) -> None:
    full_sorted = sorted(full_records, key=lambda record: record["width"] / record["r_outer"])
    ap_sorted = sorted(aperture_records, key=lambda record: record["width"] / record["r_outer"])
    with out.open("w", encoding="utf-8") as fobj:
        fobj.write(
            "width_over_rout,width,r_inner,r_outer,"
            "full_dK_dlogr,full_err,full_n_points,full_zero_fallback,"
            "aperture_dK_dlogr,aperture_err,aperture_n_points,aperture_zero_fallback,"
            "aperture_n_pixels,aperture_path_min,aperture_path_max,aperture_path_median\n"
        )
        for full, ap in zip(full_sorted, ap_sorted):
            fobj.write(
                f"{full['width'] / full['r_outer']:.8f},{full['width']:.8f},"
                f"{full['r_inner']:.8f},{full['r_outer']:.8f},"
                f"{full['slope']:.8f},{full['slope_err']:.8f},{full['n_points']},{int(full['used_zero_fallback'])},"
                f"{ap['slope']:.8f},{ap['slope_err']:.8f},{ap['n_points']},{int(ap['used_zero_fallback'])},"
                f"{ap['n_proj_pixels']},{ap['path_min']:.8f},{ap['path_max']:.8f},{ap['path_median']:.8f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    parser.add_argument("--aperture-fraction", type=float, default=0.8)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    full_records = []
    aperture_records = []
    for path in shell_paths_in_dir(args.shell_dir):
        shell = load_shell(path, rng)
        full = {**shell, **fit_k_vs_logr_above_floor(shell)}
        full_records.append(full)
        aperture = aperture_record_from_npz(path, args.aperture_fraction, args.bin_resolution, args.n_jobs)
        aperture_records.append(aperture)
        print(
            f"width/rout={full['width'] / full['r_outer']:.4f} "
            f"full={full['slope']:.3f} (n={full['n_points']:2d}) "
            f"aperture={aperture['slope']:.3f} (n={aperture['n_points']:2d}, "
            f"zero={aperture['used_zero_fallback']})"
        )

    out_fig = args.shell_dir / "kurtosis_dK_dlogr_slope_vs_shell_width_aperture08_overlay.png"
    out_csv = args.shell_dir / "kurtosis_dK_dlogr_slope_vs_shell_width_aperture08_overlay.csv"
    plot_comparison(full_records, aperture_records, args.aperture_fraction, out_fig)
    write_csv(full_records, aperture_records, out_csv)
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
