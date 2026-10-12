#!/usr/bin/env python3
"""Compare full-disk and central-aperture logK-logr fits for shell widths."""

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
from plot_shell_kurtosis_loglog_vs_thickness import fit_logk_logr_above_floor  # noqa: E402
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
    shell = {
        "path": path,
        "r": stats["r"],
        "kk": stats["kk"],
        "kk_lo": np.full_like(stats["kk"], np.nan, dtype=float),
        "kk_hi": np.full_like(stats["kk"], np.nan, dtype=float),
        "r_inner": r_inner,
        "r_outer": r_outer,
        "width": r_outer - r_inner,
        "n_proj_pixels": stats["n_pixels"],
        "path_min": stats["path_min"],
        "path_max": stats["path_max"],
        "path_median": stats["path_median"],
    }
    fit = fit_logk_logr_above_floor(shell)
    return {**shell, **fit}


def plot_comparison(full_records: list[dict], aperture_records: list[dict], threshold: float, aperture_fraction: float, out: Path) -> None:
    full = sorted(full_records, key=lambda record: record["width"] / record["r_outer"])
    ap = sorted(aperture_records, key=lambda record: record["width"] / record["r_outer"])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cmap = plt.colormaps["viridis"]
    rms_full = np.array([record["rms_log_resid"] for record in full])
    finite_rms = rms_full[np.isfinite(rms_full)]
    norm = plt.Normalize(
        vmin=float(np.nanmin(finite_rms)) if finite_rms.size else 0.0,
        vmax=float(np.nanmax(finite_rms)) if finite_rms.size else 1.0,
    )

    for record in full:
        width_norm = record["width"] / record["r_outer"]
        alpha = 1.0 if width_norm <= threshold else 0.25
        color = cmap(norm(record["rms_log_resid"])) if np.isfinite(record["rms_log_resid"]) else "0.5"
        ax.errorbar(
            width_norm,
            record["slope"],
            yerr=record["slope_err"],
            fmt="o",
            ms=6,
            capsize=3,
            color=color,
            alpha=alpha,
            label="full disk" if record is full[0] else None,
            zorder=2,
        )

    full_widths = np.array([record["width"] / record["r_outer"] for record in full])
    full_slopes = np.array([record["slope"] for record in full])
    ax.plot(full_widths, full_slopes, color="0.45", lw=1.1, alpha=0.45, zorder=1)

    if ap:
        ap_widths = np.array([record["width"] / record["r_outer"] for record in ap])
        ap_slopes = np.array([record["slope"] for record in ap])
        ap_errs = np.array([record["slope_err"] for record in ap])
        ax.errorbar(
            ap_widths,
            ap_slopes,
            yerr=ap_errs,
            fmt="s-",
            ms=6,
            lw=1.4,
            capsize=3,
            color="C3",
            label=fr"central aperture $R_{{proj}}\leq {aperture_fraction:.1f}r_{{out}}$",
            zorder=4,
        )

    ax.axvline(threshold, color="0.25", ls=":", lw=1.1, label=fr"$\Delta R/r_{{out}}={threshold:g}$")
    ax.axhline(0, color="0.35", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$(r_{out}-r_{in})/r_{out}$")
    ax.set_ylabel(r"Power-law slope, $d\log_{10}K/d\log_{10}r$")
    ax.set_title("Log-log kurtosis fit: full disk vs central-aperture control")
    ax.grid(True, which="both", alpha=0.3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Full-disk RMS residual in log10(K)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_csv(full_records: list[dict], aperture_records: list[dict], out: Path) -> None:
    aperture_by_width = {
        round(record["width"] / record["r_outer"], 8): record for record in aperture_records
    }
    with out.open("w", encoding="utf-8") as fobj:
        fobj.write(
            "width_over_rout,width,r_inner,r_outer,"
            "full_slope,full_slope_err,full_rms_log,full_n_points,full_n_pixels,"
            "aperture_slope,aperture_slope_err,aperture_rms_log,aperture_n_points,aperture_n_pixels,"
            "aperture_path_min,aperture_path_max,aperture_path_median\n"
        )
        for full in sorted(full_records, key=lambda record: record["width"] / record["r_outer"]):
            width_norm = round(full["width"] / full["r_outer"], 8)
            ap = aperture_by_width.get(width_norm)
            if ap is None:
                ap_vals = [np.nan] * 8
            else:
                ap_vals = [
                    ap["slope"],
                    ap["slope_err"],
                    ap["rms_log_resid"],
                    ap["n_points"],
                    ap["n_proj_pixels"],
                    ap["path_min"],
                    ap["path_max"],
                    ap["path_median"],
                ]
            fobj.write(
                f"{full['width'] / full['r_outer']:.8f},{full['width']:.8f},"
                f"{full['r_inner']:.8f},{full['r_outer']:.8f},"
                f"{full['slope']:.8f},{full['slope_err']:.8f},{full['rms_log_resid']:.8f},"
                f"{full['n_points']},{full['n_proj_pixels']},"
                + ",".join("nan" if not np.isfinite(v) else f"{v:.8f}" for v in ap_vals)
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--aperture-fraction", type=float, default=0.8)
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    full_records = []
    aperture_records = []
    for path in shell_paths_in_dir(args.shell_dir):
        shell = load_shell(path, rng)
        fit = fit_logk_logr_above_floor(shell)
        full = {**shell, **fit}
        full_records.append(full)
        width_norm = full["width"] / full["r_outer"]
        print(
            f"full width/rout={width_norm:.4f} slope={full['slope']:.3f} "
            f"n={full['n_points']:2d} alpha={'normal' if width_norm <= args.threshold else 'faded'}"
        )
        if width_norm > args.threshold:
            ap = aperture_record_from_npz(path, args.aperture_fraction, args.bin_resolution, args.n_jobs)
            aperture_records.append(ap)
            print(
                f"  aperture R<={args.aperture_fraction:.2f}rout slope={ap['slope']:.3f} "
                f"n={ap['n_points']:2d} pixels={ap['n_proj_pixels']:5d} "
                f"path={ap['path_min']:.0f}..{ap['path_max']:.0f}"
            )

    usable_full = [record for record in full_records if record["n_points"] >= MIN_FIT_POINTS and np.isfinite(record["slope"])]
    usable_ap = [record for record in aperture_records if record["n_points"] >= MIN_FIT_POINTS and np.isfinite(record["slope"])]
    out_fig = args.shell_dir / "kurtosis_logk_logr_slope_vs_shell_width_aperture08_overlay.png"
    out_csv = args.shell_dir / "kurtosis_logk_logr_slope_vs_shell_width_aperture08_overlay.csv"
    plot_comparison(usable_full, usable_ap, args.threshold, args.aperture_fraction, out_fig)
    write_csv(full_records, aperture_records, out_csv)
    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
