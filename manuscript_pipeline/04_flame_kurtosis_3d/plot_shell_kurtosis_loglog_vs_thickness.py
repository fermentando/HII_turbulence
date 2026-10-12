#!/usr/bin/env python3
"""Fit log10(K) vs log10(r) for synthetic shell kurtosis curves."""

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
sys.path.insert(0, str(HERE))

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


def fit_logk_logr_above_floor(shell: dict, floor: float = KURTOSIS_FLOOR) -> dict:
    mask = (
        np.isfinite(shell["r"])
        & np.isfinite(shell["kk"])
        & (shell["r"] > 0)
        & (shell["kk"] >= floor)
    )
    result = {
        "slope": np.nan,
        "slope_err": np.nan,
        "intercept": np.nan,
        "intercept_err": np.nan,
        "rms_log_resid": np.nan,
        "n_points": int(np.count_nonzero(mask)),
        "fit_mask": mask,
        "r_fit": shell["r"][mask],
        "kk_fit": shell["kk"][mask],
    }
    if result["n_points"] < MIN_FIT_POINTS:
        return result

    x = np.log10(shell["r"][mask])
    y = np.log10(shell["kk"][mask])
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
    result["rms_log_resid"] = float(np.sqrt(np.nanmean((y - model) ** 2)))
    return result


def plot_slope_vs_thickness(records: list[dict], out: Path) -> None:
    widths_norm = np.array([record["width"] / record["r_outer"] for record in records])
    slopes = np.array([record["slope"] for record in records])
    slope_err = np.array([record["slope_err"] for record in records])
    rms = np.array([record["rms_log_resid"] for record in records])
    order = np.argsort(widths_norm)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sc = ax.scatter(widths_norm[order], slopes[order], c=rms[order], cmap="viridis", zorder=3)
    ax.errorbar(widths_norm[order], slopes[order], yerr=slope_err[order], fmt="none", color="0.25", capsize=3, zorder=2)
    ax.plot(widths_norm[order], slopes[order], color="0.35", lw=1.1, zorder=1)
    ax.axhline(0, color="0.35", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$(r_{out}-r_{in})/r_{out}$")
    ax.set_ylabel(r"Power-law slope, $d\log_{10}K/d\log_{10}r$")
    ax.set_title(r"Log-log kurtosis fit vs. normalized shell thickness")
    ax.grid(True, which="both", alpha=0.3)
    fig.colorbar(sc, ax=ax, label="RMS residual in log10(K)")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_all_fit_examples(records: list[dict], out: Path) -> None:
    n = len(records)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.25 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, shell in zip(axes, records):
        yerr_all = np.vstack([
            np.clip(shell["kk"] - shell["kk_lo"], 0, None),
            np.clip(shell["kk_hi"] - shell["kk"], 0, None),
        ])
        ax.errorbar(shell["r"], shell["kk"], yerr=yerr_all, fmt="o", ms=3, color="0.6", ecolor="0.78", capsize=1.5)
        mask = shell["fit_mask"]
        if np.count_nonzero(mask) >= MIN_FIT_POINTS and np.isfinite(shell["slope"]):
            ax.plot(shell["r_fit"], shell["kk_fit"], "o", ms=3.2, color="C0")
            r_line = np.geomspace(shell["r_fit"].min(), shell["r_fit"].max(), 200)
            model = 10 ** shell["intercept"] * r_line ** shell["slope"]
            ax.plot(
                r_line,
                model,
                "-",
                color="C3",
                lw=1.8,
                label=fr"$b={shell['slope']:.2f}$, rms={shell['rms_log_resid']:.3f}",
            )
            ax.legend(fontsize=7, loc="best")
        ax.axhline(KURTOSIS_FLOOR, color="0.4", ls="--", lw=0.9, alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(fr"$\Delta R/r_{{out}}={shell['width'] / shell['r_outer']:.3f}$", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$r_{\rm proj}$")
    for ax in axes[::ncols]:
        ax.set_ylabel("Kurtosis, $K$")

    fig.suptitle(r"Log-log linear fits to kurtosis for all shell widths", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    records = []
    for path in shell_paths_in_dir(args.shell_dir):
        shell = load_shell(path, rng)
        fit = fit_logk_logr_above_floor(shell)
        record = {**shell, **fit}
        records.append(record)
        print(
            f"width/rout={record['width'] / record['r_outer']:.4f} "
            f"n_above_floor={record['n_points']:3d} "
            f"slope={record['slope']:.3f} +/- {record['slope_err']:.3f} "
            f"rms_log={record['rms_log_resid']:.4f}"
        )

    usable = [record for record in records if record["n_points"] >= MIN_FIT_POINTS and np.isfinite(record["slope"])]
    if not usable:
        raise RuntimeError(f"No shells had at least {MIN_FIT_POINTS} bins above K>={KURTOSIS_FLOOR:g}")

    slope_out = args.shell_dir / "kurtosis_logk_logr_slope_vs_normalized_shell_thickness.png"
    examples_out = args.shell_dir / "kurtosis_logk_logr_fit_examples_all_widths.png"
    csv_out = args.shell_dir / "kurtosis_logk_logr_slopes.csv"
    plot_slope_vs_thickness(usable, slope_out)
    plot_all_fit_examples(usable, examples_out)

    with csv_out.open("w", encoding="utf-8") as fobj:
        fobj.write("width_over_rout,width,r_inner,r_outer,slope,slope_err,intercept,intercept_err,rms_log_resid,n_points,n_projected_pixels\n")
        for record in records:
            fobj.write(
                f"{record['width'] / record['r_outer']:.8f},{record['width']:.8f},"
                f"{record['r_inner']:.8f},{record['r_outer']:.8f},"
                f"{record['slope']:.8f},{record['slope_err']:.8f},"
                f"{record['intercept']:.8f},{record['intercept_err']:.8f},"
                f"{record['rms_log_resid']:.8f},{record['n_points']},{record['n_proj_pixels']}\n"
            )

    print(f"Wrote {slope_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {examples_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {csv_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
