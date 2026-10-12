#!/usr/bin/env python3
"""Linear K(r) fits for fixed-outer-radius synthetic shell projections."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from plot_shell_kurtosis_slope_vs_thickness import (  # noqa: E402
    KURTOSIS_FLOOR,
    MIN_FIT_POINTS,
    RNG_SEED,
    bootstrap_kurtosis_from_pdf,
)


DEFAULT_SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
FNAME_RE = re.compile(r"Synthetic_x_projected_shell_rin([\dp]+)_rout([\dp]+)_order1_pdfkk\.npz")


def parse_shell_radii(path: Path) -> tuple[float, float]:
    match = FNAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected shell filename: {path.name}")
    return float(match.group(1).replace("p", ".")), float(match.group(2).replace("p", "."))


def load_shell(path: Path, rng: np.random.Generator) -> dict:
    with np.load(path) as data:
        r_inner, r_outer = parse_shell_radii(path)
        log_centers = np.asarray(data["log_centers"], dtype=float)
        r = 10**log_centers
        kk = np.asarray(data["kk"], dtype=float)
        pdf = np.asarray(data["pdf"], dtype=float)
        cnt = np.asarray(data["cnt"], dtype=float)
        dv_centers = np.asarray(data["dv_centers"], dtype=float)
        n_proj_pixels = int(np.asarray(data["y"]).size)

    valid = np.isfinite(r) & np.isfinite(kk) & (r > 0)
    order = np.argsort(r[valid])
    idx = np.flatnonzero(valid)[order]
    kk_lo_all, kk_hi_all = bootstrap_kurtosis_from_pdf(pdf, dv_centers, cnt, rng)
    return {
        "path": path,
        "r": r[idx],
        "kk": kk[idx],
        "kk_lo": kk_lo_all[idx],
        "kk_hi": kk_hi_all[idx],
        "r_inner": r_inner,
        "r_outer": r_outer,
        "width": r_outer - r_inner,
        "n_proj_pixels": n_proj_pixels,
    }


def fit_radius_linear_above_floor(shell: dict, floor: float = KURTOSIS_FLOOR) -> dict:
    mask = np.isfinite(shell["r"]) & np.isfinite(shell["kk"]) & (shell["kk"] >= floor)
    result = {
        "slope": np.nan,
        "slope_err": np.nan,
        "intercept": np.nan,
        "intercept_err": np.nan,
        "n_points": int(np.count_nonzero(mask)),
        "fit_mask": mask,
        "r_fit": shell["r"][mask],
        "kk_fit": shell["kk"][mask],
    }
    if result["n_points"] < MIN_FIT_POINTS:
        return result

    try:
        coeffs, cov = np.polyfit(shell["r"][mask], shell["kk"][mask], 1, cov=True)
        slope_err, intercept_err = np.sqrt(np.diag(cov))
    except (ValueError, np.linalg.LinAlgError):
        coeffs = np.polyfit(shell["r"][mask], shell["kk"][mask], 1)
        slope_err = np.nan
        intercept_err = np.nan

    result["slope"] = float(coeffs[0])
    result["intercept"] = float(coeffs[1])
    result["slope_err"] = float(slope_err)
    result["intercept_err"] = float(intercept_err)
    return result


def plot_slope_vs_thickness(records: list[dict], out: Path) -> None:
    widths = np.array([record["width"] for record in records])
    slopes = np.array([record["slope"] for record in records])
    slope_err = np.array([record["slope_err"] for record in records])
    n_points = np.array([record["n_points"] for record in records])
    order = np.argsort(widths)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.errorbar(widths[order], slopes[order], yerr=slope_err[order], fmt="o-", ms=5, lw=1.4, capsize=3)
    for w, slope, n in zip(widths[order], slopes[order], n_points[order]):
        ax.annotate(str(n), (w, slope), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center", color="0.4")
    ax.axhline(0, color="0.35", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Shell thickness, $r_{out} - r_{in}$")
    ax.set_ylabel(r"Linear slope, $dK/dr$")
    ax.set_title(r"Linear kurtosis slope vs. shell thickness")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_example_fits(records: list[dict], out: Path) -> None:
    example_idx = [0, len(records) // 2, len(records) - 1]
    example_idx = sorted(set(min(max(i, 0), len(records) - 1) for i in example_idx))

    fig, axes = plt.subplots(1, len(example_idx), figsize=(4.6 * len(example_idx), 4.2), sharey=True)
    if len(example_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, example_idx):
        shell = records[idx]
        yerr_all = np.vstack([
            np.clip(shell["kk"] - shell["kk_lo"], 0, None),
            np.clip(shell["kk_hi"] - shell["kk"], 0, None),
        ])
        ax.errorbar(shell["r"], shell["kk"], yerr=yerr_all, fmt="o", ms=4, color="0.6", ecolor="0.75", capsize=2)
        mask = shell["fit_mask"]
        if np.count_nonzero(mask) >= MIN_FIT_POINTS and np.isfinite(shell["slope"]):
            yerr_fit = np.vstack([
                np.clip(shell["kk"][mask] - shell["kk_lo"][mask], 0, None),
                np.clip(shell["kk_hi"][mask] - shell["kk"][mask], 0, None),
            ])
            ax.errorbar(shell["r_fit"], shell["kk_fit"], yerr=yerr_fit, fmt="o", ms=4, color="C0", ecolor="C0", capsize=2.5)
            r_line = np.linspace(shell["r_fit"].min(), shell["r_fit"].max(), 200)
            model = shell["intercept"] + shell["slope"] * r_line
            ax.plot(r_line, model, "-", color="C3", lw=2, label=fr"$m={shell['slope']:.2f}$")
        ax.axhline(KURTOSIS_FLOOR, color="0.4", ls="--", lw=1.0, alpha=0.8)
        ax.set_xlabel(r"Projected radius, $r_{\rm proj}$")
        ax.set_title(fr"$\Delta R={shell['width']:.3f}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Kurtosis, $K$")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    args = parser.parse_args()

    shell_dir = args.shell_dir
    rng = np.random.default_rng(RNG_SEED)
    shells = [load_shell(path, rng) for path in sorted(shell_dir.glob("Synthetic_x_projected_shell_rin*_order1_pdfkk.npz"))]
    if not shells:
        raise FileNotFoundError(f"No shell NPZ files found in {shell_dir}")

    records = []
    for shell in sorted(shells, key=lambda item: item["width"]):
        fit = fit_radius_linear_above_floor(shell)
        record = {**shell, **fit}
        records.append(record)
        print(
            f"width={record['width']:.4f} n_proj_pixels={record['n_proj_pixels']:6d} "
            f"n_above_floor={record['n_points']:3d} slope={record['slope']:.3f} +/- {record['slope_err']:.3f}"
        )

    usable = [record for record in records if record["n_points"] >= MIN_FIT_POINTS and np.isfinite(record["slope"])]
    if not usable:
        raise RuntimeError(f"No shells had at least {MIN_FIT_POINTS} bins above K>={KURTOSIS_FLOOR:g}")

    slope_out = shell_dir / "kurtosis_radius_linear_slope_vs_shell_thickness.png"
    examples_out = shell_dir / "kurtosis_radius_linear_fit_examples.png"
    csv_out = shell_dir / "kurtosis_radius_linear_slopes.csv"
    plot_slope_vs_thickness(usable, slope_out)
    plot_example_fits(usable, examples_out)

    with csv_out.open("w", encoding="utf-8") as fobj:
        fobj.write("width,r_inner,r_outer,slope,slope_err,intercept,intercept_err,n_points,n_projected_pixels\n")
        for record in records:
            fobj.write(
                f"{record['width']:.8f},{record['r_inner']:.8f},{record['r_outer']:.8f},"
                f"{record['slope']:.8f},{record['slope_err']:.8f},"
                f"{record['intercept']:.8f},{record['intercept_err']:.8f},"
                f"{record['n_points']},{record['n_proj_pixels']}\n"
            )

    print(f"Wrote {slope_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {examples_out.relative_to(REPO_ROOT)}")
    print(f"Wrote {csv_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
