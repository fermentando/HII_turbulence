#!/usr/bin/env python3
"""Exponential-fit K(log r) above a kurtosis floor, and plot scale vs shell thickness.

Every shell shares the same outer radius (R_OUTER_FIXED) and only r_inner
varies with thickness. The set of projected (y, z) pixels that see any
shell gas along the line of sight is, in the continuum limit, simply the
disk of radius r_outer - independent of r_inner/thickness. Fixing r_outer
therefore keeps the number of projected v_x datapoints feeding the K(r)
statistics the same across all shell widths, so the only thing that varies
between shells is how much line-of-sight averaging each one applies.

For each shell, restrict the K(r) curve to bins with kurtosis >=
KURTOSIS_FLOOR, fit K(r) = KURTOSIS_FLOOR + A exp(-log10(r)/tau), and take
tau as the characteristic excess-kurtosis decay scale in log-radius.

Per-bin K values get bootstrap error bars (resampling each bin's P(dv) at
its effective pair count) so the example fit plots show the real
bin-to-bin uncertainty, not just point estimates.
"""

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
SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)

KURTOSIS_FLOOR = 3.0
MIN_FIT_POINTS = 3
R_OUTER_FIXED = 1.0
WIDTH_MIN = 0.02
WIDTH_MAX = 0.5
N_WIDTHS = 12

BOOTSTRAP_DRAWS = 200
MAX_BOOTSTRAP_COUNT = 50_000
RNG_SEED = 2718
FNAME_RE = re.compile(r"Synthetic_x_projected_shell_rin([\dp]+)_rout([\dp]+)_order1_pdfkk\.npz")


def width_grid() -> np.ndarray:
    return np.logspace(np.log10(WIDTH_MIN), np.log10(WIDTH_MAX), N_WIDTHS)


def shell_path_for_width(width: float) -> Path:
    r_inner = R_OUTER_FIXED - width
    r_outer = R_OUTER_FIXED
    tag = f"rin{r_inner:.4f}_rout{r_outer:.4f}".replace(".", "p")
    return SHELL_DIR / f"Synthetic_x_projected_shell_{tag}_order1_pdfkk.npz"


def shell_paths_in_dir(shell_dir: Path) -> list[Path]:
    paths = [
        path for path in shell_dir.glob("Synthetic_x_projected_shell_rin*_rout*_order1_pdfkk.npz")
        if FNAME_RE.match(path.name)
    ]

    def width_from_path(path: Path) -> float:
        match = FNAME_RE.match(path.name)
        r_inner = float(match.group(1).replace("p", "."))
        r_outer = float(match.group(2).replace("p", "."))
        return r_outer - r_inner

    return sorted(paths, key=width_from_path)


def centered_kurtosis_from_probabilities(probability: np.ndarray, dv: np.ndarray) -> float:
    p_sum = np.sum(probability)
    if p_sum <= 0:
        return np.nan
    p = probability / p_sum
    mean = np.sum(p * dv)
    centered = dv - mean
    m2 = np.sum(p * centered**2)
    if m2 <= 0:
        return np.nan
    m4 = np.sum(p * centered**4)
    return float(m4 / m2**2)


def bootstrap_kurtosis_from_pdf(
    pdf: np.ndarray, dv_centers: np.ndarray, counts: np.ndarray, rng: np.random.Generator,
    n_boot: int = BOOTSTRAP_DRAWS,
) -> tuple[np.ndarray, np.ndarray]:
    """16th/84th percentile band for the centered kurtosis of each r bin."""
    lo = np.full(pdf.shape[0], np.nan)
    hi = np.full(pdf.shape[0], np.nan)
    for i, row in enumerate(pdf):
        finite = np.isfinite(row) & np.isfinite(dv_centers) & (row > 0)
        if np.count_nonzero(finite) < 3:
            continue
        p = row[finite].astype(float)
        p_sum = np.sum(p)
        if p_sum <= 0:
            continue
        p /= p_sum
        dv = dv_centers[finite]

        count = counts[i] if i < counts.size else np.nan
        if not np.isfinite(count) or count <= 1:
            continue
        n_eff = int(np.clip(round(count), 20, MAX_BOOTSTRAP_COUNT))

        draws = np.empty(n_boot, dtype=float)
        for j in range(n_boot):
            sampled = rng.multinomial(n_eff, p)
            draws[j] = centered_kurtosis_from_probabilities(sampled.astype(float), dv)

        finite_draws = draws[np.isfinite(draws)]
        if finite_draws.size >= 10:
            lo[i], hi[i] = np.nanpercentile(finite_draws, [16, 84])
    return lo, hi


def load_shell(path: Path, rng: np.random.Generator) -> dict:
    data = np.load(path)
    log_centers = np.asarray(data["log_centers"], dtype=float)
    kk = np.asarray(data["kk"], dtype=float)
    pdf = np.asarray(data["pdf"], dtype=float)
    cnt = np.asarray(data["cnt"], dtype=float)
    dv_centers = np.asarray(data["dv_centers"], dtype=float)
    r = 10**log_centers

    valid = np.isfinite(r) & np.isfinite(kk) & (r > 0)
    order = np.argsort(r[valid])
    idx = np.flatnonzero(valid)[order]

    kk_lo_all, kk_hi_all = bootstrap_kurtosis_from_pdf(pdf, dv_centers, cnt, rng)

    r = r[idx]
    kk = kk[idx]
    kk_lo = kk_lo_all[idx]
    kk_hi = kk_hi_all[idx]

    shell_mask = np.asarray(data["shell_mask"])
    n_proj_pixels = int(np.asarray(data["y"]).size)
    return {
        "path": path,
        "r": r,
        "log_r": np.log10(r),
        "kk": kk,
        "kk_lo": kk_lo,
        "kk_hi": kk_hi,
        "r_inner": float(data["r_inner"]),
        "r_outer": float(data["r_outer"]),
        "width": float(data["r_outer"]) - float(data["r_inner"]),
        "fill_fraction": float(np.mean(shell_mask)),
        "n_proj_pixels": n_proj_pixels,
    }


def fit_logradius_exponential_above_floor(shell: dict, floor: float = KURTOSIS_FLOOR) -> dict:
    mask = shell["kk"] > floor
    n = int(np.count_nonzero(mask))
    result = {
        "n_points": n,
        "amplitude": np.nan,
        "amplitude_err": np.nan,
        "tau": np.nan,
        "tau_err": np.nan,
        "decay_rate": np.nan,
        "decay_rate_err": np.nan,
    }
    if n < MIN_FIT_POINTS:
        return result

    r = shell["r"][mask]
    log_r = shell["log_r"][mask]
    kk = shell["kk"][mask]
    excess = kk - floor
    finite = np.isfinite(r) & np.isfinite(log_r) & np.isfinite(excess) & (r > 0) & (excess > 0)
    if np.count_nonzero(finite) < MIN_FIT_POINTS:
        result["n_points"] = int(np.count_nonzero(finite))
        return result

    r = r[finite]
    log_r = log_r[finite]
    kk = kk[finite]
    excess = excess[finite]
    log_excess = np.log(excess)
    coeffs, cov = np.polyfit(log_r, log_excess, 1, cov=True)
    decay_rate = float(coeffs[0])
    log_amplitude = float(coeffs[1])
    if decay_rate >= 0:
        return result

    amplitude = float(np.exp(log_amplitude))
    scale = float(-1.0 / decay_rate)
    decay_rate_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) and cov[0, 0] >= 0 else np.nan
    log_amplitude_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) and cov[1, 1] >= 0 else np.nan

    result["n_points"] = int(r.size)
    result["amplitude"] = amplitude
    result["amplitude_err"] = amplitude * log_amplitude_err if np.isfinite(log_amplitude_err) else np.nan
    result["tau"] = scale
    result["tau_err"] = decay_rate_err / decay_rate**2 if np.isfinite(decay_rate_err) else np.nan
    result["decay_rate"] = decay_rate
    result["decay_rate_err"] = decay_rate_err
    fit_mask = np.zeros_like(shell["kk"], dtype=bool)
    fit_mask[np.flatnonzero(mask)[finite]] = True
    result["fit_mask"] = fit_mask
    result["r_fit"] = r
    result["log_r_fit"] = log_r
    result["kk_fit"] = kk
    return result


def plot_example_fits(shells_results: list[tuple[dict, dict]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(shells_results), figsize=(4.6 * len(shells_results), 4.2), sharey=True)
    if len(shells_results) == 1:
        axes = [axes]

    for ax, (shell, fit) in zip(axes, shells_results):
        yerr_all = np.vstack([
            np.clip(shell["kk"] - shell["kk_lo"], 0, None),
            np.clip(shell["kk_hi"] - shell["kk"], 0, None),
        ])
        ax.errorbar(
            shell["r"], shell["kk"], yerr=yerr_all, fmt="o", ms=4, color="0.6", ecolor="0.75",
            elinewidth=1.0, capsize=2, alpha=0.8, label="all bins",
        )
        if fit["n_points"] >= MIN_FIT_POINTS and np.isfinite(fit["tau"]):
            mask = fit["fit_mask"]
            yerr_fit = np.vstack([
                np.clip(shell["kk"][mask] - shell["kk_lo"][mask], 0, None),
                np.clip(shell["kk_hi"][mask] - shell["kk"][mask], 0, None),
            ])
            ax.errorbar(
                fit["r_fit"],
                fit["kk_fit"],
                yerr=yerr_fit,
                fmt="o",
                ms=4,
                color="C0",
                ecolor="C0", elinewidth=1.2, capsize=2.5, label=f"$K\\geq{KURTOSIS_FLOOR:g}$ (fit)",
            )
            r_line = np.geomspace(fit["r_fit"].min(), fit["r_fit"].max(), 200)
            log_r_line = np.log10(r_line)
            model = KURTOSIS_FLOOR + fit["amplitude"] * np.exp(-log_r_line / fit["tau"])
            ax.plot(
                r_line,
                model,
                "-",
                color="C3",
                lw=2,
                label=fr"$\tau={fit['tau']:.3f}$",
            )
        ax.axhline(KURTOSIS_FLOOR, color="0.4", ls="--", lw=1.0, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Projected radius, $r_{\rm proj}$")
        ax.set_title(f"width={shell['width']:.3f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Kurtosis, $K$")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_all_fit_examples(shells_results: list[tuple[dict, dict]], out: Path) -> None:
    n = len(shells_results)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.25 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (shell, fit) in zip(axes, shells_results):
        yerr_all = np.vstack([
            np.clip(shell["kk"] - shell["kk_lo"], 0, None),
            np.clip(shell["kk_hi"] - shell["kk"], 0, None),
        ])
        ax.errorbar(
            shell["r"], shell["kk"], yerr=yerr_all, fmt="o", ms=3, color="0.6", ecolor="0.78",
            elinewidth=0.8, capsize=1.5, alpha=0.8,
        )
        if fit["n_points"] >= MIN_FIT_POINTS and np.isfinite(fit["tau"]):
            mask = fit["fit_mask"]
            yerr_fit = np.vstack([
                np.clip(shell["kk"][mask] - shell["kk_lo"][mask], 0, None),
                np.clip(shell["kk_hi"][mask] - shell["kk"][mask], 0, None),
            ])
            ax.errorbar(
                fit["r_fit"], fit["kk_fit"], yerr=yerr_fit, fmt="o", ms=3.2, color="C0",
                ecolor="C0", elinewidth=1.0, capsize=2.0,
            )
            r_line = np.geomspace(fit["r_fit"].min(), fit["r_fit"].max(), 200)
            model = KURTOSIS_FLOOR + fit["amplitude"] * np.exp(-np.log10(r_line) / fit["tau"])
            ax.plot(r_line, model, "-", color="C3", lw=1.8, label=fr"$\tau={fit['tau']:.3f}$")
            ax.legend(fontsize=7, loc="best")
        ax.axhline(KURTOSIS_FLOOR, color="0.4", ls="--", lw=0.9, alpha=0.8)
        ax.set_xscale("log")
        ax.set_title(fr"$\Delta R={shell['width']:.4f}$", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$r_{\rm proj}$")
    for ax in axes[::ncols]:
        ax.set_ylabel("Kurtosis, $K$")

    fig.suptitle(fr"Exponential fits to excess kurtosis for all shell widths ($r_{{out}}={R_OUTER_FIXED:g}$ fixed)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_scale_vs_thickness(records: list[dict], out: Path) -> None:
    widths = np.array([r["width"] / r["r_outer"] for r in records])
    scales = np.array([r["tau"] for r in records])
    errs = np.array([r["tau_err"] for r in records])
    n_points = np.array([r["n_points"] for r in records])

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.errorbar(widths, scales, yerr=errs, fmt="o-", ms=5, lw=1.4, capsize=3, color="C0")
    for w, scale, n in zip(widths, scales, n_points):
        ax.annotate(str(n), (w, scale), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center", color="0.4")

    ax.set_xscale("log")
    ax.set_xlabel(r"$(r_{out}-r_{in})/r_{out}$")
    ax.set_ylabel(r"Log-radius exponential scale, $\tau$ in $K=3+A e^{-\log_{10}r/\tau}$")
    ax.set_title(f"Excess-kurtosis log-radius scale vs. shell thickness ($r_{{out}}={R_OUTER_FIXED:g}$ fixed)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    global SHELL_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=SHELL_DIR)
    args = parser.parse_args()
    SHELL_DIR = args.shell_dir

    rng = np.random.default_rng(RNG_SEED)
    shells = []
    missing = []
    paths = shell_paths_in_dir(SHELL_DIR)
    if not paths:
        widths = width_grid()
        for w in widths:
            path = shell_path_for_width(w)
            if not path.exists():
                missing.append((w, path))
                continue
            paths.append(path)

    for path in paths:
        shells.append(load_shell(path, rng))

    if missing:
        print("Missing shell NPZ files (run projected_shell_kurtosis.py for these widths first):")
        for w, path in missing:
            print(f"  width={w:.4f} -> {path.name}")

    if not shells:
        print("No shell data available.")
        sys.exit(1)

    pixel_counts = [s["n_proj_pixels"] for s in shells]
    print(f"2D projected pixel counts (should be ~constant across widths now): {pixel_counts}")
    print(f"  min={min(pixel_counts)}  max={max(pixel_counts)}  spread={max(pixel_counts)-min(pixel_counts)}")

    records = []
    for shell in shells:
        fit = fit_logradius_exponential_above_floor(shell)
        records.append({**shell, **fit})
        status = "OK" if fit["n_points"] >= MIN_FIT_POINTS else "SKIP (too few points)"
        print(
            f"width={shell['width']:.4f}  n_proj_pixels={shell['n_proj_pixels']:6d}  "
            f"n_above_floor={fit['n_points']:3d}  "
            f"tau={fit['tau']:.3f} +/- {fit['tau_err']:.3f}  "
            f"A={fit['amplitude']:.3f}  [{status}]"
        )

    usable = [r for r in records if r["n_points"] >= MIN_FIT_POINTS]
    skipped = len(records) - len(usable)
    if skipped:
        print(f"Skipped {skipped}/{len(records)} widths with fewer than {MIN_FIT_POINTS} points above K>={KURTOSIS_FLOOR:g}")
    if not usable:
        print("No shell had enough points above the kurtosis floor to fit a line.")
        sys.exit(1)

    scale_out = SHELL_DIR / "kurtosis_logradius_exponential_scale_vs_shell_thickness.png"
    plot_scale_vs_thickness(usable, scale_out)
    print(f"Wrote {scale_out.relative_to(REPO_ROOT)}")

    example_idx = [0, len(usable) // 2, len(usable) - 1]
    example_idx = sorted(set(min(max(i, 0), len(usable) - 1) for i in example_idx))
    examples = [(usable[i], usable[i]) for i in example_idx]
    examples_out = SHELL_DIR / "kurtosis_logradius_exponential_fit_examples.png"
    plot_example_fits(examples, examples_out)
    print(f"Wrote {examples_out.relative_to(REPO_ROOT)}")

    all_examples = [(r, r) for r in usable]
    all_examples_out = SHELL_DIR / "kurtosis_logradius_exponential_fit_examples_all_widths.png"
    plot_all_fit_examples(all_examples, all_examples_out)
    print(f"Wrote {all_examples_out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
