#!/usr/bin/env python3
"""Kurtosis-vs-radius and kurtosis-slope-vs-radius plots for synthetic shells.

Reads the NPZ files produced by projected_shell_kurtosis.py (one per shell
width), plots the kurtosis curve K(r) for a single reference shell, checks
whether its local slope dK/d(log r) is smooth, and then compares that slope
curve across all available shell widths.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)

MIN_SMOOTH_POINTS = 5
MAX_NOISE_RATIO = 0.3

FNAME_RE = re.compile(r"Synthetic_x_projected_shell_rin([\dp]+)_rout([\dp]+)_order1_pdfkk\.npz")


def trim_sparse_edge_bins(log_r: np.ndarray, kk: np.ndarray, gap_factor: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """Drop trailing bins separated by an anomalously large gap in log r.

    A handful of skipped (empty) bins near the outer edge of a thin shell's
    projected footprint show up as one bin with a much larger-than-typical
    spacing to its neighbor; that bin also tends to carry much weaker
    statistics, so it is trimmed rather than treated as a real feature.
    """
    while log_r.size > 3:
        gaps = np.diff(log_r)
        median_gap = np.median(gaps)
        if gaps[-1] <= gap_factor * median_gap:
            break
        log_r = log_r[:-1]
        kk = kk[:-1]
    return log_r, kk


def load_shell(path: Path) -> dict:
    data = np.load(path)
    log_centers = np.asarray(data["log_centers"], dtype=float)
    kk = np.asarray(data["kk"], dtype=float)
    r = 10**log_centers

    valid = np.isfinite(r) & np.isfinite(kk) & (r > 0) & (kk > 0)
    order = np.argsort(r[valid])
    r = r[valid][order]
    kk = kk[valid][order]
    log_r = np.log10(r)
    log_r, kk = trim_sparse_edge_bins(log_r, kk)
    r = 10**log_r

    r_inner = float(data["r_inner"])
    r_outer = float(data["r_outer"])
    return {
        "path": path,
        "r": r,
        "log_r": log_r,
        "kk": kk,
        "r_inner": r_inner,
        "r_outer": r_outer,
        "width": r_outer - r_inner,
    }


def discover_shells(shell_dir: Path) -> list[dict]:
    shells = []
    for f in sorted(shell_dir.glob("Synthetic_x_projected_shell_rin*_order1_pdfkk.npz")):
        if not FNAME_RE.match(f.name):
            continue
        shells.append(load_shell(f))
    return sorted(shells, key=lambda s: s["width"])


def local_slope(log_r: np.ndarray, kk: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """dK / d(log10 r), via a Savitzky-Golay smoothed derivative.

    Raw point-to-point finite differences are dominated by per-bin shot
    noise (each bin has a different number of shell pixels), so the slope
    is estimated from a local polynomial fit instead of a bare np.gradient.
    """
    n = log_r.size
    if n < 2:
        return np.full_like(kk, np.nan)
    if n < 4:
        return np.gradient(kk, log_r)

    win = min(window, n if n % 2 == 1 else n - 1)
    win = max(win, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win <= polyorder:
        return np.gradient(kk, log_r)

    delta = float(np.median(np.diff(log_r)))
    return savgol_filter(kk, window_length=win, polyorder=polyorder, deriv=1, delta=delta, mode="interp")


def smoothness_diagnostics(
    log_r: np.ndarray, kk: np.ndarray, smoothed_slope: np.ndarray,
    min_points: int = MIN_SMOOTH_POINTS, max_noise_ratio: float = MAX_NOISE_RATIO,
) -> tuple[bool, str]:
    """Is the smoothed slope estimate signal-dominated rather than noise-dominated?

    Real turnovers in K(log r) make a derivative curve's own point-to-point
    jumps large even when it is clean, so jump-size thresholds on the slope
    itself can't tell noise from genuine curvature. Instead compare the
    smoothed (Savitzky-Golay) slope against the raw finite-difference slope:
    if they diverge by only a small fraction of the smoothed curve's own
    dynamic range, the bin-to-bin shot noise is small relative to the
    signal and the slope curve is trustworthy.
    """
    if log_r.size < min_points:
        return False, f"only {log_r.size} points"

    raw_slope = np.gradient(kk, log_r)
    resid = raw_slope - smoothed_slope
    finite = np.isfinite(resid) & np.isfinite(smoothed_slope)
    if np.count_nonzero(finite) < min_points:
        return False, "too few finite slope points"

    rmse = float(np.sqrt(np.mean(resid[finite] ** 2)))
    signal_range = float(np.ptp(smoothed_slope[finite]))
    if signal_range <= 1e-9:
        return False, "slope curve is flat (no dynamic range)"

    ratio = rmse / signal_range
    is_smooth = ratio <= max_noise_ratio
    reason = f"noise/signal={ratio:.2f} (rmse={rmse:.3f}, range={signal_range:.3f})"
    return is_smooth, reason


def plot_kurtosis_vs_radius(shell: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(shell["r"], shell["kk"], "o-", ms=4, lw=1.4, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel(r"Projected radius, $r_{\rm proj}$")
    ax.set_ylabel(r"Kurtosis, $K$")
    width = shell["width"]
    ax.set_title(fr"Synthetic shell, width$=${width:.2f} ($r_{{in}}={shell['r_inner']:.3f}$, $r_{{out}}={shell['r_outer']:.3f}$)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_slope_vs_width(shells: list[dict], out: Path) -> list[dict]:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    cmap = plt.colormaps["viridis"]
    widths = [s["width"] for s in shells]
    wmin, wmax = min(widths), max(widths)

    kept = []
    for shell in shells:
        slope = local_slope(shell["log_r"], shell["kk"])
        is_smooth, reason = smoothness_diagnostics(shell["log_r"], shell["kk"], slope)
        tag = "KEEP" if is_smooth else "SKIP"
        print(f"{tag} width={shell['width']:.2f}: slope {reason}")
        if not is_smooth:
            continue

        color = cmap(0.0 if wmax == wmin else (shell["width"] - wmin) / (wmax - wmin))
        ax.plot(
            shell["r"],
            slope,
            "o-",
            ms=3.5,
            lw=1.4,
            color=color,
            label=f"width={shell['width']:.2f}",
        )
        kept.append(shell)

    ax.set_xscale("log")
    ax.axhline(0, color="0.4", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel(r"Projected radius, $r_{\rm proj}$")
    ax.set_ylabel(r"$dK / d(\log_{10} r)$")
    ax.set_title("Kurtosis slope vs. projected radius, by shell width")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, title="Shell width", loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return kept


def main() -> None:
    shells = discover_shells(SHELL_DIR)
    if not shells:
        print(f"No shell NPZ files found in {SHELL_DIR}")
        sys.exit(1)

    # Reference shell: the thinnest one (most recently requested).
    reference = min(shells, key=lambda s: s["width"])
    rin_tag = f"{reference['r_inner']:.3f}".replace(".", "p")
    rout_tag = f"{reference['r_outer']:.3f}".replace(".", "p")
    kurt_out = SHELL_DIR / f"kurtosis_vs_radius_rin{rin_tag}_rout{rout_tag}.png"
    plot_kurtosis_vs_radius(reference, kurt_out)
    print(f"Wrote {kurt_out.relative_to(REPO_ROOT)}")

    slope = local_slope(reference["log_r"], reference["kk"])
    is_smooth, reason = smoothness_diagnostics(reference["log_r"], reference["kk"], slope)
    status = "smooth" if is_smooth else "NOT smooth"
    print(f"Reference shell (width={reference['width']:.2f}) slope is {status}: {reason}")

    slope_out = SHELL_DIR / "kurtosis_slope_vs_log_r_by_shell_width.png"
    kept = plot_slope_vs_width(shells, slope_out)
    print(f"Wrote {slope_out.relative_to(REPO_ROOT)}")
    print(f"Kept {len(kept)}/{len(shells)} shell widths with a smooth slope curve")


if __name__ == "__main__":
    main()
