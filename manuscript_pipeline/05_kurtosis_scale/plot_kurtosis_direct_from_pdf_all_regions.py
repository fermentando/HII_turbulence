#!/usr/bin/env python3
"""Plot direct kurtosis vs separation for all regions.

This script computes kurtosis directly from the velocity-difference PDF at each
separation bin, using central moments of the distribution in dv.

Input: NPZ files in NPZ_kurtosis with keys: pdf, dv_bins (or dv_centers),
       log_centers.
Output: combined all-region figure + one figure per region in a separate folder.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

RADIUS_CSV = Path("manuscript_notes/adopted_hii_region_radii.csv")
LINE_ORDER = ("NII", "SII", "SIII")
LINE_STYLES = {
    "NII": "-",
    "SII": "--",
    "SIII": ":",
}
MAX_MEDIAN_KURTOSIS = 10.0
MAX_PEAK_KURTOSIS = 30.0
MAX_LARGE_JUMP_FRACTION = 0.30
LARGE_JUMP_KURTOSIS = 1.5
LEADING_STABLE_WINDOW = 6
LEADING_MAX_KURTOSIS = 10.0
LEADING_MAX_JUMP = 3.0
BOOTSTRAP_DRAWS = 200
MAX_BOOTSTRAP_COUNT = 50_000
RNG_SEED = 1729


def parse_name(stem: str) -> tuple[str, str]:
    # Expected pattern: <region>_<line>_order1_pdfkk
    if not stem.endswith("_order1_pdfkk"):
        raise ValueError("Unexpected filename pattern")
    core = stem[: -len("_order1_pdfkk")]

    for candidate in LINE_ORDER:
        suffix = "_" + candidate
        if core.endswith(suffix):
            return core[: -len(suffix)], candidate

    raise ValueError("Could not parse line tracer from filename")


def load_adopted_radii() -> dict[str, float]:
    if not RADIUS_CSV.exists():
        raise FileNotFoundError(RADIUS_CSV)

    radii = {}
    with RADIUS_CSV.open("r", encoding="utf-8", newline="") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            try:
                radius = float(row["radius_pc"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(radius) and radius > 0:
                radii[row["object"]] = radius
    return radii


def direct_kurtosis_from_pdf(pdf: np.ndarray, dv_centers: np.ndarray) -> np.ndarray:
    """Compute kurtosis from PDF rows using central moments.

    For each separation bin r:
      mu  = sum p(dv|r) * dv
      m2  = sum p(dv|r) * (dv - mu)^2
      m4  = sum p(dv|r) * (dv - mu)^4
      kk  = m4 / m2^2
    """
    kk = np.full(pdf.shape[0], np.nan, dtype=float)

    for i in range(pdf.shape[0]):
        p = np.asarray(pdf[i], dtype=float)
        finite = np.isfinite(p) & np.isfinite(dv_centers)
        if np.count_nonzero(finite) < 2:
            continue

        p = p[finite]
        dv = dv_centers[finite]

        s = p.sum()
        if s <= 0:
            continue
        p = p / s

        mu = np.sum(p * dv)
        d = dv - mu
        m2 = np.sum(p * d**2)
        if m2 <= 0:
            continue
        m4 = np.sum(p * d**4)
        kk[i] = m4 / (m2**2)

    return kk


def row_kurtosis_from_probabilities(probability: np.ndarray, dv: np.ndarray) -> float:
    s = np.sum(probability)
    if s <= 0:
        return np.nan
    p = probability / s
    mu = np.sum(p * dv)
    delta = dv - mu
    m2 = np.sum(p * delta**2)
    if m2 <= 0:
        return np.nan
    m4 = np.sum(p * delta**4)
    return float(m4 / m2**2)


def bootstrap_kurtosis_from_pdf(
    pdf: np.ndarray,
    dv_centers: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = BOOTSTRAP_DRAWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate 16/84 percentile bands for PDF-derived kurtosis."""
    lo = np.full(pdf.shape[0], np.nan, dtype=float)
    hi = np.full(pdf.shape[0], np.nan, dtype=float)

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
            draws[j] = row_kurtosis_from_probabilities(sampled.astype(float), dv)

        finite_draws = draws[np.isfinite(draws)]
        if finite_draws.size >= 10:
            lo[i], hi[i] = np.nanpercentile(finite_draws, [16, 84])

    return lo, hi


def load_direct_kurtosis(npz_path: Path, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    pdf = np.asarray(data["pdf"], dtype=float)
    log_centers = np.asarray(data["log_centers"], dtype=float)
    counts = np.asarray(data["cnt"], dtype=float) if "cnt" in data else np.full(pdf.shape[0], np.nan)

    if "dv_centers" in data:
        dv_centers = np.asarray(data["dv_centers"], dtype=float)
    elif "dv_bins" in data:
        dv_bins = np.asarray(data["dv_bins"], dtype=float)
        dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    else:
        raise KeyError("NPZ missing both dv_centers and dv_bins")

    kk_direct = direct_kurtosis_from_pdf(pdf, dv_centers)
    kk_lo, kk_hi = bootstrap_kurtosis_from_pdf(pdf, dv_centers, counts, rng)
    r = 10**log_centers

    valid = np.isfinite(r) & np.isfinite(kk_direct) & (r > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("not enough finite points")

    r = r[valid]
    kk_direct = kk_direct[valid]
    kk_lo = kk_lo[valid]
    kk_hi = kk_hi[valid]

    order = np.argsort(r)
    return r[order], kk_direct[order], kk_lo[order], kk_hi[order]


def passes_quality_filter(kk_direct: np.ndarray) -> tuple[bool, str]:
    """Reject curves dominated by extreme spikes or bin-to-bin oscillations."""
    y = np.asarray(kk_direct, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size < 8:
        return False, f"too few finite points ({y.size})"

    median = float(np.nanmedian(y))
    peak = float(np.nanmax(y))
    jumps = np.abs(np.diff(y))
    large_jump_fraction = float(np.count_nonzero(jumps > LARGE_JUMP_KURTOSIS) / jumps.size)

    if median > MAX_MEDIAN_KURTOSIS:
        return False, f"median K={median:.2f} > {MAX_MEDIAN_KURTOSIS:g}"
    if peak > MAX_PEAK_KURTOSIS:
        return False, f"peak K={peak:.2f} > {MAX_PEAK_KURTOSIS:g}"
    if large_jump_fraction > MAX_LARGE_JUMP_FRACTION:
        return False, (
            f"{large_jump_fraction:.0%} adjacent jumps exceed "
            f"{LARGE_JUMP_KURTOSIS:g}"
        )
    return True, "ok"


def trim_leading_noisy_bins(
    r: np.ndarray,
    kk_direct: np.ndarray,
    kk_lo: np.ndarray,
    kk_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop noisy small-scale leading bins before the first stable window."""
    y = np.asarray(kk_direct, dtype=float)
    if y.size <= LEADING_STABLE_WINDOW:
        return r, kk_direct, kk_lo, kk_hi, 0

    max_start = y.size - LEADING_STABLE_WINDOW
    for start in range(max_start + 1):
        window = y[start : start + LEADING_STABLE_WINDOW]
        finite = np.isfinite(window)
        if not np.all(finite):
            continue
        jumps = np.abs(np.diff(window))
        if np.nanmax(window) <= LEADING_MAX_KURTOSIS and np.nanmax(jumps) <= LEADING_MAX_JUMP:
            return r[start:], kk_direct[start:], kk_lo[start:], kk_hi[start:], start

    return r, kk_direct, kk_lo, kk_hi, 0


def plot_curve_with_band(
    ax,
    x: np.ndarray,
    kk: np.ndarray,
    kk_lo: np.ndarray,
    kk_hi: np.ndarray,
    color,
    linestyle: str,
    label: str | None = None,
    marker_size: float = 3.0,
    line_width: float = 1.25,
    alpha: float = 0.88,
) -> None:
    band = np.isfinite(x) & np.isfinite(kk_lo) & np.isfinite(kk_hi) & (kk_hi >= kk_lo)
    if np.count_nonzero(band) >= 2:
        ax.fill_between(
            x[band],
            kk_lo[band],
            kk_hi[band],
            color=color,
            alpha=0.24,
            linewidth=0,
        )
    ax.plot(
        x,
        kk,
        linestyle=linestyle,
        marker="o",
        markersize=marker_size,
        linewidth=line_width,
        color=color,
        alpha=alpha,
        label=label,
    )


def main() -> None:
    src_dir = Path("NPZ_kurtosis")
    out_dir = Path("Figures") / "manuscript" / "05_kurtosis_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("custom_plot.mplstyle")
    except OSError:
        pass

    files = sorted(src_dir.glob("*_order1_pdfkk.npz"))
    if not files:
        print(f"No input files found in {src_dir}")
        return

    regions = sorted({parse_name(f.stem)[0] for f in files})
    radii = load_adopted_radii()
    missing_radii = sorted(set(regions) - set(radii))
    if missing_radii:
        raise KeyError(f"Missing adopted radii for: {', '.join(missing_radii)}")

    cmap = plt.colormaps["tab20"]
    region_color = {r: cmap(i / max(len(regions) - 1, 1)) for i, r in enumerate(regions)}

    rng = np.random.default_rng(RNG_SEED)
    region_data: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {r: [] for r in regions}
    kept_curves: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    skipped_noisy: list[str] = []
    trimmed_curves: list[str] = []

    fig, ax = plt.subplots(figsize=(11.5, 7))

    for f in files:
        try:
            region, line = parse_name(f.stem)
            r, kk_direct, kk_lo, kk_hi = load_direct_kurtosis(f, rng)
            keep, reason = passes_quality_filter(kk_direct)
            if not keep:
                skipped_noisy.append(f"{region} {line}: {reason}")
                continue
            r, kk_direct, kk_lo, kk_hi, n_trimmed = trim_leading_noisy_bins(r, kk_direct, kk_lo, kk_hi)
            if n_trimmed:
                trimmed_curves.append(f"{region} {line}: removed first {n_trimmed} bins")
            r_norm = r / radii[region]

            plot_curve_with_band(
                ax,
                r_norm,
                kk_direct,
                kk_lo,
                kk_hi,
                linestyle=LINE_STYLES.get(line, "-"),
                color=region_color[region],
            )
            region_data[region].append((line, r, r_norm, kk_direct, kk_lo, kk_hi))
            kept_curves.append((region, line, r, r_norm, kk_direct, kk_lo, kk_hi))

        except Exception as exc:
            print(f"SKIP {f.name}: {exc}")

    ax.set_xscale("log")
    ax.set_xlabel(r"Normalized separation, $r / R_{\rm HII}$")
    ax.set_ylabel("Direct kurtosis from PDF")
    ax.set_ylim(0, 10)
    ax.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax.set_title(r"Direct Kurtosis vs Separation Normalized by HII-region Radius")
    ax.grid(True, which="both", alpha=0.25)

    region_handles = [
        plt.Line2D(
            [0],
            [0],
            color=region_color[r],
            linestyle="-",
            linewidth=2.2,
            label=r,
        )
        for r in regions
    ]

    line_handles = [
        plt.Line2D([0], [0], color="0.2", lw=2, linestyle=LINE_STYLES[l], label=l)
        for l in LINE_ORDER
    ]

    leg1 = ax.legend(
        handles=region_handles,
        title="Region (color)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=line_handles,
        title="Line (style)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out_png = out_dir / "kurtosis_direct_vs_normalized_separation_all_regions.png"
    out_pdf = out_dir / "kurtosis_direct_vs_normalized_separation_all_regions.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Normalized separations by adopted radii from: {RADIUS_CSV}")
    if skipped_noisy:
        print("Skipped noisy/extreme curves:")
        for item in skipped_noisy:
            print(f"  - {item}")
    if trimmed_curves:
        print("Trimmed noisy small-separation heads:")
        for item in trimmed_curves:
            print(f"  - {item}")

    fig_pc, ax_pc = plt.subplots(figsize=(11.5, 7))
    for region, line, r_pc, _, kk_direct, kk_lo, kk_hi in kept_curves:
        plot_curve_with_band(
            ax_pc,
            r_pc,
            kk_direct,
            kk_lo,
            kk_hi,
            linestyle=LINE_STYLES.get(line, "-"),
            color=region_color[region],
        )

    ax_pc.set_xscale("log")
    ax_pc.set_xlabel("Physical separation, r [pc]")
    ax_pc.set_ylabel("Direct kurtosis from PDF")
    ax_pc.set_ylim(0, 10)
    ax_pc.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
    ax_pc.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax_pc.set_title("Direct Kurtosis vs Physical Separation")
    ax_pc.grid(True, which="both", alpha=0.25)

    leg1_pc = ax_pc.legend(
        handles=region_handles,
        title="Region (color)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax_pc.add_artist(leg1_pc)
    ax_pc.legend(
        handles=line_handles,
        title="Line (style)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        borderaxespad=0.0,
    )
    fig_pc.tight_layout(rect=[0, 0, 0.78, 1])

    out_pc_png = out_dir / "kurtosis_direct_vs_physical_separation_all_regions_cleaned.png"
    out_pc_pdf = out_dir / "kurtosis_direct_vs_physical_separation_all_regions_cleaned.pdf"
    fig_pc.savefig(out_pc_png, dpi=220)
    fig_pc.savefig(out_pc_pdf)
    plt.close(fig_pc)

    print(f"Saved: {out_pc_png}")
    print(f"Saved: {out_pc_pdf}")

    per_region_dir = out_dir / "per_region"
    per_region_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        entries = region_data.get(region, [])
        if not entries:
            continue

        fig_r, ax_r = plt.subplots(figsize=(9, 6))
        for line, _, r, kk_direct, kk_lo, kk_hi in entries:
            plot_curve_with_band(
                ax_r,
                r,
                kk_direct,
                kk_lo,
                kk_hi,
                linestyle=LINE_STYLES.get(line, "-"),
                color=region_color[region],
                label=line,
                marker_size=3.5,
                line_width=1.4,
                alpha=0.9,
            )

        ax_r.set_xscale("log")
        ax_r.set_xlabel(r"Normalized separation, $r / R_{\rm HII}$")
        ax_r.set_ylabel("Direct kurtosis from PDF")
        ax_r.set_ylim(0, 10)
        ax_r.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
        ax_r.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
        ax_r.set_title(
            rf"Direct Kurtosis vs Normalized Separation: {region} "
            rf"($R_{{\rm HII}}={radii[region]:.2f}$ pc)"
        )
        ax_r.grid(True, which="both", alpha=0.25)

        handles, labels = ax_r.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_r.legend(unique.values(), unique.keys(), title="Line", fontsize=9)

        fig_r.tight_layout()
        safe_region = region.replace("/", "_").replace(" ", "_")
        out_region_png = per_region_dir / f"kurtosis_direct_vs_normalized_separation_{safe_region}.png"
        out_region_pdf = per_region_dir / f"kurtosis_direct_vs_normalized_separation_{safe_region}.pdf"
        fig_r.savefig(out_region_png, dpi=220)
        fig_r.savefig(out_region_pdf)
        plt.close(fig_r)

        print(f"Saved: {out_region_png}")
        print(f"Saved: {out_region_pdf}")


if __name__ == "__main__":
    main()
