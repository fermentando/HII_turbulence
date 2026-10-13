#!/usr/bin/env python3
"""Plot kurtosis (kk) as a function of separation for all regions.

Uses NPZ files generated in NPZ_kurtosis. Colors distinguish regions and
linestyles distinguish emission lines. Spiky curves are filtered out.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

matplotlib_backend = "Agg"
plt.switch_backend(matplotlib_backend)

Y_MAX = 10.0
NPZ_DIR = Path("NPZ_kurtosis")
LINE_ORDER = ("NII", "SII", "SIII")
LINE_STYLES = {
    "NII": "-",
    "SII": "--",
    "SIII": ":",
}
MIN_SMOOTH_POINTS = 8
MAX_FRACTION_ABOVE_YMAX = 0.10
MAX_P95_KURTOSIS = 15.0
MAX_VISIBLE_ADJACENT_JUMP = 5.0
BOOTSTRAP_DRAWS = 200
MAX_BOOTSTRAP_COUNT = 50_000
RNG_SEED = 2718


def parse_name(stem: str) -> tuple[str, str]:
    # Expected pattern: <region>_<line>_order1_pdfkk
    if not stem.endswith("_order1_pdfkk"):
        raise ValueError("Unexpected filename pattern")
    core = stem[: -len("_order1_pdfkk")]

    line = None
    for candidate in LINE_ORDER:
        suffix = "_" + candidate
        if core.endswith(suffix):
            line = candidate
            region = core[: -len(suffix)]
            return region, line

    raise ValueError("Could not parse line tracer from filename")


def load_adopted_radii() -> dict[str, float]:
    """R_HII [pc] per region, as stored by kurtosis_plot.py (get_hii_radius_pc, LVM catalogue)."""
    radii = {}
    for path in sorted(NPZ_DIR.glob("*_order1_pdfkk.npz")):
        with np.load(path) as data:
            if "r_hii_pc" in data:
                radius = float(data["r_hii_pc"])
                if np.isfinite(radius) and radius > 0:
                    radii[parse_name(path.stem)[0]] = radius
    return radii


def raw_kurtosis_from_probabilities(probability: np.ndarray, dv: np.ndarray) -> float:
    p_sum = np.sum(probability)
    if p_sum <= 0:
        return np.nan
    p = probability / p_sum
    m2 = np.sum(p * dv**2)
    if m2 <= 0:
        return np.nan
    m4 = np.sum(p * dv**4)
    return float(m4 / m2**2)


def bootstrap_raw_kurtosis_from_pdf(
    pdf: np.ndarray,
    dv_centers: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = BOOTSTRAP_DRAWS,
) -> tuple[np.ndarray, np.ndarray]:
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
            draws[j] = raw_kurtosis_from_probabilities(sampled.astype(float), dv)

        finite_draws = draws[np.isfinite(draws)]
        if finite_draws.size >= 10:
            lo[i], hi[i] = np.nanpercentile(finite_draws, [16, 84])

    return lo, hi


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
    line_width: float = 1.4,
) -> None:
    band = np.isfinite(x) & np.isfinite(kk_lo) & np.isfinite(kk_hi) & (kk_hi >= kk_lo)
    if np.count_nonzero(band) >= 2:
        ax.fill_between(x[band], kk_lo[band], kk_hi[band], color=color, alpha=0.24, linewidth=0)
    ax.plot(
        x,
        kk,
        linestyle=linestyle,
        marker="o",
        markersize=marker_size,
        linewidth=line_width,
        color=color,
        alpha=0.9,
        label=label,
    )


def smooth_curve_diagnostics(kk: np.ndarray) -> tuple[bool, str]:
    """Return whether a positive-kurtosis curve is smooth enough to show."""
    if kk.size < MIN_SMOOTH_POINTS:
        return False, f"only {kk.size} positive points"

    frac_above = float(np.mean(kk > Y_MAX))
    p95 = float(np.nanpercentile(kk, 95))
    visible = kk[kk <= Y_MAX]

    if visible.size >= 2:
        max_visible_jump = float(np.nanmax(np.abs(np.diff(visible))))
    else:
        max_visible_jump = np.inf

    if frac_above > MAX_FRACTION_ABOVE_YMAX:
        return False, f"{frac_above:.0%} of points exceed y={Y_MAX:g}"
    if p95 > MAX_P95_KURTOSIS:
        return False, f"95th percentile kk={p95:.2f} exceeds {MAX_P95_KURTOSIS:g}"
    if max_visible_jump > MAX_VISIBLE_ADJACENT_JUMP:
        return False, f"visible adjacent jump={max_visible_jump:.2f} exceeds {MAX_VISIBLE_ADJACENT_JUMP:g}"

    return True, (
        f"frac_above={frac_above:.0%}, p95={p95:.2f}, "
        f"max_visible_jump={max_visible_jump:.2f}"
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
    rng = np.random.default_rng(RNG_SEED)

    cmap = plt.colormaps["tab20"]
    region_color = {r: cmap(i / max(len(regions) - 1, 1)) for i, r in enumerate(regions)}

    region_data: dict[str, list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        r: [] for r in regions
    }

    fig = plt.figure(figsize=(6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.25], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.set_axis_off()

    for f in files:
        try:
            region, line = parse_name(f.stem)
            data = np.load(f)
            kk = np.asarray(data["kk"], dtype=float)
            log_centers = np.asarray(data["log_centers"], dtype=float)
            pdf = np.asarray(data["pdf"], dtype=float)
            counts = np.asarray(data["cnt"], dtype=float) if "cnt" in data else np.full_like(kk, np.nan)
            if "dv_centers" in data:
                dv_centers = np.asarray(data["dv_centers"], dtype=float)
            else:
                dv_bins = np.asarray(data["dv_bins"], dtype=float)
                dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
            kk_lo, kk_hi = bootstrap_raw_kurtosis_from_pdf(pdf, dv_centers, counts, rng)
            r = 10 ** log_centers

            valid = np.isfinite(r) & np.isfinite(kk) & (r > 0) & (kk > 0)
            if np.count_nonzero(valid) < 2:
                print(f"SKIP {f.name}: not enough finite points")
                continue

            r = r[valid]
            kk = kk[valid]
            kk_lo = kk_lo[valid]
            kk_hi = kk_hi[valid]

            order = np.argsort(r)
            r = r[order]
            kk = kk[order]
            kk_lo = kk_lo[order]
            kk_hi = kk_hi[order]

            is_smooth, smooth_reason = smooth_curve_diagnostics(kk)
            if not is_smooth:
                print(f"SKIP {region} {line}: spiky curve ({smooth_reason})")
                continue
            print(f"KEEP {region} {line}: smooth curve ({smooth_reason})")

            r_norm = r / radii[region]
            plot_curve_with_band(
                ax,
                r_norm,
                kk,
                kk_lo,
                kk_hi,
                linestyle=LINE_STYLES.get(line, "-"),
                color=region_color[region],
            )
            region_data[region].append((line, r_norm, kk, kk_lo, kk_hi))

        except Exception as exc:
            print(f"SKIP {f.name}: {exc}")

    ax.set_xscale("log")
    ax.set_xlabel(r"Normalized separation, $r / R_{\rm HII}$")
    ax.set_ylabel("$K (S_4 / S_2^2)$")
    ax.set_ylim(0, Y_MAX)
    ax.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax.grid(True, which="both", alpha=0.25)

    plotted_regions = [r for r in regions if region_data.get(r)]
    if not plotted_regions:
        print("No smooth curves passed the filter")
        plt.close(fig)
        return
    plotted_lines = sorted({line for entries in region_data.values() for line, _, _, _, _ in entries})

    # Build compact legends: color=region and linestyle=line.
    region_handles = [
        plt.Line2D(
            [0],
            [0],
            color=region_color[r],
            linestyle="-",
            linewidth=2,
            label=r.replace("/", "_").replace(" ", "_")
        )
        for r in plotted_regions
    ]

    leg1 = ax_legend.legend(
        handles=region_handles,
        title="Region (color)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.00),
        borderaxespad=0.0,
        frameon=True,
    )
    ax_legend.add_artist(leg1)

    line_handles = [
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES[line], lw=2, label=line)
        for line in plotted_lines
    ]
    ax_legend.legend(
        handles=line_handles,
        title="Line (style)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.45),
        borderaxespad=0.0,
        frameon=True,
    )

    fig.subplots_adjust(left=0.16, right=0.965, bottom=0.22, top=0.95, wspace=0.03)

    out_png = out_dir / "kurtosis_vs_normalized_separation_all_regions.png"
    out_pdf = out_dir / "kurtosis_vs_normalized_separation_all_regions.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # Also save one line plot per region.
    per_region_dir = out_dir / "per_region"
    per_region_dir.mkdir(parents=True, exist_ok=True)

    for region in plotted_regions:
        entries = region_data.get(region, [])
        if not entries:
            continue

        fig_r, ax_r = plt.subplots(figsize=(9, 6))
        for line, r, kk, kk_lo, kk_hi in entries:
            plot_curve_with_band(
                ax_r,
                r,
                kk,
                kk_lo,
                kk_hi,
                linestyle=LINE_STYLES.get(line, "-"),
                color=region_color[region],
                label=line,
                marker_size=3.5,
                line_width=1.6,
            )

        ax_r.set_xscale("log")
        ax_r.set_xlabel(r"Normalized separation, $r / R_{\rm HII}$")
        ax_r.set_ylabel("Kurtosis (kk)")
        ax_r.set_ylim(0, Y_MAX)
        ax_r.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
        ax_r.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
        ax_r.set_title(f"Smooth Kurtosis vs Normalized Separation: {region}")
        ax_r.grid(True, which="both", alpha=0.25)

        # Keep legend entries unique.
        handles, labels = ax_r.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_r.legend(unique.values(), unique.keys(), title="Line", fontsize=9)

        fig_r.tight_layout()
        safe_region = region.replace("/", "_").replace(" ", "_")
        out_region_png = per_region_dir / f"kurtosis_vs_normalized_separation_{safe_region}.png"
        out_region_pdf = per_region_dir / f"kurtosis_vs_normalized_separation_{safe_region}.pdf"
        fig_r.savefig(out_region_png, dpi=220)
        fig_r.savefig(out_region_pdf)
        plt.close(fig_r)

        print(f"Saved: {out_region_png}")
        print(f"Saved: {out_region_pdf}")


if __name__ == "__main__":
    main()
