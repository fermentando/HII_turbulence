#!/usr/bin/env python3
"""Plot direct kurtosis vs separation for all regions.

This script computes kurtosis directly from the velocity-difference PDF at each
separation bin, using central moments of the distribution in dv.

Input: NPZ files in NPZ_kurtosis with keys: pdf, dv_bins (or dv_centers),
       log_centers.
Output: combined all-region figure + one figure per region in a separate folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def parse_name(stem: str) -> tuple[str, str]:
    # Expected pattern: <region>_<line>_order1_pdfkk
    if not stem.endswith("_order1_pdfkk"):
        raise ValueError("Unexpected filename pattern")
    core = stem[: -len("_order1_pdfkk")]

    for candidate in ("NII", "SII", "SIII"):
        suffix = "_" + candidate
        if core.endswith(suffix):
            return core[: -len(suffix)], candidate

    raise ValueError("Could not parse line tracer from filename")


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


def load_direct_kurtosis(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    pdf = np.asarray(data["pdf"], dtype=float)
    log_centers = np.asarray(data["log_centers"], dtype=float)

    if "dv_centers" in data:
        dv_centers = np.asarray(data["dv_centers"], dtype=float)
    elif "dv_bins" in data:
        dv_bins = np.asarray(data["dv_bins"], dtype=float)
        dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    else:
        raise KeyError("NPZ missing both dv_centers and dv_bins")

    kk_direct = direct_kurtosis_from_pdf(pdf, dv_centers)
    r = 10**log_centers

    valid = np.isfinite(r) & np.isfinite(kk_direct) & (r > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("not enough finite points")

    r = r[valid]
    kk_direct = kk_direct[valid]

    order = np.argsort(r)
    return r[order], kk_direct[order]


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
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8", "p"]
    region_marker = {r: marker_cycle[i % len(marker_cycle)] for i, r in enumerate(regions)}

    line_color = {
        "NII": "tab:blue",
        "SII": "tab:red",
        "SIII": "tab:green",
    }

    region_data: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {r: [] for r in regions}

    fig, ax = plt.subplots(figsize=(11, 7))

    for f in files:
        try:
            region, line = parse_name(f.stem)
            r, kk_direct = load_direct_kurtosis(f)

            ax.scatter(
                r,
                kk_direct,
                marker=region_marker[region],
                s=18,
                color=line_color.get(line, "black"),
                alpha=0.85,
            )
            region_data[region].append((line, r, kk_direct))

        except Exception as exc:
            print(f"SKIP {f.name}: {exc}")

    ax.set_xscale("log")
    ax.set_xlabel("Separation")
    ax.set_ylabel("Direct kurtosis from PDF")
    ax.set_ylim(0, 10)
    ax.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax.set_title("Direct Kurtosis vs Separation for All Regions")
    ax.grid(True, which="both", alpha=0.25)

    region_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=region_marker[r],
            color="black",
            linestyle="None",
            markersize=6,
            label=r,
        )
        for r in regions
    ]

    line_handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=l)
        for l, c in line_color.items()
    ]

    leg1 = ax.legend(
        handles=region_handles,
        title="Region (marker)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=line_handles,
        title="Line (color)",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0, 0, 0.78, 1])

    out_png = out_dir / "kurtosis_direct_vs_separation_all_regions.png"
    out_pdf = out_dir / "kurtosis_direct_vs_separation_all_regions.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    per_region_dir = out_dir / "per_region"
    per_region_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        entries = region_data.get(region, [])
        if not entries:
            continue

        fig_r, ax_r = plt.subplots(figsize=(9, 6))
        for line, r, kk_direct in entries:
            ax_r.scatter(
                r,
                kk_direct,
                marker=region_marker[region],
                s=22,
                color=line_color.get(line, "black"),
                alpha=0.9,
                label=line,
            )

        ax_r.set_xscale("log")
        ax_r.set_xlabel("Separation")
        ax_r.set_ylabel("Direct kurtosis from PDF")
        ax_r.set_ylim(0, 10)
        ax_r.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
        ax_r.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
        ax_r.set_title(f"Direct Kurtosis vs Separation: {region}")
        ax_r.grid(True, which="both", alpha=0.25)

        handles, labels = ax_r.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_r.legend(unique.values(), unique.keys(), title="Line", fontsize=9)

        fig_r.tight_layout()
        safe_region = region.replace("/", "_").replace(" ", "_")
        out_region_png = per_region_dir / f"kurtosis_direct_vs_separation_{safe_region}.png"
        out_region_pdf = per_region_dir / f"kurtosis_direct_vs_separation_{safe_region}.pdf"
        fig_r.savefig(out_region_png, dpi=220)
        fig_r.savefig(out_region_pdf)
        plt.close(fig_r)

        print(f"Saved: {out_region_png}")
        print(f"Saved: {out_region_pdf}")


if __name__ == "__main__":
    main()
