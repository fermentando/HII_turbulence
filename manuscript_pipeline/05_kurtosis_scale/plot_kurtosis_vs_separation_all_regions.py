#!/usr/bin/env python3
"""Plot kurtosis (kk) as a function of separation for all regions.

Uses NPZ files generated in NPZ_kurtosis and assigns a unique marker shape
per region. Colors distinguish emission lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

matplotlib_backend = "Agg"
plt.switch_backend(matplotlib_backend)


def parse_name(stem: str) -> tuple[str, str]:
    # Expected pattern: <region>_<line>_order1_pdfkk
    if not stem.endswith("_order1_pdfkk"):
        raise ValueError("Unexpected filename pattern")
    core = stem[: -len("_order1_pdfkk")]

    line = None
    for candidate in ("NII", "SII", "SIII"):
        suffix = "_" + candidate
        if core.endswith(suffix):
            line = candidate
            region = core[: -len(suffix)]
            return region, line

    raise ValueError("Could not parse line tracer from filename")


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

    region_data: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
        r: [] for r in regions
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    for f in files:
        try:
            region, line = parse_name(f.stem)
            data = np.load(f)
            kk = np.asarray(data["kk"], dtype=float)
            log_centers = np.asarray(data["log_centers"], dtype=float)
            r = 10 ** log_centers

            valid = np.isfinite(r) & np.isfinite(kk) & (r > 0)
            if np.count_nonzero(valid) < 2:
                print(f"SKIP {f.name}: not enough finite points")
                continue

            r = r[valid]
            kk = kk[valid]

            order = np.argsort(r)
            r = r[order]
            kk = kk[order]

            ax.scatter(
                r,
                kk,
                marker=region_marker[region],
                s=18,
                color=line_color.get(line, "black"),
                alpha=0.85,
            )
            region_data[region].append((line, r, kk))

        except Exception as exc:
            print(f"SKIP {f.name}: {exc}")

    ax.set_xscale("log")
    ax.set_xlabel("Separation")
    ax.set_ylabel("Kurtosis (kk)")
    ax.set_ylim(0, 10)
    ax.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
    ax.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
    ax.set_title("Kurtosis vs Separation for All Regions")
    ax.grid(True, which="both", alpha=0.25)

    # Build compact legends: marker=region and color=line.
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

    out_png = out_dir / "kurtosis_vs_separation_all_regions.png"
    out_pdf = out_dir / "kurtosis_vs_separation_all_regions.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

    # Also save one scatter plot per region.
    per_region_dir = out_dir / "per_region"
    per_region_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        entries = region_data.get(region, [])
        if not entries:
            continue

        fig_r, ax_r = plt.subplots(figsize=(9, 6))
        for line, r, kk in entries:
            ax_r.scatter(
                r,
                kk,
                marker=region_marker[region],
                s=22,
                color=line_color.get(line, "black"),
                alpha=0.9,
                label=line,
            )

        ax_r.set_xscale("log")
        ax_r.set_xlabel("Separation")
        ax_r.set_ylabel("Kurtosis (kk)")
        ax_r.set_ylim(0, 10)
        ax_r.axhline(6, color="0.35", ls="--", lw=1.0, alpha=0.8)
        ax_r.axhline(3, color="0.35", ls=":", lw=1.0, alpha=0.8)
        ax_r.set_title(f"Kurtosis vs Separation: {region}")
        ax_r.grid(True, which="both", alpha=0.25)

        # Keep legend entries unique by line tracer.
        handles, labels = ax_r.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_r.legend(unique.values(), unique.keys(), title="Line", fontsize=9)

        fig_r.tight_layout()
        safe_region = region.replace("/", "_").replace(" ", "_")
        out_region_png = per_region_dir / f"kurtosis_vs_separation_{safe_region}.png"
        out_region_pdf = per_region_dir / f"kurtosis_vs_separation_{safe_region}.pdf"
        fig_r.savefig(out_region_png, dpi=220)
        fig_r.savefig(out_region_pdf)
        plt.close(fig_r)

        print(f"Saved: {out_region_png}")
        print(f"Saved: {out_region_pdf}")


if __name__ == "__main__":
    main()
