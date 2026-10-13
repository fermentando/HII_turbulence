"""Kurtosis of the velocity increments vs separation / HII-region radius, for [NII], [SII]
and [SIII] together. K = 3 (Gaussian) is marked with a dashed line.

Reads NPZ/KURT_<region>_<LINE>.npz written by kurtosis_plot.py (run that first for each
region and line). The grid mode also writes every curve to
Figures/manuscript/04_flame_kurtosis_3d/kurtosis_vs_l_over_rhii_all_regions.csv
(region, line, l_pc, l_over_rhii, kurtosis, counts, r_hii_pc) and, with --per-region,
one single-region figure per region.

Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_kurtosis_vs_l_over_rhii.py                       # grid, all regions
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_kurtosis_vs_l_over_rhii.py --region Flame_nebula  # one region
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

LINES = ("NII", "SII", "SIII")
OUT_DIR = Path("Figures/manuscript/04_flame_kurtosis_3d")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--region", default=None, help="Single region (e.g. Flame_nebula); default: grid of all regions.")
parser.add_argument("--per-region", action="store_true", help="In grid mode, also write one figure per region.")
args = parser.parse_args()


def load(region, line):
    path = Path(f"NPZ/KURT_{region}_{line}.npz")
    if not path.exists():
        return None
    with np.load(path) as d:
        return {"x": d["l_pc"] / float(d["r_hii_pc"]), "k": d["kurtosis"], "r_hii": float(d["r_hii_pc"]),
                "l_pc": d["l_pc"], "counts": d["counts"],
                "k_err": d["kurtosis_err"] if "kurtosis_err" in d.files else np.full(len(d["kurtosis"]), np.nan)}


def draw_panel(ax, region, fontsize_title=18, show_legend=True):
    r_hii = None
    for line in LINES:
        curve = load(region, line)
        if curve is None:
            continue
        r_hii = curve["r_hii"]
        ax.fill_between(curve["x"], curve["k"] - curve["k_err"], curve["k"] + curve["k_err"],
                        color=LINE_COLORS[line], alpha=0.18, lw=0)
        ax.plot(curve["x"], curve["k"], "o-", ms=3.5, lw=1.8, color=LINE_COLORS[line], label=LINE_LABELS[line])
    ax.axhline(3.0, color="0.3", ls="--", lw=1.2, zorder=0)
    ax.text(0.02, 3.0, "Gaussian ($K=3$)", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=11, color="0.3")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in (2, 5) else ""))
    ax.grid(True, alpha=0.3)
    title = region.replace("_", " ") + (f" ($R_{{\\rm HII}} = {r_hii:.1f}$ pc)" if r_hii else "")
    ax.set_title(title, fontsize=fontsize_title)
    if show_legend:
        ax.legend(fontsize=11, loc="upper right")


plt.style.use("./custom_plot.mplstyle")
OUT_DIR.mkdir(parents=True, exist_ok=True)
xlabel, ylabel = r"$\ell / R_{\rm HII}$", r"$K = \langle \Delta v^4 \rangle / \langle \Delta v^2 \rangle^2$"

def single_region_figure(region):
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_panel(ax, region)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    out = OUT_DIR / f"{region.replace('_nebula', '')}_kurtosis_vs_l_over_rhii.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if args.region:
    region = args.region.replace("LVM_", "").replace(".fits", "")
    if all(load(region, line) is None for line in LINES):
        sys.exit(f"No NPZ/KURT_{region}_*.npz: run kurtosis_plot.py --region {region} first")
    single_region_figure(region)
    sys.exit(0)
else:
    regions = sorted({p.stem[len("KURT_"):].rsplit("_", 1)[0] for p in Path("NPZ").glob("KURT_*_*.npz")})
    if not regions:
        sys.exit("No NPZ/KURT_*.npz files: run kurtosis_plot.py first")
    n_cols = 3
    n_rows = int(np.ceil(len(regions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    for idx, region in enumerate(regions):
        ax = axes[divmod(idx, n_cols)]
        draw_panel(ax, region, fontsize_title=16, show_legend=(idx == 0))
        ax.set_xlabel(xlabel)
        if idx % n_cols == 0:
            ax.set_ylabel(r"$K$")
    for idx in range(len(regions), n_rows * n_cols):
        axes[divmod(idx, n_cols)].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18)
    out = OUT_DIR / "kurtosis_vs_l_over_rhii_all_regions.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Every curve as a table.
    import csv
    csv_out = OUT_DIR / "kurtosis_vs_l_over_rhii_all_regions.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.writer(fobj)
        writer.writerow(["region", "line", "l_pc", "l_over_rhii", "kurtosis", "kurtosis_err", "counts", "r_hii_pc"])
        for region in regions:
            for line in LINES:
                curve = load(region, line)
                if curve is None:
                    continue
                for lp, x, k, ke, n in zip(curve["l_pc"], curve["x"], curve["k"], curve["k_err"], curve["counts"]):
                    writer.writerow([region, line, f"{lp:.5g}", f"{x:.5g}", f"{k:.5g}", f"{ke:.4g}", int(n),
                                     f"{curve['r_hii']:.4g}"])
    print(f"Wrote {csv_out}")

    if args.per_region:
        for region in regions:
            single_region_figure(region)
