#!/usr/bin/env python3
"""Create 3D PDF/kurtosis plots for all regions and emission lines.

This generalizes the Flame NII kurtosis figure to every available
``NPZ_kurtosis/*_<LINE>_order1_pdfkk.npz`` file and writes one subfolder per
region and line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "NPZ_kurtosis"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
NPZ_RE = re.compile(r"(?P<region>.+)_(?P<line>NII|SII|SIII)_order1_pdfkk\.npz$")


def parse_npz_name(path: Path) -> tuple[str, str]:
    match = NPZ_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected kurtosis filename: {path.name}")
    return match.group("region"), match.group("line")


def finite_kurtosis_norm(kk: np.ndarray, default_min: float = 3.0, default_max: float = 6.0):
    finite = kk[np.isfinite(kk) & (kk > 0)]
    if finite.size == 0:
        return plt.Normalize(default_min, default_max)

    vmin = min(default_min, float(np.nanpercentile(finite, 5)))
    vmax = max(default_max, float(np.nanpercentile(finite, 95)))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return plt.Normalize(default_min, default_max)
    return plt.Normalize(vmin, vmax)


def load_pdfkk(npz_path: Path):
    with np.load(npz_path) as data:
        required = {"pdf", "dv_bins", "log_centers", "kk"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"missing keys: {sorted(missing)}")

        pdf = np.asarray(data["pdf"], dtype=float)
        dv_bins = np.asarray(data["dv_bins"], dtype=float)
        log_centers = np.asarray(data["log_centers"], dtype=float)
        kk = np.asarray(data["kk"], dtype=float)

    if pdf.ndim != 2:
        raise ValueError(f"pdf must be 2D; got {pdf.shape}")
    if pdf.shape[0] != log_centers.size or pdf.shape[0] != kk.size:
        raise ValueError("pdf/log_centers/kk radial dimensions do not match")
    if pdf.shape[1] != dv_bins.size - 1:
        raise ValueError("pdf velocity dimension does not match dv_bins")

    return pdf, dv_bins, log_centers, kk


def make_kurtosis_plot(npz_path: Path, output_root: Path, x_min: float, x_max: float) -> Path:
    region, line = parse_npz_name(npz_path)
    pdf, dv_bins, log_centers, kk = load_pdfkk(npz_path)

    r = 10**log_centers
    dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    mask = (dv_centers >= x_min) & (dv_centers <= x_max)
    if not np.any(mask):
        raise ValueError(f"no dv bins inside requested range {x_min:g} to {x_max:g}")

    out_dir = output_root / region / line
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.03, 1.0], hspace=0.18)
    cax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], projection="3d")

    cmap = cm.viridis
    norm = finite_kurtosis_norm(kk)

    for i in range(len(r)):
        y = pdf[i][mask]
        good = np.isfinite(y) & (y > 0)
        if not np.any(good):
            continue
        ax.plot(
            dv_centers[mask][good],
            np.log10(np.full(np.sum(good), r[i])),
            y[good],
            color=cmap(norm(kk[i])),
            linewidth=1.0,
        )

    ax.set_xlabel(r"$(\Delta v)$ (km/s)", fontsize=14, labelpad=10)
    ax.set_ylabel(r"$\log r$ (pc)", fontsize=14, labelpad=10)
    ax.set_zlabel(r"$\mathcal{P}(\Delta v)$", fontsize=14, labelpad=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(np.log10(r[0]), np.log10(r[-1]))
    ax.view_init(elev=25, azim=45)
    try:
        ax.dist = 9
    except Exception:
        pass
    ax.set_zscale("log")

    yvals = pdf[:, mask]
    positive = yvals[np.isfinite(yvals) & (yvals > 0)]
    if positive.size:
        ax.set_zlim(max(float(np.nanpercentile(positive, 1)), 1e-4), min(float(np.nanmax(positive)), 1.0))
    else:
        ax.set_zlim(1e-4, 1.0)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=12)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("w")

    ax.set_title(f"{region} {line}", pad=18)

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(r"Kurtosis", labelpad=3, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    out_png = out_dir / f"{region}_{line}_order1_3d_kurtosis.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=-10.0)
    parser.add_argument("--x-max", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    files = sorted(args.source_dir.glob("*_order1_pdfkk.npz"))
    files = [path for path in files if NPZ_RE.match(path.name)]
    if not files:
        raise FileNotFoundError(f"No *_order1_pdfkk.npz files found in {args.source_dir}")

    ok = 0
    skipped = 0
    for path in files:
        try:
            out = make_kurtosis_plot(path, args.output_dir, args.x_min, args.x_max)
            ok += 1
            print(f"OK   {out.relative_to(REPO_ROOT)}")
        except Exception as exc:
            skipped += 1
            print(f"SKIP {path.name}: {exc}")

    print(f"\nDone. Plotted: {ok}, Skipped: {skipped}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
