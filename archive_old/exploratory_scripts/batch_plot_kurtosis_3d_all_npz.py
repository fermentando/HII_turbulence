#!/usr/bin/env python3
"""Batch-generate 3D PDF/kurtosis plots for NPZ files.

This reproduces the plotting logic from the provided notebook cell and applies it
across all NPZ files that contain: pdf, dv_bins, log_centers, kk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator


def make_plot(npz_path: Path, out_dir: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)

    required = ("pdf", "dv_bins", "log_centers", "kk")
    if not all(k in data.files for k in required):
        missing = [k for k in required if k not in data.files]
        raise KeyError(f"Missing keys: {missing}")

    pdf = np.asarray(data["pdf"], dtype=float)
    dv_bins = np.asarray(data["dv_bins"], dtype=float)
    log_centers = np.asarray(data["log_centers"], dtype=float)
    kk = np.asarray(data["kk"], dtype=float)

    if pdf.ndim != 2:
        raise ValueError(f"pdf must be 2D; got shape {pdf.shape}")
    if dv_bins.ndim != 1 or log_centers.ndim != 1 or kk.ndim != 1:
        raise ValueError("dv_bins, log_centers, and kk must be 1D arrays")

    n_r, n_dv = pdf.shape
    if len(log_centers) != n_r or len(kk) != n_r or len(dv_bins) != n_dv + 1:
        raise ValueError(
            "Inconsistent shapes among pdf, dv_bins, log_centers, and kk"
        )

    dv_centers = 0.5 * (dv_bins[:-1] + dv_bins[1:])
    r = 10 ** log_centers

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.viridis
    norm = plt.Normalize(3, 6)

    x_min, x_max = -10, 10
    mask = (dv_centers >= x_min) & (dv_centers <= x_max)

    for i in range(len(r)):
        color = cmap(norm(kk[i]))
        ax.plot(
            dv_centers[mask],
            np.log10(np.full(np.sum(mask), r[i])),
            pdf[i][mask],
            color=color,
        )

    ax.set_xlabel(r"$(\Delta v)^p$")
    ax.set_ylabel("Separation r")
    ax.set_zlabel("PDF")
    ax.set_xlim(9, -9)

    ax.view_init(elev=20, azim=60)
    ax.dist = 10
    ax.set_zscale("log")

    max_pdf = float(np.nanmax(pdf)) if np.isfinite(np.nanmax(pdf)) else 0.0
    zmax = max_pdf * 1.5
    if zmax <= 1e-2:
        zmax = 1.0
    ax.set_zlim(1e-2, zmax)

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

    ax.tick_params(axis="both", which="major")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    cax = fig.add_axes([0.25, 0.92, 0.5, 0.02])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("Bin index", labelpad=5)

    out_png = out_dir / f"{npz_path.stem}_3d_kurtosis.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch 3D kurtosis plots from NPZ files")
    p.add_argument(
        "--source-dir",
        type=Path,
        default=Path("."),
        help="Directory to scan recursively for NPZ files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Figures") / "kurtosis_3d_all_npz",
        help="Directory where PNG plots are saved",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    base = args.source_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.style.use("kurtosis_plot.mplstyle")
    except OSError:
        pass

    files = sorted(base.rglob("*.npz"))
    files = [
        f
        for f in files
        if "kurtosis_3d_all_npz" not in f.parts and "__pycache__" not in f.parts
    ]
    if not files:
        print("No NPZ files found in workspace")
        return

    ok = 0
    skipped = 0

    for f in files:
        try:
            make_plot(f, out_dir)
            ok += 1
            print(f"OK   {f.name}")
        except Exception as exc:
            skipped += 1
            print(f"SKIP {f.name}: {exc}")

    print(f"\nDone. Plotted: {ok}, Skipped: {skipped}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
