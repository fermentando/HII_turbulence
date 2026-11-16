#!/usr/bin/env python3
"""All-region false-color atlas using the validated row-script pipeline.

Same pipeline as plot_three_region_composite_row.py (the one you approved:
palette-based dark_additive_rgb color mix, low-SNR fiber filling with a
max-distance cap, cosmic-ray/color-outlier rejection, angular-scale
smoothing, no post-hoc blur or median filter), just looped over every
LVM_*.fits region instead of 4 selected ones.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_three_region_composite_row import (  # noqa: E402
    dark_additive_rgb,
    load_region_rgb,
    rasterize_rgb_image,
    tighten_sky_axes,
)

REPO_ROOT = HERE.parents[1]
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "three_line_false_color_row_pipeline_all_regions.png"


def main():
    fits_paths = sorted(REPO_ROOT.glob("LVM_*.fits"))
    if not fits_paths:
        raise FileNotFoundError("No LVM_*.fits files found.")

    n_cols = 4
    n_rows = math.ceil(len(fits_paths) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.4 * n_cols, 4.1 * n_rows), constrained_layout=True, squeeze=False,
    )
    fig.patch.set_facecolor("black")
    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.set_axis_off()

    for idx, fits_path in enumerate(fits_paths):
        region_name = fits_path.stem.replace("LVM_", "").replace("_nebula", "")
        print(f"Processing {fits_path.name}...")
        ra, dec, rgb = load_region_rgb(fits_path)
        ax = axes.ravel()[idx]
        image, extent = rasterize_rgb_image(ra, dec, rgb)
        ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
        ax.set_title(region_name, fontsize=13, fontweight="bold", pad=4, color="white")
        tighten_sky_axes(ax, ra, dec)

    for idx in range(len(fits_paths), n_rows * n_cols):
        axes.ravel()[idx].set_axis_off()

    fig.text(
        0.995, 0.002, "SIII=R, NII=G, SII=B", color="white", fontsize=12,
        ha="right", va="bottom", alpha=0.85,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=260, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
