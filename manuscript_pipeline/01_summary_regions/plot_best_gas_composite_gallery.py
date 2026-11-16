#!/usr/bin/env python3
"""All-region gallery using the refined best-gas-composite pipeline.

Applies the same pipeline validated on Rosette in plot_best_gas_composite.py
(per-fiber outlier rejection, proper HSV luminance/chrominance construction,
CLAHE local-contrast enhancement) to every LVM_*.fits region, laid out as a
grid -- the "best effort" successor to plot_three_line_composite_all_regions.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_best_gas_composite import (  # noqa: E402
    enhance_local_contrast,
    luminance_chrominance_rgb,
    rasterize_rgb_image,
    reject_outlier_fibers,
    stretched_channel,
    tighten_sky_axes,
)

REPO_ROOT = HERE.parents[1]
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "best_gas_composite_all_regions.png"


def process_region(fits_path: Path):
    with fits.open(fits_path) as hdul:
        table = hdul[1].data
        ra = np.asarray(table["ra"], dtype=float)
        dec = np.asarray(table["dec"], dtype=float)
        sii_raw = np.asarray(table["flux_sii"], dtype=float)
        nii_raw = np.asarray(table["flux_nii"], dtype=float)
        siii_raw = np.asarray(table["flux_siii"], dtype=float)
        sii_err = table["flux_sii_err"]
        nii_err = table["flux_nii_err"]
        siii_err = table["flux_siii_err"]

    valid = np.isfinite(ra) & np.isfinite(dec)
    sii_clean, _ = reject_outlier_fibers(ra, dec, np.where(valid, sii_raw, np.nan))
    nii_clean, _ = reject_outlier_fibers(ra, dec, np.where(valid, nii_raw, np.nan))
    siii_clean, _ = reject_outlier_fibers(ra, dec, np.where(valid, siii_raw, np.nan))

    sii, good_sii = stretched_channel(sii_clean, sii_err)
    nii, good_nii = stretched_channel(nii_clean, nii_err)
    siii, good_siii = stretched_channel(siii_clean, siii_err)

    visible = valid & (good_sii | good_nii | good_siii)
    rgb, _ = luminance_chrominance_rgb(sii[visible], nii[visible], siii[visible])

    image, extent = rasterize_rgb_image(ra[visible], dec[visible], rgb)
    image = enhance_local_contrast(image)

    region_name = fits_path.stem.replace("LVM_", "").replace("_nebula", "")
    return region_name, ra[visible], dec[visible], image, extent


def main():
    fits_paths = sorted(REPO_ROOT.glob("LVM_*.fits"))
    if not fits_paths:
        raise FileNotFoundError("No LVM_*.fits files found.")

    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass

    n_cols = 4
    n_rows = math.ceil(len(fits_paths) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 4.3 * n_rows), constrained_layout=True, squeeze=False,
    )
    fig.patch.set_facecolor("black")
    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.set_axis_off()

    for idx, fits_path in enumerate(fits_paths):
        print(f"Processing {fits_path.name}...")
        name, ra, dec, image, extent = process_region(fits_path)
        ax = axes.ravel()[idx]
        ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
        ax.set_title(name, fontsize=13, fontweight="bold", pad=4, color="white")
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
