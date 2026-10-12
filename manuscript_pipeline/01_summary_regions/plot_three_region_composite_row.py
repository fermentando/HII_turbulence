"""Create a composite row for selected regions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

from plot_three_line_composite_all_regions import dark_additive_rgb, stretched_channel


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "selected_region_false_color_row.png"
INTERPOLATED_OUTPUT_PATH = OUTPUT_DIR / "selected_region_false_color_row_interpolated_balanced.png"
REGIONS = [
    REPO_ROOT / "LVM_Gum38a.fits",
    REPO_ROOT / "LVM_RCW60.fits",
    REPO_ROOT / "LVM_Rosette_nebula.fits",
    REPO_ROOT / "LVM_Sh2-311.fits",
]


def tighten_sky_axes(ax, ra_values, dec_values, pad_fraction=0.01):
    """Set tight RA/Dec limits while preserving local sky aspect."""
    ra_values = np.asarray(ra_values, dtype=float)
    dec_values = np.asarray(dec_values, dtype=float)
    valid = np.isfinite(ra_values) & np.isfinite(dec_values)
    if not np.any(valid):
        return

    ra_min, ra_max = np.nanmin(ra_values[valid]), np.nanmax(ra_values[valid])
    dec_min, dec_max = np.nanmin(dec_values[valid]), np.nanmax(dec_values[valid])
    ra_pad = max((ra_max - ra_min) * pad_fraction, 1e-6)
    dec_pad = max((dec_max - dec_min) * pad_fraction, 1e-6)

    ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
    ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
    dec_mid = 0.5 * (dec_min + dec_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(dec_mid)), adjustable="box")


def load_region_rgb(fits_path):
    """Load one region and return visible coordinates plus RGB colors."""
    with fits.open(fits_path) as hdul:
        table = hdul[1].data
        ra = np.asarray(table["ra"], dtype=float)
        dec = np.asarray(table["dec"], dtype=float)
        sii, good_sii = stretched_channel(table["flux_sii"], table["flux_sii_err"])
        nii, good_nii = stretched_channel(table["flux_nii"], table["flux_nii_err"])
        siii, good_siii = stretched_channel(table["flux_siii"], table["flux_siii_err"])

    visible = np.isfinite(ra) & np.isfinite(dec) & (good_sii | good_nii | good_siii)
    return ra[visible], dec[visible], dark_additive_rgb(sii[visible], nii[visible], siii[visible])


def rasterize_rgb_image(ra, dec, rgb, n_pix=900, smooth_sigma=1.7, min_weight=0.02):
    """Rasterize irregular RGB spaxels to a smoothed image grid."""
    ra_min, ra_max = float(np.nanmin(ra)), float(np.nanmax(ra))
    dec_min, dec_max = float(np.nanmin(dec)), float(np.nanmax(dec))
    ra_span = max(ra_max - ra_min, 1e-12)
    dec_span = max(dec_max - dec_min, 1e-12)

    n_ra = n_pix
    n_dec = max(120, int(round(n_pix * dec_span / ra_span)))
    ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)

    image = np.zeros((n_dec, n_ra, 4), dtype=float)
    counts, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    smooth_counts = gaussian_filter(counts, smooth_sigma, mode="constant", cval=0.0)
    footprint = smooth_counts > min_weight
    footprint = binary_fill_holes(binary_closing(footprint, structure=np.ones((5, 5))))

    for channel in range(3):
        weighted_sum, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges], weights=rgb[:, channel])
        smooth_sum = gaussian_filter(weighted_sum, smooth_sigma, mode="constant", cval=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(
                smooth_sum,
                smooth_counts,
                out=np.zeros_like(smooth_sum),
                where=smooth_counts > 1e-8,
            )
        image[..., channel] = np.clip(values, 0, 1)

    image[..., 3] = footprint.astype(float)
    extent = [ra_max, ra_min, dec_min, dec_max]
    return image, extent


def main():
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(15.0, 4.1), constrained_layout=True)
    fig.patch.set_facecolor("black")

    for ax, fits_path in zip(axes, REGIONS):
        ra, dec, rgb = load_region_rgb(fits_path)
        ax.set_facecolor("black")
        image, extent = rasterize_rgb_image(ra, dec, rgb)
        ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
        tighten_sky_axes(ax, ra, dec)
        ax.set_axis_off()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(INTERPOLATED_OUTPUT_PATH, dpi=360, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {INTERPOLATED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
