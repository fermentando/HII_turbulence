"""Create an all-region false-color atlas from SII, NII, and SIII fluxes."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "three_line_false_color_dark_additive_all_regions.png"
RASTER_OUTPUT_PATH = OUTPUT_DIR / "three_line_false_color_dark_additive_all_regions_rasterized.png"
PALETTE = {
    "siii": np.array([1.25, 0.22, 0.06]),  # bright red/orange: higher ionization
    "nii": np.array([0.10, 0.85, 0.25]),  # green: intermediate ionization
    "sii": np.array([0.15, 0.35, 1.35]),  # bright blue: lower ionization
}


def stretched_channel(flux, flux_err, snr_min=5.0, pmin=2.0, pmax=99.0, softening=0.08):
    """Return a robustly normalized flux channel and its valid/SNR mask."""
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0)
    good &= (flux / flux_err) > snr_min

    channel = np.zeros_like(flux, dtype=float)
    if not np.any(good):
        return channel, good

    log_flux = np.log10(flux[good])
    lo, hi = np.nanpercentile(log_flux, [pmin, pmax])
    scaled = np.clip((log_flux - lo) / max(hi - lo, 1e-12), 0, 1)
    channel[good] = np.arcsinh(scaled / softening) / np.arcsinh(1.0 / softening)
    return channel, good


def saturation_boost(rgb, amount=1.4):
    """Increase color separation while keeping total brightness recognizable."""
    rgb = np.clip(rgb, 0, 1)
    gray = np.mean(rgb, axis=1, keepdims=True)
    return np.clip(gray + amount * (rgb - gray), 0, 1)


def dark_additive_rgb(
    sii,
    nii,
    siii,
    gamma=1.08,
    brightness=1.45,
    saturation=1.90,
    shoulder=0.70,
    black_point=0.10,
    contrast=1.45,
    white_limit=0.86,
):
    """Mix channels additively so low surface-brightness regions remain dark."""
    rgb = (
        sii[:, None] * PALETTE["sii"]
        + nii[:, None] * PALETTE["nii"]
        + siii[:, None] * PALETTE["siii"]
    )
    rgb = rgb / 1.7
    rgb = rgb / (rgb + shoulder)
    rgb = brightness * rgb**gamma
    rgb = np.clip((rgb - black_point) / max(1.0 - black_point, 1e-12), 0, 1)
    rgb = np.clip(0.5 + contrast * (rgb - 0.5), 0, 1)
    rgb = saturation_boost(rgb, amount=saturation)

    peak = np.max(rgb, axis=1)
    bright = peak > white_limit
    rgb[bright] *= (white_limit / peak[bright])[:, None]
    return np.clip(rgb, 0, 1)


def tighten_sky_axes(ax, ra_values, dec_values, pad_fraction=0.015):
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

    valid_position = np.isfinite(ra) & np.isfinite(dec)
    visible = valid_position & (good_sii | good_nii | good_siii)
    rgb = dark_additive_rgb(sii[visible], nii[visible], siii[visible])
    region_name = fits_path.stem.replace("LVM_", "").replace("_nebula", "")
    return region_name, ra[visible], dec[visible], rgb


def rasterize_rgb_image(ra, dec, rgb, n_pix=520, smooth_sigma=1.8, min_weight=0.03):
    """Rasterize irregular RGB spaxels to a smoothed image grid."""
    ra_min, ra_max = float(np.nanmin(ra)), float(np.nanmax(ra))
    dec_min, dec_max = float(np.nanmin(dec)), float(np.nanmax(dec))
    ra_span = max(ra_max - ra_min, 1e-12)
    dec_span = max(dec_max - dec_min, 1e-12)

    n_ra = n_pix
    n_dec = max(100, int(round(n_pix * dec_span / ra_span)))
    ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)

    image = np.zeros((n_dec, n_ra, 4), dtype=float)
    counts, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    smooth_counts = gaussian_filter(counts, smooth_sigma, mode="constant", cval=0.0)

    for channel in range(3):
        weighted_sum, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges], weights=rgb[:, channel])
        smooth_sum = gaussian_filter(weighted_sum, smooth_sigma, mode="constant", cval=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(
                smooth_sum,
                smooth_counts,
                out=np.zeros_like(smooth_sum),
                where=smooth_counts > min_weight,
            )
        image[..., channel] = np.clip(values, 0, 1)

    image[..., 3] = (smooth_counts > min_weight).astype(float)
    extent = [ra_max, ra_min, dec_min, dec_max]
    return image, extent


def main():
    fits_paths = sorted(REPO_ROOT.glob("LVM_*.fits"))
    if not fits_paths:
        raise FileNotFoundError("No LVM_*.fits files found.")

    regions = [load_region_rgb(path) for path in fits_paths]
    regions = [region for region in regions if region[1].size > 0]
    if not regions:
        raise RuntimeError("No valid SII/NII/SIII spaxels available.")

    n_cols = 4
    n_rows = math.ceil(len(regions) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.9 * n_cols, 3.7 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.tick_params(labelsize=7, length=2.5)

    for idx, (name, ra, dec, rgb) in enumerate(regions):
        ax = axes.ravel()[idx]
        image, extent = rasterize_rgb_image(ra, dec, rgb)
        ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
        ax.set_title(name, fontsize=10, fontweight="bold", pad=3)
        ax.grid(True, color="white", alpha=0.10, linewidth=0.5)
        tighten_sky_axes(ax, ra, dec)

        row, col = divmod(idx, n_cols)
        if row == n_rows - 1:
            ax.set_xlabel("RA (deg)", fontsize=8)
        if col == 0:
            ax.set_ylabel("Dec (deg)", fontsize=8)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=np.clip(PALETTE["siii"], 0, 1), markersize=6, label="SIII"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=np.clip(PALETTE["nii"], 0, 1), markersize=6, label="NII"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=np.clip(PALETTE["sii"], 0, 1), markersize=6, label="SII"),
    ]
    for idx in range(len(regions), n_rows * n_cols):
        ax = axes.ravel()[idx]
        ax.set_axis_off()
        if idx == len(regions):
            legend = ax.legend(
                handles=handles,
                loc="center",
                frameon=False,
                fontsize=11,
                title="RGB channels",
                title_fontsize=12,
            )
            for text in legend.get_texts():
                text.set_color("black")
            legend.get_title().set_color("black")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RASTER_OUTPUT_PATH, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {RASTER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
