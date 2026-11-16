"""Create an all-region false-color atlas from SII, NII, and SIII fluxes."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "three_line_false_color_dark_additive_all_regions.png"
RASTER_OUTPUT_PATH = OUTPUT_DIR / "three_line_false_color_dark_additive_all_regions_rasterized.png"


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


def saturation_boost(rgb, amount=1.4, weight=None):
    """Increase color separation while keeping total brightness recognizable.

    `weight` (0-1 per pixel), when given, scales the boost so faint pixels
    stay closer to gray/black and only bright pixels get the full color
    push -- otherwise faint, noisy background gets boosted to the same
    vivid saturation as real bright nebulosity, washing every panel in a
    uniform colored haze regardless of actual signal strength.
    """
    rgb = np.clip(rgb, 0, 1)
    gray = np.mean(rgb, axis=1, keepdims=True)
    effective = amount if weight is None else 1.0 + (amount - 1.0) * weight[:, None]
    return np.clip(gray + effective * (rgb - gray), 0, 1)


def dark_additive_rgb(
    sii,
    nii,
    siii,
    intensity_softening=0.22,
    saturation=2.8,
    gamma_lum=0.75,
    white_pctl=98.5,
):
    """Separate brightness (total flux) from color (line ratios).

    HII region cores are typically bright in SII, NII, and SIII all at
    once -- a straight additive RGB mix (sii*blue + nii*green + siii*red)
    washes those cores to pale white/pink no matter how the tone curve is
    tuned, since "bright in all three channels" literally *is* white.
    Normalizing each pixel's color by its own total flux (so color encodes
    only the *relative* line ratios) and using the total flux separately,
    asinh-stretched, for brightness keeps real color structure visible
    even in bright cores -- the standard astrophotography "luminance +
    chrominance" split.
    """
    channels = np.column_stack([siii, nii, sii])  # R=SIII, G=NII, B=SII
    total = np.sum(channels, axis=1)

    intensity = np.zeros_like(total)
    positive = total > 0
    if np.any(positive):
        hi = np.nanpercentile(total[positive], white_pctl)
        scaled = np.clip(total[positive] / max(hi, 1e-12), 0, 1)
        intensity[positive] = (
            np.arcsinh(scaled / intensity_softening) / np.arcsinh(1.0 / intensity_softening)
        ) ** gamma_lum

    color = np.zeros_like(channels)
    color[positive] = channels[positive] / total[positive, None]
    color *= 3.0  # so a pixel with exactly 1/3-1/3-1/3 ratio stays unsaturated gray

    rgb = np.clip(color * intensity[:, None], 0, 1)
    return saturation_boost(rgb, amount=saturation, weight=intensity)


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


def rasterize_rgb_image(ra, dec, rgb, n_pix=900, smooth_sigma_deg=0.007, min_fiber_weight=0.35):
    """Rasterize irregular RGB spaxels to a smoothed image grid.

    smooth_sigma is specified in ANGULAR units (degrees on sky), not raster
    pixels, and converted to pixels internally. This matters: at n_pix~900
    over a ~1-2 deg field, each pixel is far smaller than the ~0.0103 deg
    fiber pitch, so a fixed *pixel* sigma (e.g. the old default of 1.8 px)
    is narrower than the gap between fibers and never actually averages
    neighboring fibers together -- every cell just reflects 0 or 1 raw
    fiber, producing a speckled, noise-dominated image (this was especially
    bad for Flame, whose real footprint is a thin 3-armed "tripod" that the
    old under-smoothed rasterization turned into a noisy false hexagon).
    Tying sigma to a physical angular scale (~0.7x the fiber pitch) fixes
    this for every region regardless of its angular size or point density.

    Low-SNR fibers (masked to 0 in every channel by stretched_channel) and
    the sparse raster grid both leave small gaps -- binary_closing +
    binary_fill_holes patches those into the surrounding footprint instead
    of leaving visible black holes/pinholes punched through nebulosity.
    """
    ra_min, ra_max = float(np.nanmin(ra)), float(np.nanmax(ra))
    dec_min, dec_max = float(np.nanmin(dec)), float(np.nanmax(dec))
    ra_span = max(ra_max - ra_min, 1e-12)
    dec_span = max(dec_max - dec_min, 1e-12)

    n_ra = n_pix
    n_dec = max(100, int(round(n_pix * dec_span / ra_span)))
    ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)
    smooth_sigma = smooth_sigma_deg / (ra_span / n_ra)

    image = np.zeros((n_dec, n_ra, 4), dtype=float)
    counts, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    smooth_counts = gaussian_filter(counts, smooth_sigma, mode="constant", cval=0.0)
    # A lone fiber's own smoothed peak is 1/(2*pi*sigma^2) (2D Gaussian kernel
    # normalized to unit integral) -- thresholding at a FIXED absolute count
    # (the old min_weight=0.03) only worked by accident at the old fixed
    # pixel-sigma=1.8; with sigma now set per-region in angular units, a
    # bigger region/finer n_pix gives a wider sigma_pix, which dilutes the
    # peak and pushes it below a fixed threshold -- emptying the footprint
    # entirely for larger regions (this silently deleted BFS65 and
    # shrank Gum31/RCW52/RCW79/Sh2-311 to thin slivers). Scaling the
    # threshold by the kernel's own peak keeps "at least ~0.35 fibers'
    # worth of local density" meaningful regardless of sigma_pix.
    min_weight = min_fiber_weight / (2.0 * np.pi * smooth_sigma**2)
    footprint = smooth_counts > min_weight
    close_size = max(3, int(round(5 * n_pix / 640)))
    if close_size % 2 == 0:
        close_size += 1
    footprint = binary_fill_holes(binary_closing(footprint, structure=np.ones((close_size, close_size))))

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
        figsize=(4.4 * n_cols, 4.1 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("black")

    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.set_axis_off()

    for idx, (name, ra, dec, rgb) in enumerate(regions):
        ax = axes.ravel()[idx]
        image, extent = rasterize_rgb_image(ra, dec, rgb)
        ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
        ax.set_title(name, fontsize=13, fontweight="bold", pad=4, color="white")
        tighten_sky_axes(ax, ra, dec)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=(1.0, 0.1, 0.1), markersize=7, label="SIII"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=(0.1, 1.0, 0.1), markersize=7, label="NII"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=(0.1, 0.1, 1.0), markersize=7, label="SII"),
    ]
    for idx in range(len(regions), n_rows * n_cols):
        ax = axes.ravel()[idx]
        if idx == len(regions):
            legend = ax.legend(
                handles=handles,
                loc="center",
                frameon=False,
                fontsize=13,
                title="RGB channels",
                title_fontsize=14,
            )
            for text in legend.get_texts():
                text.set_color("white")
            legend.get_title().set_color("white")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RASTER_OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {RASTER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
