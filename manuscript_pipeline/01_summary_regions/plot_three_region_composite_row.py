"""Create a composite row for selected regions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

from plot_best_gas_composite import fill_low_snr_fibers, reject_color_outliers, reject_outlier_fibers
from plot_three_line_composite_all_regions import stretched_channel


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

# Original palette-based additive color mix -- kept as a local copy (not
# imported from plot_three_line_composite_all_regions) since that module's
# dark_additive_rgb was since changed to a different (luminance/chrominance
# ratio) algorithm for the all-region gallery. This script intentionally
# stays on the original version, which gave a better result for this row.
PALETTE = {
    "siii": np.array([1.25, 0.22, 0.06]),  # bright red/orange: higher ionization
    "nii": np.array([0.10, 0.85, 0.25]),  # green: intermediate ionization
    "sii": np.array([0.15, 0.35, 1.35]),  # bright blue: lower ionization
}


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
    brightness=1.60,
    saturation=1.90,
    shoulder=0.60,
    black_point=0.05,
    contrast=1.30,
    white_limit=0.90,
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
        sii_raw = np.asarray(table["flux_sii"], dtype=float)
        nii_raw = np.asarray(table["flux_nii"], dtype=float)
        siii_raw = np.asarray(table["flux_siii"], dtype=float)
        sii_err = table["flux_sii_err"]
        nii_err = table["flux_nii_err"]
        siii_err = table["flux_siii_err"]

    valid = np.isfinite(ra) & np.isfinite(dec)

    # Fill low-SNR fibers from nearby well-measured neighbors, but fall
    # back to the fiber's own raw value (rather than excluding it) when
    # nothing trustworthy is close by -- the combination that actually
    # gets full field-of-view AND no fabricated colors: a plain SNR mask
    # hard-zeroes a failed line and fabricates a pure color; requiring all
    # three lines to pass independently drops huge swaths of real data
    # wherever one line is systematically fainter (SIII, usually); filling
    # with a hard distance cap and no fallback does the same, just less
    # aggressively. Falling back to the fiber's own (possibly noisy, but
    # real) value only in that last-resort case keeps every fiber in the
    # footprint, with a realistic color almost everywhere.
    sii_filled, _ = fill_low_snr_fibers(ra, dec, np.where(valid, sii_raw, np.nan), sii_err, fallback_to_raw=True)
    nii_filled, _ = fill_low_snr_fibers(ra, dec, np.where(valid, nii_raw, np.nan), nii_err, fallback_to_raw=True)
    siii_filled, _ = fill_low_snr_fibers(ra, dec, np.where(valid, siii_raw, np.nan), siii_err, fallback_to_raw=True)

    # Reject outlier fibers (cosmic-ray-like spikes, bad pixels) at the
    # source -- each fiber's flux is compared to its own k nearest
    # neighbors, and only genuine point-defects get replaced. This removes
    # the colorful speckle without needing heavy blur or a post-hoc median
    # filter, so real sharp structure stays sharp.
    sii_clean, _ = reject_outlier_fibers(ra, dec, sii_filled)
    nii_clean, _ = reject_outlier_fibers(ra, dec, nii_filled)
    siii_clean, _ = reject_outlier_fibers(ra, dec, siii_filled)
    # Catches the remaining stray colored dots: fibers whose individual line
    # fluxes aren't extreme enough to trip the per-channel check above, but
    # whose overall color (line ratio) is inconsistent with its neighbors.
    sii_clean, nii_clean, siii_clean, _ = reject_color_outliers(ra, dec, sii_clean, nii_clean, siii_clean)

    # Every channel now has a value for every valid fiber (real if it
    # originally passed SNR, neighbor-filled otherwise), so a permissive
    # stretch (snr_min=0) is enough -- the trustworthiness decision already
    # happened above.
    sii, _ = stretched_channel(sii_clean, sii_clean, snr_min=0.0)
    nii, _ = stretched_channel(nii_clean, nii_clean, snr_min=0.0)
    siii, _ = stretched_channel(siii_clean, siii_clean, snr_min=0.0)

    has_data = np.isfinite(sii_clean) & np.isfinite(nii_clean) & np.isfinite(siii_clean)
    visible = valid & has_data
    return ra[visible], dec[visible], dark_additive_rgb(sii[visible], nii[visible], siii[visible])


def rasterize_rgb_image(ra, dec, rgb, n_pix=900, smooth_sigma_deg=0.006, min_fiber_weight=0.35):
    """Rasterize irregular RGB spaxels to a smoothed image grid.

    smooth_sigma is specified in ANGULAR units (degrees on sky) and
    converted to raster pixels internally -- the previous fixed *pixel*
    sigma of 1.7 converted to only ~0.003-0.005 deg for these regions,
    2-4x NARROWER than the actual LVM fiber pitch (~0.0103 deg). That
    meant each raster cell mostly saw 0 or 1 raw fiber, since the blur
    kernel was too narrow to average neighboring fibers together at all --
    producing a grainy, undersampled look regardless of output dpi. Tying
    sigma to the fiber pitch itself fixes this for any region/pixel count.
    min_weight is similarly scaled by the kernel's own peak response
    (1/(2*pi*sigma_px^2)) instead of a fixed count, so the footprint
    threshold stays meaningful across different sigma_px.
    """
    ra_min, ra_max = float(np.nanmin(ra)), float(np.nanmax(ra))
    dec_min, dec_max = float(np.nanmin(dec)), float(np.nanmax(dec))
    ra_span = max(ra_max - ra_min, 1e-12)
    dec_span = max(dec_max - dec_min, 1e-12)

    n_ra = n_pix
    n_dec = max(120, int(round(n_pix * dec_span / ra_span)))
    ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)
    smooth_sigma = smooth_sigma_deg / (ra_span / n_ra)

    image = np.zeros((n_dec, n_ra, 4), dtype=float)
    counts, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    smooth_counts = gaussian_filter(counts, smooth_sigma, mode="constant", cval=0.0)
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
