#!/usr/bin/env python3
"""Best-effort SII/NII/SIII false-color composite, tuned to reveal gas structure.

Builds on the lessons from the other false-color scripts in this directory:
- dark_additive_rgb (plot_three_region_composite_row.py) washes bright
  cores to white and leaves faint structure nearly black.
- The luminance/chrominance split (plot_three_line_composite_all_regions.py)
  keeps color visible in bright cores, but on its own doesn't make faint
  filamentary structure easier to SEE.
- Post-hoc median filtering (despeckle) cleans small outlier dots but can't
  remove larger multi-fiber artifacts without blurring real structure.

This script instead:
1. Rejects outlier fibers at the SOURCE, before any color is computed --
   each fiber's log-flux is compared to the median of its own k nearest
   neighbors (KDTree), and fibers that deviate by more than a few MAD from
   their local neighborhood are replaced by that neighborhood median. This
   fixes cosmic-ray-like artifacts and bad pixels directly, rather than
   trying to blur them away after the fact.
2. Splits luminance (total flux) from chrominance (line ratios) as before,
   so bright cores keep real color instead of washing to white.
3. Applies unsharp masking to the rasterized luminance channel -- boosting
   the difference between the image and a heavily blurred version of
   itself -- which is the standard astrophotography trick for making
   filamentary gas structure pop without needing to crank global contrast
   (which would blow out the core and crush the faint outskirts instead).

Defaults to Rosette (the most data-rich single region in this dataset,
35,080 fibers) but REGION can be pointed at any LVM_*.fits file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO_ROOT = HERE.parents[1]
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
REGION = REPO_ROOT / "LVM_Rosette_nebula.fits"
OUTPUT_PATH = OUTPUT_DIR / f"{REGION.stem.replace('LVM_', '').replace('_nebula', '')}_best_gas_composite.png"


def reject_outlier_fibers(ra, dec, flux, k=8, mad_threshold=3.5):
    """Replace fibers whose flux deviates sharply from their k nearest
    neighbors with that neighborhood's median -- fixes cosmic-ray-like
    single-fiber spikes without touching genuine smoothly-varying nebular
    structure (real structure is, by definition, similar to its neighbors;
    only true point-defects get flagged and replaced).
    """
    flux = np.asarray(flux, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    out = flux.copy()
    if good.sum() < k + 1:
        return out, np.zeros_like(good)

    coords = np.column_stack([ra[good], dec[good]])
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # includes self at idx[:, 0]
    neighbor_flux = flux[good][idx[:, 1:]]
    log_flux = np.log10(flux[good])
    log_neighbors = np.log10(neighbor_flux)
    neighbor_median = np.median(log_neighbors, axis=1)
    mad = np.median(np.abs(log_neighbors - neighbor_median[:, None]), axis=1)
    mad = np.maximum(mad, 1e-6)
    deviation = np.abs(log_flux - neighbor_median) / mad
    is_outlier = deviation > mad_threshold

    replaced = np.zeros_like(good)
    good_idx = np.flatnonzero(good)
    out[good_idx[is_outlier]] = 10 ** neighbor_median[is_outlier]
    replaced[good_idx[is_outlier]] = True
    return out, replaced


def fill_low_snr_fibers(ra, dec, flux, flux_err, snr_min=5.0, k=10, max_fill_dist_deg=0.03, fallback_to_raw=False):
    """Replace a fiber's flux with the local neighborhood's (SNR-passing)
    median whenever that fiber's own SNR is too low to trust, instead of
    leaving it to be hard-zeroed later by stretched_channel.

    This matters specifically for 3-line composites: if a fiber clears the
    SNR cut in two lines but not the third, hard-zeroing the failed line
    (the old behavior) doesn't mean "this line is absent" -- it just means
    "not well measured here" -- but the composite renders it as if the
    line truly were zero, producing a fabricated, artificially pure color
    (e.g. zero-red "pure cyan") at that fiber. Requiring all three lines to
    individually pass SNR instead just throws the whole fiber away,
    shrinking the usable footprint and leaving ragged edges wherever one
    line is systematically fainter (SIII, usually). Filling the low-SNR
    value from well-measured neighbors keeps the fiber in the composite
    with a realistic color and keeps the footprint intact.

    `max_fill_dist_deg` caps how far away a "trustworthy" neighbor can be
    (default ~3x the fiber pitch). Without this cap, a WIDE contiguous
    low-SNR patch (e.g. the genuinely faint outskirts of a region, where
    nothing nearby is trustworthy either) gets filled from whatever
    trustworthy cluster happens to be nearest -- often a distant, brighter
    part of the field -- painting a large, uniform, fabricated color patch
    across the whole area instead of many small, locally-accurate fills.
    Fibers with no trustworthy neighbor within range are left unfilled by
    default (NaN, meant to be excluded downstream) rather than guessed at
    from afar. With `fallback_to_raw=True`, those fibers keep their own
    original (noisy but real) value instead of being dropped -- trading a
    little visible grain in the rare case nothing nearby is trustworthy
    for never losing field-of-view coverage.
    """
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    finite = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0)
    snr = np.divide(flux, flux_err, out=np.full_like(flux, np.nan), where=finite)
    trustworthy = finite & (snr >= snr_min)

    out = flux.copy()
    if trustworthy.sum() < k:
        out[~trustworthy] = np.nan
        return out, np.zeros_like(trustworthy)

    all_coords = np.column_stack([ra, dec])
    good_coords = all_coords[trustworthy]
    tree = cKDTree(good_coords)

    needs_fill = np.isfinite(ra) & np.isfinite(dec) & ~trustworthy
    dist, idx = tree.query(all_coords[needs_fill], k=k)
    neighbor_flux = flux[trustworthy][idx]
    filled_value = np.median(neighbor_flux, axis=1)
    within_range = dist[:, -1] <= max_fill_dist_deg  # k-th (farthest used) neighbor distance

    fill_idx = np.flatnonzero(needs_fill)
    out[fill_idx[within_range]] = filled_value[within_range]
    if not fallback_to_raw:
        out[fill_idx[~within_range]] = np.nan  # too far from any trustworthy data -- exclude, don't fabricate
    # else: leave out[fill_idx[~within_range]] as the original raw value

    actually_filled = np.zeros_like(trustworthy)
    actually_filled[fill_idx[within_range]] = True
    return out, actually_filled


def reject_color_outliers(ra, dec, sii, nii, siii, k=10, ratio_threshold=5.0):
    """Fix fibers whose COLOR (line-ratio) is inconsistent with their
    neighbors, even when no single channel's flux is extreme enough to
    trip reject_outlier_fibers on its own.

    A fiber can have unremarkable brightness in all three lines and still
    look like a stray colored dot once rendered, if the *relative* mix
    between SII/NII/SIII is off from its surroundings (e.g. one line
    moderately overestimated due to sky-subtraction residuals or blending
    with a faint background source) -- flux-level outlier rejection, which
    checks each line independently, has no way to catch that. This checks
    each fiber's normalized (sii, nii, siii) fraction vector against the
    median fraction vector of its k nearest neighbors, flags fibers whose
    distance from that neighborhood median is large relative to how
    scattered the neighbors already are amongst themselves (so it adapts
    to how color-homogeneous the local nebulosity actually is, rather than
    using one fixed distance for the whole image), and replaces only those
    fibers' three fluxes with values matching the neighborhood's color
    while preserving the fiber's own total brightness.
    """
    sii = np.asarray(sii, dtype=float)
    nii = np.asarray(nii, dtype=float)
    siii = np.asarray(siii, dtype=float)
    good = (
        np.isfinite(sii) & np.isfinite(nii) & np.isfinite(siii)
        & (sii > 0) & (nii > 0) & (siii > 0)
    )
    out_sii, out_nii, out_siii = sii.copy(), nii.copy(), siii.copy()
    if good.sum() < k + 1:
        return out_sii, out_nii, out_siii, np.zeros_like(good)

    coords = np.column_stack([ra[good], dec[good]])
    tree = cKDTree(coords)
    total = sii[good] + nii[good] + siii[good]
    frac = np.column_stack([sii[good], nii[good], siii[good]]) / total[:, None]

    _, idx = tree.query(coords, k=k + 1)
    neighbor_frac = frac[idx[:, 1:]]
    neighbor_median = np.median(neighbor_frac, axis=1)
    dist = np.linalg.norm(frac - neighbor_median, axis=1)
    neighbor_spread = np.median(np.linalg.norm(neighbor_frac - neighbor_median[:, None, :], axis=2), axis=1)
    neighbor_spread = np.maximum(neighbor_spread, 1e-4)
    is_outlier = (dist / neighbor_spread) > ratio_threshold

    replaced = np.zeros_like(good)
    good_idx = np.flatnonzero(good)
    fixed_idx = good_idx[is_outlier]
    fixed_frac = neighbor_median[is_outlier]
    fixed_total = total[is_outlier]
    out_sii[fixed_idx] = fixed_frac[:, 0] * fixed_total
    out_nii[fixed_idx] = fixed_frac[:, 1] * fixed_total
    out_siii[fixed_idx] = fixed_frac[:, 2] * fixed_total
    replaced[fixed_idx] = True
    return out_sii, out_nii, out_siii, replaced


def stretched_channel(flux, flux_err, snr_min=5.0, pmin=1.0, pmax=99.5, softening=0.08):
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


def saturation_boost(rgb, amount, weight=None):
    rgb = np.clip(rgb, 0, 1)
    gray = np.mean(rgb, axis=1, keepdims=True)
    effective = amount if weight is None else 1.0 + (amount - 1.0) * weight[:, None]
    return np.clip(gray + effective * (rgb - gray), 0, 1)


def luminance_chrominance_rgb(sii, nii, siii, intensity_softening=0.18, saturation=1.8, gamma_lum=0.85, white_pctl=99.0):
    """Build HSV directly so Value always equals the true stretched
    brightness, by construction -- fixes a real bug in the earlier
    "color*3.0 then clip" version, where ANY pixel with one even mildly
    dominant channel (not just bright ones) got clipped to max(R,G,B)=1.0
    regardless of actual brightness. That silently destroyed the dynamic
    range needed for any later local-contrast enhancement (CLAHE) to find
    real structure in -- most of the image was already flatly saturated
    before CLAHE ever ran.

    Hue comes from the three channels' relative fractions (SIII/NII/SII
    mapped to the standard 0/120/240 degree RGB primary angles, combined
    by a circular weighted mean). Saturation is how dominant one channel
    is over the others (max fraction - min fraction, boosted for vividness
    but capped at 1). Value is the independently-stretched total
    brightness. matplotlib's hsv_to_rgb guarantees max(R,G,B) == Value
    exactly, so brightness and color purity never contaminate each other.
    """
    channels = np.column_stack([siii, nii, sii])  # order: SIII, NII, SII
    total = np.sum(channels, axis=1)

    value = np.zeros_like(total)
    positive = total > 0
    if np.any(positive):
        hi = np.nanpercentile(total[positive], white_pctl)
        scaled = np.clip(total[positive] / max(hi, 1e-12), 0, 1)
        value[positive] = (
            np.arcsinh(scaled / intensity_softening) / np.arcsinh(1.0 / intensity_softening)
        ) ** gamma_lum

    frac = np.zeros_like(channels)
    frac[positive] = channels[positive] / total[positive, None]

    angles = np.deg2rad([0.0, 120.0, 240.0])  # SIII=red, NII=green, SII=blue
    x = frac @ np.cos(angles)
    y = frac @ np.sin(angles)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0

    raw_sat = frac.max(axis=1) - frac.min(axis=1)  # 0 (equal mix) to 1 (pure single line)
    sat = np.clip(raw_sat * saturation, 0, 1)

    from matplotlib.colors import hsv_to_rgb

    rgb = hsv_to_rgb(np.column_stack([hue, sat, value]))
    return rgb, value


def tighten_sky_axes(ax, ra_values, dec_values, pad_fraction=0.01):
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


def rasterize_rgb_image(ra, dec, rgb, n_pix=1000, smooth_sigma_deg=0.010, min_fiber_weight=0.35):
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
    min_weight = min_fiber_weight / (2.0 * np.pi * smooth_sigma**2)
    footprint = smooth_counts > min_weight
    footprint = binary_fill_holes(binary_closing(footprint, structure=np.ones((7, 7))))

    for channel in range(3):
        weighted_sum, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges], weights=rgb[:, channel])
        smooth_sum = gaussian_filter(weighted_sum, smooth_sigma, mode="constant", cval=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(smooth_sum, smooth_counts, out=np.zeros_like(smooth_sum), where=smooth_counts > 1e-8)
        image[..., channel] = np.clip(values, 0, 1)

    image[..., 3] = footprint.astype(float)
    extent = [ra_max, ra_min, dec_min, dec_max]
    return image, extent


def enhance_local_contrast(image, clip_limit=0.035, kernel_fraction=0.03, blend=0.85):
    """Reveal faint filamentary structure via CLAHE on the luminance (value)
    channel, without altering hue/saturation.

    A plain unsharp mask subtracts a blurred copy of the image from itself
    -- it boosts edges, but a large smoothly-saturated region (like a
    bright HII core sitting at or near the white point) has no edges to
    boost in the first place, since every nearby pixel is already close to
    the same clipped value. CLAHE instead locally re-normalizes contrast
    within small tiles, so even a region that's "merely bright" everywhere
    at the global scale gets its own *local* faint variations stretched
    out -- the standard astrophotography technique (e.g. PixInsight's
    LocalHistogramEqualization) for pulling filamentary detail out of an
    otherwise flat-looking bright nebula core.
    """
    from skimage.exposure import equalize_adapthist

    rgb = image[..., :3]
    footprint = image[..., 3] > 0
    value = rgb.max(axis=2)
    hue_sat_ratio = np.divide(rgb, value[..., None], out=np.zeros_like(rgb), where=value[..., None] > 1e-8)

    kernel_size = max(8, int(round(min(value.shape) * kernel_fraction)))
    equalized = equalize_adapthist(np.clip(value, 0, 1), kernel_size=kernel_size, clip_limit=clip_limit)
    boosted_value = np.where(footprint, blend * equalized + (1 - blend) * value, value)

    boosted_rgb = hue_sat_ratio * boosted_value[..., None]
    out = image.copy()
    out[..., :3] = np.clip(boosted_rgb, 0, 1)
    return out


def main():
    with fits.open(REGION) as hdul:
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
    sii_clean, n_sii = reject_outlier_fibers(ra, dec, np.where(valid, sii_raw, np.nan))
    nii_clean, n_nii = reject_outlier_fibers(ra, dec, np.where(valid, nii_raw, np.nan))
    siii_clean, n_siii = reject_outlier_fibers(ra, dec, np.where(valid, siii_raw, np.nan))
    print(f"Outlier fibers replaced: SII={n_sii.sum()}, NII={n_nii.sum()}, SIII={n_siii.sum()}")

    sii, good_sii = stretched_channel(sii_clean, sii_err)
    nii, good_nii = stretched_channel(nii_clean, nii_err)
    siii, good_siii = stretched_channel(siii_clean, siii_err)

    visible = valid & (good_sii | good_nii | good_siii)
    rgb, _ = luminance_chrominance_rgb(sii[visible], nii[visible], siii[visible])

    image, extent = rasterize_rgb_image(ra[visible], dec[visible], rgb)
    image = enhance_local_contrast(image)

    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(10.0, 9.2))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
    tighten_sky_axes(ax, ra[visible], dec[visible])
    ax.set_axis_off()

    region_name = REGION.stem.replace("LVM_", "").replace("_nebula", "")
    ax.text(
        0.02, 0.02, f"{region_name}  —  SIII=R, NII=G, SII=B", transform=ax.transAxes,
        color="white", fontsize=11, alpha=0.85, va="bottom",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=320, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
