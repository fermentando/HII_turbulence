"""Create an RGB false-color emission map for the Rosette nebula."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
FITS_PATH = REPO_ROOT / "LVM_Rosette_nebula.fits"
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "LVM_Rosette_nebula_false_color_rgb.png"
RATIO_OUTPUT_PATH = OUTPUT_DIR / "LVM_Rosette_nebula_false_color_ratio_rgb.png"


def stretched_channel(flux, flux_err, snr_min=5.0, pmin=2.0, pmax=99.0, softening=0.08):
    """Return a robustly normalized image channel in the range [0, 1]."""
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0)
    good &= (flux / flux_err) > snr_min

    channel = np.zeros_like(flux, dtype=float)
    if not np.any(good):
        return channel, good

    log_flux = np.full_like(flux, np.nan, dtype=float)
    log_flux[good] = np.log10(flux[good])
    lo, hi = np.nanpercentile(log_flux[good], [pmin, pmax])
    scaled = np.clip((log_flux[good] - lo) / max(hi - lo, 1e-12), 0, 1)
    channel[good] = np.arcsinh(scaled / softening) / np.arcsinh(1.0 / softening)
    return channel, good


def saturation_boost(rgb, amount=2.2):
    """Increase color separation while keeping brightness fixed."""
    rgb = np.clip(rgb, 0, 1)
    gray = np.mean(rgb, axis=1, keepdims=True)
    return np.clip(gray + amount * (rgb - gray), 0, 1)


def ratio_composite(red, green, blue, intensity_softening=0.10, saturation=2.2):
    """Use total emission for brightness and line ratios for color."""
    channels = np.column_stack([red, green, blue])
    total = np.sum(channels, axis=1)

    intensity = np.zeros_like(total)
    positive = total > 0
    scaled_total = np.zeros_like(total)
    if np.any(positive):
        hi = np.nanpercentile(total[positive], 99.5)
        scaled_total[positive] = np.clip(total[positive] / max(hi, 1e-12), 0, 1)
        intensity[positive] = (
            np.arcsinh(scaled_total[positive] / intensity_softening)
            / np.arcsinh(1.0 / intensity_softening)
        )

    color = np.zeros_like(channels)
    color[positive] = channels[positive] / total[positive, None]
    color *= 3.0
    rgb = np.clip(color * intensity[:, None], 0, 1)
    return saturation_boost(rgb, amount=saturation)


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


def main():
    with fits.open(FITS_PATH) as hdul:
        table = hdul[1].data
        ra = np.asarray(table["ra"], dtype=float)
        dec = np.asarray(table["dec"], dtype=float)

        red, good_sii = stretched_channel(table["flux_sii"], table["flux_sii_err"])
        green, good_nii = stretched_channel(table["flux_nii"], table["flux_nii_err"])
        blue, good_siii = stretched_channel(table["flux_siii"], table["flux_siii_err"])

    valid_position = np.isfinite(ra) & np.isfinite(dec)
    visible = valid_position & (good_sii | good_nii | good_siii)
    flux_rgb = np.column_stack([red[visible], green[visible], blue[visible]])
    ratio_rgb = ratio_composite(red[visible], green[visible], blue[visible])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax, rgb, title in zip(
        axes,
        [flux_rgb, ratio_rgb],
        ["Independent flux stretch", "Ratio-preserving color"],
    ):
        ax.set_facecolor("black")
        ax.scatter(
            ra[visible],
            dec[visible],
            c=rgb,
            s=8,
            linewidths=0,
            alpha=0.96,
        )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("RA (deg)")
        ax.grid(True, color="white", alpha=0.16, linewidth=0.6)
        tighten_sky_axes(ax, ra[visible], dec[visible])

    axes[0].set_ylabel("Dec (deg)")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=(1, 0, 0), markersize=7, label="SII"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=(0, 1, 0), markersize=7, label="NII"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=(0.2, 0.35, 1), markersize=7, label="SIII"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

    fig, ax = plt.subplots(figsize=(7.0, 6.4), constrained_layout=True)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("white")
    ax.scatter(ra[visible], dec[visible], c=ratio_rgb, s=8, linewidths=0, alpha=0.96)
    ax.set_title("Rosette nebula ratio-preserving false color", fontsize=13, fontweight="bold")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.grid(True, color="white", alpha=0.16, linewidth=0.6)
    tighten_sky_axes(ax, ra[visible], dec[visible])
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)
    fig.savefig(RATIO_OUTPUT_PATH, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {RATIO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
