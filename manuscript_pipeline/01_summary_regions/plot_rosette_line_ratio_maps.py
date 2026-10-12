"""Create physically interpretable emission-line ratio maps for Rosette."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[2]
FITS_PATH = REPO_ROOT / "LVM_Rosette_nebula.fits"
OUTPUT_DIR = REPO_ROOT / "Figures/manuscript/01_summary_regions"
OUTPUT_PATH = OUTPUT_DIR / "LVM_Rosette_nebula_line_ratio_maps.png"
SIII_SII_OUTPUT_PATH = OUTPUT_DIR / "LVM_Rosette_nebula_siii_sii_ratio_map.png"


def valid_flux(table, line, snr_min=5.0):
    """Return positive finite flux values and a valid/SNR mask for one line."""
    flux = np.asarray(table[f"flux_{line}"], dtype=float)
    flux_err = np.asarray(table[f"flux_{line}_err"], dtype=float)
    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err > 0)
    good &= (flux / flux_err) > snr_min
    return flux, good


def log_ratio(numerator, numerator_good, denominator, denominator_good):
    """Return log10(numerator / denominator) where both lines are usable."""
    good = numerator_good & denominator_good
    ratio = np.full_like(numerator, np.nan, dtype=float)
    ratio[good] = np.log10(numerator[good] / denominator[good])
    return ratio, good


def symmetric_percentile_limits(values, percentile=98.0):
    """Get symmetric display limits around zero for a diverging ratio map."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = np.nanpercentile(np.abs(finite), percentile)
    vmax = max(float(vmax), 0.05)
    return -vmax, vmax


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

        flux_nii, good_nii = valid_flux(table, "nii")
        flux_sii, good_sii = valid_flux(table, "sii")
        flux_siii, good_siii = valid_flux(table, "siii")

    ratio_siii_nii, good_siii_nii = log_ratio(flux_siii, good_siii, flux_nii, good_nii)
    ratio_sii_nii, good_sii_nii = log_ratio(flux_sii, good_sii, flux_nii, good_nii)
    ratio_siii_sii, good_siii_sii = log_ratio(flux_siii, good_siii, flux_sii, good_sii)

    valid_position = np.isfinite(ra) & np.isfinite(dec)
    ratio_maps = [
        (ratio_siii_nii, good_siii_nii & valid_position, r"$\log_{10}(\mathrm{SIII}/\mathrm{NII})$"),
        (ratio_sii_nii, good_sii_nii & valid_position, r"$\log_{10}(\mathrm{SII}/\mathrm{NII})$"),
    ]

    all_ratio_values = np.concatenate([ratio[mask] for ratio, mask, _ in ratio_maps])
    vmin, vmax = symmetric_percentile_limits(all_ratio_values, percentile=98.0)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    scatter = None
    for ax, (ratio, mask, title) in zip(axes, ratio_maps):
        scatter = ax.scatter(
            ra[mask],
            dec[mask],
            c=ratio[mask],
            s=8,
            cmap="coolwarm",
            norm=norm,
            linewidths=0,
            alpha=0.96,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("RA (deg)")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        tighten_sky_axes(ax, ra[mask], dec[mask])

    axes[0].set_ylabel("Dec (deg)")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.92, pad=0.02)
        cbar.set_label(r"$\log_{10}$ line ratio")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Common color range: {vmin:.3f} to {vmax:.3f} dex")
    print(f"SIII/NII points: {np.sum(good_siii_nii & valid_position):,}")
    print(f"SII/NII points: {np.sum(good_sii_nii & valid_position):,}")

    mask = good_siii_sii & valid_position
    vmin_single, vmax_single = symmetric_percentile_limits(ratio_siii_sii[mask], percentile=98.0)
    norm_single = TwoSlopeNorm(vmin=vmin_single, vcenter=0.0, vmax=vmax_single)

    fig, ax = plt.subplots(figsize=(7.0, 6.4), constrained_layout=True)
    scatter = ax.scatter(
        ra[mask],
        dec[mask],
        c=ratio_siii_sii[mask],
        s=8,
        cmap="coolwarm",
        norm=norm_single,
        linewidths=0,
        alpha=0.96,
    )
    ax.set_title(r"$\log_{10}(\mathrm{SIII}/\mathrm{SII})$", fontsize=13, fontweight="bold")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    tighten_sky_axes(ax, ra[mask], dec[mask])
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label(r"$\log_{10}(\mathrm{SIII}/\mathrm{SII})$")
    fig.savefig(SIII_SII_OUTPUT_PATH, dpi=220)
    plt.close(fig)

    print(f"Saved: {SIII_SII_OUTPUT_PATH}")
    print(f"SIII/SII color range: {vmin_single:.3f} to {vmax_single:.3f} dex")
    print(f"SIII/SII points: {np.sum(mask):,}")


if __name__ == "__main__":
    main()
