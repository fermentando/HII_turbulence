"""Create full-field NII emission flux maps for all LVM region FITS files.

This manuscript figure intentionally does not use brightest-source positions,
fixed log-flux cuts, contours, or occupied-subregion selections.
"""

from __future__ import annotations

import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def load_nii_map_data(fits_path: str, snr_min: float = 5.0):
    """Return RA, Dec, and NII flux arrays after basic finite/SNR filtering."""
    with fits.open(fits_path) as hdul:
        table = hdul[1].data
        required = {"ra", "dec", "flux_nii", "flux_nii_err"}
        missing = required.difference(table.names)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        flux = np.asarray(table["flux_nii"], dtype=float)
        flux_err = np.asarray(table["flux_nii_err"], dtype=float)
        ra = np.asarray(table["ra"], dtype=float)
        dec = np.asarray(table["dec"], dtype=float)

    valid = np.isfinite(flux) & np.isfinite(flux_err) & np.isfinite(ra) & np.isfinite(dec)
    valid &= (flux_err > 0) & (flux > 0)
    snr = np.zeros_like(flux)
    snr[valid] = flux[valid] / flux_err[valid]
    mask = valid & (snr > snr_min)
    return ra[mask], dec[mask], flux[mask]


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
    fits_files = sorted(glob.glob("LVM_*.fits"))
    if not fits_files:
        raise FileNotFoundError("No LVM_*.fits files found in the current directory.")

    region_data = []
    for path in fits_files:
        try:
            ra, dec, flux = load_nii_map_data(path)
            if flux.size == 0:
                print(f"Skipping {os.path.basename(path)}: no valid NII points after filtering")
                continue
            region_data.append((path, ra, dec, flux))
            print(f"Loaded {os.path.basename(path)}: {flux.size:,} valid points")
        except Exception as exc:
            print(f"Skipping {os.path.basename(path)} due to error: {exc}")

    if not region_data:
        raise RuntimeError("No valid subregion data available for NII plotting.")

    all_log_flux = np.concatenate([np.log10(item[3]) for item in region_data])
    vmin = np.nanpercentile(all_log_flux, 5)
    vmax = np.nanpercentile(all_log_flux, 95)

    n_regions = len(region_data)
    n_cols = 3
    n_rows = math.ceil(n_regions / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 5.3 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    scatter_for_cbar = None
    for idx, (path, ra, dec, flux) in enumerate(region_data):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        base = os.path.splitext(os.path.basename(path))[0].replace("LVM_", "")

        scatter_for_cbar = ax.scatter(
            ra,
            dec,
            c=np.log10(flux),
            s=7,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.95,
        )

        ax.set_title(base, fontsize=11, fontweight="bold")
        ax.set_xlabel("RA (deg)")
        if col == 0:
            ax.set_ylabel("Dec (deg)")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        tighten_sky_axes(ax, ra, dec)

    for idx in range(n_regions, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    if scatter_for_cbar is not None:
        cbar = fig.colorbar(scatter_for_cbar, ax=axes.ravel().tolist(), shrink=0.95, pad=0.01)
        cbar.set_label(r"$\log_{10}(\mathrm{NII\ flux})$")

    fig.suptitle("NII Emission Flux Maps (full valid field; S/N > 5)", fontsize=15, fontweight="bold")

    output_dir = "Figures/manuscript/01_summary_regions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nii_flux_maps_all_subregions_fullfield.png")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
