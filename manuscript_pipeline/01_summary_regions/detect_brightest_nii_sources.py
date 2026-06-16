"""Detect brightest NII emission regions and estimate source positions.

Algorithm overview:
1. Load NII flux spaxels with S/N filtering.
2. Rasterize the irregular spaxels into a regular RA/Dec grid using weighted means.
3. Threshold the grid at a high percentile in log10(flux) to keep only the brightest emission.
4. Run connected-component labeling on the threshold mask.
5. Rank components by integrated flux and report weighted centroids as source coordinates.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def load_nii_points(fits_path: str, snr_min: float = 5.0):
    """Return filtered RA/Dec/flux arrays for NII."""
    with fits.open(fits_path) as hdul:
        table = hdul[1].data
        required = {"ra", "dec", "flux_nii", "flux_nii_err"}
        missing = required.difference(table.names)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        ra = np.asarray(table["ra"], dtype=float)
        dec = np.asarray(table["dec"], dtype=float)
        flux = np.asarray(table["flux_nii"], dtype=float)
        ferr = np.asarray(table["flux_nii_err"], dtype=float)

    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(flux) & np.isfinite(ferr)
    valid &= ferr > 0

    snr = np.zeros_like(flux)
    snr[valid] = flux[valid] / ferr[valid]

    mask = valid & (snr >= snr_min) & (flux > 0)
    return ra[mask], dec[mask], flux[mask]


def rasterize_weighted_mean(ra: np.ndarray, dec: np.ndarray, flux: np.ndarray, n_grid: int = 260):
    """Rasterize point cloud to a regular grid using weighted mean flux per cell."""
    ra_min, ra_max = float(np.min(ra)), float(np.max(ra))
    dec_min, dec_max = float(np.min(dec)), float(np.max(dec))

    # Keep square-ish cells in sky coordinates.
    ra_span = max(ra_max - ra_min, 1e-12)
    dec_span = max(dec_max - dec_min, 1e-12)
    n_ra = n_grid
    n_dec = max(64, int(round(n_grid * dec_span / ra_span)))

    ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)

    sum_flux, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges], weights=flux)
    counts, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_flux = np.where(counts > 0, sum_flux / counts, np.nan)

    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    return mean_flux, counts, ra_centers, dec_centers


def connected_components(mask: np.ndarray):
    """Return list of components (each as list of (row, col)) using 8-connectivity."""
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    n_rows, n_cols = mask.shape
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for r in range(n_rows):
        for c in range(n_cols):
            if not mask[r, c] or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            comp = []

            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))

                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            components.append(comp)

    return components


def detect_bright_regions(
    mean_flux: np.ndarray,
    counts: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    bright_percentile: float = 99.0,
    min_cells: int = 8,
):
    """Detect bright connected regions and return ranked source candidates."""
    valid = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
    if not np.any(valid):
        return [], np.zeros_like(mean_flux, dtype=bool)

    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
    log_flux[valid] = np.log10(mean_flux[valid])
    threshold = np.nanpercentile(log_flux[valid], bright_percentile)

    bright_mask = valid & (log_flux >= threshold)
    comps = connected_components(bright_mask)

    sources = []
    for comp in comps:
        if len(comp) < min_cells:
            continue

        rows = np.array([rc[0] for rc in comp], dtype=int)
        cols = np.array([rc[1] for rc in comp], dtype=int)
        flux_vals = mean_flux[rows, cols]

        integrated_flux = float(np.nansum(flux_vals))
        peak_idx = int(np.nanargmax(flux_vals))
        peak_row = rows[peak_idx]
        peak_col = cols[peak_idx]
        peak_flux = float(flux_vals[peak_idx])

        weights = np.maximum(flux_vals, 0)
        wsum = float(np.nansum(weights))
        if wsum <= 0:
            continue

        ra_vals = ra_centers[cols]
        dec_vals = dec_centers[rows]
        ra_centroid = float(np.nansum(ra_vals * weights) / wsum)
        dec_centroid = float(np.nansum(dec_vals * weights) / wsum)

        sources.append(
            {
                "n_cells": len(comp),
                "integrated_flux": integrated_flux,
                "peak_flux": peak_flux,
                "peak_ra": float(ra_centers[peak_col]),
                "peak_dec": float(dec_centers[peak_row]),
                "centroid_ra": ra_centroid,
                "centroid_dec": dec_centroid,
                "rows": rows,
                "cols": cols,
            }
        )

    sources.sort(key=lambda s: s["integrated_flux"], reverse=True)
    return sources, bright_mask


def make_overlay_plot(
    out_path: str,
    region_name: str,
    mean_flux: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    bright_mask: np.ndarray,
    sources: list[dict],
    max_marked: int = 3,
):
    """Save overlay plot showing brightest mask and source positions."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    plot_data = np.log10(np.where(mean_flux > 0, mean_flux, np.nan))
    im = ax.imshow(
        plot_data,
        origin="lower",
        extent=[ra_centers[0], ra_centers[-1], dec_centers[0], dec_centers[-1]],
        cmap="inferno",
        aspect="auto",
    )

    # Outline thresholded bright mask.
    yy, xx = np.where(bright_mask)
    if yy.size > 0:
        ax.scatter(
            ra_centers[xx],
            dec_centers[yy],
            s=3,
            c="cyan",
            alpha=0.35,
            linewidths=0,
            label="bright-mask cells",
        )

    for i, src in enumerate(sources[:max_marked], start=1):
        ax.scatter(src["peak_ra"], src["peak_dec"], marker="*", s=240, c="white", edgecolor="black", linewidth=0.8)
        ax.scatter(src["centroid_ra"], src["centroid_dec"], marker="x", s=90, c="deepskyblue", linewidth=2.0)
        ax.text(
            src["peak_ra"],
            src["peak_dec"],
            f" S{i}",
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    ax.invert_xaxis()
    ax.set_title(f"{region_name}: brightest NII sources", fontweight="bold")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label(r"$\log_{10}(\mathrm{NII\ flux})$")

    if sources:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main():
    fits_files = sorted(glob.glob("LVM_*.fits"))
    if not fits_files:
        raise FileNotFoundError("No LVM_*.fits files found in current directory")

    os.makedirs("Figures/bright_sources", exist_ok=True)
    plot_dir = "Figures/manuscript/01_summary_regions/bright_sources"
    os.makedirs(plot_dir, exist_ok=True)

    percentile_candidates = [99.0, 98.5, 98.0, 97.0]
    min_cells_candidates = [8, 5, 3]
    top_k_export = 5

    csv_rows = []
    for fits_path in fits_files:
        base = os.path.splitext(os.path.basename(fits_path))[0]
        region_name = base.replace("LVM_", "")

        try:
            ra, dec, flux = load_nii_points(fits_path, snr_min=5.0)
            if flux.size < 100:
                print(f"Skipping {base}: not enough valid NII points ({flux.size})")
                continue

            mean_flux, counts, ra_centers, dec_centers = rasterize_weighted_mean(ra, dec, flux, n_grid=260)
            sources = []
            bright_mask = np.zeros_like(mean_flux, dtype=bool)
            chosen_percentile = None
            chosen_min_cells = None

            for pct in percentile_candidates:
                for min_cells in min_cells_candidates:
                    trial_sources, trial_mask = detect_bright_regions(
                        mean_flux,
                        counts,
                        ra_centers,
                        dec_centers,
                        bright_percentile=pct,
                        min_cells=min_cells,
                    )
                    if trial_sources:
                        sources = trial_sources
                        bright_mask = trial_mask
                        chosen_percentile = pct
                        chosen_min_cells = min_cells
                        break
                if sources:
                    break

            # Last-resort fallback for sparse fields: use global maximum cell.
            if not sources:
                valid_cells = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
                if np.any(valid_cells):
                    rr, cc = np.unravel_index(np.nanargmax(np.where(valid_cells, mean_flux, np.nan)), mean_flux.shape)
                    peak_flux = float(mean_flux[rr, cc])
                    sources = [
                        {
                            "n_cells": 1,
                            "integrated_flux": peak_flux,
                            "peak_flux": peak_flux,
                            "peak_ra": float(ra_centers[cc]),
                            "peak_dec": float(dec_centers[rr]),
                            "centroid_ra": float(ra_centers[cc]),
                            "centroid_dec": float(dec_centers[rr]),
                            "rows": np.array([rr]),
                            "cols": np.array([cc]),
                        }
                    ]
                    bright_mask = np.zeros_like(mean_flux, dtype=bool)
                    bright_mask[rr, cc] = True
                    chosen_percentile = float("nan")
                    chosen_min_cells = 1

            out_plot = os.path.join(plot_dir, f"{base}_brightest_sources.png")
            make_overlay_plot(out_plot, region_name, mean_flux, ra_centers, dec_centers, bright_mask, sources)

            if not sources:
                print(f"{base}: no bright connected sources passed size threshold")
                continue

            for rank, src in enumerate(sources[:top_k_export], start=1):
                csv_rows.append(
                    {
                        "region": region_name,
                        "rank": rank,
                        "n_cells": src["n_cells"],
                        "integrated_flux": src["integrated_flux"],
                        "peak_flux": src["peak_flux"],
                        "peak_ra_deg": src["peak_ra"],
                        "peak_dec_deg": src["peak_dec"],
                        "centroid_ra_deg": src["centroid_ra"],
                        "centroid_dec_deg": src["centroid_dec"],
                    }
                )

            best = sources[0]
            threshold_text = "fallback-peak" if chosen_percentile is None or math.isnan(chosen_percentile) else f"{chosen_percentile:.1f}p"
            print(
                f"{base}: {len(sources)} bright sources | "
                f"threshold={threshold_text}, min_cells={chosen_min_cells} | "
                f"best peak=({best['peak_ra']:.6f}, {best['peak_dec']:.6f}) "
                f"centroid=({best['centroid_ra']:.6f}, {best['centroid_dec']:.6f})"
            )

        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"Skipping {base} due to error: {exc}")

    csv_path = "Figures/bright_sources/nii_bright_source_catalog.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fobj:
            writer = csv.DictWriter(fobj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved source catalog: {csv_path}")
    else:
        print("No source entries to save.")


if __name__ == "__main__":
    main()
