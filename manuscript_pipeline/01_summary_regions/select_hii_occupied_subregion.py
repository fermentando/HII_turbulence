"""Select the connected subregion occupied by each detected HII source.

Inputs:
- LVM_*.fits files (same folder)
- Figures/bright_sources/nii_bright_source_catalog.csv (rank-1 source positions)

Outputs:
- Figures/bright_sources/hii_occupied_subregions.csv
- Figures/manuscript/01_summary_regions/bright_sources/<region>_occupied_subregion.png
"""

from __future__ import annotations

import csv
import glob
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.path import Path


def load_rank1_sources(catalog_path: str):
    """Load rank-1 source per region from CSV."""
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Source catalog not found: {catalog_path}")

    out = {}
    with open(catalog_path, "r", newline="", encoding="utf-8") as fobj:
        reader = csv.DictReader(fobj)
        for row in reader:
            try:
                rank = int(row.get("rank", "0"))
            except ValueError:
                continue
            if rank != 1:
                continue

            region = row.get("region", "")
            if not region:
                continue

            out[region] = {
                "peak_ra": float(row["peak_ra_deg"]),
                "peak_dec": float(row["peak_dec_deg"]),
                "centroid_ra": float(row["centroid_ra_deg"]),
                "centroid_dec": float(row["centroid_dec_deg"]),
            }
    return out


def load_valid_points(fits_path: str, snr_min: float = 5.0):
    """Load NII-valid points (same filter as previous source step)."""
    with fits.open(fits_path) as hdul:
        table = hdul[1].data
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


def rasterize_flux_grid(ra: np.ndarray, dec: np.ndarray, flux: np.ndarray, n_grid: int = 220):
    """Rasterize to a regular grid using mean flux per occupied cell."""
    ra_min, ra_max = float(np.min(ra)), float(np.max(ra))
    dec_min, dec_max = float(np.min(dec)), float(np.max(dec))
    ra_span = max(ra_max - ra_min, 1e-12)
    dec_span = max(dec_max - dec_min, 1e-12)

    n_ra = n_grid
    n_dec = max(100, int(round(n_grid * dec_span / ra_span)))

    ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)

    sum_flux, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges], weights=flux)
    counts, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_flux = np.where(counts > 0, sum_flux / counts, np.nan)

    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    return mean_flux, counts, ra_edges, dec_edges, ra_centers, dec_centers


def polygon_area(vertices: np.ndarray):
    """Shoelace area in degree^2 for polygon vertices [x=RA, y=Dec]."""
    if vertices.shape[0] < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def smooth_field(field: np.ndarray, iterations: int = 2):
    """Simple 3x3 box smoothing (no SciPy dependency)."""
    out = field.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="edge")
        acc = np.zeros_like(out)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                acc += padded[1 + dr : 1 + dr + out.shape[0], 1 + dc : 1 + dc + out.shape[1]]
        out = acc / 9.0
    return out


def select_substructure_by_contours(
    mean_flux: np.ndarray,
    counts: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    src_ra: float,
    src_dec: float,
    percentile_levels=(99.5, 99.0, 98.5, 98.0, 97.0, 95.0, 93.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0),
    min_cells: int = 3,
):
    """Pick contour-delimited substructure containing source centroid.

    Returns a dict with selected cells and contour metadata, or None if no contour works.
    """
    valid = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
    if not np.any(valid):
        return None

    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
    log_flux[valid] = np.log10(mean_flux[valid])
    # Fill gaps with a low floor so contour polygons can still close on sparse tilings.
    contour_field = np.where(valid, log_flux, np.nanmin(log_flux[valid]) - 0.2)
    contour_field = smooth_field(contour_field, iterations=2)

    RA, DEC = np.meshgrid(ra_centers, dec_centers)

    for pct in percentile_levels:
        level = float(np.nanpercentile(log_flux[valid], pct))

        fig_tmp, ax_tmp = plt.subplots(figsize=(3, 3))
        cs = ax_tmp.contour(RA, DEC, contour_field, levels=[level])

        best = None
        if len(cs.allsegs) > 0:
            for verts in cs.allsegs[0]:
                if verts.shape[0] < 3:
                    continue

                path = Path(verts)
                if not path.contains_point((src_ra, src_dec)):
                    continue

                inside = path.contains_points(np.column_stack([RA.ravel(), DEC.ravel()])).reshape(mean_flux.shape)
                inside &= valid
                n_cells = int(np.sum(inside))
                if n_cells < min_cells:
                    continue

                area = polygon_area(verts)
                # Prefer the most localized contour (smallest area) at this level.
                candidate = {
                    "percentile": float(pct),
                    "level_log_flux": level,
                    "cells_mask": inside,
                    "n_cells": n_cells,
                    "area_deg2": area,
                    "verts": verts,
                }
                if best is None or candidate["area_deg2"] < best["area_deg2"]:
                    best = candidate

        plt.close(fig_tmp)

        if best is not None:
            rows, cols = np.where(best["cells_mask"])
            best["cell_ra"] = ra_centers[cols]
            best["cell_dec"] = dec_centers[rows]
            return best

    return None


def seed_component(mask: np.ndarray, r0: int, c0: int):
    """Return connected component containing seed using 8-neighborhood."""
    if r0 < 0 or r0 >= mask.shape[0] or c0 < 0 or c0 >= mask.shape[1]:
        return np.array([], dtype=int), np.array([], dtype=int)
    if not mask[r0, c0]:
        return np.array([], dtype=int), np.array([], dtype=int)

    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(r0, c0)])
    visited[r0, c0] = True
    rows = []
    cols = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while q:
        rr, cc = q.popleft()
        rows.append(rr)
        cols.append(cc)
        for dr, dc in neighbors:
            nr, nc = rr + dr, cc + dc
            if nr < 0 or nr >= mask.shape[0] or nc < 0 or nc >= mask.shape[1]:
                continue
            if visited[nr, nc] or not mask[nr, nc]:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))

    return np.array(rows, dtype=int), np.array(cols, dtype=int)


def select_substructure_by_threshold_component(
    mean_flux: np.ndarray,
    counts: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    src_ra: float,
    src_dec: float,
    percentile_levels=(99.5, 99.0, 98.0, 97.0, 95.0, 93.0, 90.0, 85.0, 80.0, 75.0, 70.0),
    min_cells: int = 3,
):
    """Fallback: choose isoflux connected component containing or nearest to centroid cell."""
    valid = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
    if not np.any(valid):
        return None

    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
    log_flux[valid] = np.log10(mean_flux[valid])

    c0 = int(np.clip(np.searchsorted(ra_centers, src_ra), 0, len(ra_centers) - 1))
    r0 = int(np.clip(np.searchsorted(dec_centers, src_dec), 0, len(dec_centers) - 1))

    for pct in percentile_levels:
        level = float(np.nanpercentile(log_flux[valid], pct))
        mask = valid & (log_flux >= level)
        rows, cols = seed_component(mask, r0, c0)
        if rows.size < min_cells:
            continue

        return {
            "percentile": float(pct),
            "level_log_flux": level,
            "cell_ra": ra_centers[cols],
            "cell_dec": dec_centers[rows],
            "n_cells": int(rows.size),
            "contains_centroid": True,
        }

    # Secondary fallback: choose nearest connected threshold component.
    for pct in percentile_levels:
        level = float(np.nanpercentile(log_flux[valid], pct))
        mask = valid & (log_flux >= level)
        comps = connected_components(mask)
        best = None
        for cells in comps:
            if len(cells) < min_cells:
                continue
            rows = np.array([rc[0] for rc in cells], dtype=int)
            cols = np.array([rc[1] for rc in cells], dtype=int)
            d2 = np.min((rows - r0) ** 2 + (cols - c0) ** 2)
            if best is None or d2 < best[0]:
                best = (d2, rows, cols)
        if best is not None:
            _, rows, cols = best
            return {
                "percentile": float(pct),
                "level_log_flux": level,
                "cell_ra": ra_centers[cols],
                "cell_dec": dec_centers[rows],
                "n_cells": int(rows.size),
                "contains_centroid": False,
            }

    return None


def connected_components(mask: np.ndarray):
    """Connected components with 8-connectivity."""
    visited = np.zeros_like(mask, dtype=bool)
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    comps = []
    n_rows, n_cols = mask.shape

    for r in range(n_rows):
        for c in range(n_cols):
            if not mask[r, c] or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            comps.append(cells)

    return comps


def dilate_mask(mask: np.ndarray, iterations: int = 1):
    """Binary dilation with 8-neighborhood, implemented without SciPy."""
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= padded[1 + dr : 1 + dr + out.shape[0], 1 + dc : 1 + dc + out.shape[1]]
        out = grown
    return out


def nearest_occupied_cell(r0: int, c0: int, occupied: np.ndarray):
    """Return nearest occupied cell index (rr, cc) using squared distance."""
    ys, xs = np.where(occupied)
    if ys.size == 0:
        return None
    d2 = (ys - r0) ** 2 + (xs - c0) ** 2
    k = int(np.argmin(d2))
    return int(ys[k]), int(xs[k])


def main():
    source_catalog = "Figures/bright_sources/nii_bright_source_catalog.csv"
    rank1 = load_rank1_sources(source_catalog)

    fits_files = sorted(glob.glob("LVM_*.fits"))
    if not fits_files:
        raise FileNotFoundError("No LVM_*.fits files found in current directory")

    os.makedirs("Figures/bright_sources", exist_ok=True)
    plot_dir = "Figures/manuscript/01_summary_regions/bright_sources"
    os.makedirs(plot_dir, exist_ok=True)

    results = []
    n_grid = 220

    for fits_path in fits_files:
        region = os.path.splitext(os.path.basename(fits_path))[0].replace("LVM_", "")
        src = rank1.get(region)
        if src is None:
            print(f"Skipping {region}: no rank-1 source in catalog")
            continue

        ra, dec, flux = load_valid_points(fits_path, snr_min=5.0)
        if ra.size == 0:
            print(f"Skipping {region}: no valid points")
            continue

        mean_flux, counts, ra_edges, dec_edges, ra_centers, dec_centers = rasterize_flux_grid(
            ra, dec, flux, n_grid=n_grid
        )

        contour_pick = select_substructure_by_contours(
            mean_flux,
            counts,
            ra_centers,
            dec_centers,
            src["centroid_ra"],
            src["centroid_dec"],
        )

        method = "contour"
        chosen_id = 1
        n_components_total = 1

        if contour_pick is not None:
            comp_ra = contour_pick["cell_ra"]
            comp_dec = contour_pick["cell_dec"]
            area_deg2 = float(contour_pick["area_deg2"])
            component_cells = int(contour_pick["n_cells"])
            contour_pct = float(contour_pick["percentile"])
            contour_level = float(contour_pick["level_log_flux"])
            contour_verts = contour_pick["verts"]
        else:
            thresh_pick = select_substructure_by_threshold_component(
                mean_flux,
                counts,
                ra_centers,
                dec_centers,
                src["centroid_ra"],
                src["centroid_dec"],
            )
            if thresh_pick is None:
                print(f"Skipping {region}: no threshold component encloses centroid")
                continue

            comp_ra = thresh_pick["cell_ra"]
            comp_dec = thresh_pick["cell_dec"]
            cell_area = (ra_edges[1] - ra_edges[0]) * (dec_edges[1] - dec_edges[0])
            area_deg2 = float(thresh_pick["n_cells"] * abs(cell_area))
            component_cells = int(thresh_pick["n_cells"])
            n_components_total = 1
            method = "fallback_threshold_component"
            if not bool(thresh_pick.get("contains_centroid", True)):
                method = "fallback_nearest_threshold_component"
            contour_pct = float(thresh_pick["percentile"])
            contour_level = float(thresh_pick["level_log_flux"])
            contour_verts = None

        results.append(
            {
                "region": region,
                "selection_method": method,
                "component_id": chosen_id,
                "n_components_total": n_components_total,
                "component_cells": component_cells,
                "component_area_deg2": area_deg2,
                "contour_percentile": contour_pct,
                "contour_level_log_flux": contour_level,
                "component_ra_min": float(np.min(comp_ra)),
                "component_ra_max": float(np.max(comp_ra)),
                "component_dec_min": float(np.min(comp_dec)),
                "component_dec_max": float(np.max(comp_dec)),
                "source_peak_ra": src["peak_ra"],
                "source_peak_dec": src["peak_dec"],
                "source_centroid_ra": src["centroid_ra"],
                "source_centroid_dec": src["centroid_dec"],
            }
        )

        # Diagnostic plot
        fig, ax = plt.subplots(figsize=(6.5, 5.9))
        ax.scatter(ra, dec, c=np.log10(flux), s=4, cmap="inferno", alpha=0.3, linewidths=0)
        ax.scatter(comp_ra, comp_dec, s=9, c="cyan", alpha=0.9, linewidths=0, label="occupied subregion")
        if contour_verts is not None:
            ax.plot(contour_verts[:, 0], contour_verts[:, 1], color="white", linewidth=1.6, alpha=0.9, label="selected contour")
        ax.scatter(src["peak_ra"], src["peak_dec"], marker="*", s=220, c="white", edgecolor="black", linewidth=0.9, zorder=10, label="source peak")
        ax.scatter(src["centroid_ra"], src["centroid_dec"], marker="x", s=90, c="deepskyblue", linewidth=2.0, zorder=10, label="source centroid")
        ax.invert_xaxis()
        if method == "contour":
            ax.set_title(
                f"{region}: contour-selected substructure (p={contour_pct:.1f})",
                fontweight="bold",
            )
        else:
            ax.set_title(f"{region}: fallback occupied subregion (ID {chosen_id})", fontweight="bold")
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        fig.tight_layout()
        out_plot = os.path.join(plot_dir, f"{region}_occupied_subregion.png")
        fig.savefig(out_plot, dpi=170)
        plt.close(fig)

        print(
            f"{region}: {method} selection with {component_cells} cells "
            f"(area={area_deg2:.4f} deg^2)"
        )

    if not results:
        print("No occupied subregions were selected.")
        return

    out_csv = "Figures/bright_sources/hii_occupied_subregions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved occupied-subregion catalog: {out_csv}")


if __name__ == "__main__":
    main()
