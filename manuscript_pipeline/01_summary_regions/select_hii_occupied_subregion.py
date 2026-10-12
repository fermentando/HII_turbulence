"""Select the connected subregion occupied by each ionizing source.

Inputs:
- LVM_*.fits files (same folder)
- ionizing_sources.xlsx (region-name source coordinates)

Outputs:
- Figures/bright_sources/hii_occupied_subregions.csv
- Figures/manuscript/01_summary_regions/bright_sources/<region>_occupied_subregion.png
"""

from __future__ import annotations

import csv
import glob
import os
from collections import deque
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib.path import Path


REGION_ALIASES = {
    "Flame_nebula": "Sh 2-277",
    "Gum38a": "NGC 3576",
    "Orion": "Sh 2-281",
    "Rosette_nebula": "Sh 2-275",
}

NEBULA_PERCENTILE_LEVELS = (95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0, 30.0)


def normalize_region_name(name: str):
    """Normalize object names for matching LVM filenames to workbook rows."""
    return "".join(ch.lower() for ch in name if ch.isalnum())


def xlsx_column_index(cell_ref: str):
    """Return zero-based column index from an Excel cell reference."""
    col = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        col = col * 26 + ord(ch.upper()) - ord("A") + 1
    return col - 1


def load_xlsx_rows(path: str):
    """Read the first worksheet of a simple XLSX file without extra dependencies."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ionizing-source workbook not found: {path}")

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                text = "".join(
                    node.text or ""
                    for node in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")
                )
                shared_strings.append(text)

        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            values = []
            for cell in row.findall("m:c", ns):
                idx = xlsx_column_index(cell.attrib["r"])
                while len(values) <= idx:
                    values.append("")

                val_node = cell.find("m:v", ns)
                if val_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(val_node.text)]
                else:
                    value = val_node.text or ""
                values[idx] = value.strip() if isinstance(value, str) else value
            rows.append(values)
    return rows


def parse_angle(value: str):
    """Parse a decimal degree string from the workbook."""
    cleaned = str(value).replace("\xa0", "").strip()
    if not cleaned:
        return None
    return float(cleaned)


def load_ionizing_sources(workbook_path: str):
    """Load ionizing-source centers keyed by normalized region name."""
    rows = load_xlsx_rows(workbook_path)
    if not rows:
        return {}

    headers = [str(item).strip().lower() for item in rows[0]]
    region_idx = headers.index("region")
    source_idx = headers.index("source")
    glon_idx = next(i for i, item in enumerate(headers) if "galactic l" in item)
    glat_idx = next(i for i, item in enumerate(headers) if "galactic b" in item)
    type_idx = headers.index("type")

    out = {}
    for row in rows[1:]:
        if len(row) <= max(region_idx, source_idx, glon_idx, glat_idx, type_idx):
            continue

        region = row[region_idx].strip()
        if not region:
            continue

        glon = parse_angle(row[glon_idx])
        glat = parse_angle(row[glat_idx])
        if glon is None or glat is None:
            continue

        coord = SkyCoord(l=glon * u.deg, b=glat * u.deg, frame="galactic").icrs
        out[normalize_region_name(region)] = {
            "catalog_region": region,
            "source": row[source_idx].strip(),
            "source_type": row[type_idx].strip(),
            "gal_l": float(glon),
            "gal_b": float(glat),
            "ra": float(coord.ra.deg),
            "dec": float(coord.dec.deg),
        }
    return out


def get_region_source(region: str, ionizing_sources: dict):
    """Return the ionizing-source center for an LVM region name."""
    lookup = REGION_ALIASES.get(region, region)
    candidates = [lookup]
    if lookup.startswith("Sh2-"):
        candidates.append(lookup.replace("Sh2-", "Sh 2-"))
    if lookup.startswith("Sh 2-"):
        candidates.append(lookup.replace("Sh 2-", "Sh2-"))

    for candidate in candidates:
        source = ionizing_sources.get(normalize_region_name(candidate))
        if source is not None:
            return source

    return None


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
    percentile_levels=NEBULA_PERCENTILE_LEVELS,
    min_cells: int = 25,
):
    """Pick a broad contour-delimited nebula containing the ionizing source.

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
    best_overall = None

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
                candidate = {
                    "percentile": float(pct),
                    "level_log_flux": level,
                    "cells_mask": inside,
                    "n_cells": n_cells,
                    "area_deg2": area,
                    "verts": verts,
                }
                # At a fixed level, use the smallest enclosing contour if nested contours exist.
                if best is None or candidate["area_deg2"] < best["area_deg2"]:
                    best = candidate

        plt.close(fig_tmp)

        if best is not None:
            # Across levels, prefer the broader nebula around the source, not the first bright knot.
            if best_overall is None or best["n_cells"] > best_overall["n_cells"]:
                best_overall = best

    if best_overall is None:
        return None

    rows, cols = np.where(best_overall["cells_mask"])
    best_overall["cell_ra"] = ra_centers[cols]
    best_overall["cell_dec"] = dec_centers[rows]
    return best_overall


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
    percentile_levels=NEBULA_PERCENTILE_LEVELS,
    min_cells: int = 25,
    bridge_iterations: int = 2,
):
    """Fallback: choose a broad isoflux component containing or nearest the source."""
    valid = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
    if not np.any(valid):
        return None

    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
    log_flux[valid] = np.log10(mean_flux[valid])

    c0 = int(np.clip(np.searchsorted(ra_centers, src_ra), 0, len(ra_centers) - 1))
    r0 = int(np.clip(np.searchsorted(dec_centers, src_dec), 0, len(dec_centers) - 1))
    best_containing = None

    for pct in percentile_levels:
        level = float(np.nanpercentile(log_flux[valid], pct))
        mask = dilate_mask(valid & (log_flux >= level), iterations=bridge_iterations) & valid
        rows, cols = seed_component(mask, r0, c0)
        if rows.size < min_cells:
            continue

        candidate = {
            "percentile": float(pct),
            "level_log_flux": level,
            "cell_ra": ra_centers[cols],
            "cell_dec": dec_centers[rows],
            "n_cells": int(rows.size),
            "contains_centroid": True,
        }
        if best_containing is None or candidate["n_cells"] > best_containing["n_cells"]:
            best_containing = candidate

    if best_containing is not None:
        return best_containing

    # Secondary fallback: choose nearest connected threshold component.
    best_nearest = None
    for pct in percentile_levels:
        level = float(np.nanpercentile(log_flux[valid], pct))
        mask = dilate_mask(valid & (log_flux >= level), iterations=bridge_iterations) & valid
        comps = connected_components(mask)
        nearest_at_level = None
        for cells in comps:
            if len(cells) < min_cells:
                continue
            rows = np.array([rc[0] for rc in cells], dtype=int)
            cols = np.array([rc[1] for rc in cells], dtype=int)
            d2 = np.min((rows - r0) ** 2 + (cols - c0) ** 2)
            if nearest_at_level is None or d2 < nearest_at_level[0]:
                nearest_at_level = (d2, rows, cols)
        if nearest_at_level is not None:
            d2, rows, cols = nearest_at_level
            candidate = {
                "percentile": float(pct),
                "level_log_flux": level,
                "cell_ra": ra_centers[cols],
                "cell_dec": dec_centers[rows],
                "n_cells": int(rows.size),
                "contains_centroid": False,
            }
            if (
                best_nearest is None
                or candidate["n_cells"] > best_nearest["n_cells"]
                or (candidate["n_cells"] == best_nearest["n_cells"] and d2 < best_nearest["distance2"])
            ):
                candidate["distance2"] = float(d2)
                best_nearest = candidate

    if best_nearest is not None:
        best_nearest.pop("distance2", None)
    return best_nearest


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


def select_source_aperture_nebula(
    mean_flux: np.ndarray,
    counts: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    src_ra: float,
    src_dec: float,
    min_cells: int = 100,
):
    """Select observed nebular cells in a growing aperture around the ionizing source."""
    valid = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
    if not np.any(valid):
        return None

    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
    log_flux[valid] = np.log10(mean_flux[valid])
    floor = float(np.nanpercentile(log_flux[valid], 30.0))
    RA, DEC = np.meshgrid(ra_centers, dec_centers)
    cos_dec = max(np.cos(np.deg2rad(src_dec)), 0.2)
    dist = np.hypot((RA - src_ra) * cos_dec, DEC - src_dec)
    field_radius = 0.5 * np.hypot(
        (float(np.max(ra_centers)) - float(np.min(ra_centers))) * cos_dec,
        float(np.max(dec_centers)) - float(np.min(dec_centers)),
    )

    best = None
    for radius in np.linspace(0.08 * field_radius, 0.35 * field_radius, 8):
        mask = valid & (dist <= radius) & (log_flux >= floor)
        n_cells = int(np.sum(mask))
        if n_cells == 0:
            continue
        candidate = {
            "percentile": 30.0,
            "level_log_flux": floor,
            "mask": mask,
            "radius_deg": float(radius),
            "n_cells": n_cells,
        }
        best = candidate
        if n_cells >= min_cells:
            break

    if best is None:
        return None

    rows, cols = np.where(best["mask"])
    best["cell_ra"] = ra_centers[cols]
    best["cell_dec"] = dec_centers[rows]
    return best


def main():
    source_catalog = "ionizing_sources.xlsx"
    ionizing_sources = load_ionizing_sources(source_catalog)

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
        src = get_region_source(region, ionizing_sources)
        if src is None:
            print(f"Skipping {region}: no ionizing-source coordinates in {source_catalog}")
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
            src["ra"],
            src["dec"],
        )
        thresh_pick = select_substructure_by_threshold_component(
            mean_flux,
            counts,
            ra_centers,
            dec_centers,
            src["ra"],
            src["dec"],
        )

        method = "contour"
        chosen_id = 1
        n_components_total = 1

        use_contour = contour_pick is not None and (
            thresh_pick is None or int(contour_pick["n_cells"]) >= int(thresh_pick["n_cells"])
        )

        if use_contour:
            comp_ra = contour_pick["cell_ra"]
            comp_dec = contour_pick["cell_dec"]
            area_deg2 = float(contour_pick["area_deg2"])
            component_cells = int(contour_pick["n_cells"])
            contour_pct = float(contour_pick["percentile"])
            contour_level = float(contour_pick["level_log_flux"])
            contour_verts = contour_pick["verts"]
        else:
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

        if component_cells < 100:
            aperture_pick = select_source_aperture_nebula(
                mean_flux,
                counts,
                ra_centers,
                dec_centers,
                src["ra"],
                src["dec"],
                min_cells=100,
            )
            if aperture_pick is not None and int(aperture_pick["n_cells"]) > component_cells:
                comp_ra = aperture_pick["cell_ra"]
                comp_dec = aperture_pick["cell_dec"]
                cell_area = (ra_edges[1] - ra_edges[0]) * (dec_edges[1] - dec_edges[0])
                area_deg2 = float(aperture_pick["n_cells"] * abs(cell_area))
                component_cells = int(aperture_pick["n_cells"])
                method = "source_aperture_nebula"
                contour_pct = float(aperture_pick["percentile"])
                contour_level = float(aperture_pick["level_log_flux"])
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
                "source_catalog_region": src["catalog_region"],
                "source_name": src["source"],
                "source_type": src["source_type"],
                "source_gal_l": src["gal_l"],
                "source_gal_b": src["gal_b"],
                "source_ra": src["ra"],
                "source_dec": src["dec"],
            }
        )

        # Diagnostic plot
        fig, ax = plt.subplots(figsize=(6.5, 5.9))
        ax.scatter(ra, dec, c=np.log10(flux), s=4, cmap="inferno", alpha=0.3, linewidths=0)
        ax.scatter(comp_ra, comp_dec, s=9, c="cyan", alpha=0.9, linewidths=0, label="occupied subregion")
        if contour_verts is not None:
            ax.plot(contour_verts[:, 0], contour_verts[:, 1], color="white", linewidth=1.6, alpha=0.9, label="selected contour")
        ax.scatter(
            src["ra"],
            src["dec"],
            marker="*",
            s=220,
            c="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=10,
            label="ionizing source",
        )
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
