"""Create a multi-panel NII emission flux map for all subregion FITS files.

This script scans for ``LVM_*.fits`` files in the current directory,
filters spaxels by S/N > 5 in NII, and builds a single figure containing one
panel per subregion.
"""

import glob
import math
import os
import csv
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.path import Path

CONTOUR_LOGFLUX_THRESHOLD = 2.8
HIGH_LOGFLUX_THRESHOLD = 3.2


def load_nii_map_data(fits_path: str):
    """Return RA, Dec, and NII flux arrays after S/N filtering."""
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
    valid &= flux_err > 0
    snr = np.zeros_like(flux)
    snr[valid] = flux[valid] / flux_err[valid]

    mask = valid & (snr > 5) & (flux > 0)
    return ra[mask], dec[mask], flux[mask]


def load_rank1_sources(catalog_path: str):
    """Load rank-1 source rows indexed by region name."""
    if not os.path.exists(catalog_path):
        return {}

    by_region = {}
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
            by_region[region] = {
                "peak_ra": float(row["peak_ra_deg"]),
                "peak_dec": float(row["peak_dec_deg"]),
                "centroid_ra": float(row["centroid_ra_deg"]),
                "centroid_dec": float(row["centroid_dec_deg"]),
            }
    return by_region


def rasterize_flux_grid(ra: np.ndarray, dec: np.ndarray, flux: np.ndarray, n_grid: int = 220):
    """Rasterize irregular points to a regular mean-flux grid."""
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
    return mean_flux, counts, ra_centers, dec_centers


def polygon_area(vertices: np.ndarray):
    """Shoelace polygon area in deg^2."""
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


def dilate_mask(mask: np.ndarray, iterations: int = 1):
    """Binary dilation with 8-neighborhood without external dependencies."""
    out = mask.copy()
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= padded[1 + dr : 1 + dr + out.shape[0], 1 + dc : 1 + dc + out.shape[1]]
        out = grown
    return out


def connected_components(mask: np.ndarray):
    """Connected-component labeling using 8-connectivity."""
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
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


def nearest_occupied_cell(r0: int, c0: int, occupied: np.ndarray):
    """Get nearest occupied grid cell to requested (row, col)."""
    ys, xs = np.where(occupied)
    if ys.size == 0:
        return None
    d2 = (ys - r0) ** 2 + (xs - c0) ** 2
    k = int(np.argmin(d2))
    return int(ys[k]), int(xs[k])


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


def select_structures_fixed_contour(
    ra: np.ndarray,
    dec: np.ndarray,
    flux: np.ndarray,
    level_logflux: float = CONTOUR_LOGFLUX_THRESHOLD,
    central_only: bool = False,
    src_ra: float | None = None,
    src_dec: float | None = None,
    center_mode: bool = False,
):
    """Return cells and contour polygons for structures above a fixed log10(NII flux) threshold.

    If ``central_only`` is True, select only one contour substructure.
    - default: use source centroid (src_ra, src_dec)
    - if center_mode=True: use FoV geometric center
    """
    mean_flux, counts, ra_centers, dec_centers = rasterize_flux_grid(ra, dec, flux, n_grid=220)
    valid = np.isfinite(mean_flux) & (counts > 0) & (mean_flux > 0)
    if not np.any(valid):
        return np.array([]), np.array([]), []

    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
    log_flux[valid] = np.log10(mean_flux[valid])
    contour_field = np.where(valid, log_flux, np.nanmin(log_flux[valid]) - 0.2)
    contour_field = smooth_field(contour_field, iterations=2)

    RA, DEC = np.meshgrid(ra_centers, dec_centers)
    mask = valid & (log_flux >= level_logflux)
    rows, cols = np.where(mask)

    fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
    cs = ax_tmp.contour(RA, DEC, contour_field, levels=[level_logflux])
    contour_polygons = []
    if len(cs.allsegs) > 0:
        for verts in cs.allsegs[0]:
            if verts.shape[0] >= 3:
                contour_polygons.append(verts)
    plt.close(fig_tmp)

    if central_only and contour_polygons:
        if center_mode:
            target_ra = 0.5 * (float(np.min(ra)) + float(np.max(ra)))
            target_dec = 0.5 * (float(np.min(dec)) + float(np.max(dec)))
        else:
            if src_ra is None or src_dec is None:
                target_ra = 0.5 * (float(np.min(ra)) + float(np.max(ra)))
                target_dec = 0.5 * (float(np.min(dec)) + float(np.max(dec)))
            else:
                target_ra = src_ra
                target_dec = src_dec

        # Prefer contour that contains target point. If none contain it, use nearest contour centroid.
        chosen = None
        for verts in contour_polygons:
            if Path(verts).contains_point((target_ra, target_dec)):
                area = polygon_area(verts)
                if chosen is None or area < chosen[0]:
                    chosen = (area, verts)

        if chosen is None:
            best_dist = None
            best_verts = None
            for verts in contour_polygons:
                cx = float(np.mean(verts[:, 0]))
                cy = float(np.mean(verts[:, 1]))
                d2 = (cx - target_ra) ** 2 + (cy - target_dec) ** 2
                if best_dist is None or d2 < best_dist:
                    best_dist = d2
                    best_verts = verts
            if best_verts is not None:
                chosen = (polygon_area(best_verts), best_verts)

        if chosen is not None:
            verts = chosen[1]
            inside = Path(verts).contains_points(np.column_stack([RA.ravel(), DEC.ravel()])).reshape(mask.shape)
            inside &= mask
            r_sel, c_sel = np.where(inside)
            return ra_centers[c_sel], dec_centers[r_sel], [verts]

    return ra_centers[cols], dec_centers[rows], contour_polygons


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
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"Skipping {os.path.basename(path)} due to error: {exc}")

    if not region_data:
        raise RuntimeError("No valid subregion data available for NII plotting.")

    source_catalog = "Figures/bright_sources/nii_bright_source_catalog.csv"
    rank1_sources = load_rank1_sources(source_catalog)

    # Use common color scaling across all subregions in log10(flux).
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

        log_flux = np.log10(flux)
        scatter_for_cbar = ax.scatter(
            ra,
            dec,
            c=log_flux,
            s=7,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=0.95,
        )

        base = os.path.splitext(os.path.basename(path))[0].replace("LVM_", "")
        ax.set_title(base, fontsize=11, fontweight="bold")

        src = rank1_sources.get(base)
        special_center_regions = {"Orion", "RCW52"}
        central_only = base in special_center_regions
        local_threshold = HIGH_LOGFLUX_THRESHOLD if base in special_center_regions else CONTOUR_LOGFLUX_THRESHOLD
        occ_ra, occ_dec, contour_polys = select_structures_fixed_contour(
            ra,
            dec,
            flux,
            level_logflux=local_threshold,
            central_only=central_only,
            src_ra=(src["centroid_ra"] if src is not None else None),
            src_dec=(src["centroid_dec"] if src is not None else None),
            center_mode=(base in special_center_regions),
        )
        if occ_ra.size > 0:
            ax.scatter(
                occ_ra,
                occ_dec,
                marker="s",
                s=2.0,
                c="cyan",
                alpha=0.35,
                linewidths=0,
                zorder=8,
                label=f"log flux > {local_threshold:.1f}",
            )
        for i_poly, verts in enumerate(contour_polys):
            ax.plot(
                verts[:, 0],
                verts[:, 1],
                color="white",
                linewidth=1.2,
                alpha=0.95,
                zorder=9,
                label="contour" if i_poly == 0 else None,
            )

        if src is not None:
            ax.scatter(
                src["peak_ra"],
                src["peak_dec"],
                marker="*",
                s=180,
                c="white",
                edgecolor="black",
                linewidth=0.9,
                zorder=10,
                label="peak",
            )
            ax.scatter(
                src["centroid_ra"],
                src["centroid_dec"],
                marker="x",
                s=80,
                c="deepskyblue",
                linewidth=1.8,
                zorder=10,
                label="centroid",
            )
            ax.text(
                src["peak_ra"],
                src["peak_dec"],
                " S1",
                color="white",
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=11,
            )

        ax.set_xlabel("RA (deg)")
        if col == 0:
            ax.set_ylabel("Dec (deg)")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        if src is not None or occ_ra.size > 0:
            ax.legend(loc="upper right", fontsize=7, frameon=True)

    # Hide empty panels.
    for idx in range(n_regions, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    if scatter_for_cbar is not None:
        cbar = fig.colorbar(scatter_for_cbar, ax=axes.ravel().tolist(), shrink=0.95, pad=0.01)
        cbar.set_label(r"$\log_{10}(\mathrm{NII\ flux})$")

    fig.suptitle(
        f"NII Emission Flux Maps (S/N > 5, contours at log10 flux > {CONTOUR_LOGFLUX_THRESHOLD:.1f})",
        fontsize=15,
        fontweight="bold",
    )

    os.makedirs("Figures", exist_ok=True)
    output_dir = "Figures/manuscript/01_summary_regions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nii_flux_maps_all_subregions_with_sources_and_subregions.png")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
