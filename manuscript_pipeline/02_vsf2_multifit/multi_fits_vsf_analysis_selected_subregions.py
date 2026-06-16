"""Multi-FITS VSF analysis on selected subregions.

Subregions are defined by the fixed criterion:
log10(NII flux) > 2.8  (i.e., flux_nii > 10**2.8), with NII S/N > 5.

For each emission line (NII/SII/SIII), we compute full-FOV VSFs using only spaxels
that belong to this selected NII subregion and pass the line S/N filter.
"""

import glob
import os
import sys
import warnings
from pathlib import Path as FSPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.path import Path

warnings.filterwarnings("ignore")

CORE_DIR = FSPath(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D

plt.style.use("./custom_plot.mplstyle")

NPZ_OUTPUT_DIR = "NPZ"
VSF_ERROR_MODE = "bootstrap"
VSF_TEMP_OUTNAME = "/private/tmp/ignore_selected_subregions"
N_BOOTSTRAP = 100
MAX_REGIONS = int(sys.argv[1]) if len(sys.argv) > 1 else None
NII_LOGFLUX_THRESHOLD = 2.8
NII_FLUX_THRESHOLD = 10 ** NII_LOGFLUX_THRESHOLD
NII_LOGFLUX_THRESHOLD_HIGH = 3.2
SPECIAL_CENTER_REGIONS = {"Orion", "RCW52"}


def get_heliocentric_distance(filename):
    """Get heliocentric distance for an object from the HII regions table."""
    special_distances = {
        "Flame_nebula": 400.0,
        "Orion": 400.0,
        "Rosette_nebula": 1500.0,
    }

    base_name = os.path.splitext(os.path.basename(filename))[0]
    object_name = base_name.replace("LVM_", "")

    if object_name in special_distances:
        return special_distances[object_name]

    with fits.open("extra_HII_regions.fits") as hdul_dist:
        dist_data = hdul_dist[1].data

    name_mapping = {
        "Flame_nebula": "Flame",
        "Rosette_nebula": "Rosette",
    }
    lookup_name = name_mapping.get(object_name, object_name)

    try:
        index = list(dist_data["HII Region"]).index(lookup_name)
        distance_kpc = float(dist_data["Heliocentric_distance"][index])
        return distance_kpc * 1e3
    except ValueError:
        print(f"Warning: Distance not found for {object_name}, using default 1500 pc")
        return 1500.0


def get_available_lines(filename):
    """Get available emission lines in a FITS file."""
    with fits.open(filename) as hdul:
        cols = hdul[1].data.names
    lines = []
    if "vel_nii" in cols:
        lines.append("nii")
    if "vel_sii" in cols:
        lines.append("sii")
    if "vel_siii" in cols:
        lines.append("siii")
    return lines


def load_selected_subregion_line_data(filename, line):
    """Load line data restricted to selected NII subregions and line S/N > 5."""
    object_name = os.path.splitext(os.path.basename(filename))[0].replace("LVM_", "")
    local_log_threshold = NII_LOGFLUX_THRESHOLD_HIGH if object_name in SPECIAL_CENTER_REGIONS else NII_LOGFLUX_THRESHOLD
    local_flux_threshold = 10 ** local_log_threshold

    with fits.open(filename) as hdul:
        table_data = hdul[1].data

        flux_nii = np.asarray(table_data["flux_nii"], dtype=float)
        flux_nii_err = np.asarray(table_data["flux_nii_err"], dtype=float)

        # Base selection from NII substructures.
        valid_nii = np.isfinite(flux_nii) & np.isfinite(flux_nii_err) & (flux_nii_err > 0)
        nii_snr = np.zeros_like(flux_nii)
        nii_snr[valid_nii] = flux_nii[valid_nii] / flux_nii_err[valid_nii]
        selected_subregion = valid_nii & (nii_snr > 5) & (flux_nii > local_flux_threshold)

        # For Orion and RCW52: keep only the contour region centered on FoV center.
        if object_name in SPECIAL_CENTER_REGIONS and np.any(selected_subregion):
            ra_all = np.asarray(table_data["ra"], dtype=float)
            dec_all = np.asarray(table_data["dec"], dtype=float)

            # Build contouring field from all high-quality NII spaxels (not only thresholded ones).
            contour_valid = valid_nii & (nii_snr > 5) & np.isfinite(ra_all) & np.isfinite(dec_all) & (flux_nii > 0)
            if np.any(contour_valid):
                ra_c = ra_all[contour_valid]
                dec_c = dec_all[contour_valid]
                flux_c = flux_nii[contour_valid]

                # Rasterize mean flux field.
                ra_min, ra_max = float(np.min(ra_c)), float(np.max(ra_c))
                dec_min, dec_max = float(np.min(dec_c)), float(np.max(dec_c))
                ra_span = max(ra_max - ra_min, 1e-12)
                dec_span = max(dec_max - dec_min, 1e-12)
                n_ra = 220
                n_dec = max(100, int(round(n_ra * dec_span / ra_span)))
                ra_edges = np.linspace(ra_min, ra_max, n_ra + 1)
                dec_edges = np.linspace(dec_min, dec_max, n_dec + 1)
                sum_flux, _, _ = np.histogram2d(dec_c, ra_c, bins=[dec_edges, ra_edges], weights=flux_c)
                counts_c, _, _ = np.histogram2d(dec_c, ra_c, bins=[dec_edges, ra_edges])
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_flux = np.where(counts_c > 0, sum_flux / counts_c, np.nan)

                valid_grid = np.isfinite(mean_flux) & (mean_flux > 0) & (counts_c > 0)
                if np.any(valid_grid):
                    log_flux = np.full_like(mean_flux, np.nan, dtype=float)
                    log_flux[valid_grid] = np.log10(mean_flux[valid_grid])
                    fill_val = np.nanmin(log_flux[valid_grid]) - 0.2
                    contour_field = np.where(valid_grid, log_flux, fill_val)

                    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
                    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
                    RA, DEC = np.meshgrid(ra_centers, dec_centers)

                    fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
                    cs = ax_tmp.contour(RA, DEC, contour_field, levels=[local_log_threshold])
                    polygons = [seg for seg in cs.allsegs[0] if seg.shape[0] >= 3] if len(cs.allsegs) > 0 else []
                    plt.close(fig_tmp)

                    if polygons:
                        target_ra = 0.5 * (ra_min + ra_max)
                        target_dec = 0.5 * (dec_min + dec_max)
                        chosen = None
                        for verts in polygons:
                            if Path(verts).contains_point((target_ra, target_dec)):
                                area = 0.5 * abs(np.dot(verts[:, 0], np.roll(verts[:, 1], -1)) - np.dot(verts[:, 1], np.roll(verts[:, 0], -1)))
                                if chosen is None or area < chosen[0]:
                                    chosen = (area, verts)

                        if chosen is None:
                            best_d2 = None
                            best_verts = None
                            for verts in polygons:
                                cx = float(np.mean(verts[:, 0]))
                                cy = float(np.mean(verts[:, 1]))
                                d2 = (cx - target_ra) ** 2 + (cy - target_dec) ** 2
                                if best_d2 is None or d2 < best_d2:
                                    best_d2 = d2
                                    best_verts = verts
                            if best_verts is not None:
                                chosen = (0.0, best_verts)

                        if chosen is not None:
                            chosen_path = Path(chosen[1])
                            inside = chosen_path.contains_points(np.column_stack([ra_all, dec_all]))
                            selected_subregion &= inside

        flux_col = f"flux_{line}"
        flux_err_col = f"flux_{line}_err"
        vel_col = f"vel_{line}"
        vel_err_col = f"vel_{line}_err"

        flux_line = np.asarray(table_data[flux_col], dtype=float)
        flux_line_err = np.asarray(table_data[flux_err_col], dtype=float)
        ra_all = np.asarray(table_data["ra"], dtype=float)
        dec_all = np.asarray(table_data["dec"], dtype=float)
        vel_all = np.asarray(table_data[vel_col], dtype=float)
        vel_err_all = np.asarray(table_data[vel_err_col], dtype=float)

        valid_line = np.isfinite(flux_line) & np.isfinite(flux_line_err) & (flux_line_err > 0)
        line_snr = np.zeros_like(flux_line)
        line_snr[valid_line] = flux_line[valid_line] / flux_line_err[valid_line]

        mask = selected_subregion & valid_line & (line_snr > 5)
        mask &= np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(vel_all) & np.isfinite(vel_err_all)

        return (
            ra_all[mask],
            dec_all[mask],
            flux_line[mask],
            vel_all[mask],
            vel_err_all[mask],
            int(np.sum(selected_subregion)),
            local_log_threshold,
        )


def plot_vsf_with_errorbars(ax, rvals, vsf, vsf_err, color, label, marker="o", linestyle="-", linewidth=2.2):
    """Plot VSF curve, uncertainty band, and sampled error bars."""
    ax.plot(rvals, vsf, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    ax.fill_between(rvals, np.maximum(vsf - vsf_err, 1e0), vsf + vsf_err, color=color, alpha=0.18)

    n_err = min(12, len(rvals))
    err_idx = np.unique(np.linspace(0, len(rvals) - 1, n_err, dtype=int))
    yerr_low = np.minimum(vsf_err[err_idx], np.maximum(vsf[err_idx] - 1e-12, 0.0))
    yerr = np.vstack([yerr_low, vsf_err[err_idx]])

    ax.errorbar(
        rvals[err_idx],
        vsf[err_idx],
        yerr=yerr,
        fmt=marker,
        ms=3,
        capsize=2,
        elinewidth=1,
        color=color,
        alpha=0.9,
    )
    ax.set_ylim(bottom=1)


def finite_positive_max(values):
    """Return max finite positive value, or nan if none."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return np.nan
    return float(np.max(arr))


print("=" * 90)
print(
    "MULTI-FITS VSF ANALYSIS - SELECTED SUBREGIONS "
    f"(default log10 NII flux > {NII_LOGFLUX_THRESHOLD:.1f}; "
    f"Orion/RCW52 center contour > {NII_LOGFLUX_THRESHOLD_HIGH:.1f})"
)
print("=" * 90)
print()

fits_files = sorted(glob.glob("LVM_*.fits"))
files_to_process = fits_files[:MAX_REGIONS] if MAX_REGIONS is not None else fits_files

print(f"Found {len(fits_files)} FITS files")
for f in files_to_process:
    print(f"  {f}")
print()

print(f"Total files to process: {len(files_to_process)}")
for filename in files_to_process:
    base = os.path.splitext(os.path.basename(filename))[0]
    lines = get_available_lines(filename)
    print(f"  {base}: {', '.join(l.upper() for l in lines)}")
print()

os.makedirs("Figures", exist_ok=True)
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

line_colors = {"nii": "blue", "sii": "red", "siii": "green"}
line_names = {"nii": "NII", "sii": "SII", "siii": "SIII"}

n_cols = 3
n_rows = int(np.ceil(len(files_to_process) / n_cols)) if files_to_process else 1
fig_all, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)
fig_disp, axes_disp = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.2 * n_rows), squeeze=False)

plot_idx = 0
for filename in files_to_process:
    print(f"Processing {os.path.basename(filename)}...")

    base_name = os.path.splitext(os.path.basename(filename))[0]
    object_name = base_name.replace("LVM_", "")
    available_lines = get_available_lines(filename)
    distance = get_heliocentric_distance(filename)
    print(f"  Distance: {distance:.1f} pc")

    row = plot_idx // n_cols
    col = plot_idx % n_cols
    ax_main = axes[row, col]
    ax_disp = axes_disp[row, col]
    region_y_max = np.nan
    region_disp_y_max = np.nan
    n_selected_subregion = None
    region_log_threshold = NII_LOGFLUX_THRESHOLD

    for line in available_lines:
        try:
            ra, dec, flux, vel, vel_err, n_selected, local_log_threshold = load_selected_subregion_line_data(filename, line)
            if n_selected_subregion is None:
                n_selected_subregion = n_selected
            region_log_threshold = local_log_threshold
            print(f"  {line.upper()}: {len(ra):,} points in selected subregions")

            if len(ra) < 50:
                print(f"    Skipping {line.upper()}: too few points after subregion selection")
                continue

            vsf_total, vsf_total_err, log_centers_total, counts_total = VSF2D(
                ra, dec, vel, dist=distance, vel_err=vel_err
            ).generate_vsf_2d(
                order=2,
                outname=VSF_TEMP_OUTNAME,
                n_jobs=4,
                min_pairs=3,
                bin_resolution=20,
                error_mode=VSF_ERROR_MODE,
                n_bootstrap=N_BOOTSTRAP,
            )
            with np.load(VSF_TEMP_OUTNAME + "_order2.npz", allow_pickle=True) as vsf_diagnostics:
                vsf_std_total = np.asarray(vsf_diagnostics["vsf_std"], dtype=float)
                vsf_sem_total = np.asarray(vsf_diagnostics["vsf_sem"], dtype=float)
                vsf_propagated_total = np.asarray(vsf_diagnostics["vsf_propagated_mean_err"], dtype=float)
                vsf_bootstrap_total = np.asarray(vsf_diagnostics["vsf_bootstrap_err"], dtype=float)

            valid = np.isfinite(vsf_total) & np.isfinite(vsf_total_err) & (counts_total > 0)
            if not np.any(valid):
                print(f"    No valid VSF bins for {line.upper()} in selected subregions")
                continue

            r_total = 10 ** log_centers_total[valid]
            vsf_total = vsf_total[valid]
            vsf_total_err = vsf_total_err[valid]
            vsf_std_total = vsf_std_total[valid]
            vsf_sem_total = vsf_sem_total[valid]
            vsf_propagated_total = vsf_propagated_total[valid]
            vsf_bootstrap_total = vsf_bootstrap_total[valid]
            counts_total = counts_total[valid]
            with np.errstate(divide="ignore", invalid="ignore"):
                vsf_dispersion_over_mean = vsf_std_total / vsf_total
                vsf_sem_over_mean = vsf_sem_total / vsf_total
                vsf_propagated_over_mean = vsf_propagated_total / vsf_total
                vsf_bootstrap_over_mean = vsf_bootstrap_total / vsf_total

            npz_out = os.path.join(
                NPZ_OUTPUT_DIR,
                f"VSF_{object_name}_{line.upper()}_selected_subregions_logflux{region_log_threshold:.1f}.npz",
            )
            np.savez(
                npz_out,
                vsf=vsf_total,
                vsf_err=vsf_total_err,
                vsf_std=vsf_std_total,
                vsf_sem=vsf_sem_total,
                vsf_propagated_mean_err=vsf_propagated_total,
                vsf_bootstrap_err=vsf_bootstrap_total,
                vsf_dispersion_over_mean=vsf_dispersion_over_mean,
                vsf_sem_over_mean=vsf_sem_over_mean,
                vsf_propagated_over_mean=vsf_propagated_over_mean,
                vsf_bootstrap_over_mean=vsf_bootstrap_over_mean,
                error_mode=VSF_ERROR_MODE,
                n_bootstrap=N_BOOTSTRAP,
                log_centers=np.log10(r_total),
                counts=counts_total,
            )
            print(f"    Saved {npz_out}")

            color = line_colors[line]
            plot_vsf_with_errorbars(
                ax_main,
                r_total,
                vsf_total,
                vsf_total_err,
                color,
                f"{line_names[line]} (selected subregions)",
                marker="s",
                linestyle="-",
                linewidth=2.3,
            )

            total_max = finite_positive_max(vsf_total + vsf_total_err)
            if np.isfinite(total_max):
                region_y_max = total_max if not np.isfinite(region_y_max) else max(region_y_max, total_max)

            positive_disp = vsf_dispersion_over_mean[
                np.isfinite(vsf_dispersion_over_mean) & (vsf_dispersion_over_mean > 0)
            ]
            if positive_disp.size > 0:
                ax_disp.plot(
                    r_total,
                    vsf_dispersion_over_mean,
                    color=color,
                    linewidth=2.0,
                    marker="s",
                    ms=3,
                    label=line_names[line],
                )
                disp_max = float(np.nanmax(positive_disp))
                region_disp_y_max = (
                    disp_max if not np.isfinite(region_disp_y_max) else max(region_disp_y_max, disp_max)
                )

        except Exception as exc:
            print(f"  Error processing {line.upper()}: {exc}")
            continue

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel(r"$l_{3D}$ [pc]", fontsize=12)
    if col == 0:
        ax_main.set_ylabel(r"$\bar{S}_\mathrm{2} = \langle | v(r + \ell) - v(r) | \rangle ^{2}$ (km/s)$^2$", fontsize=12)
    if np.isfinite(region_y_max):
        ax_main.set_ylim(bottom=1, top=max(1.2, 1.25 * region_y_max))
    else:
        ax_main.set_ylim(bottom=1)
    ax_main.grid(True, alpha=0.3)

    ax_disp.set_xscale("log")
    ax_disp.set_yscale("log")
    ax_disp.set_xlabel(r"$l_{3D}$ [pc]", fontsize=12)
    if col == 0:
        ax_disp.set_ylabel(r"$\sigma_\mathrm{pair}(S_2) / \bar{S}_2$", fontsize=12)
    if np.isfinite(region_disp_y_max):
        ax_disp.set_ylim(bottom=1e-3, top=max(1.1e-3, 1.25 * region_disp_y_max))
    else:
        ax_disp.set_ylim(bottom=1e-3)
    ax_disp.grid(True, alpha=0.3)

    n_selected_subregion = 0 if n_selected_subregion is None else n_selected_subregion
    ax_main.set_title(
        f"{object_name} (selected NII subregions)\n"
        f"NII-selected spaxels: {n_selected_subregion:,} | log10>{region_log_threshold:.1f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_disp.set_title(
        f"{object_name}: intrinsic pair dispersion / VSF mean",
        fontsize=11,
        fontweight="bold",
    )

    if ax_main.has_data():
        ax_main.legend(fontsize=7, loc="best")
    if ax_disp.has_data():
        ax_disp.legend(fontsize=7, loc="best")

    plot_idx += 1

for idx in range(plot_idx, n_rows * n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].set_visible(False)
    axes_disp[r, c].set_visible(False)

fig_all.tight_layout()
fig_disp.tight_layout()
figure_dir = "Figures/manuscript/02_vsf2_multifit"
os.makedirs(figure_dir, exist_ok=True)
multi_out = os.path.join(figure_dir, "multi_fits_vsf_absolute_selected_subregions_mixed_thresholds.png")
fig_all.savefig(multi_out, dpi=150, bbox_inches="tight")
disp_out = os.path.join(figure_dir, "multi_fits_vsf_dispersion_over_mean_selected_subregions_mixed_thresholds.png")
fig_disp.savefig(disp_out, dpi=150, bbox_inches="tight")
plt.close(fig_all)
plt.close(fig_disp)

print()
print(f"Saved global figure as {multi_out}")
print(f"Saved dispersion diagnostic as {disp_out}")
print(f"Saved VSF data in {NPZ_OUTPUT_DIR}/ for selected subregions")
print()

print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Processed {len(files_to_process)} FITS files")
print(f"Created a {n_rows}x{n_cols} global subplot figure (one region per subplot)")
print(
    "Each subplot shows VSFs using selected NII subregions "
    f"(default log10>{NII_LOGFLUX_THRESHOLD:.1f}; Orion/RCW52 center contour log10>{NII_LOGFLUX_THRESHOLD_HIGH:.1f})"
)
print("=" * 90)
