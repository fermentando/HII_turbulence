"""
Multi-FITS VSF Analysis (Full FOV)
Computes one VSF per line using the full field of view (no subwindow aggregation).
"""

import glob
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

warnings.filterwarnings('ignore')

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D

plt.style.use('./custom_plot.mplstyle')

NPZ_OUTPUT_DIR = 'NPZ'
VSF_ERROR_MODE = 'bootstrap'
MIN_SEPARATION_ARCSEC = 37.0
MAX_REGIONS = int(sys.argv[1]) if len(sys.argv) > 1 else None


def arcsec_to_pc(angle_arcsec, distance_pc):
    """Convert an angular separation at distance_pc to a physical separation."""
    return distance_pc * np.deg2rad(angle_arcsec / 3600.0)


def get_heliocentric_distance(filename):
    """Get heliocentric distance for an object from the HII regions table."""
    special_distances = {
        'Flame_nebula': 400.0,
        'Orion': 400.0,
        'Rosette_nebula': 1500.0,
    }

    base_name = os.path.splitext(os.path.basename(filename))[0]
    object_name = base_name.replace('LVM_', '')

    if object_name in special_distances:
        return special_distances[object_name]

    hdul_dist = fits.open('extra_HII_regions.fits')
    dist_data = hdul_dist[1].data
    hdul_dist.close()

    name_mapping = {
        'Flame_nebula': 'Flame',
        'Rosette_nebula': 'Rosette',
    }
    lookup_name = name_mapping.get(object_name, object_name)

    try:
        index = list(dist_data['HII Region']).index(lookup_name)
        distance_kpc = float(dist_data['Heliocentric_distance'][index])
        return distance_kpc * 1e3
    except ValueError:
        print(f"Warning: Distance not found for {object_name}, using default 1500 pc")
        return 1500.0


def get_available_lines(filename):
    """Get available emission lines in a FITS file."""
    hdul = fits.open(filename)
    cols = hdul[1].data.names
    lines = []
    if 'vel_nii' in cols:
        lines.append('nii')
    if 'vel_sii' in cols:
        lines.append('sii')
    if 'vel_siii' in cols:
        lines.append('siii')
    hdul.close()
    return lines


def load_and_filter_data(filename, line):
    """Load and filter data for a specific line."""
    hdul = fits.open(filename)
    table_data = hdul[1].data

    flux_col = f'flux_{line}'
    flux_err_col = f'flux_{line}_err'
    vel_col = f'vel_{line}'
    vel_err_col = f'vel_{line}_err'

    filter_data = table_data[flux_col] / table_data[flux_err_col] > 5

    ra = table_data['ra'][filter_data]
    dec = table_data['dec'][filter_data]
    flux = table_data[flux_col][filter_data]
    vel = table_data[vel_col][filter_data]
    vel_err = table_data[vel_err_col][filter_data]

    hdul.close()
    return ra, dec, flux, vel, vel_err


def plot_vsf_with_errorbars(ax, rvals, vsf, vsf_err, color, label,
                            marker='o', linestyle='-', linewidth=2.2):
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


print('=' * 80)
print('MULTI-FITS VSF ANALYSIS - FULL FOV (No Subwindow Median)')
print('=' * 80)
print()

fits_files = sorted(glob.glob('LVM_*.fits'))
files_to_process = [f for f in fits_files if 'Orion' not in os.path.basename(f)]
if MAX_REGIONS is not None:
    files_to_process = files_to_process[:MAX_REGIONS]

print(f"Found {len(fits_files)} FITS files, skipping Orion")
for f in files_to_process:
    print(f"  {f}")
print()

print(f"Total files to process: {len(files_to_process)}")
for filename in files_to_process:
    base = os.path.splitext(os.path.basename(filename))[0]
    lines = get_available_lines(filename)
    print(f"  {base}: {', '.join(l.upper() for l in lines)}")
print()

os.makedirs('Figures', exist_ok=True)
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

line_colors = {'nii': 'blue', 'sii': 'red', 'siii': 'green'}
line_names = {'nii': 'NII', 'sii': 'SII', 'siii': 'SIII'}

n_cols = 3
n_rows = int(np.ceil(len(files_to_process) / n_cols)) if files_to_process else 1
fig_all, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)

plot_idx = 0
for filename in files_to_process:
    print(f"Processing {os.path.basename(filename)}...")

    base_name = os.path.splitext(os.path.basename(filename))[0]
    object_name = base_name.replace('LVM_', '')
    available_lines = get_available_lines(filename)
    distance = get_heliocentric_distance(filename)
    print(f"  Distance: {distance:.1f} pc")
    min_separation_pc = arcsec_to_pc(MIN_SEPARATION_ARCSEC, distance)
    print(f"  Minimum separation: {MIN_SEPARATION_ARCSEC:.1f} arcsec = {min_separation_pc:.4f} pc")

    row = plot_idx // n_cols
    col = plot_idx % n_cols
    ax_main = axes[row, col]
    region_y_max = np.nan

    for line in available_lines:
        try:
            ra, dec, flux, vel, vel_err = load_and_filter_data(filename, line)
            print(f"  {line.upper()}: {len(ra):,} points")

            vsf_total, vsf_total_err, log_centers_total, counts_total = VSF2D(
                ra, dec, vel, dist=distance, vel_err=vel_err
            ).generate_vsf_2d(
                order=2,
                outname='/private/tmp/ignore_fullfov',
                n_jobs=4,
                min_pairs=3,
                bin_resolution=20,
                error_mode=VSF_ERROR_MODE,
                min_separation=min_separation_pc,
            )

            valid = np.isfinite(vsf_total) & np.isfinite(vsf_total_err) & (counts_total > 0)
            if not np.any(valid):
                print(f"    No valid full-FOV VSF bins for {line.upper()}")
                continue

            r_total = 10**log_centers_total[valid]
            vsf_total = vsf_total[valid]
            vsf_total_err = vsf_total_err[valid]
            counts_total = counts_total[valid]

            npz_out = os.path.join(
                NPZ_OUTPUT_DIR,
                f'VSF_{object_name}_{line.upper()}_total_fullfov.npz'
            )
            np.savez(
                npz_out,
                vsf=vsf_total,
                vsf_err=vsf_total_err,
                log_centers=np.log10(r_total),
                counts=counts_total,
                error_mode=VSF_ERROR_MODE,
                min_separation_arcsec=MIN_SEPARATION_ARCSEC,
                min_separation_pc=min_separation_pc,
            )
            print(f"    Saved {npz_out}")

            color = line_colors[line]
            plot_vsf_with_errorbars(
                ax_main,
                r_total,
                vsf_total,
                vsf_total_err,
                color,
                f"{line_names[line]} total (full FOV)",
                marker='s',
                linestyle='-',
                linewidth=2.3,
            )

            total_max = finite_positive_max(vsf_total + vsf_total_err)
            if np.isfinite(total_max):
                region_y_max = total_max if not np.isfinite(region_y_max) else max(region_y_max, total_max)

        except Exception as exc:
            print(f"  Error processing {line.upper()}: {exc}")
            continue

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlabel(r'$l_{3D}$ [pc]', fontsize=12)
    if col == 0:
        ax_main.set_ylabel(r'$\bar{S}_\mathrm{2} = \langle | v(r + \ell) - v(r) | \rangle ^{2}$ (km/s)$^2$', fontsize=12)
    if np.isfinite(region_y_max):
        ax_main.set_ylim(bottom=1, top=max(1.2, 1.25 * region_y_max))
    else:
        ax_main.set_ylim(bottom=1)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title(f"{object_name} (full FOV)", fontsize=12, fontweight='bold')
    if row == n_rows - 1 and col == n_cols - 1:
        ax_main.legend(fontsize=7, loc='best')

    plot_idx += 1

for idx in range(plot_idx, n_rows * n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].set_visible(False)

fig_all.tight_layout()
figure_dir = 'Figures/manuscript/02_vsf2_multifit'
os.makedirs(figure_dir, exist_ok=True)
multi_out = os.path.join(figure_dir, 'multi_fits_vsf_absolute_fullfov.png')
fig_all.savefig(multi_out, dpi=150, bbox_inches='tight')
plt.close(fig_all)

print()
print(f"Saved global figure as {multi_out}")
print(f"Saved full-FOV VSF data in {NPZ_OUTPUT_DIR}/ as VSF_{{object}}_{{line}}_total_fullfov.npz")
print()

print('=' * 80)
print('SUMMARY')
print('=' * 80)
print(f"Processed {len(files_to_process)} FITS files")
print(f"Created a {n_rows}x{n_cols} global subplot figure (one region per subplot)")
print('Each subplot shows only full-FOV total VSFs (no subwindow mean/median)')
print('=' * 80)
