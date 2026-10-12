"""
Multi-FITS VSF Analysis - Compare all emission lines across all FITS files
Creates subplots for each file/line combination
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import os
import glob
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D, VSF2DSubwindows

plt.style.use('./custom_plot.mplstyle')

NPZ_OUTPUT_DIR = 'NPZ'
VSF_ERROR_MODE = 'bootstrap'
MIN_SEPARATION_ARCSEC = 37.0

# Optional CLI arg: limit to first N regions for quick preview/testing.
MAX_REGIONS = int(sys.argv[1]) if len(sys.argv) > 1 else None

def arcsec_to_pc(angle_arcsec, distance_pc):
    """Convert an angular separation at distance_pc to a physical separation."""
    return distance_pc * np.deg2rad(angle_arcsec / 3600.0)

def get_heliocentric_distance(filename):
    """Get heliocentric distance for an object from the HII regions table"""
    # Special cases with known distances
    special_distances = {
        'Flame_nebula': 400.0,  # pc
        'Orion': 400.0,         # pc  
        'Rosette_nebula': 1500.0 # pc
    }
    
    # Extract object name from filename (remove LVM_ prefix and .fits suffix)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    object_name = base_name.replace('LVM_', '')
    
    # Check for special cases first
    if object_name in special_distances:
        return special_distances[object_name]
    
    # Load the distance table for other objects
    hdul_dist = fits.open('extra_HII_regions.fits')
    dist_data = hdul_dist[1].data
    hdul_dist.close()
    
    # Handle special cases
    name_mapping = {
        'Flame_nebula': 'Flame',  # Try alternative name
        'Rosette_nebula': 'Rosette'
    }
    lookup_name = name_mapping.get(object_name, object_name)
    
    # Find the object in the table
    try:
        index = list(dist_data['HII Region']).index(lookup_name)
        distance_kpc = float(dist_data['Heliocentric_distance'][index])
        distance_pc = distance_kpc * 1e3  # Convert kpc to pc
        return distance_pc
    except ValueError:
        print(f"Warning: Distance not found for {object_name}, using default 1500 pc")
        return 1500.0

def get_available_lines(filename):
    """Get available emission lines in a FITS file"""
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
    """Load and filter data for a specific line"""
    hdul = fits.open(filename)
    table_data = hdul[1].data

    flux_col = f'flux_{line}'
    flux_err_col = f'flux_{line}_err'
    vel_col = f'vel_{line}'
    vel_err_col = f'vel_{line}_err'

    # Quality filter
    filter_data = table_data[flux_col] / table_data[flux_err_col] > 5

    ra = table_data['ra'][filter_data]
    dec = table_data['dec'][filter_data]
    flux = table_data[flux_col][filter_data]
    vel = table_data[vel_col][filter_data]
    vel_err = table_data[vel_err_col][filter_data]

    hdul.close()
    return ra, dec, flux, vel, vel_err

def _build_peak_centered_edges(values, center, window_size):
    """Build fixed-width edges so one cell is centered exactly at center."""
    vmin = values.min()
    vmax = values.max()

    first_edge = center - 0.5 * window_size
    while first_edge > vmin:
        first_edge -= window_size

    edges = [first_edge]
    while edges[-1] < vmax:
        edges.append(edges[-1] + window_size)

    return np.array(edges)

def _passes_fill_fraction(xw, yw, x0, x1, y0, y1, fill_fraction=0.7, n_sub=5):
    """Replicate VSF2DSubwindows fill-fraction criterion."""
    sub_x = np.linspace(x0, x1, n_sub + 1)
    sub_y = np.linspace(y0, y1, n_sub + 1)

    occupied = 0
    total_cells = n_sub * n_sub
    for ii in range(n_sub):
        for jj in range(n_sub):
            submask = (
                (xw >= sub_x[ii]) & (xw < sub_x[ii + 1]) &
                (yw >= sub_y[jj]) & (yw < sub_y[jj + 1])
            )
            if np.any(submask):
                occupied += 1

    return (occupied / total_cells) >= fill_fraction

def compute_peak_aligned_subwindow_vsfs(vsf_obj, window_size, peak_ra, peak_dec,
                                        order=2, n_jobs=4, bin_resolution=20,
                                        min_points=50, fill_fraction=0.7, n_sub=5,
                                        min_separation=None):
    """Compute subwindow VSFs on a grid anchored so one cell is centered at peak."""
    x_all = vsf_obj.x_physical.value if hasattr(vsf_obj.x_physical, 'value') else vsf_obj.x_physical
    y_all = vsf_obj.y_physical.value if hasattr(vsf_obj.y_physical, 'value') else vsf_obj.y_physical

    peak_match_idx = np.argmin((vsf_obj.ra - peak_ra)**2 + (vsf_obj.dec - peak_dec)**2)
    x_center = x_all[peak_match_idx]
    y_center = y_all[peak_match_idx]

    x_edges = _build_peak_centered_edges(x_all, x_center, window_size)
    y_edges = _build_peak_centered_edges(y_all, y_center, window_size)

    peak_i = np.searchsorted(x_edges, x_center, side='right') - 1
    peak_j = np.searchsorted(y_edges, y_center, side='right') - 1

    results = []
    seeds = np.arange(n_jobs)

    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            x0, x1 = x_edges[i], x_edges[i + 1]
            y0, y1 = y_edges[j], y_edges[j + 1]

            mask = (
                (x_all >= x0) & (x_all < x1) &
                (y_all >= y0) & (y_all < y1)
            )

            if np.sum(mask) < min_points:
                continue

            xw = x_all[mask]
            yw = y_all[mask]
            velw = vsf_obj.vel[mask]

            if not _passes_fill_fraction(xw, yw, x0, x1, y0, y1,
                                         fill_fraction=fill_fraction, n_sub=n_sub):
                continue

            data = {'x': xw, 'y': yw}
            try:
                log_centers = vsf_obj.generate_bins_2d(
                    data,
                    bin_resolution=bin_resolution,
                    min_separation=min_separation,
                )
            except ValueError:
                continue
            edges = vsf_obj.generate_bins_edges(log_centers)
            bin_edges = 10**edges

            vsf_sum = np.zeros(len(bin_edges) - 1)
            counts = np.zeros_like(vsf_sum)

            chunks = [
                vsf_obj.compute_vsf_chunk_2d(
                    xw, yw, velw, np.zeros_like(velw), bin_edges, seed, order,
                    min_separation
                ) for seed in seeds
            ]

            for v, _, _, c in chunks:
                vsf_sum += v
                counts += c

            with np.errstate(divide='ignore', invalid='ignore'):
                vsf = np.where(counts > 0, vsf_sum / counts, np.nan)

            results.append((i, j, vsf, log_centers))

    peak_included = any((i == peak_i and j == peak_j) for i, j, _, _ in results)
    return results, peak_i, peak_j, peak_included

def _aggregate_vsf_from_results(results, bin_resolution=20, min_valid_fraction=0.5,
                                statistic='mean'):
    """Aggregate subwindow VSFs into mean/median curve with uncertainty."""
    if not results:
        return None, None, None, None

    log_centers_list = [log_c for _, _, _, log_c in results]
    if not log_centers_list:
        return None, None, None, None

    r_min_common = max(log_c.min() for log_c in log_centers_list)
    r_max_common = min(log_c.max() for log_c in log_centers_list)

    if not np.isfinite(r_min_common) or not np.isfinite(r_max_common) or r_max_common <= r_min_common:
        return None, None, None, None

    n_decades = max(r_max_common - r_min_common, 1e-6)
    n_bins = int(np.ceil(n_decades * bin_resolution))
    common_log_centers = np.linspace(r_min_common, r_max_common, max(n_bins, 10))
    common_r = 10**common_log_centers

    vsfs_interpolated = []
    for _, _, vsf, log_c in results:
        r = 10**log_c
        vsfs_interpolated.append(np.interp(common_r, r, vsf, left=np.nan, right=np.nan))

    if not vsfs_interpolated:
        return None, None, None, None

    vsfs_array = np.vstack(vsfs_interpolated)
    mean_vsf = np.nanmean(vsfs_array, axis=0)
    median_vsf = np.nanmedian(vsfs_array, axis=0)
    n_valid = np.sum(~np.isnan(vsfs_array), axis=0)

    valid_mask = n_valid >= len(results) * min_valid_fraction
    if not np.any(valid_mask):
        return None, None, None, None

    if statistic == 'mean':
        # Bootstrap CI error estimate for mean (68% CI half-width).
        n_boot = 300
        ci = 68.0
        low_pct = 0.5 * (100.0 - ci)
        high_pct = 100.0 - low_pct

        rng = np.random.default_rng(12345)
        mean_ci_err = np.full(vsfs_array.shape[1], np.nan)

        for k in range(vsfs_array.shape[1]):
            vals = vsfs_array[:, k]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue

            boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
            boot_means = np.nanmean(boot, axis=1)
            low = np.percentile(boot_means, low_pct)
            high = np.percentile(boot_means, high_pct)
            mean_ci_err[k] = 0.5 * (high - low)

        vsf = mean_vsf
        vsf_err = mean_ci_err
    elif statistic == 'median':
        # Bootstrap CI error estimate for median (68% CI half-width).
        n_boot = 300
        ci = 68.0
        low_pct = 0.5 * (100.0 - ci)
        high_pct = 100.0 - low_pct

        rng = np.random.default_rng(12345)
        median_ci_err = np.full(vsfs_array.shape[1], np.nan)

        for k in range(vsfs_array.shape[1]):
            vals = vsfs_array[:, k]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue

            boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
            boot_medians = np.nanmedian(boot, axis=1)
            low = np.percentile(boot_medians, low_pct)
            high = np.percentile(boot_medians, high_pct)
            median_ci_err[k] = 0.5 * (high - low)

        vsf = median_vsf
        vsf_err = median_ci_err
    else:
        raise ValueError("statistic must be 'mean' or 'median'")

    vsf = vsf[valid_mask]
    vsf_err = vsf_err[valid_mask]
    log_centers = common_log_centers[valid_mask]
    counts = n_valid[valid_mask]
    return vsf, vsf_err, log_centers, counts

def _median_relative_uncertainty_pct(vsf, vsf_err):
    """Return median percent uncertainty: median(100 * err / VSF)."""
    if vsf is None or vsf_err is None or len(vsf) == 0:
        return np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = 100.0 * vsf_err / vsf
    rel = rel[np.isfinite(rel) & (rel >= 0)]
    if len(rel) == 0:
        return np.nan
    return np.nanmedian(rel)

def compute_vsf_for_line(ra, dec, flux, vel, vel_err, window_size, distance,
                         min_separation):
    """Compute VSF variants for a single line: mean and median (all subwindows)."""
    vsf_subwindows = VSF2DSubwindows(ra, dec, vel, dist=distance)

    peak_idx = np.nanargmax(flux)
    peak_ra = ra[peak_idx]
    peak_dec = dec[peak_idx]

    bin_resolution = 20
    results, peak_i, peak_j, peak_included = compute_peak_aligned_subwindow_vsfs(
        vsf_subwindows,
        window_size=window_size,
        peak_ra=peak_ra,
        peak_dec=peak_dec,
        order=2,
        n_jobs=4,
        bin_resolution=bin_resolution,
        min_points=50,
        fill_fraction=0.7,
        n_sub=5,
        min_separation=min_separation,
    )

    mean_data = _aggregate_vsf_from_results(
        results,
        bin_resolution=bin_resolution,
        min_valid_fraction=0.5,
        statistic='mean'
    )
    median_data = _aggregate_vsf_from_results(
        results,
        bin_resolution=bin_resolution,
        min_valid_fraction=0.5,
        statistic='median'
    )

    diagnostics = {
        'peak_ra': peak_ra,
        'peak_dec': peak_dec,
        'peak_i': peak_i,
        'peak_j': peak_j,
        'peak_included': peak_included,
        'results_all': results,
        'n_windows_all': len(results),
        'mean_unc_pct': _median_relative_uncertainty_pct(mean_data[0], mean_data[1]),
        'median_unc_pct': _median_relative_uncertainty_pct(median_data[0], median_data[1]),
    }

    return mean_data, median_data, diagnostics

def plot_vsf_with_errorbars(ax, rvals, vsf, vsf_err, color, label,
                            marker='o', alpha_line=1.0, linestyle='-', linewidth=2.2):
    """Plot VSF curve, ±1 sigma band, and sampled error bars."""
    ax.plot(rvals, vsf, color=color, linewidth=linewidth, linestyle=linestyle,
            label=label, alpha=alpha_line)
    ax.fill_between(rvals, np.maximum(vsf - vsf_err, 1e0), vsf + vsf_err,
                    color=color, alpha=0.18)

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
        alpha=0.9 * alpha_line
    )
    ax.set_ylim(bottom=1)

def plot_subregion_vsfs(ax, results, color, alpha=0.20, label=None, max_curves=3):
    """Overlay the first few subregion VSFs in same color for direct comparison."""
    if not results:
        return

    first = True
    plotted = 0
    for _, _, vsf, log_c in results:
        if plotted >= max_curves:
            break
        if vsf is None or log_c is None:
            continue
        valid = np.isfinite(vsf)
        if np.sum(valid) < 3:
            continue
        rvals = 10**log_c[valid]
        yvals = vsf[valid]
        ax.plot(
            rvals,
            yvals,
            color=color,
            linewidth=1.0,
            alpha=alpha,
            label=label if first else None
        )
        ax.set_ylim(bottom=1)
        first = False
        plotted += 1

# Main execution
print("=" * 80)
print("MULTI-FITS VSF ANALYSIS - All Lines, All Files")
print("=" * 80)
print()

# Find all FITS files and skip Orion
fits_files = sorted(glob.glob("LVM_*.fits"))
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

# Ensure output directories exist.
os.makedirs('Figures', exist_ok=True)
os.makedirs(NPZ_OUTPUT_DIR, exist_ok=True)

# Colors for different lines
line_colors = {'nii': 'blue', 'sii': 'red', 'siii': 'green'}
line_names = {'nii': 'NII', 'sii': 'SII', 'siii': 'SIII'}

# Create one global figure with one subplot per region.
n_cols = 3
n_rows = int(np.ceil(len(files_to_process) / n_cols))
fig_all, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows), squeeze=False)


def _finite_positive_max(values):
    """Return max finite positive value, or nan if none."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return np.nan
    return float(np.max(arr))

# Process each file
plot_idx = 0
for filename in files_to_process:
    print(f"Processing {os.path.basename(filename)}...")

    base_name = os.path.splitext(os.path.basename(filename))[0]
    available_lines = get_available_lines(filename)
    
    # Get heliocentric distance for this object
    distance = get_heliocentric_distance(filename)
    print(f"  Distance: {distance:.1f} pc")
    min_separation_pc = arcsec_to_pc(MIN_SEPARATION_ARCSEC, distance)
    print(f"  Minimum separation: {MIN_SEPARATION_ARCSEC:.1f} arcsec = {min_separation_pc:.4f} pc")
    
    # Select region subplot in the global grid.
    row = plot_idx // n_cols
    col = plot_idx % n_cols
    ax_main = axes[row, col]
    region_y_max = np.nan
    
    # Process each line for this file
    for line in available_lines:
        try:
            # Load data
            ra, dec, flux, vel, vel_err = load_and_filter_data(filename, line)
            print(f"  {line.upper()}: {len(ra):,} points")

            # Get physical coordinates for window size
            vsf_phys = VSF2D(ra, dec, vel, dist=distance, vel_err=vel_err)
            x_phys = vsf_phys.x_physical
            y_phys = vsf_phys.y_physical

            x_range = x_phys.max() - x_phys.min()
            y_range = y_phys.max() - y_phys.min()
            window_size = max(x_range, y_range) / 5

            # Compute VSFs from all subwindows: mean and median aggregations.
            mean_data, median_data, diag = compute_vsf_for_line(
                ra, dec, flux, vel, vel_err, window_size, distance,
                min_separation_pc
            )

            peak_msg = "included" if diag['peak_included'] else "rejected"
            print(
                f"    Peak-centered cell (i={diag['peak_i']}, j={diag['peak_j']}) {peak_msg}; "
                f"windows used = {diag['n_windows_all']}"
            )
            print(
                f"    Mean/median relative uncertainty: "
                f"{diag['mean_unc_pct']:.2f}% / {diag['median_unc_pct']:.2f}%"
            )

            vsf_mean, err_mean, log_mean, cnt_mean = mean_data
            vsf_median, err_median, log_median, cnt_median = median_data

            if vsf_mean is None and vsf_median is None:
                print(f"    No valid VSF data for {line.upper()} (mean and median)")
                continue

            if vsf_mean is not None:
                r_mean = 10**log_mean
            if vsf_median is not None:
                r_median = 10**log_median

            # Save NPZ files for mean/median variants.
            object_name = base_name.replace('LVM_', '')
            npz_include = os.path.join(NPZ_OUTPUT_DIR, f'VSF_{object_name}_{line.upper()}_mean_allsubwindows.npz')
            np.savez(
                npz_include,
                vsf=vsf_mean,
                vsf_err=err_mean,
                log_centers=log_mean,
                counts=cnt_mean,
                min_separation_arcsec=MIN_SEPARATION_ARCSEC,
                min_separation_pc=min_separation_pc,
            )

            npz_exclude = os.path.join(NPZ_OUTPUT_DIR, f'VSF_{object_name}_{line.upper()}_median_allsubwindows.npz')
            np.savez(
                npz_exclude,
                vsf=vsf_median,
                vsf_err=err_median,
                log_centers=log_median,
                counts=cnt_median,
                min_separation_arcsec=MIN_SEPARATION_ARCSEC,
                min_separation_pc=min_separation_pc,
            )
            print(f"    ✓ Saved {npz_include} and {npz_exclude}")

            color = line_colors[line]

            # Overlay individual subregion VSFs (same color, lower alpha) on main panel.
            plot_subregion_vsfs(
                ax_main,
                diag['results_all'],
                color,
                alpha=0.20,
                label=f"{line_names[line]} subregions (first 3)",
                max_curves=3
            )

            # Track subregion maxima for dynamic y-limit assignment.
            subregion_max = np.nan
            for _, _, sub_vsf, _ in diag['results_all']:
                if sub_vsf is None:
                    continue
                vmax = _finite_positive_max(sub_vsf)
                if np.isfinite(vmax):
                    subregion_max = vmax if not np.isfinite(subregion_max) else max(subregion_max, vmax)
            if np.isfinite(subregion_max):
                region_y_max = subregion_max if not np.isfinite(region_y_max) else max(region_y_max, subregion_max)

            if vsf_mean is not None:
                mean_max = _finite_positive_max(vsf_mean + err_mean)
                if np.isfinite(mean_max):
                    region_y_max = mean_max if not np.isfinite(region_y_max) else max(region_y_max, mean_max)
                plot_vsf_with_errorbars(
                    ax_main,
                    r_mean,
                    vsf_mean,
                    err_mean,
                    color,
                    f"{line_names[line]} mean",
                    marker='o',
                    alpha_line=1.0,
                    linestyle='-',
                    linewidth=2.4
                )
            if vsf_median is not None:
                median_max = _finite_positive_max(vsf_median + err_median)
                if np.isfinite(median_max):
                    region_y_max = median_max if not np.isfinite(region_y_max) else max(region_y_max, median_max)
                plot_vsf_with_errorbars(
                    ax_main,
                    r_median,
                    vsf_median,
                    err_median,
                    color,
                    f"{line_names[line]} median",
                    marker='^',
                    alpha_line=1.0,
                    linestyle='--',
                    linewidth=2.2
                )

            # --- Compute and plot total VSF for the whole region (order=2) ---
            vsf_total, vsf_total_err, log_centers_total, counts_total = VSF2D(ra, dec, vel, dist=distance, vel_err=vel_err).generate_vsf_2d(
                order=2,
                outname="/private/tmp/ignore",
                n_jobs=4,
                min_pairs=3,
                bin_resolution=20,
                error_mode=VSF_ERROR_MODE,
                min_separation=min_separation_pc,
            )
            valid = np.isfinite(vsf_total) & (counts_total > 0)
            if np.any(valid):
                r_total = 10**log_centers_total[valid]
                vsf_total = vsf_total[valid]
                vsf_total_err = vsf_total_err[valid]

                total_max = _finite_positive_max(vsf_total + vsf_total_err)
                if np.isfinite(total_max):
                    region_y_max = total_max if not np.isfinite(region_y_max) else max(region_y_max, total_max)

                plot_vsf_with_errorbars(
                    ax_main,
                    r_total,
                    vsf_total,
                    vsf_total_err,
                    color,
                    f"{line_names[line]} (total)",
                    marker='s',
                    alpha_line=0.85,
                    linestyle=':',
                    linewidth=2.2
                )

        except Exception as e:
            print(f"  Error processing {line.upper()}: {e}")
            continue
    
    # Format region subplot with former left-panel content.
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
    ax_main.set_title(base_name.replace('LVM_', ''), fontsize=12, fontweight='bold')
    if row == n_rows - 1 and col == n_cols - 1:
        ax_main.legend(fontsize=7, loc='best')

    plot_idx += 1

# Hide any unused subplot slots.
for idx in range(plot_idx, n_rows * n_cols):
    r = idx // n_cols
    c = idx % n_cols
    axes[r, c].set_visible(False)

fig_all.tight_layout()
figure_dir = 'Figures/manuscript/02_vsf2_multifit'
os.makedirs(figure_dir, exist_ok=True)
multi_out = os.path.join(figure_dir, 'multi_fits_vsf_absolute_subwindows.png')
fig_all.savefig(multi_out, dpi=150, bbox_inches='tight')
plt.close(fig_all)

print()
print(f"✓ Global subplot figure saved as {multi_out}")
print(f"✓ Individual VSF data saved in {NPZ_OUTPUT_DIR}/ as VSF_{{object}}_{{line}}_{{mean|median}}_allsubwindows.npz")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Processed {len(files_to_process)} FITS files")
print(f"Created a {n_rows}x{n_cols} global subplot figure (one region per subplot)")
print("Each subplot shows the former left-panel content: subregions + mean/median/total")
print("=" * 80)
