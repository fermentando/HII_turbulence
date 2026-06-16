"""
Orion Subwindow VSF Analysis - Fast demonstration
Uses 1/5 FOV windows (minimal subwindows) and sequential processing
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import seaborn as sns
import sys
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D, VSF2DSubwindows

plt.style.use('./custom_plot.mplstyle')
# Get input file from command line argument
filename = sys.argv[1] if len(sys.argv) > 1 else "LVM_RCW52.fits"
base_name = os.path.splitext(os.path.basename(filename))[0]

print("=" * 70)
print(f"{base_name.upper()} SUBWINDOW VSF ANALYSIS - FAST DEMO")
print("=" * 70)
print()

# Load data with quality filter for all emission lines
print(f"Loading {base_name} data (with S/N > 5 filter for all lines)...")
hdul = fits.open(filename)
table_data = hdul[1].data

# Define emission lines
lines = ['nii', 'sii', 'siii']
line_names = ['NII', 'SII', 'SIII']

# Load data for each line
ra_all = []
dec_all = []
vel_all = []
vel_err_all = []
line_data = {}

for line, name in zip(lines, line_names):
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

    peak_idx = np.nanargmax(flux)
    peak_ra = ra[peak_idx]
    peak_dec = dec[peak_idx]
    peak_flux = flux[peak_idx]
    
    line_data[name] = {
        'ra': ra,
        'dec': dec,
        'flux': flux,
        'vel': vel,
        'vel_err': vel_err,
        'peak_ra': peak_ra,
        'peak_dec': peak_dec,
        'peak_flux': peak_flux
    }
    
    print(f"  {name}: {len(ra):,} points")
    print(f"    Peak flux @ RA={peak_ra:.6f}, Dec={peak_dec:.6f}, flux={peak_flux:.3e}")

# Use NII for physical coordinate calculation and velocity field
ra = line_data['NII']['ra']
dec = line_data['NII']['dec']
vel_nii = line_data['NII']['vel']
vel_err = line_data['NII']['vel_err']

# Get physical coordinates and compute window size
vsf_phys = VSF2D(ra, dec, vel_nii, dist=1500, vel_err=vel_err)
x_phys = vsf_phys.x_physical
y_phys = vsf_phys.y_physical

x_range = x_phys.max() - x_phys.min()
y_range = y_phys.max() - y_phys.min()

print(f"Spatial extent (1500 pc):")
print(f"  X range: {x_range:.1f} pc")
print(f"  Y range: {y_range:.1f} pc")

# Use 1/5 of FOV (only 5x5 grid = 25 windows total)
window_size = max(x_range, y_range) / 5
print(f"  Window size (1/5 FOV): {window_size:.1f} pc")
print()

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
                                        order=1, n_jobs=1, bin_resolution=30,
                                        min_points=50, fill_fraction=0.7, n_sub=5):
    """Compute subwindow VSFs on a grid anchored so one cell is centered at the peak."""
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
            log_centers = vsf_obj.generate_bins_2d(data, bin_resolution=bin_resolution)
            edges = vsf_obj.generate_bins_edges(log_centers)
            bin_edges = 10**edges

            vsf_sum = np.zeros(len(bin_edges) - 1)
            counts = np.zeros_like(vsf_sum)

            chunks = [
                vsf_obj.compute_vsf_chunk_2d(
                    xw, yw, velw, np.zeros_like(velw), bin_edges, seed, order
                ) for seed in seeds
            ]

            for v, _, _, c in chunks:
                vsf_sum += v
                counts += c

            with np.errstate(divide='ignore', invalid='ignore'):
                vsf = np.where(counts > 0, vsf_sum / counts, np.nan)

            results.append((i, j, vsf, log_centers))

    peak_included = any((i == peak_i and j == peak_j) for i, j, _, _ in results)

    return results, x_edges, y_edges, peak_i, peak_j, peak_included

# Create and visualize subwindows
print("Creating VSF2DSubwindows object...")
vsf_subwindows = VSF2DSubwindows(ra, dec, vel_nii, dist=1500)

# Compute subwindow VSFs for all lines
print(f"Computing individual subwindow VSFs in sequence for all lines...")
print(f"  Window size: {window_size:.0f} pc (1/5 of FOV)")
print(f"  Please wait (2-3 minutes)...")

# Store results for each line
all_results = {}
all_vsfs_array = {}
all_common_r = {}
all_mean_vsf = {}
all_std_vsf = {}
all_median_vsf = {}
all_median_err = {}
all_lower_vsf = {}
all_upper_vsf = {}
all_x_edges = {}
all_y_edges = {}
all_peak_window = {}

for name in line_names:
    print(f"  Computing VSFs for {name}...")
    
    # Create VSF2DSubwindows object for this line
    vsf_subwindows = VSF2DSubwindows(line_data[name]['ra'], line_data[name]['dec'], line_data[name]['vel'], dist=1500)
    
    results, x_edges, y_edges, peak_i, peak_j, peak_included = compute_peak_aligned_subwindow_vsfs(
        vsf_subwindows,
        window_size=window_size,
        peak_ra=line_data[name]['peak_ra'],
        peak_dec=line_data[name]['peak_dec'],
        order=1,
        n_jobs=1,
        bin_resolution=30,
        min_points=50,
        fill_fraction=0.7,
        n_sub=5
    )
    if peak_included:
        print(f"    {name}: peak-centered grid cell included at (i={peak_i}, j={peak_j})")
    else:
        print(f"    {name}: peak-centered grid cell rejected by cuts (i={peak_i}, j={peak_j})")
    
    all_results[name] = results
    all_x_edges[name] = x_edges
    all_y_edges[name] = y_edges
    all_peak_window[name] = (peak_i, peak_j, peak_included)
    
    # Extract data and build common radius grid
    log_centers_list = [log_c for _, _, _, log_c in results]
    r_min_common = max(log_c.min() for log_c in log_centers_list)
    r_max_common = min(log_c.max() for log_c in log_centers_list)
    common_log_centers = np.linspace(r_min_common, r_max_common, 80)
    common_r = 10**common_log_centers
    
    vsfs_interpolated = []
    for _, _, vsf, log_c in results:
        r = 10**log_c
        vsfs_interpolated.append(np.interp(common_r, r, vsf, left=np.nan, right=np.nan))
    
    vsfs_array = np.vstack(vsfs_interpolated)
    
    # Debug: Check for NaN values
    nan_mask = np.isnan(vsfs_array)
    nan_fraction = np.mean(nan_mask, axis=0)
    print(f"    {name}: NaN fraction per bin: min={nan_fraction.min():.2f}, max={nan_fraction.max():.2f}")
    
    n_valid = np.sum(~np.isnan(vsfs_array), axis=0)
    mean_vsf = np.nanmean(vsfs_array, axis=0)
    std_raw = np.nanstd(vsfs_array, axis=0)
    std_vsf = np.divide(std_raw, np.sqrt(n_valid), out=np.full_like(std_raw, np.nan), where=n_valid > 0)

    median_vsf = np.nanmedian(vsfs_array, axis=0)
    mad_vsf = np.nanmedian(np.abs(vsfs_array - median_vsf[None, :]), axis=0)
    median_err = np.divide(1.4826 * mad_vsf, np.sqrt(n_valid), out=np.full_like(mad_vsf, np.nan), where=n_valid > 0)

    lower_vsf = mean_vsf - std_vsf
    upper_vsf = mean_vsf + std_vsf
    
    all_vsfs_array[name] = vsfs_array
    all_common_r[name] = common_r
    all_mean_vsf[name] = mean_vsf
    all_std_vsf[name] = std_vsf
    all_median_vsf[name] = median_vsf
    all_median_err[name] = median_err
    all_lower_vsf[name] = lower_vsf
    all_upper_vsf[name] = upper_vsf

print(f"\n✓ Computed VSFs for {len(line_names)} lines, {len(results)} subwindows each")
print()

# Use common_r from the last computation (they should be the same)
# Update statistics using NII as reference
vsfs_array = all_vsfs_array['NII']
mean_vsf = all_mean_vsf['NII']
std_vsf = all_std_vsf['NII']
lower_vsf = all_lower_vsf['NII']
upper_vsf = all_upper_vsf['NII']

# Error band statistics using standard deviation
range_pct = np.where(mean_vsf > 0, (2 * std_vsf) / mean_vsf * 100, np.nan)

print(f"Subwindow statistics (NII reference):")
print(f"  Number of subwindows: {len(results)}")
print(f"  Common radial bins: {len(common_log_centers)}")
print(f"  Mean VSF range: {np.nanmin(mean_vsf):.2e} to {np.nanmax(mean_vsf):.2e}")
print()

# Error band statistics using standard deviation
print(f"Error bar analysis:")
print(f"  Median ±1σ range: {np.nanmedian(range_pct):.1f}% of mean")
print(f"  Interpretation: ±1σ envelope of subwindow VSFs")
print()

# Analyze uncertainty vs scale
print("Uncertainty vs scale analysis:")
small_scale_idx = common_r < 10  # Small scales (< 10 pc)
large_scale_idx = common_r > 30  # Large scales (> 30 pc)

print(f"  Scale ranges:")
print(f"    Small scales (< 10 pc): {np.sum(small_scale_idx)} bins, r = {common_r[small_scale_idx].min():.1f} to {common_r[small_scale_idx].max():.1f} pc")
if np.sum(large_scale_idx) > 0:
    print(f"    Large scales (> 30 pc): {np.sum(large_scale_idx)} bins, r = {common_r[large_scale_idx].min():.1f} to {common_r[large_scale_idx].max():.1f} pc")
else:
    print(f"    Large scales (> 30 pc): No data points in this range")
print()

if np.sum(small_scale_idx) > 0:
    small_range = np.nanmedian(range_pct[small_scale_idx])
    print(f"  Small scales (< 10 pc): ±1σ range = {small_range:.1f}% of mean")

if np.sum(large_scale_idx) > 0:
    large_range = np.nanmedian(range_pct[large_scale_idx])
    print(f"  Large scales (> 30 pc): ±1σ range = {large_range:.1f}% of mean")
    ratio = small_range / large_range if large_range > 0 else np.nan
    print(f"  Ratio (small/large): σ width = {ratio:.1f}x")
    print(f"  → Small-scale wiggles are captured by a broader σ envelope")
else:
    print(f"  Large scales (> 30 pc): No data points in this range")
    print(f"  → Dataset too small for large-scale analysis")

print()

# Create the two-panel figure
print("Creating summary figure with velocity field and absolute VSFs...")

from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(17, 7))
gs = gridspec.GridSpec(1, 3, width_ratios=[0.05, 1.0, 1.0], wspace=0.3)
ax_cbar = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# Subplot 1: Velocity field with NII flux contours
plt.style.use('./custom_plot.mplstyle')
vel_cmap = sns.color_palette('crest', as_cmap=True)
from scipy.interpolate import griddata

sc = ax1.scatter(
    ra,
    dec,
    c=vel_nii,
    cmap=vel_cmap,
    s=18,
    vmin=np.percentile(vel_nii, 10),
    vmax=np.percentile(vel_nii, 90)
)
ax1.invert_xaxis()
ax1.set_xlabel('RA (°)')
ax1.set_ylabel('Dec (°)')
ax1.grid(True, alpha=0.3)

# Overlay selected NII subwindow divisions from VSF computation.
nii_results = all_results['NII']
nii_x_edges = all_x_edges['NII']
nii_y_edges = all_y_edges['NII']

# Approximate inverse mapping from physical x/y to RA/Dec for edge plotting.
ra_from_x = np.poly1d(np.polyfit(x_phys, ra, 1))
dec_from_y = np.poly1d(np.polyfit(y_phys, dec, 1))

# Draw complete 5x5 grid boundaries faintly.
for x_edge in nii_x_edges:
    ra_edge = ra_from_x(x_edge)
    ax1.plot([ra_edge, ra_edge], [dec.min(), dec.max()], color='white', linewidth=0.7, alpha=0.25)
for y_edge in nii_y_edges:
    dec_edge = dec_from_y(y_edge)
    ax1.plot([ra.min(), ra.max()], [dec_edge, dec_edge], color='white', linewidth=0.7, alpha=0.25)

# Highlight accepted subwindows that were used in VSF statistics.
for i, j, _, _ in nii_results:
    ra0 = ra_from_x(nii_x_edges[i])
    ra1 = ra_from_x(nii_x_edges[i + 1])
    dec0 = dec_from_y(nii_y_edges[j])
    dec1 = dec_from_y(nii_y_edges[j + 1])
    ax1.plot([ra0, ra1, ra1, ra0, ra0], [dec0, dec0, dec1, dec1, dec0], color='white', linewidth=1.5, alpha=0.85)

# Mark the NII peak used to center one VSF subwindow.
nii_peak_ra = line_data['NII']['peak_ra']
nii_peak_dec = line_data['NII']['peak_dec']
ax1.scatter(nii_peak_ra, nii_peak_dec, marker='*', s=260, c='yellow', edgecolor='black', linewidth=1.2, zorder=10)
ax1.text(nii_peak_ra, nii_peak_dec, ' peak', color='yellow', fontsize=11, fontweight='bold', zorder=11)

nii_peak_i, nii_peak_j, nii_peak_included = all_peak_window['NII']
peak_ra0 = ra_from_x(nii_x_edges[nii_peak_i])
peak_ra1 = ra_from_x(nii_x_edges[nii_peak_i + 1])
peak_dec0 = dec_from_y(nii_y_edges[nii_peak_j])
peak_dec1 = dec_from_y(nii_y_edges[nii_peak_j + 1])
ax1.plot(
    [peak_ra0, peak_ra1, peak_ra1, peak_ra0, peak_ra0],
    [peak_dec0, peak_dec0, peak_dec1, peak_dec1, peak_dec0],
    color='yellow', linewidth=2.0, linestyle='--', alpha=0.95
)
if not nii_peak_included:
    ax1.text(nii_peak_ra, nii_peak_dec, ' peak (rejected)', color='yellow', fontsize=10, fontweight='bold', zorder=11)

# Overlay NII flux contours
nii_flux = table_data['flux_nii'][table_data['flux_nii'] / table_data['flux_nii_err'] > 5]
contour_levels = np.percentile(nii_flux, [70, 80, 90, 97])
grid_ra = np.linspace(ra.min(), ra.max(), 200)
grid_dec = np.linspace(dec.min(), dec.max(), 200)
RA, DEC = np.meshgrid(grid_ra, grid_dec)
grid_flux = griddata((ra, dec), nii_flux, (RA, DEC), method='linear')
cont = ax1.contour(RA, DEC, grid_flux, levels=contour_levels, colors='orange', linewidths=2.5, alpha=0.95)
ax1.clabel(cont, inline=True, fmt='%.1e')

# Colorbar on dedicated narrow subplot (left-most axis).
cbar = fig.colorbar(sc, cax=ax_cbar, orientation='vertical')
ax_cbar.yaxis.set_ticks_position('left')
ax_cbar.yaxis.set_label_position('left')
cbar.set_label(r'$v_\mathrm{LOS}$ [km/s]')

# Subplot 2: Absolute VSFs with subregions + mean/median/total
vsf_colors = sns.color_palette('rocket', n_colors=3)
for i, name in enumerate(line_names):
    color = vsf_colors[i]
    r_line = all_common_r[name]

    # Plot first 3 subwindow curves for visual context.
    plotted = 0
    for _, _, vsf_sub, log_c_sub in all_results[name]:
        if plotted >= 3:
            break
        valid_sub = np.isfinite(vsf_sub)
        if np.sum(valid_sub) < 3:
            continue
        ax2.plot(10**log_c_sub[valid_sub], vsf_sub[valid_sub], color=color, linewidth=1.0, alpha=0.20,
                 label=f'{name} subregions (first 3)' if plotted == 0 else None)
        plotted += 1

    mean_vsf_line = all_mean_vsf[name]
    mean_err_line = all_std_vsf[name]
    median_vsf_line = all_median_vsf[name]
    median_err_line = all_median_err[name]

    ax2.plot(r_line, mean_vsf_line, color=color, linestyle='-', linewidth=2.2, label=f'{name} mean')
    ax2.fill_between(r_line, np.maximum(mean_vsf_line - mean_err_line, 1e-12), mean_vsf_line + mean_err_line,
                     color=color, alpha=0.18)

    ax2.plot(r_line, median_vsf_line, color=color, linestyle=':', linewidth=2.2, label=f'{name} median')
    ax2.fill_between(r_line, np.maximum(median_vsf_line - median_err_line, 1e-12), median_vsf_line + median_err_line,
                     color=color, alpha=0.10)

    # Compute and overlay full-region total VSF.
    vsf_total, vsf_total_err, log_centers_total, counts_total = VSF2D(
        line_data[name]['ra'], line_data[name]['dec'], line_data[name]['vel'], dist=1500,
        vel_err=line_data[name]['vel_err']
    ).generate_vsf_2d(order=1, outname='/tmp/ignore_orion_total', n_jobs=1, min_pairs=3, bin_resolution=30)
    valid_total = np.isfinite(vsf_total) & (counts_total > 0)
    if np.any(valid_total):
        r_total = 10**log_centers_total[valid_total]
        total_line = vsf_total[valid_total]
        total_err_line = vsf_total_err[valid_total]
        ax2.plot(r_total, total_line, color=color, linestyle='--', linewidth=2.2, label=f'{name} total')
        ax2.fill_between(r_total, np.maximum(total_line - total_err_line, 1e-12), total_line + total_err_line,
                         color=color, alpha=0.08)


ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'$l_{2D} = \left| r_i - r_j \right| [pc]$')
ax2.set_ylabel(r'$\bar{S}_\mathrm{2} = \langle | v(r + l) - v(r) | \rangle ^{2}$')
ax2.grid(True, alpha=0.3)
# Increase tick density on log x-axis
ax2.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

# Custom legend: style meaning + line colors (middle panel).
style_handles_ax2 = [
    Line2D([0], [0], color='black', lw=2.2, linestyle='-', label='mean'),
    Line2D([0], [0], color='black', lw=2.2, linestyle=':', label='median'),
    Line2D([0], [0], color='black', lw=2.2, linestyle='--', label='total'),
]
color_handles = [
    Line2D([0], [0], color=vsf_colors[i], lw=3.0, linestyle='-', label=name)
    for i, name in enumerate(line_names)
]
ax2.legend(handles=style_handles_ax2 + color_handles, fontsize=8, loc='best', frameon=True)

plt.tight_layout()
figure_dir = 'Figures/manuscript/01_summary_regions'
os.makedirs(figure_dir, exist_ok=True)
figure_out = os.path.join(figure_dir, f'{base_name}_summary.png')
plt.savefig(figure_out, dpi=150, bbox_inches='tight')
print("✓ Summary figure saved")
print()

print("=" * 70)
print("FILE CREATED")
print("=" * 70)
print()
print(figure_out)
print("   → Two-panel figure:")
print("     • Left: Velocity field (RA/Dec coordinates)")
print("     • Middle: Subregions + mean/median/total VSFs")
print()
print("=" * 70)
