"""
Orion-style Full-FOV VSF Analysis - Fast demonstration
Computes one VSF per line over the full field of view (no subwindow median).
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D

plt.style.use('./custom_plot.mplstyle')

filename = sys.argv[1] if len(sys.argv) > 1 else 'LVM_RCW52.fits'
base_name = os.path.splitext(os.path.basename(filename))[0]
object_name = base_name.replace('LVM_', '')

print('=' * 70)
print(f"{base_name.upper()} FULL-FOV VSF ANALYSIS - FAST DEMO")
print('=' * 70)
print()

print(f"Loading {base_name} data (with S/N > 5 filter for all lines)...")
hdul = fits.open(filename)
table_data = hdul[1].data

candidate_lines = ['nii', 'sii', 'siii']
line_names_map = {'nii': 'NII', 'sii': 'SII', 'siii': 'SIII'}
line_colors = {'nii': '#2C7FB8', 'sii': '#D95F0E', 'siii': '#238B45'}

available_lines = [line for line in candidate_lines if f'vel_{line}' in table_data.names]
line_names = [line_names_map[line] for line in available_lines]

if not available_lines:
    raise RuntimeError('No supported velocity lines found (NII/SII/SIII).')

line_data = {}
for line in available_lines:
    name = line_names_map[line]
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

    peak_idx = np.nanargmax(flux)
    peak_ra = ra[peak_idx]
    peak_dec = dec[peak_idx]
    peak_flux = flux[peak_idx]

    line_data[name] = {
        'line_key': line,
        'ra': ra,
        'dec': dec,
        'flux': flux,
        'vel': vel,
        'vel_err': vel_err,
        'peak_ra': peak_ra,
        'peak_dec': peak_dec,
        'peak_flux': peak_flux,
    }

    print(f"  {name}: {len(ra):,} points")
    print(f"    Peak flux @ RA={peak_ra:.6f}, Dec={peak_dec:.6f}, flux={peak_flux:.3e}")

reference_name = 'NII' if 'NII' in line_data else line_names[0]
ra_ref = line_data[reference_name]['ra']
dec_ref = line_data[reference_name]['dec']
vel_ref = line_data[reference_name]['vel']
vel_err_ref = line_data[reference_name]['vel_err']

vsf_phys = VSF2D(ra_ref, dec_ref, vel_ref, dist=1500, vel_err=vel_err_ref)
x_phys = vsf_phys.x_physical
y_phys = vsf_phys.y_physical

x_range = x_phys.max() - x_phys.min()
y_range = y_phys.max() - y_phys.min()

print('Spatial extent (1500 pc):')
print(f"  X range: {x_range:.1f} pc")
print(f"  Y range: {y_range:.1f} pc")
print()

print('Computing full-FOV VSFs for all available lines...')

all_total = {}
for name in line_names:
    d = line_data[name]
    print(f"  Computing {name} full-FOV VSF...")

    vsf_total, vsf_total_err, log_centers_total, counts_total = VSF2D(
        d['ra'], d['dec'], d['vel'], dist=1500, vel_err=d['vel_err']
    ).generate_vsf_2d(order=1, outname='/tmp/ignore_orion_fullfov', n_jobs=1, min_pairs=3, bin_resolution=30)

    valid = np.isfinite(vsf_total) & np.isfinite(vsf_total_err) & (counts_total > 0)
    if not np.any(valid):
        print(f"    {name}: no valid bins")
        continue

    r_total = 10**log_centers_total[valid]
    y_total = vsf_total[valid]
    yerr_total = vsf_total_err[valid]
    cnt_total = counts_total[valid]

    all_total[name] = {
        'r': r_total,
        'vsf': y_total,
        'vsf_err': yerr_total,
        'counts': cnt_total,
    }

    npz_name = f"NPZ/VSF_{object_name}_{name}_total_fullfov_fast.npz"
    np.savez(
        npz_name,
        vsf=y_total,
        vsf_err=yerr_total,
        log_centers=np.log10(r_total),
        counts=cnt_total,
    )
    print(f"    Saved {npz_name}")

print()

print('Creating summary figure with velocity field, absolute VSFs, and normalized ratios...')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'wspace': 0.30})

vel_cmap = sns.color_palette('crest', as_cmap=True)
sc = ax1.scatter(
    ra_ref,
    dec_ref,
    c=vel_ref,
    cmap=vel_cmap,
    s=18,
    vmin=np.percentile(vel_ref, 10),
    vmax=np.percentile(vel_ref, 90),
)
ax1.invert_xaxis()
ax1.set_xlabel('RA (deg)')
ax1.set_ylabel('Dec (deg)')
ax1.grid(True, alpha=0.3)
ax1.set_title(f'{reference_name} velocity field')

peak_ra = line_data[reference_name]['peak_ra']
peak_dec = line_data[reference_name]['peak_dec']
ax1.scatter(peak_ra, peak_dec, marker='*', s=260, c='yellow', edgecolor='black', linewidth=1.2, zorder=10)
ax1.text(peak_ra, peak_dec, ' peak', color='yellow', fontsize=11, fontweight='bold', zorder=11)

ref_flux = line_data[reference_name]['flux']
contour_levels = np.percentile(ref_flux, [70, 80, 90, 97])
grid_ra = np.linspace(ra_ref.min(), ra_ref.max(), 200)
grid_dec = np.linspace(dec_ref.min(), dec_ref.max(), 200)
ra_grid, dec_grid = np.meshgrid(grid_ra, grid_dec)
grid_flux = griddata((ra_ref, dec_ref), ref_flux, (ra_grid, dec_grid), method='linear')
contours = ax1.contour(ra_grid, dec_grid, grid_flux, levels=contour_levels, colors='orange', linewidths=2.5, alpha=0.95)
ax1.clabel(contours, inline=True, fmt='%.1e')

cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label(r'$v_\mathrm{LOS}$ [km/s]')

for name in line_names:
    if name not in all_total:
        continue
    line_key = line_data[name]['line_key']
    color = line_colors[line_key]

    r = all_total[name]['r']
    y = all_total[name]['vsf']
    yerr = all_total[name]['vsf_err']

    ax2.plot(r, y, color=color, linestyle='-', linewidth=2.2, label=f'{name} total (full FOV)')
    ax2.fill_between(r, np.maximum(y - yerr, 1e-12), y + yerr, color=color, alpha=0.16)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'$l_{2D} = |r_i - r_j|$ [pc]')
ax2.set_ylabel(r'$\bar{S}_\mathrm{2} = \langle |v(r + l) - v(r)| \rangle^2$')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax2.legend(fontsize=9, loc='best', frameon=True)

if reference_name in all_total:
    r_ref_curve = all_total[reference_name]['r']
    y_ref_curve = all_total[reference_name]['vsf']

    for name in line_names:
        if name not in all_total:
            continue
        line_key = line_data[name]['line_key']
        color = line_colors[line_key]

        r = all_total[name]['r']
        y = all_total[name]['vsf']

        r_min = max(np.min(r_ref_curve), np.min(r))
        r_max = min(np.max(r_ref_curve), np.max(r))
        if r_max <= r_min:
            continue

        r_overlap = np.logspace(np.log10(r_min), np.log10(r_max), 80)
        y_interp = np.interp(r_overlap, r, y)
        y_ref_interp = np.interp(r_overlap, r_ref_curve, y_ref_curve)

        ratio = np.divide(y_interp, y_ref_interp, out=np.full_like(y_interp, np.nan), where=y_ref_interp > 0)
        valid_ratio = np.isfinite(ratio)
        if np.any(valid_ratio):
            label = f'{name}/{reference_name}' if name != reference_name else f'{name}/{name}'
            ax3.plot(r_overlap[valid_ratio], ratio[valid_ratio], color=color, linewidth=2.2, label=label)

ax3.set_xscale('log')
ax3.set_xlabel(r'$l_{2D} = |r_i - r_j|$ [pc]')
ax3.set_ylabel(f'Total VSF ratio to {reference_name}')
ax3.grid(True, alpha=0.3)
ax3.axhline(1.0, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
ax3.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

style_handle = Line2D([0], [0], color='black', lw=2.2, linestyle='-', label='total/full-FOV')
color_handles = []
for name in line_names:
    if name not in all_total:
        continue
    line_key = line_data[name]['line_key']
    color_handles.append(Line2D([0], [0], color=line_colors[line_key], lw=3.0, linestyle='-', label=name))

ax3.legend(handles=[style_handle] + color_handles, fontsize=8, loc='best', frameon=True)

plt.tight_layout()
figure_dir = 'Figures/manuscript/01_summary_regions'
os.makedirs(figure_dir, exist_ok=True)
figure_out = os.path.join(figure_dir, f'{base_name}_summary_fullfov.png')
plt.savefig(figure_out, dpi=150, bbox_inches='tight')

hdul.close()

print('Saved full-FOV summary figure:')
print(f'  {figure_out}')
print()
print('=' * 70)
print('FILES CREATED')
print('=' * 70)
print(figure_out)
print(f'NPZ/VSF_{object_name}_<LINE>_total_fullfov_fast.npz')
print('  -> Full-FOV curves only (no subwindow mean/median statistics)')
print('=' * 70)
