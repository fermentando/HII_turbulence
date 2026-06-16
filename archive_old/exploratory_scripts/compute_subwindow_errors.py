"""
Compute error bars for the mean VSF from residuals across subwindows

Error bar methodology:
- For each radial separation r, we have multiple window VSF values: {VSF₁(r), VSF₂(r), ..., VSFₙ(r)}
- Standard Deviation σ(r) = std({VSFᵢ(r)}) → shows scatter/variation across windows
- Standard Error SE(r) = σ(r) / √n → shows uncertainty in the mean estimate
- We can plot either ±σ(r) [shows actual spatial variation] or ±SE(r) [shows statistical uncertainty]

Why this works:
- The variation in VSF values across windows tells us how spatially heterogeneous the velocity structure is
- Large σ means the velocity structure differs significantly from window to window
- Small σ means the velocity structure is uniform across the FOV
- The SEM tells us how well our mean estimate constrains the true average

This is more robust than computing error propagation from measurement uncertainties,
because it captures real spatial variation rather than just instrumental noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

sys.path.insert(0, '/Users/fernando/PhDResearch/LVM stuff')
from vsf_module import VSF2D, VSF2DSubwindows


# Load results from previous run
print("=" * 70)
print("COMPUTING ERROR BARS ON MEAN SUBWINDOW VSF")
print("=" * 70)
print()

print("Loading Orion data...")
rosette_hdul = fits.open("LVM_Orion.fits")
table_rosette = rosette_hdul[1].data
filter_orion = table_rosette['flux_nii'] / table_rosette['flux_nii_err'] > 5

ra_orion = table_rosette['ra'][filter_orion]
dec_orion = table_rosette['dec'][filter_orion]
vel_nii_orion = table_rosette['vel_nii'][filter_orion]
vel_err_orion = table_rosette['vel_nii_err'][filter_orion]

print(f"  Points: {len(ra_orion):,}")
print()

# Compute physical FOV
print("Computing FOV and window size...")
vsf_obj = VSF2D(ra_orion, dec_orion, vel_nii_orion, dist=1500, vel_err=vel_err_orion)
x_phys = vsf_obj.x_physical
y_phys = vsf_obj.y_physical

x_range = x_phys.max() - x_phys.min()
y_range = y_phys.max() - y_phys.min()
window_size = max(x_range, y_range) / 20  # 1/20 FOV

print(f"  FOV: {x_range:.0f} × {y_range:.0f} pc")
print(f"  Window size (1/20): {window_size:.0f} pc")
print()

# Compute subwindow VSFs
print("Computing subwindow VSFs...")
vsf_subwindows = VSF2DSubwindows(ra_orion, dec_orion, vel_nii_orion, dist=1500)

results, x_edges, y_edges = vsf_subwindows.compute_subwindow_vsfs(
    window_size=window_size,
    order=1,
    n_jobs=2,
    bin_resolution=30
)

print(f"✓ {len(results)} subwindow VSFs computed")
print()

# Extract individual VSFs and their radii
print("Extracting and interpolating to common radius grid...")
vsfs_individual = []
radii_individual = []

for _, _, vsf, log_c in results:
    r = 10**log_c
    vsfs_individual.append(vsf)
    radii_individual.append(r)

# Create common radius grid (use intersection of all windows' ranges)
r_min = max(r.min() for r in radii_individual)
r_max = min(r.max() for r in radii_individual)
n_bins_common = 70
r_common = np.logspace(np.log10(r_min), np.log10(r_max), n_bins_common)

print(f"  Common radius range: {r_min:.1f} to {r_max:.1f} pc")
print(f"  Interpolating {len(results)} windows to {n_bins_common} common bins...")

# Interpolate all VSFs to common grid
vsfs_interp = []
for vsf, r in zip(vsfs_individual, radii_individual):
    # Filter out NaN values
    valid = ~np.isnan(vsf) & (vsf > 0)
    if np.sum(valid) > 3:
        vsf_interp = np.interp(r_common, r[valid], vsf[valid], left=np.nan, right=np.nan)
        vsfs_interp.append(vsf_interp)

print(f"  {len(vsfs_interp)} windows successfully interpolated")
print()

# Compute statistics on common grid
vsfs_array = np.array(vsfs_interp)
mean_vsf = np.nanmean(vsfs_array, axis=0)
std_vsf = np.nanstd(vsfs_array, axis=0)
n_windows = np.sum(~np.isnan(vsfs_array), axis=0)
sem_vsf = std_vsf / np.sqrt(n_windows)  # Standard error of the mean

print("Error bar statistics:")
print(f"  Mean VSF: {np.nanmin(mean_vsf):.2e} to {np.nanmax(mean_vsf):.2e}")
print(f"  Std Dev (σ): {np.nanmin(std_vsf):.2e} to {np.nanmax(std_vsf):.2e}")
print(f"  Std Error (SEM): {np.nanmin(sem_vsf):.2e} to {np.nanmax(sem_vsf):.2e}")
print()

# Relative uncertainties
rel_std = std_vsf / mean_vsf * 100
rel_sem = sem_vsf / mean_vsf * 100

print(f"Relative uncertainties:")
print(f"  Median σ/mean: {np.nanmedian(rel_std):.1f}%")
print(f"  Median SEM/mean: {np.nanmedian(rel_sem):.1f}%")
print()

# Create visualization comparing error types
print("Creating error bar comparison plot...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Mean VSF with ±σ (shows actual spatial variation)
ax = axes[0]
for vsf in vsfs_interp:
    ax.plot(r_common, vsf, '-', color='C0', alpha=0.1, linewidth=0.8)

ax.plot(r_common, mean_vsf, 'r-', linewidth=3, label='Mean VSF', zorder=10)
ax.fill_between(r_common, 
                np.maximum(mean_vsf - std_vsf, 1e-5),
                mean_vsf + std_vsf,
                color='red', alpha=0.25, label='±1σ (spatial variation)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Separation (pc)', fontweight='bold', fontsize=11)
ax.set_ylabel('VSF (km²/s²)', fontweight='bold', fontsize=11)
ax.set_title('Error Bars: ±1σ (Spatial Variation)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best')

# Plot 2: Mean VSF with ±SEM (uncertainty in the mean)
ax = axes[1]
for vsf in vsfs_interp:
    ax.plot(r_common, vsf, '-', color='C0', alpha=0.1, linewidth=0.8)

ax.plot(r_common, mean_vsf, 'b-', linewidth=3, label='Mean VSF', zorder=10)
ax.fill_between(r_common,
                np.maximum(mean_vsf - sem_vsf, 1e-5),
                mean_vsf + sem_vsf,
                color='blue', alpha=0.25, label='±SEM (mean uncertainty)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Separation (pc)', fontweight='bold', fontsize=11)
ax.set_ylabel('VSF (km²/s²)', fontweight='bold', fontsize=11)
ax.set_title('Error Bars: ±SEM (Uncertainty in Mean)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best')

# Plot 3: Relative uncertainties
ax = axes[2]
ax.plot(r_common, rel_std, 's-', linewidth=2.5, markersize=4, label='σ/mean', color='red', alpha=0.8)
ax.plot(r_common, rel_sem, 'o-', linewidth=2.5, markersize=4, label='SEM/mean', color='blue', alpha=0.8)
ax.axhline(5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='5% (reference)')

ax.set_xscale('log')
ax.set_xlabel('Separation (pc)', fontweight='bold', fontsize=11)
ax.set_ylabel('Relative Uncertainty (%)', fontweight='bold', fontsize=11)
ax.set_title('Relative Uncertainties', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('Figures/orion_errorbar_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: Figures/orion_errorbar_analysis.png")
print()

# Compare with full FOV
print("Computing full FOV VSF for comparison...")
vsf_full = VSF2D(ra_orion, dec_orion, vel_nii_orion, dist=1500, vel_err=vel_err_orion)
vsf_full_data, vsf_full_err, log_centers_full, counts_full = vsf_full.generate_vsf_2d(
    order=1,
    outname='Figures/orion_full_fov_with_errors',
    bin_resolution=30,
    n_jobs=2
)
r_full = 10**log_centers_full
print("✓ Full FOV VSF computed")
print()

# Comparison plot
print("Creating final comparison with full FOV...")
fig, ax = plt.subplots(figsize=(12, 8))

# Individual windows (thin)
for vsf in vsfs_interp:
    ax.plot(r_common, vsf, '-', color='C0', alpha=0.1, linewidth=0.6, label='Individual windows' if vsf is vsfs_interp[0] else '')

# Mean with error bands
ax.plot(r_common, mean_vsf, 'r-', linewidth=3.5, label='Subwindow mean (1/20 FOV)', zorder=10)
ax.fill_between(r_common,
                np.maximum(mean_vsf - sem_vsf, 1e-5),
                mean_vsf + sem_vsf,
                color='red', alpha=0.3, label='±SEM')

# Full FOV VSF
ax.plot(r_full, vsf_full_data, 'b-', linewidth=3.5, label='Full FOV VSF', zorder=9)
ax.fill_between(r_full,
                np.maximum(vsf_full_data - vsf_full_err, 1e-5),
                vsf_full_data + vsf_full_err,
                color='blue', alpha=0.2, label='Full FOV error')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Separation (pc)', fontweight='bold', fontsize=12)
ax.set_ylabel('VSF (km²/s²)', fontweight='bold', fontsize=12)
ax.set_title('Orion NII: Subwindow Mean with Error Bars vs Full FOV', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best', framealpha=0.95)

plt.tight_layout()
plt.savefig('Figures/orion_mean_vs_full_with_errors.png', dpi=150, bbox_inches='tight')
print("✓ Saved: Figures/orion_mean_vs_full_with_errors.png")
print()

# Detailed summary
print("=" * 70)
print("ERROR BAR SUMMARY")
print("=" * 70)
print()
print("Two types of error bars were computed:")
print()
print("1. ±σ (Standard Deviation)")
print("   Interpretation: Actual range of VSF values across subwindows")
print("   Use this to: Visualize spatial heterogeneity in velocity structure")
print(f"   Magnitude: {np.nanmedian(rel_std):.1f}% of mean (median)")
print()
print("2. ±SEM (Standard Error of Mean)")
print("   Interpretation: Uncertainty in the mean due to finite sampling")
print("   Formula: SEM(r) = σ(r) / √n")
print("   Use this to: Assess statistical uncertainty in the mean estimate")
print(f"   Magnitude: {np.nanmedian(rel_sem):.1f}% of mean (median)")
print()
print("Physical meaning:")
print(f"  • If σ is large: FOV has significant spatial structure")
print(f"  • If σ is small: FOV is homogeneous (VSF is uniform)")
print(f"  • SEM is always smaller than σ by factor √n ≈ √{len(vsfs_interp):.0f} ≈ {np.sqrt(len(vsfs_interp)):.1f}")
print()
print(f"Number of windows: {len(vsfs_interp)}")
print(f"Number of radial bins: {n_bins_common}")
print(f"Coverage: {np.sum(~np.isnan(mean_vsf))} / {n_bins_common} bins with valid data")
print()
print("=" * 70)
