"""
Orion Subwindow VSF Analysis with Refactored Module

This script:
1. Loads Orion NII data
2. Subdivides the FOV into windows of size 1/20 of the total range
3. Computes individual VSFs for each window
4. Computes and plots the mean VSF
5. Compares with the original full FOV VSF
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys

sys.path.insert(0, '/Users/fernando/PhDResearch/LVM stuff')
from vsf_module import VSF2D, VSF2DSubwindows


def compute_window_size_from_fov():
    """
    Load Orion data and compute appropriate window size (1/20 of FOV).
    """
    print("=" * 70)
    print("ORION SUBWINDOW VSF ANALYSIS")
    print("=" * 70)
    print()
    
    # Load Orion data
    print("Loading Orion NII data...")
    rosette_hdul = fits.open("LVM_Orion.fits")
    table_rosette = rosette_hdul[1].data
    filter_orion = table_rosette['flux_nii'] / table_rosette['flux_nii_err'] > 5
    
    ra_orion = table_rosette['ra'][filter_orion]
    dec_orion = table_rosette['dec'][filter_orion]
    vel_nii_orion = table_rosette['vel_nii'][filter_orion]
    vel_err_orion = table_rosette['vel_nii_err'][filter_orion]
    
    print(f"  Data points: {len(ra_orion)}")
    print(f"  RA range: {ra_orion.min():.2f} to {ra_orion.max():.2f}")
    print(f"  Dec range: {dec_orion.min():.2f} to {dec_orion.max():.2f}")
    print()
    
    # Calculate FOV in degrees
    ra_range = ra_orion.max() - ra_orion.min()
    dec_range = dec_orion.max() - dec_orion.min()
    
    print(f"FOV ranges:")
    print(f"  RA range:  {ra_range:.3f} degrees")
    print(f"  Dec range: {dec_range:.3f} degrees")
    print()
    
    # Create VSF object to get physical coordinates
    vsf_phys = VSF2D(ra_orion, dec_orion, vel_nii_orion, dist=1500, vel_err=vel_err_orion)
    x_phys = vsf_phys.x_physical
    y_phys = vsf_phys.y_physical
    
    # Calculate physical FOV ranges
    x_range = x_phys.max() - x_phys.min()
    y_range = y_phys.max() - y_phys.min()
    
    print(f"Physical FOV ranges (1500 pc distance):")
    print(f"  X range: {x_range:.1f} pc")
    print(f"  Y range: {y_range:.1f} pc")
    print()
    
    # Window size = 1/20 of FOV range
    window_size = max(x_range, y_range) / 20
    
    print(f"Window size (1/20 of max FOV): {window_size:.1f} pc")
    print()
    
    return ra_orion, dec_orion, vel_nii_orion, vel_err_orion, window_size


def run_subwindow_vsf(ra_orion, dec_orion, vel_nii_orion, vel_err_orion, window_size):
    """
    Compute subwindow VSFs and generate plots.
    """
    print("Creating VSF2DSubwindows object...")
    vsf_subwindows = VSF2DSubwindows(ra_orion, dec_orion, vel_nii_orion, dist=1500)
    print("✓ Object created")
    print()
    
    # Visualize the subwindows
    print("Generating FOV visualization with subwindows...")
    vsf_subwindows.plot_subwindows_fov(
        window_size=window_size, 
        shape="square", 
        show_all=True
    )
    print("✓ FOV plot saved")
    print()
    
    # Compute individual subwindow VSFs
    print(f"Computing VSFs for each subwindow (window_size={window_size:.1f} pc)...")
    print("  This may take a few minutes for large datasets...")
    results, x_edges, y_edges = vsf_subwindows.compute_subwindow_vsfs(
        window_size=window_size,
        order=1,
        n_jobs=2,  # Reduced from 4 to manage memory
        bin_resolution=30
    )
    
    print(f"✓ Computed {len(results)} subwindow VSFs")
    print()
    
    # Extract and compute statistics
    print("Computing statistics across subwindows...")
    vsfs_list = [vsf for _, _, vsf, _ in results]
    log_centers_list = [log_c for _, _, _, log_c in results]
    
    # Use first log_centers for binning
    log_centers = log_centers_list[0]
    r_phys = 10**log_centers
    
    print(f"  Number of bins per VSF: {len(log_centers)}")
    print(f"  Number of subwindows: {len(vsfs_list)}")
    print()
    
    # Compute mean VSF
    vsfs_array = np.array(vsfs_list)
    mean_vsf = np.nanmean(vsfs_array, axis=0)
    std_vsf = np.nanstd(vsfs_array, axis=0)
    
    print(f"Mean VSF range: {np.nanmin(mean_vsf):.2e} to {np.nanmax(mean_vsf):.2e}")
    print()
    
    return vsf_subwindows, results, log_centers, r_phys, mean_vsf, std_vsf, window_size


def plot_subwindow_vsfs_detailed(vsf_subwindows, results, log_centers, r_phys, mean_vsf, std_vsf, window_size):
    """
    Create detailed subwindow VSF plots.
    """
    print("Creating subwindow VSF overlay plot...")
    vsf_subwindows.plot_subwindow_vsfs(
        window_size=window_size,
        outname="Figures/orion_subwindow",
        order=1,
        alpha=0.15,
        show_mean=True,
        n_common=60
    )
    print("✓ Overlay plot saved")
    print()
    
    # Create custom comparison plot
    print("Creating detailed comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Individual subwindow VSFs with mean
    for _, _, vsf, log_c in results:
        r = 10**log_c
        ax1.plot(r, vsf, '-', color='C0', alpha=0.15, linewidth=1)
    
    ax1.plot(r_phys, mean_vsf, 'r-', linewidth=3, label='Mean VSF', zorder=10)
    ax1.fill_between(r_phys, 
                     np.nanmax([mean_vsf - std_vsf, np.ones_like(mean_vsf) * 1e-5], axis=0),
                     mean_vsf + std_vsf,
                     color='red', alpha=0.2, label='±1 std')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Separation (pc)', fontsize=12)
    ax1.set_ylabel('VSF', fontsize=12)
    ax1.set_title(f'Subwindow VSFs (window={window_size:.0f} pc)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Standard deviation / Mean ratio (quality metric)
    snr = np.where(mean_vsf > 0, std_vsf / mean_vsf, np.nan)
    ax2.plot(r_phys, snr, 'o-', color='purple', linewidth=2, markersize=4)
    ax2.set_xscale('log')
    ax2.set_xlabel('Separation (pc)', fontsize=12)
    ax2.set_ylabel('Std / Mean', fontsize=12)
    ax2.set_title('Relative Uncertainty Across Subwindows', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures/orion_subwindow_detailed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Detailed plot saved")
    print()


def compare_with_full_fov(ra_orion, dec_orion, vel_nii_orion, vel_err_orion, log_centers, r_phys, mean_vsf):
    """
    Compare subwindow mean VSF with full FOV VSF.
    """
    print("Computing full FOV VSF for comparison...")
    vsf_full = VSF2D(ra_orion, dec_orion, vel_nii_orion, dist=1500, vel_err=vel_err_orion)
    vsf_full_data, vsf_full_err, log_centers_full, counts_full = vsf_full.generate_vsf_2d(
        order=1,
        outname='Figures/orion_full_fov',
        bin_resolution=30
    )
    
    r_full = 10**log_centers_full
    
    print("✓ Full FOV VSF computed")
    print()
    
    # Create comparison plot
    print("Creating comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot individual subwindow VSFs
    results_dummy = []  # We'll skip individual plots for clarity
    
    # Plot full FOV
    ax.plot(r_full, vsf_full_data, 'b-', linewidth=2.5, 
            label='Full FOV VSF', marker='o', markersize=4, alpha=0.8)
    
    # Plot mean of subwindows
    ax.plot(r_phys, mean_vsf, 'r--', linewidth=2.5, 
            label=f'Mean Subwindow VSF (1/20 FOV)', marker='s', markersize=4, alpha=0.8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Separation (pc)', fontsize=12)
    ax.set_ylabel('VSF', fontsize=12)
    ax.set_title('Orion NII: Full FOV vs Subwindow Mean VSF', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('Figures/orion_full_vs_subwindow.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Comparison plot saved")
    print()
    
    # Compute differences
    print("Numerical comparison:")
    # Interpolate to common grid
    common_r = np.logspace(np.log10(max(r_phys.min(), r_full.min())), 
                          np.log10(min(r_phys.max(), r_full.max())), 50)
    
    full_interp = np.interp(common_r, r_full, vsf_full_data, left=np.nan, right=np.nan)
    mean_interp = np.interp(common_r, r_phys, mean_vsf, left=np.nan, right=np.nan)
    
    diff = np.abs(full_interp - mean_interp) / np.abs(full_interp)
    
    print(f"  Mean relative difference: {np.nanmean(diff):.2%}")
    print(f"  Max relative difference: {np.nanmax(diff):.2%}")
    print(f"  RMS relative difference: {np.nanstd(diff):.2%}")
    print()


def main():
    """Main execution function."""
    try:
        # Step 1: Calculate window size
        ra, dec, vel, vel_err, win_size = compute_window_size_from_fov()
        
        # Step 2: Run subwindow analysis
        vsf_sub, results, log_c, r, mean_v, std_v, w_size = run_subwindow_vsf(
            ra, dec, vel, vel_err, win_size
        )
        
        # Step 3: Plot subwindow VSFs
        plot_subwindow_vsfs_detailed(vsf_sub, results, log_c, r, mean_v, std_v, w_size)
        
        # Step 4: Compare with full FOV
        compare_with_full_fov(ra, dec, vel, vel_err, log_c, r, mean_v)
        
        # Summary
        print("=" * 70)
        print("✓ ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        print("Output files:")
        print("  - Figures/orion_subwindow_vsfs_overlay.png")
        print("  - Figures/orion_subwindow_detailed.png")
        print("  - Figures/orion_full_vs_subwindow.png")
        print("  - Figures/orion_full_fov_order1.npz")
        print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
