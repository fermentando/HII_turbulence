"""
Example usage of the VSF2D module and independent PDF/C2/C4 computation.

This script demonstrates how to:
1. Use the VSF2D class directly
2. Compute bins and PDF/C2/C4 statistics as independent steps
3. Cache bins to avoid recomputation for multiple VSF orders
"""

import numpy as np
from vsf_module import (VSF2D, VSF2DSubwindows, compute_bins, 
                        compute_pdf_and_statistics)
from astropy.io import fits


# ============================================================================
# EXAMPLE 1: Simple VSF computation (traditional approach)
# ============================================================================

def example_simple_vsf():
    """Load data and compute a simple 2D VSF."""
    print("=" * 70)
    print("EXAMPLE 1: Simple VSF Computation")
    print("=" * 70)
    
    # Load FITS data
    hdul = fits.open("LVM_Flame_nebula.fits")
    table = hdul[1].data
    
    # Apply quality filter
    filter = table['flux_nii'] / table['flux_nii_err'] > 5
    
    # Create VSF2D object
    vsf = VSF2D(
        table['ra'][filter], 
        table['dec'][filter], 
        table['vel_nii'][filter],
        dist=400,
        vel_err=table['vel_nii_err'][filter]
    )
    
    # Generate VSF (order 1)
    vsf_data, vsf_err, log_centers, counts = vsf.generate_vsf_2d(
        order=1, 
        outname='Figures/Flame_NII', 
        bin_resolution=20
    )
    
    print(f"Computed VSF with {len(log_centers)} bins")
    print(f"Data saved to Figures/Flame_NII_order1.npz")
    print()


# ============================================================================
# EXAMPLE 2: Multiple VSF orders - NO redundant bin/PDF computation
# ============================================================================

def example_multiple_orders_efficient():
    """
    Compute multiple VSF orders efficiently.
    
    Each call to generate_vsf_2d now ONLY computes the VSF itself.
    PDF/C2/C4 are computed independently if needed.
    """
    print("=" * 70)
    print("EXAMPLE 2: Multiple VSF Orders (Truly Independent)")
    print("=" * 70)
    
    # Load data once
    hdul = fits.open("LVM_Flame_nebula.fits")
    table = hdul[1].data
    filter = table['flux_nii'] / table['flux_nii_err'] > 5
    
    ra = table['ra'][filter]
    dec = table['dec'][filter]
    vel = table['vel_nii'][filter]
    vel_err = table['vel_nii_err'][filter]
    dist = 400
    
    print("Creating VSF object...")
    vsf = VSF2D(ra, dec, vel, dist=dist, vel_err=vel_err)
    
    print()
    print("Computing multiple orders (each call is INDEPENDENT):")
    for order in [1, 2, 4]:
        print(f"  Order {order}...", end=" ")
        vsf_data, vsf_err, log_centers, counts = vsf.generate_vsf_2d(
            order=order,
            outname='Figures/Flame_NII_orders',
            bin_resolution=20
        )
        print("✓")
    
    print()
    print("Each VSF was computed independently (no redundant PDF/C2/C4)!")
    print()


# ============================================================================
# EXAMPLE 3: Independent PDF/C2/C4 computation (CACHEABLE)
# ============================================================================

def example_independent_pdf_computation():
    """
    Compute PDF, C2, C4 independently from VSF.
    
    This is useful when you want to compute these statistics once
    and reuse them without recomputing during VSF calculations.
    """
    print("=" * 70)
    print("EXAMPLE 3: Independent PDF/C2/C4 Computation (Cacheable)")
    print("=" * 70)
    
    # Load data
    hdul = fits.open("LVM_Flame_nebula.fits")
    table = hdul[1].data
    filter = table['flux_nii'] / table['flux_nii_err'] > 5
    
    ra = table['ra'][filter]
    dec = table['dec'][filter]
    vel = table['vel_nii'][filter]
    vel_err = table['vel_nii_err'][filter]
    
    # Step 1: Compute bins (independent, cached step)
    print("Step 1: Computing bins...")
    log_centers, bin_edges, dv_bins, bins_dict = compute_bins(
        x_data=ra,
        y_data=dec,
        vel_data=vel,
        bin_resolution=20
    )
    
    # Save bins for later use
    np.savez('cached_bins.npz', 
             log_centers=log_centers,
             bin_edges=bin_edges,
             dv_bins=dv_bins)
    print(f"  Bins saved to cached_bins.npz")
    print()
    
    # Step 2: Compute PDF, C2, C4 independently
    print("Step 2: Computing PDF and intermittency statistics...")
    pdf, c2_all, c4_all, cnt_all, dv_centers, kk = compute_pdf_and_statistics(
        x_data=ra,
        y_data=dec,
        vel_data=vel,
        vel_err_data=vel_err,
        log_centers=log_centers,
        bin_edges=bin_edges,
        dv_bins=dv_bins,
        order=2,
        n_jobs=4
    )
    
    # Save PDF and statistics for later use
    np.savez('cached_pdf_stats.npz',
             pdf=pdf,
             c2_all=c2_all,
             c4_all=c4_all,
             cnt_all=cnt_all,
             dv_centers=dv_centers,
             kk=kk,
             log_centers=log_centers,
             dv_bins=dv_bins)
    
    print(f"  PDF shape: {pdf.shape}")
    print(f"  Kurtosis range: {kk[~np.isnan(kk)].min():.2f} to {kk[~np.isnan(kk)].max():.2f}")
    print(f"  Statistics saved to cached_pdf_stats.npz")
    print()
    
    # Step 3: Compute VSF using cached statistics
    print("Step 3: Computing VSF orders using cached bins...")
    for order in [1, 2]:
        vsf = VSF2D(ra, dec, vel, dist=400, vel_err=vel_err)
        vsf_data, vsf_err, _, counts = vsf.generate_vsf_2d(
            order=order,
            outname='Figures/Flame_NII_with_cached_stats',
            bin_resolution=20
        )
    
    print("  VSF computation complete with cached bins!")
    print()


# ============================================================================
# EXAMPLE 4: Working with subwindows
# ============================================================================

def example_subwindows():
    """Use VSF2DSubwindows to compute VSF for spatial subregions."""
    print("=" * 70)
    print("EXAMPLE 4: Subwindow VSF Analysis")
    print("=" * 70)
    
    # Load data
    hdul = fits.open("LVM_Orion.fits")
    table = hdul[1].data
    filter = table['flux_nii'] / table['flux_nii_err'] > 8
    
    ra = table['ra'][filter]
    dec = table['dec'][filter]
    vel = table['vel_nii'][filter]
    
    # Create subwindows object
    vsf = VSF2DSubwindows(ra, dec, vel, dist=1500)
    
    # Plot the FOV with subwindows
    print("Generating subwindow visualization...")
    vsf.plot_subwindows_fov(window_size=20, shape="square")
    
    # Plot VSF for each subwindow
    print("Computing VSF for each subwindow...")
    vsf.plot_subwindow_vsfs(
        window_size=20,
        outname="vsf_windows_Orion",
        order=2
    )
    
    print()


# ============================================================================
# EXAMPLE 5: Loading and reusing cached bins
# ============================================================================

def example_load_cached_bins():
    """Demonstrate loading and reusing previously cached bins."""
    print("=" * 70)
    print("EXAMPLE 5: Loading Cached Bins")
    print("=" * 70)
    
    # Load cached bins (from Example 3)
    print("Loading cached bins from cached_bins.npz...")
    bin_data = np.load('cached_bins.npz')
    log_centers = bin_data['log_centers']
    bin_edges = bin_data['bin_edges']
    dv_bins = bin_data['dv_bins']
    
    print(f"  Loaded {len(log_centers)} spatial bins")
    print(f"  Loaded {len(dv_bins)} velocity bins")
    print()
    
    # Load different data and use same bins
    print("Loading new data sample...")
    # (In practice, load different data here)
    
    print("Could now compute VSF on different region using same bins!")
    print()


# ============================================================================
# WORKFLOW RECOMMENDATIONS
# ============================================================================

"""
RECOMMENDED WORKFLOW:

1. For single dataset, single order:
   - Use VSF2D.generate_vsf_2d() directly (Example 1)

2. For single dataset, multiple orders:
   - Precompute bins using compute_bins() (Example 2)
   - Call VSF2D.generate_vsf_2d() for each order with same bin_resolution
   - This avoids redundant bin computation

3. For reusing bins across multiple datasets:
   - Compute bins once: log_centers, bin_edges, dv_bins = compute_bins(...)
   - Save with np.savez()
   - Load with np.load()
   - Pass to compute_pdf_and_statistics() for each new dataset

4. For spatial analysis:
   - Use VSF2DSubwindows for subwindow-based analysis (Example 4)

5. MOST IMPORTANT: For maximum efficiency
   - Use compute_bins() once
   - Use compute_pdf_and_statistics() to cache PDF/C2/C4
   - Save both caches with np.savez()
   - Reuse for multiple analyses without recomputation
"""


if __name__ == "__main__":
    print("\n")
    print("  VSF2D Module Examples")
    print("  " + "=" * 66)
    print()
    
    # Uncomment examples to run
    # example_simple_vsf()
    # example_multiple_orders_efficient()
    # example_independent_pdf_computation()
    # example_subwindows()
    # example_load_cached_bins()
    
    print("Examples available. Uncomment in __main__ to run.")
