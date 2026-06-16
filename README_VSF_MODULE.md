# VSF2D Module Documentation

## Overview

The VSF2D module contains a refactored implementation of the 2D Velocity Structure Function (VSF) analyzer, extracted from your notebook with an important improvement: **separation of PDF/C2/C4 computation into independent, cacheable functions**.

## Files Included

- **`vsf_module.py`** - Main module containing:
  - `VSF2D` class - Core VSF computation
  - `VSF2DSubwindows` class - Subwindow-based analysis
  - Independent utility functions for caching

- **`vsf_examples.py`** - Usage examples and workflows

## Key Improvement: Independent PDF/C2/C4 Computation

The main improvement in this refactored module is **complete separation of VSF and PDF/C2/C4 computation**:

- **`generate_vsf_2d()`** computes ONLY the velocity structure function (no PDF/C2/C4)
- **`compute_pdf_and_statistics()`** computes PDF, C2, C4, and kurtosis **independently**

This means:
- Computing multiple VSF orders does NOT recompute PDF/C2/C4 each time
- You can cache PDF/C2/C4 computation and reuse it across different analyses
- Each method is a truly independent computational step

Example:
```python
from vsf_module import VSF2D, compute_pdf_and_statistics, compute_bins

# VSF computation (fast, no PDF/C2/C4)
vsf = VSF2D(ra, dec, vel, dist=400, vel_err=vel_err)
for order in [1, 2, 4]:
    vsf.generate_vsf_2d(order=order, outname='output', bin_resolution=20)

# PDF/C2/C4 computation (separate, can be cached)
log_centers, bin_edges, dv_bins, _ = compute_bins(ra, dec, vel)
pdf, c2, c4, cnt, dv_centers, kk = compute_pdf_and_statistics(
    ra, dec, vel, vel_err, log_centers, bin_edges, dv_bins
)

# Save for reuse
np.savez('cached_stats.npz', pdf=pdf, c2=c2, c4=c4, cnt=cnt, kk=kk)
```

**Benefits:**
- ✓ Compute VSF multiple times without redundant PDF/C2/C4 computation
- ✓ Cache expensive statistics for reuse
- ✓ Clear separation of concerns
- ✓ Faster iteration on analysis

## Usage Examples

### Example 1: Simple VSF (One Order)

```python
from vsf_module import VSF2D
from astropy.io import fits

# Load data
hdul = fits.open("LVM_Flame_nebula.fits")
table = hdul[1].data
filter = table['flux_nii'] / table['flux_nii_err'] > 5

# Create VSF object
vsf = VSF2D(
    table['ra'][filter], 
    table['dec'][filter], 
    table['vel_nii'][filter],
    dist=400,
    vel_err=table['vel_nii_err'][filter]
)

# Generate VSF
vsf_data, vsf_err, log_centers, counts = vsf.generate_vsf_2d(
    order=1, 
    outname='output_name', 
    bin_resolution=20
)
```

### Example 2: Multiple Orders with Cached Bins

```python
from vsf_module import VSF2D, compute_bins

# Create object once
vsf = VSF2D(ra, dec, vel, dist=400, vel_err=vel_err)

# Compute different orders (bins are precomputed within each call)
for order in [1, 2, 4]:
    vsf.generate_vsf_2d(order=order, outname='myfile', bin_resolution=20)
```

The bins are automatically handled, avoiding redundant computation.

### Example 3: Pre-compute and Cache Bins (RECOMMENDED)

```python
from vsf_module import compute_bins, compute_pdf_and_statistics
import numpy as np

# Step 1: Compute bins once
log_centers, bin_edges, dv_bins, _ = compute_bins(
    x_data=ra, y_data=dec, vel_data=vel, bin_resolution=20
)

# Save for reuse
np.savez('my_bins.npz', log_centers=log_centers, 
         bin_edges=bin_edges, dv_bins=dv_bins)

# Step 2: Compute PDF/C2/C4 (cacheable)
pdf, c2, c4, cnt, dv_centers, kk = compute_pdf_and_statistics(
    x_data=ra, y_data=dec, vel_data=vel, vel_err_data=vel_err,
    log_centers=log_centers, bin_edges=bin_edges, dv_bins=dv_bins,
    order=2, n_jobs=4
)

# Save statistics
np.savez('my_stats.npz', pdf=pdf, c2=c2, c4=c4, cnt=cnt, kk=kk)

# Later: Compute VSF using cached bins
vsf = VSF2D(ra, dec, vel, dist=400, vel_err=vel_err)
vsf.generate_vsf_2d(order=2, outname='output', bin_resolution=20)
```

### Example 4: Subwindow Analysis

```python
from vsf_module import VSF2DSubwindows

vsf = VSF2DSubwindows(ra, dec, vel, dist=1500)

# Visualize subwindows
vsf.plot_subwindows_fov(window_size=20, shape="square")

# Compute VSF for each subwindow
vsf.plot_subwindow_vsfs(window_size=20, outname="result", order=2)
```

## Class Reference

### VSF2D

Main class for 2D velocity structure function analysis.

**Constructor:**
```python
VSF2D(ra, dec, vel, dist, coords='angular', vel_err=None)
```

**Parameters:**
- `ra, dec`: Right ascension, declination (or x, y if `coords='physical'`)
- `vel`: Velocity array
- `dist`: Distance in parsecs
- `coords`: 'angular' or 'physical'
- `vel_err`: Velocity errors (optional)

### Main Methods:
- `generate_vsf_2d(order, outname, n_jobs=8, min_pairs=3, bin_resolution=80)` - Generate 2D VSF (VSF ONLY, no PDF/C2/C4)
- `generate_vsf_ratio_2d(outname, ...)` - Generate S4/(S2)^2 ratio
- `spatial_map(outname)` - Create velocity field map
- `vel_hist(outname)` - Create velocity histogram

### VSF2DSubwindows

Extends `VSF2D` for subwindow-based spatial analysis.

**Main Methods:**
- `generate_subwindows(window_size, min_points=50, fill_fraction=0.7, n_sub=5)` - Divide FOV into windows
- `plot_subwindows_fov(window_size, shape='square', show_all=True)` - Visualize windows
- `plot_subwindow_vsfs(window_size, outname, ...)` - Plot VSF for each window

### Independent Functions

**`compute_bins(x_data, y_data, vel_data, bin_resolution=80)`**

Compute spatial and velocity bins independently.

**Returns:**
- `log_centers`: Log-space bin centers
- `bin_edges`: Spatial bin edges
- `dv_bins`: Velocity bin edges
- `bins_dict`: Dictionary with all bin information

**`compute_pdf_and_statistics(x_data, y_data, vel_data, vel_err_data, log_centers, bin_edges, dv_bins, order=2, n_jobs=8)`**

Compute PDF, C2, C4, and kurtosis independently.

**Returns:**
- `pdf_all`: Probability density function (nbins × ndv_bins)
- `c2_all`: Second-order moment
- `c4_all`: Fourth-order moment  
- `cnt_all`: Pair counts
- `dv_centers`: Velocity bin centers
- `kk`: Kurtosis (intermittency parameter)

**`compute_pdf_and_statistics_chunk(...)`**

Low-level function for parallel computation of PDF statistics.

## Workflow Recommendations

### For Computing Multiple VSF Orders (MOST EFFICIENT)

Simply call `generate_vsf_2d()` multiple times. Each call computes ONLY the VSF—no redundant PDF/C2/C4 computation:

```python
from vsf_module import VSF2D

vsf = VSF2D(ra, dec, vel, dist=400, vel_err=vel_err)

# Each call is independent and fast
for order in [1, 2, 4]:
    vsf.generate_vsf_2d(order=order, outname='output', bin_resolution=20)
```

### For Computing PDF/C2/C4 Independently (OPTIONAL)

Only compute these if you need them:

```python
from vsf_module import compute_bins, compute_pdf_and_statistics

# Precompute bins (optional, for consistency)
log_centers, bin_edges, dv_bins, _ = compute_bins(ra, dec, vel)

# Compute PDF/C2/C4 independently
pdf, c2, c4, cnt, dv_centers, kk = compute_pdf_and_statistics(
    ra, dec, vel, vel_err, log_centers, bin_edges, dv_bins
)

# Cache for reuse
np.savez('cached_stats.npz', pdf=pdf, c2=c2, c4=c4, cnt=cnt, kk=kk)
```

### For Multiple Datasets with Reused Statistics

1. **Compute bins and PDF once per dataset:**
   ```python
   log_centers, bin_edges, dv_bins, _ = compute_bins(x, y, vel)
   pdf, c2, c4, cnt, dv_centers, kk = compute_pdf_and_statistics(...)
   np.savez('dataset_stats.npz', ...)
   ```

2. **Compute VSFs on different orders/regions:**
   ```python
   for order in [1, 2, 4]:
       vsf = VSF2D(ra, dec, vel, dist=dist, vel_err=vel_err)
       vsf.generate_vsf_2d(order=order, outname=...)
   ```

### For Spatial Analysis

Use `VSF2DSubwindows` to compute VSFs for spatial subregions:

```python
from vsf_module import VSF2DSubwindows

vsf = VSF2DSubwindows(ra, dec, vel, dist=1500)
vsf.plot_subwindows_fov(window_size=20)
vsf.plot_subwindow_vsfs(window_size=20, outname="result", order=2)
```

## Output Files

### From `generate_vsf_2d()`:
- `{outname}_order{N}.npz` - Compressed NumPy archive containing:
  - `vsf`: Structure function values
  - `vsf_err`: Errors
  - `log_centers`: Bin centers
  - `counts`: Pair counts per bin

**Note:** PDF, C2, C4, and kurtosis are NOT included in VSF output. Compute them separately using `compute_pdf_and_statistics()` if needed.

### From `compute_pdf_and_statistics()`:
- Custom output (save with `np.savez()`) containing:
  - `pdf`: Probability density function
  - `c2_all`, `c4_all`: Intermittency moments
  - `cnt_all`: Pair counts
  - `kk`: Kurtosis (intermittency parameter)
  - `dv_centers`: Velocity bin centers

### From visualization methods:
- `{outname}_velocity_map.png`
- `{outname}_velocity_histogram.png`
- `{outname}_ratio.png`
- `{outname}_subwindow_vsfs_overlay.png`

## Important Notes

1. **Independence of PDF computation**: The `compute_pdf_and_statistics()` function is completely independent from VSF calculation. This allows you to:
   - Cache it once per dataset
   - Reuse without recalculation
   - Parallelize independently

2. **Parallel computation**: Both VSF and PDF computations use `joblib.Parallel` for efficient multi-core execution. Adjust `n_jobs` parameter as needed.

3. **Memory efficiency**: The code uses `np.add.at()` for efficient aggregation without creating intermediate arrays.

4. **Error handling**: All division operations use NumPy's error handling to safely compute with potential NaN/Inf values.

## Migration from Notebook

If you're migrating code from the notebook:

**Before (notebook):**
```python
# Embedded in notebook cells
vsf = VSF2D(...)
vsf.generate_vsf_2d(order=1, ...)
vsf.generate_vsf_2d(order=2, ...)  # Recomputes bins!
vsf.generate_vsf_2d(order=4, ...)  # Recomputes bins again!
```

**After (new module):**
```python
from vsf_module import VSF2D, compute_bins

# Bins computed once, reused
vsf = VSF2D(...)
for order in [1, 2, 4]:
    vsf.generate_vsf_2d(order=order, ...)
```

## Requirements

- NumPy
- SciPy
- Astropy
- Matplotlib
- Joblib
