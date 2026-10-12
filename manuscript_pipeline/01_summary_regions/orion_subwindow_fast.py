"""
Orion Subwindow VSF Analysis - Fast demonstration
Uses 1/5 FOV windows (minimal subwindows) and sequential processing
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import seaborn as sns
import sys
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D, VSF2DSubwindows

style_path = REPO_ROOT / "custom_plot.mplstyle"
if style_path.exists():
    plt.style.use(str(style_path))


def resolve_input_path(path_arg: str):
    """Resolve FITS paths from either repo root or this script directory."""
    path = Path(path_arg)
    if path.exists():
        return str(path)
    root_path = REPO_ROOT / path_arg
    if root_path.exists():
        return str(root_path)
    return str(path)


# Get input file from command line argument
filename = resolve_input_path(sys.argv[1] if len(sys.argv) > 1 else "LVM_Orion.fits")
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
SPECIAL_CENTER_REGIONS = {"Orion", "RCW52"}
N_TOTAL_BOOTSTRAP = 100
NPZ_DIR = REPO_ROOT / "NPZ"
NPZ_DIR.mkdir(exist_ok=True)


def total_vsf_cache_paths(object_name, line_name):
    """Prefer the fast cache, but allow reuse of the older full-FOV cache."""
    return [
        NPZ_DIR / f"VSF_{object_name}_{line_name}_total_fullfov_fast.npz",
        NPZ_DIR / f"VSF_{object_name}_{line_name}_total_fullfov.npz",
    ]


def load_total_vsf_cache(object_name, line_name):
    for npz_name in total_vsf_cache_paths(object_name, line_name):
        if not npz_name.exists():
            continue

        try:
            with np.load(npz_name) as data:
                missing = {"vsf", "vsf_err", "log_centers", "counts"} - set(data.files)
                if missing:
                    raise KeyError(f"missing keys: {sorted(missing)}")

                vsf = np.asarray(data["vsf"], dtype=float)
                vsf_err = np.asarray(data["vsf_err"], dtype=float)
                log_centers = np.asarray(data["log_centers"], dtype=float)
                counts = np.asarray(data["counts"], dtype=float)

            valid = (
                np.isfinite(vsf)
                & np.isfinite(vsf_err)
                & np.isfinite(log_centers)
                & np.isfinite(counts)
                & (vsf > 0)
                & (counts > 0)
            )
            if not np.any(valid):
                raise ValueError("no valid VSF bins")

            print(f"    Loading cached {line_name} total VSF: {npz_name}")
            return {
                "r": 10**log_centers[valid],
                "vsf": vsf[valid],
                "vsf_err": vsf_err[valid],
                "counts": counts[valid],
            }
        except Exception as exc:
            print(f"    Cache not usable for {line_name}: {npz_name} ({exc}); trying next option")

    return None


def save_total_vsf_cache(object_name, line_name, r, vsf, vsf_err, counts):
    npz_name = total_vsf_cache_paths(object_name, line_name)[0]
    np.savez(
        npz_name,
        vsf=vsf,
        vsf_err=vsf_err,
        log_centers=np.log10(r),
        counts=counts,
    )
    print(f"    Saved {npz_name}")


def subwindow_vsf_cache_path(object_name, line_name):
    return NPZ_DIR / f"VSF_{object_name}_{line_name}_centered_subwindows_fast.npz"


def _object_array(items):
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def load_subwindow_vsf_cache(object_name, line_name, window_size, n_points):
    npz_name = subwindow_vsf_cache_path(object_name, line_name)
    if not npz_name.exists():
        return None

    required = {
        "cache_version",
        "window_size",
        "n_points",
        "result_i",
        "result_j",
        "result_vsf",
        "result_log_centers",
        "x_edges",
        "y_edges",
        "center_i",
        "center_j",
        "center_included",
        "common_r",
        "vsfs_array",
        "mean_vsf",
        "std_vsf",
        "median_vsf",
        "median_err",
        "lower_vsf",
        "upper_vsf",
    }

    try:
        with np.load(npz_name, allow_pickle=True) as data:
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"missing keys: {sorted(missing)}")
            if int(np.asarray(data["cache_version"]).item()) != 1:
                raise ValueError("unsupported cache version")
            if not np.isclose(float(np.asarray(data["window_size"]).item()), window_size, rtol=1e-6, atol=1e-6):
                raise ValueError("window size changed")
            if int(np.asarray(data["n_points"]).item()) != int(n_points):
                raise ValueError("number of line points changed")

            result_i = np.asarray(data["result_i"], dtype=int)
            result_j = np.asarray(data["result_j"], dtype=int)
            result_vsf = data["result_vsf"]
            result_log_centers = data["result_log_centers"]
            results = [
                (
                    int(i),
                    int(j),
                    np.asarray(vsf, dtype=float),
                    np.asarray(log_centers, dtype=float),
                )
                for i, j, vsf, log_centers in zip(result_i, result_j, result_vsf, result_log_centers)
            ]

            cache = {
                "results": results,
                "x_edges": np.asarray(data["x_edges"], dtype=float),
                "y_edges": np.asarray(data["y_edges"], dtype=float),
                "center_window": (
                    int(np.asarray(data["center_i"]).item()),
                    int(np.asarray(data["center_j"]).item()),
                    bool(np.asarray(data["center_included"]).item()),
                ),
                "common_r": np.asarray(data["common_r"], dtype=float),
                "vsfs_array": np.asarray(data["vsfs_array"], dtype=float),
                "mean_vsf": np.asarray(data["mean_vsf"], dtype=float),
                "std_vsf": np.asarray(data["std_vsf"], dtype=float),
                "median_vsf": np.asarray(data["median_vsf"], dtype=float),
                "median_err": np.asarray(data["median_err"], dtype=float),
                "lower_vsf": np.asarray(data["lower_vsf"], dtype=float),
                "upper_vsf": np.asarray(data["upper_vsf"], dtype=float),
            }
    except Exception as exc:
        print(f"    Subwindow cache not usable for {line_name}: {npz_name} ({exc}); recomputing")
        return None

    if not cache["results"]:
        print(f"    Subwindow cache not usable for {line_name}: {npz_name} (no subwindows); recomputing")
        return None

    print(f"    Loading cached {line_name} subwindow VSFs: {npz_name}")
    return cache


def save_subwindow_vsf_cache(
    object_name,
    line_name,
    window_size,
    n_points,
    results,
    x_edges,
    y_edges,
    center_window,
    common_r,
    vsfs_array,
    mean_vsf,
    std_vsf,
    median_vsf,
    median_err,
    lower_vsf,
    upper_vsf,
):
    npz_name = subwindow_vsf_cache_path(object_name, line_name)
    np.savez_compressed(
        npz_name,
        cache_version=1,
        window_size=window_size,
        n_points=n_points,
        result_i=np.asarray([i for i, _, _, _ in results], dtype=int),
        result_j=np.asarray([j for _, j, _, _ in results], dtype=int),
        result_vsf=_object_array([np.asarray(vsf, dtype=float) for _, _, vsf, _ in results]),
        result_log_centers=_object_array([np.asarray(log_centers, dtype=float) for _, _, _, log_centers in results]),
        x_edges=np.asarray(x_edges, dtype=float),
        y_edges=np.asarray(y_edges, dtype=float),
        center_i=center_window[0],
        center_j=center_window[1],
        center_included=center_window[2],
        common_r=np.asarray(common_r, dtype=float),
        vsfs_array=np.asarray(vsfs_array, dtype=float),
        mean_vsf=np.asarray(mean_vsf, dtype=float),
        std_vsf=np.asarray(std_vsf, dtype=float),
        median_vsf=np.asarray(median_vsf, dtype=float),
        median_err=np.asarray(median_err, dtype=float),
        lower_vsf=np.asarray(lower_vsf, dtype=float),
        upper_vsf=np.asarray(upper_vsf, dtype=float),
    )
    print(f"    Saved {npz_name}")


def sqrt_vsf(values):
    """Convert S2 values to S2^1/2 for plotting."""
    values = np.asarray(values, dtype=float)
    return np.sqrt(np.clip(values, 0, None))


def jackknife_median_err(vsfs_array, min_valid_windows=4):
    """Leave-one-subwindow-out jackknife error on the across-window median VSF.

    Each row of `vsfs_array` is one subwindow's VSF curve (already
    interpolated onto the common r grid); treating each subwindow as one
    jackknife block mirrors the spatial block-jackknife used for the
    full-FOV VSF, just with the subwindows themselves standing in for the
    blocks instead of a separate grid.

    Returns the jackknife standard error per bin (NaN where fewer than
    `min_valid_windows` windows have finite data in that bin).
    """
    n_windows = vsfs_array.shape[0]
    if n_windows < min_valid_windows:
        return np.full(vsfs_array.shape[1], np.nan)

    replicate_medians = np.empty_like(vsfs_array)
    for i in range(n_windows):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            replicate_medians[i] = np.nanmedian(np.delete(vsfs_array, i, axis=0), axis=0)

    n_valid = np.sum(np.isfinite(replicate_medians), axis=0)
    jack_mean = np.nanmean(replicate_medians, axis=0)
    sq_dev = (replicate_medians - jack_mean)**2
    with np.errstate(invalid='ignore'):
        var = np.nansum(sq_dev, axis=0) * (n_valid - 1) / np.maximum(n_valid, 1)
    err = np.sqrt(var)
    return np.where(n_valid >= min_valid_windows, err, np.nan)


def bin_velocity_field(ra, dec, vel, oversample=1.3):
    """Bin a scattered (ra, dec, vel) point cloud into a clean weighted-mean raster.

    Replaces an alpha-blended scatter of tens of thousands of points (which
    reads as noisy/speckled, especially at the fiber-to-fiber scale) with a
    smooth image at roughly the data's own native resolution -- estimated
    from the median nearest-neighbor spacing, not an arbitrary bin count, so
    the map doesn't get over-smoothed or artificially blocky.

    Returns
    -------
    ra_edges, dec_edges : arrays
        Bin edges, for use with pcolormesh(ra_edges, dec_edges, mean_vel.T).
    mean_vel : 2D masked array
        Weighted-mean velocity per cell (shape (len(ra_edges)-1, len(dec_edges)-1)),
        NaN-masked where no points fall in a cell.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    vel = np.asarray(vel, dtype=float)

    coords = np.column_stack((ra, dec))
    tree = cKDTree(coords)
    sample = coords if len(coords) <= 8000 else coords[np.random.default_rng(0).choice(len(coords), 8000, replace=False)]
    dists, _ = tree.query(sample, k=2)
    spacing = np.median(dists[:, 1])
    bin_size = spacing / oversample

    ra_edges = np.arange(ra.min(), ra.max() + bin_size, bin_size)
    dec_edges = np.arange(dec.min(), dec.max() + bin_size, bin_size)

    counts, _, _ = np.histogram2d(ra, dec, bins=[ra_edges, dec_edges])
    sums, _, _ = np.histogram2d(ra, dec, bins=[ra_edges, dec_edges], weights=vel)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_vel = np.where(counts > 0, sums / counts, np.nan)
    return ra_edges, dec_edges, np.ma.masked_invalid(mean_vel)


def tighten_sky_axes(ax, ra_values, dec_values, pad_fraction=0.01):
    """Set tight RA/Dec limits while preserving local sky aspect."""
    ra_values = np.asarray(ra_values, dtype=float)
    dec_values = np.asarray(dec_values, dtype=float)
    valid = np.isfinite(ra_values) & np.isfinite(dec_values)
    if not np.any(valid):
        return

    ra_min, ra_max = np.nanmin(ra_values[valid]), np.nanmax(ra_values[valid])
    dec_min, dec_max = np.nanmin(dec_values[valid]), np.nanmax(dec_values[valid])
    ra_pad = max((ra_max - ra_min) * pad_fraction, 1e-6)
    dec_pad = max((dec_max - dec_min) * pad_fraction, 1e-6)

    ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
    ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
    dec_mid = 0.5 * (dec_min + dec_max)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(dec_mid)), adjustable="box")


def build_valid_nii_field_mask(table_data):
    """Use the full valid NII field; no flux-cut contour or brightest-source selection."""
    ra_all = np.asarray(table_data["ra"], dtype=float)
    dec_all = np.asarray(table_data["dec"], dtype=float)
    flux_nii = np.asarray(table_data["flux_nii"], dtype=float)
    flux_nii_err = np.asarray(table_data["flux_nii_err"], dtype=float)
    vel_nii = np.asarray(table_data["vel_nii"], dtype=float)

    valid_nii = np.isfinite(ra_all) & np.isfinite(dec_all) & np.isfinite(flux_nii) & np.isfinite(flux_nii_err)
    valid_nii &= (flux_nii_err > 0) & (flux_nii > 0) & (flux_nii / flux_nii_err > 5) & np.isfinite(vel_nii)

    return valid_nii

# Load data for each line
ra_all = []
dec_all = []
vel_all = []
vel_err_all = []
line_data = {}
object_name = base_name.replace("LVM_", "")
selected_nii_subregion = build_valid_nii_field_mask(table_data)

nii_flux_all = np.asarray(table_data['flux_nii'], dtype=float)
nii_flux_err_all = np.asarray(table_data['flux_nii_err'], dtype=float)
nii_vel_all = np.asarray(table_data['vel_nii'], dtype=float)
nii_valid_all = (
    np.isfinite(nii_flux_all)
    & np.isfinite(nii_flux_err_all)
    & (nii_flux_err_all > 0)
    & (nii_flux_all / nii_flux_err_all > 5)
    & np.isfinite(nii_vel_all)
    & np.isfinite(table_data['ra'])
    & np.isfinite(table_data['dec'])
)
velocity_background = {
    'ra': np.asarray(table_data['ra'], dtype=float)[nii_valid_all],
    'dec': np.asarray(table_data['dec'], dtype=float)[nii_valid_all],
    'vel': nii_vel_all[nii_valid_all],
    'used': selected_nii_subregion[nii_valid_all],
}

print(
    f"Full valid NII field: {int(np.sum(selected_nii_subregion)):,} spaxels "
    "(S/N > 5; no contour or brightest-source selection)"
)

for line, name in zip(lines, line_names):
    flux_col = f'flux_{line}'
    flux_err_col = f'flux_{line}_err'
    vel_col = f'vel_{line}'
    vel_err_col = f'vel_{line}_err'
    
    flux_all = np.asarray(table_data[flux_col], dtype=float)
    flux_err_all = np.asarray(table_data[flux_err_col], dtype=float)
    vel_all_line = np.asarray(table_data[vel_col], dtype=float)
    vel_err_all_line = np.asarray(table_data[vel_err_col], dtype=float)

    # Quality filter plus selected NII subregion mask.
    valid_line = np.isfinite(flux_all) & np.isfinite(flux_err_all) & (flux_err_all > 0)
    line_snr = np.zeros_like(flux_all)
    line_snr[valid_line] = flux_all[valid_line] / flux_err_all[valid_line]
    filter_data = selected_nii_subregion & valid_line & (line_snr > 5)
    filter_data &= (
        np.isfinite(table_data['ra'])
        & np.isfinite(table_data['dec'])
        & np.isfinite(vel_all_line)
        & np.isfinite(vel_err_all_line)
    )
    
    ra = table_data['ra'][filter_data]
    dec = table_data['dec'][filter_data]
    flux = flux_all[filter_data]
    vel = vel_all_line[filter_data]
    vel_err = vel_err_all_line[filter_data]

    center_ra = 0.5 * (float(np.nanmin(ra)) + float(np.nanmax(ra)))
    center_dec = 0.5 * (float(np.nanmin(dec)) + float(np.nanmax(dec)))
    
    line_data[name] = {
        'ra': ra,
        'dec': dec,
        'flux': flux,
        'vel': vel,
        'vel_err': vel_err,
        'center_ra': center_ra,
        'center_dec': center_dec,
    }
    
    print(f"  {name}: {len(ra):,} points")
    print(f"    Field center @ RA={center_ra:.6f}, Dec={center_dec:.6f}")

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

def _build_centered_edges(values, center, window_size):
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

def compute_center_aligned_subwindow_vsfs(vsf_obj, window_size, center_ra, center_dec,
                                        order=1, n_jobs=1, bin_resolution=30,
                                        min_points=50, fill_fraction=0.7, n_sub=5):
    """Compute subwindow VSFs on a grid anchored near the field center."""
    x_all = vsf_obj.x_physical.value if hasattr(vsf_obj.x_physical, 'value') else vsf_obj.x_physical
    y_all = vsf_obj.y_physical.value if hasattr(vsf_obj.y_physical, 'value') else vsf_obj.y_physical

    center_match_idx = np.argmin((vsf_obj.ra - center_ra)**2 + (vsf_obj.dec - center_dec)**2)
    x_center = x_all[center_match_idx]
    y_center = y_all[center_match_idx]

    x_edges = _build_centered_edges(x_all, x_center, window_size)
    y_edges = _build_centered_edges(y_all, y_center, window_size)

    center_i = np.searchsorted(x_edges, x_center, side='right') - 1
    center_j = np.searchsorted(y_edges, y_center, side='right') - 1

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

    center_included = any((i == center_i and j == center_j) for i, j, _, _ in results)

    return results, x_edges, y_edges, center_i, center_j, center_included

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
all_center_window = {}

for name in line_names:
    cached_subwindows = load_subwindow_vsf_cache(
        object_name,
        name,
        window_size=window_size,
        n_points=len(line_data[name]['ra']),
    )
    if cached_subwindows is not None:
        results = cached_subwindows["results"]
        all_results[name] = results
        all_x_edges[name] = cached_subwindows["x_edges"]
        all_y_edges[name] = cached_subwindows["y_edges"]
        all_center_window[name] = cached_subwindows["center_window"]
        all_vsfs_array[name] = cached_subwindows["vsfs_array"]
        all_common_r[name] = cached_subwindows["common_r"]
        all_mean_vsf[name] = cached_subwindows["mean_vsf"]
        all_std_vsf[name] = cached_subwindows["std_vsf"]
        all_median_vsf[name] = cached_subwindows["median_vsf"]
        all_median_err[name] = cached_subwindows["median_err"]
        all_lower_vsf[name] = cached_subwindows["lower_vsf"]
        all_upper_vsf[name] = cached_subwindows["upper_vsf"]
        continue

    print(f"  Computing VSFs for {name}...")
    
    # Create VSF2DSubwindows object for this line
    vsf_subwindows = VSF2DSubwindows(line_data[name]['ra'], line_data[name]['dec'], line_data[name]['vel'], dist=1500)
    
    results, x_edges, y_edges, center_i, center_j, center_included = compute_center_aligned_subwindow_vsfs(
        vsf_subwindows,
        window_size=window_size,
        center_ra=line_data[name]['center_ra'],
        center_dec=line_data[name]['center_dec'],
        order=1,
        n_jobs=1,
        bin_resolution=30,
        min_points=50,
        fill_fraction=0.7,
        n_sub=5
    )
    if center_included:
        print(f"    {name}: center grid cell included at (i={center_i}, j={center_j})")
    else:
        print(f"    {name}: center grid cell rejected by cuts (i={center_i}, j={center_j})")
    
    all_results[name] = results
    all_x_edges[name] = x_edges
    all_y_edges[name] = y_edges
    all_center_window[name] = (center_i, center_j, center_included)
    
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
    
    mean_vsf = np.nanmean(vsfs_array, axis=0)
    median_vsf = np.nanmedian(vsfs_array, axis=0)
    std_vsf = np.zeros_like(mean_vsf)
    median_err = jackknife_median_err(vsfs_array)

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
    save_subwindow_vsf_cache(
        object_name,
        name,
        window_size=window_size,
        n_points=len(line_data[name]['ra']),
        results=results,
        x_edges=x_edges,
        y_edges=y_edges,
        center_window=(center_i, center_j, center_included),
        common_r=common_r,
        vsfs_array=vsfs_array,
        mean_vsf=mean_vsf,
        std_vsf=std_vsf,
        median_vsf=median_vsf,
        median_err=median_err,
        lower_vsf=lower_vsf,
        upper_vsf=upper_vsf,
    )

print(f"\n✓ Prepared VSFs for {len(line_names)} lines")
print()

# Use common_r from the last computation (they should be the same)
# Update statistics using NII as reference
vsfs_array = all_vsfs_array['NII']
mean_vsf = all_mean_vsf['NII']
std_vsf = all_std_vsf['NII']
lower_vsf = all_lower_vsf['NII']
upper_vsf = all_upper_vsf['NII']
common_r = all_common_r['NII']
common_log_centers = np.log10(common_r)
results = all_results['NII']

print(f"Subwindow statistics (NII reference):")
print(f"  Number of subwindows: {len(results)}")
print(f"  Common radial bins: {len(common_log_centers)}")
print(f"  Mean S2^1/2 range: {np.nanmin(sqrt_vsf(mean_vsf)):.2e} to {np.nanmax(sqrt_vsf(mean_vsf)):.2e}")
print()

# Analyze available scale coverage
print("Scale coverage analysis:")
small_scale_idx = common_r < 10  # Small scales (< 10 pc)
large_scale_idx = common_r > 30  # Large scales (> 30 pc)

print(f"  Scale ranges:")
print(f"    Small scales (< 10 pc): {np.sum(small_scale_idx)} bins, r = {common_r[small_scale_idx].min():.1f} to {common_r[small_scale_idx].max():.1f} pc")
if np.sum(large_scale_idx) > 0:
    print(f"    Large scales (> 30 pc): {np.sum(large_scale_idx)} bins, r = {common_r[large_scale_idx].min():.1f} to {common_r[large_scale_idx].max():.1f} pc")
else:
    print(f"    Large scales (> 30 pc): No data points in this range")
print()

print()

# Create the two-panel figure
print("Creating summary figure with velocity field and absolute VSFs...")

from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(7, 11))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.05, 1.0], hspace=0.30)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

# Subplot 1: Velocity field with accepted subwindow grid.
# Binned raster (median nearest-neighbor spacing resolution) instead of an
# alpha-blended scatter of ~1e4-1e5 points, which reads as noisy/speckled.
# Diverging colormap centered on the field's own median velocity reads more
# intuitively as "faster/slower than the bulk motion" than a sequential one.
ra_edges, dec_edges, mean_vel_grid = bin_velocity_field(
    velocity_background['ra'], velocity_background['dec'], velocity_background['vel']
)
vmin = np.nanpercentile(velocity_background['vel'], 2)
vmax = np.nanpercentile(velocity_background['vel'], 98)
vcenter = np.nanmedian(velocity_background['vel'])
vel_cmap = sns.color_palette('RdBu_r', as_cmap=True)
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

sc = ax1.pcolormesh(
    ra_edges, dec_edges, mean_vel_grid.T,
    cmap=vel_cmap, norm=norm, shading='flat', rasterized=True,
)
ax1.invert_xaxis()
ax1.set_xlabel('RA (°)')
ax1.set_ylabel('Dec (°)')
ax1.grid(True, alpha=0.2, color='0.5')
tighten_sky_axes(ax1, velocity_background['ra'], velocity_background['dec'])

cax = make_axes_locatable(ax1).append_axes("right", size="4%", pad=0.08)
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label(r'$v_\mathrm{LOS}$ [km/s]')

# Overlay selected NII subwindow divisions from VSF computation.
nii_results = all_results['NII']
nii_x_edges = all_x_edges['NII']
nii_y_edges = all_y_edges['NII']

# Approximate inverse mapping from physical x/y to RA/Dec for edge plotting.
ra_from_x = np.poly1d(np.polyfit(x_phys, ra, 1))
dec_from_y = np.poly1d(np.polyfit(y_phys, dec, 1))

# Draw complete 5x5 grid boundaries faintly. Dark (not white) lines, since the
# diverging colormap goes pale/white near the field's median velocity, where
# white gridlines would vanish.
grid_line_color = '0.15'
for x_edge in nii_x_edges:
    ra_edge = ra_from_x(x_edge)
    ax1.plot([ra_edge, ra_edge], [dec.min(), dec.max()], color=grid_line_color, linewidth=0.7, alpha=0.35)
for y_edge in nii_y_edges:
    dec_edge = dec_from_y(y_edge)
    ax1.plot([ra.min(), ra.max()], [dec_edge, dec_edge], color=grid_line_color, linewidth=0.7, alpha=0.35)

# Highlight accepted subwindows that were used in VSF statistics.
for i, j, _, _ in nii_results:
    ra0 = ra_from_x(nii_x_edges[i])
    ra1 = ra_from_x(nii_x_edges[i + 1])
    dec0 = dec_from_y(nii_y_edges[j])
    dec1 = dec_from_y(nii_y_edges[j + 1])
    ax1.plot([ra0, ra1, ra1, ra0, ra0], [dec0, dec0, dec1, dec1, dec0], color=grid_line_color, linewidth=1.5, alpha=0.9)

nii_center_ra = line_data['NII']['center_ra']
nii_center_dec = line_data['NII']['center_dec']
ax1.scatter(nii_center_ra, nii_center_dec, marker='+', s=180, c='yellow', linewidth=2.0, zorder=10)

nii_center_i, nii_center_j, nii_center_included = all_center_window['NII']
center_ra0 = ra_from_x(nii_x_edges[nii_center_i])
center_ra1 = ra_from_x(nii_x_edges[nii_center_i + 1])
center_dec0 = dec_from_y(nii_y_edges[nii_center_j])
center_dec1 = dec_from_y(nii_y_edges[nii_center_j + 1])
ax1.plot(
    [center_ra0, center_ra1, center_ra1, center_ra0, center_ra0],
    [center_dec0, center_dec0, center_dec1, center_dec1, center_dec0],
    color='yellow', linewidth=2.0, linestyle='--', alpha=0.95
)
if not nii_center_included:
    ax1.text(nii_center_ra, nii_center_dec, ' center (rejected)', color='yellow', fontsize=10, fontweight='bold', zorder=11)

# Subplot 2: Absolute VSFs with all subregions + mean/median/total
vsf_colors = sns.color_palette('rocket', n_colors=3)
_ylim_values = []
for i, name in enumerate(line_names):
    color = vsf_colors[i]
    r_line = all_common_r[name]

    # Plot all accepted subwindow curves directly, with low alpha.
    plotted_any = False
    for _, _, vsf_sub, log_c_sub in all_results[name]:
        valid_sub = np.isfinite(vsf_sub)
        if np.sum(valid_sub) < 3:
            continue
        ax2.plot(10**log_c_sub[valid_sub], sqrt_vsf(vsf_sub[valid_sub]), color=color, linewidth=1.0, alpha=0.10,
                 label=f'{name} subwindows' if not plotted_any else None)
        plotted_any = True

    median_vsf_line = all_median_vsf[name]
    median_err_line = all_median_err[name]

    ax2.fill_between(
        r_line,
        sqrt_vsf(median_vsf_line - median_err_line),
        sqrt_vsf(median_vsf_line + median_err_line),
        color=color, alpha=0.20, linewidth=0,
    )
    ax2.plot(r_line, sqrt_vsf(median_vsf_line), color=color, linestyle=':', linewidth=2.2, label=f'{name} median')
    _finite_median = sqrt_vsf(median_vsf_line)[np.isfinite(median_vsf_line)]
    if _finite_median.size:
        _ylim_values.append(_finite_median)

    # Load or compute the full-region total VSF overlay.
    total_cache = load_total_vsf_cache(object_name, name)
    if total_cache is None:
        vsf_total, vsf_total_err, log_centers_total, counts_total = VSF2D(
            line_data[name]['ra'], line_data[name]['dec'], line_data[name]['vel'], dist=1500,
            vel_err=line_data[name]['vel_err']
        ).generate_vsf_2d(
            order=1,
            outname='/tmp/ignore_orion_total',
            n_jobs=1,
            min_pairs=3,
            bin_resolution=30,
            error_mode='block_jackknife',
            binning='shell',
        )
        valid_total = np.isfinite(vsf_total) & np.isfinite(vsf_total_err) & (counts_total > 0)
        if np.any(valid_total):
            r_total = 10**log_centers_total[valid_total]
            total_line = vsf_total[valid_total]
            total_err_line = vsf_total_err[valid_total]
            total_counts = counts_total[valid_total]
            save_total_vsf_cache(object_name, name, r_total, total_line, total_err_line, total_counts)
            total_cache = {
                "r": r_total,
                "vsf": total_line,
                "vsf_err": total_err_line,
                "counts": total_counts,
            }

    if total_cache is not None:
        r_total = total_cache["r"]
        total_line = total_cache["vsf"]
        ax2.plot(r_total, sqrt_vsf(total_line), color=color, linestyle='--', linewidth=2.2, label=f'{name} total')
        _finite_total = sqrt_vsf(total_line)[np.isfinite(total_line)]
        if _finite_total.size:
            _ylim_values.append(_finite_total)


ax2.set_xscale('log')
ax2.set_yscale('log')
if _ylim_values:
    _all_y = np.concatenate(_ylim_values)
    _all_y = _all_y[_all_y > 0]
    ax2.set_ylim(0.7 * _all_y.min(), 1.4 * _all_y.max())
else:
    ax2.set_ylim(0.1, 10)
ax2.set_xlabel(r'$l_{2D} = \left| r_i - r_j \right| [pc]$')
ax2.set_ylabel(r'$\sqrt{\bar{S}_\mathrm{2}}$ [km/s]', labelpad=2)
ax2.grid(True, alpha=0.3)
# Increase tick density on log x-axis
ax2.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

# Custom legend: style meaning + line colors (middle panel).
style_handles_ax2 = [
    Line2D([0], [0], color='black', lw=1.0, alpha=0.25, linestyle='-', label='subwindows'),
    Line2D([0], [0], color='black', lw=2.2, linestyle=':', label='median'),
    Patch(facecolor='black', alpha=0.20, label='median jackknife err'),
    Line2D([0], [0], color='black', lw=2.2, linestyle='--', label='total'),
]
color_handles = [
    Line2D([0], [0], color=vsf_colors[i], lw=3.0, linestyle='-', label=name)
    for i, name in enumerate(line_names)
]
ax2.legend(
    handles=style_handles_ax2 + color_handles, fontsize=8, loc='upper left',
    frameon=True, framealpha=0.9, borderaxespad=0.6,
)

plt.tight_layout()
fig.subplots_adjust(hspace=0.32)
figure_dir = REPO_ROOT / 'Figures/manuscript/01_summary_regions'
os.makedirs(figure_dir, exist_ok=True)
figure_out = os.path.join(str(figure_dir), f'{base_name}_summary.png')
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
