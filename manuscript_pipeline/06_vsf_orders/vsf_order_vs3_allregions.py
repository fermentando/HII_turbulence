import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D, VSF2DSubwindows


plt.style.use('./custom_plot.mplstyle')

orders = [1, 2, 3, 4, 5]
colors = ['b', 'g', 'r', 'm', 'c']

fits_files = sorted([
    f for f in os.listdir('.')
    if f.startswith('LVM_') and f.endswith('.fits')
    and 'Orion' not in f and 'Rosette' not in f
])
#fits_files = ['LVM_Flame_nebula.fits', 'LVM_Gum31.fits', 'LVM_RCW52.fits']
# Keep previous distance table for consistency.
distances = {
    'LVM_Flame_nebula.fits': 400,
    'LVM_Orion.fits': 400,
    'LVM_Gum31.fits': 2300,
    'LVM_Gum38a.fits': 2900,
    'LVM_RCW52.fits': 2600,
    'LVM_RCW60.fits': 4300,
    'LVM_RCW79.fits': 4700,
    'LVM_RCW87.fits': 6000,
    'LVM_Rosette_nebula.fits': 1600,
    'LVM_BFS65.fits': 1600,
    'LVM_Sh2-311.fits': 5000,
}


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
                                        min_points=50, fill_fraction=0.7, n_sub=5):
    """Compute subwindow VSFs on a grid anchored so one cell is centered at peak."""
    x_all = vsf_obj.x_physical.value if hasattr(vsf_obj.x_physical, 'value') else vsf_obj.x_physical
    y_all = vsf_obj.y_physical.value if hasattr(vsf_obj.y_physical, 'value') else vsf_obj.y_physical

    peak_match_idx = np.argmin((vsf_obj.ra - peak_ra) ** 2 + (vsf_obj.dec - peak_dec) ** 2)
    x_center = x_all[peak_match_idx]
    y_center = y_all[peak_match_idx]

    x_edges = _build_peak_centered_edges(x_all, x_center, window_size)
    y_edges = _build_peak_centered_edges(y_all, y_center, window_size)

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
            bin_edges = 10 ** edges

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

    return results


def aggregate_median_vsf(results, bin_resolution=20, min_valid_fraction=0.5,
                         n_boot=300, ci=68.0):
    """Aggregate subwindow VSFs using median and bootstrap CI errors."""
    if not results:
        return None, None, None, None

    log_centers_list = [log_c for _, _, _, log_c in results]
    if not log_centers_list:
        return None, None, None, None

    r_min_common = max(log_c.min() for log_c in log_centers_list)
    r_max_common = min(log_c.max() for log_c in log_centers_list)
    if (not np.isfinite(r_min_common)) or (not np.isfinite(r_max_common)) or (r_max_common <= r_min_common):
        return None, None, None, None

    n_decades = max(r_max_common - r_min_common, 1e-6)
    n_bins = int(np.ceil(n_decades * bin_resolution))
    common_log = np.linspace(r_min_common, r_max_common, max(n_bins, 10))
    common_r = 10 ** common_log

    vsfs_interp = []
    for _, _, vsf, log_c in results:
        r = 10 ** log_c
        vsfs_interp.append(np.interp(common_r, r, vsf, left=np.nan, right=np.nan))

    if not vsfs_interp:
        return None, None, None, None

    vsfs_array = np.vstack(vsfs_interp)
    med_vsf = np.nanmedian(vsfs_array, axis=0)
    n_valid = np.sum(~np.isnan(vsfs_array), axis=0)

    valid_mask = n_valid >= len(results) * min_valid_fraction
    if not np.any(valid_mask):
        return None, None, None, None

    low_pct = 0.5 * (100.0 - ci)
    high_pct = 100.0 - low_pct

    rng = np.random.default_rng(12345)
    med_ci_err = np.full(vsfs_array.shape[1], np.nan)

    for k in range(vsfs_array.shape[1]):
        vals = vsfs_array[:, k]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue

        boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True)
        boot_meds = np.nanmedian(boot, axis=1)
        low = np.percentile(boot_meds, low_pct)
        high = np.percentile(boot_meds, high_pct)
        med_ci_err[k] = 0.5 * (high - low)

    return med_vsf[valid_mask], med_ci_err[valid_mask], common_log[valid_mask], n_valid[valid_mask]


def align_orders_for_vs3(log3, vsf3, err3, logp, vsfp, errp):
    """Align VSF_p and VSF_3 onto common log-r support for p-vs-3 fitting."""
    if log3 is None or logp is None:
        return None, None, None, None

    lo = max(np.nanmin(log3), np.nanmin(logp))
    hi = min(np.nanmax(log3), np.nanmax(logp))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return None, None, None, None

    mask3 = (log3 >= lo) & (log3 <= hi)
    if np.sum(mask3) < 4:
        return None, None, None, None

    log_common = log3[mask3]
    x = np.interp(log_common, log3, vsf3, left=np.nan, right=np.nan)
    xerr = np.interp(log_common, log3, err3, left=np.nan, right=np.nan)
    y = np.interp(log_common, logp, vsfp, left=np.nan, right=np.nan)
    yerr = np.interp(log_common, logp, errp, left=np.nan, right=np.nan)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr)
    valid &= (x > 0) & (y > 0)
    valid &= (xerr >= 0) & (yerr >= 0)

    if np.sum(valid) < 4:
        return None, None, None, None

    return x[valid], y[valid], xerr[valid], yerr[valid]


def weighted_linear_fit(logx, logy, sigma):
    """Weighted fit: logy = m*logx + b with covariance matrix."""
    w = 1.0 / np.square(sigma)
    sw = np.sum(w)
    sx = np.sum(w * logx)
    sy = np.sum(w * logy)
    sxx = np.sum(w * logx * logx)
    sxy = np.sum(w * logx * logy)

    det = sw * sxx - sx * sx
    if det <= 0 or (not np.isfinite(det)):
        return np.nan, np.nan, None

    m = (sw * sxy - sx * sy) / det
    b = (sxx * sy - sx * sxy) / det

    cov = np.array([
        [sw / det, -sx / det],
        [-sx / det, sxx / det],
    ])
    return m, b, cov


def fit_loglog_with_xy_errors(x, y, xerr, yerr, n_mc=500):
    """Fit y vs x in log-log using x/y uncertainties and bootstrap slope error."""
    logx = np.log10(x)
    logy = np.log10(y)

    sx = np.maximum(xerr / (x * np.log(10.0)), 1e-12)
    sy = np.maximum(yerr / (y * np.log(10.0)), 1e-12)

    # Iterative effective-sigma treatment to include x-errors.
    m, b = np.polyfit(logx, logy, 1)
    for _ in range(4):
        sigma_eff = np.sqrt(sy ** 2 + (m * sx) ** 2)
        m_new, b_new, cov = weighted_linear_fit(logx, logy, sigma_eff)
        if not np.isfinite(m_new):
            break
        m, b = m_new, b_new

    sigma_eff = np.sqrt(sy ** 2 + (m * sx) ** 2)
    m, b, cov = weighted_linear_fit(logx, logy, sigma_eff)

    if (not np.isfinite(m)) or (not np.isfinite(b)):
        return np.nan, np.nan, np.nan

    # Parametric bootstrap over measurement errors.
    rng = np.random.default_rng(24680)
    slopes = []
    for _ in range(n_mc):
        xb = rng.normal(x, xerr)
        yb = rng.normal(y, yerr)
        valid = np.isfinite(xb) & np.isfinite(yb) & (xb > 0) & (yb > 0)
        if np.sum(valid) < 4:
            continue

        logx_b = np.log10(xb[valid])
        logy_b = np.log10(yb[valid])
        sx_b = np.maximum(xerr[valid] / (xb[valid] * np.log(10.0)), 1e-12)
        sy_b = np.maximum(yerr[valid] / (yb[valid] * np.log(10.0)), 1e-12)

        mm, bb = m, b
        for _ in range(3):
            sigma_eff_b = np.sqrt(sy_b ** 2 + (mm * sx_b) ** 2)
            mm_new, bb_new, _ = weighted_linear_fit(logx_b, logy_b, sigma_eff_b)
            if not np.isfinite(mm_new):
                break
            mm, bb = mm_new, bb_new

        if np.isfinite(mm):
            slopes.append(mm)

    if len(slopes) >= 20:
        p16, p84 = np.percentile(slopes, [16, 84])
        slope_err = 0.5 * (p84 - p16)
    elif cov is not None and cov.shape == (2, 2) and np.isfinite(cov[0, 0]):
        slope_err = np.sqrt(max(cov[0, 0], 0.0))
    else:
        slope_err = np.nan

    return m, b, slope_err


def main():
    n_regions = len(fits_files)
    ncols = 3
    nrows = int(np.ceil(n_regions / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    slope_results = []

    for idx, fitsfile in enumerate(fits_files):
        print(f"Processing region {idx + 1}/{len(fits_files)}: {fitsfile}")
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        distance = distances.get(fitsfile, 400)
        hdul = fits.open(fitsfile)
        table = hdul[1].data
        hdul.close()

        filter_mask = table['flux_nii'] / table['flux_nii_err'] > 5
        ra = table['ra'][filter_mask]
        dec = table['dec'][filter_mask]
        vel = table['vel_nii'][filter_mask]
        vel_err = table['vel_nii_err'][filter_mask]
        flux = table['flux_nii'][filter_mask]

        if len(ra) < 200:
            ax.set_title(f"{fitsfile} (insufficient points)", fontsize=11)
            ax.axis('off')
            continue

        # Physical footprint for subwindow sizing.
        vsf_phys = VSF2D(ra, dec, vel, dist=distance, vel_err=vel_err)
        x_phys = vsf_phys.x_physical
        y_phys = vsf_phys.y_physical
        x_range = x_phys.max() - x_phys.min()
        y_range = y_phys.max() - y_phys.min()
        window_size = max(x_range, y_range) / 5

        # Peak in NII flux defines grid anchoring, as in multi_fits workflow.
        peak_idx = np.nanargmax(flux)
        peak_ra = ra[peak_idx]
        peak_dec = dec[peak_idx]

        vsf_sub = VSF2DSubwindows(ra, dec, vel, dist=distance)

        order_data = {}
        for p in orders:
            print(f"  Computing order {p} median-subwindow VSF...")
            results = compute_peak_aligned_subwindow_vsfs(
                vsf_sub,
                window_size=window_size,
                peak_ra=peak_ra,
                peak_dec=peak_dec,
                order=p,
                n_jobs=4,
                bin_resolution=20,
                min_points=50,
                fill_fraction=0.7,
                n_sub=5,
            )
            med_vsf, med_err, log_c, counts = aggregate_median_vsf(
                results,
                bin_resolution=20,
                min_valid_fraction=0.5,
                n_boot=300,
                ci=68.0,
            )
            order_data[p] = (med_vsf, med_err, log_c, counts)

        vsf3, err3, log3, _ = order_data[3]
        if vsf3 is None:
            ax.set_title(f"{fitsfile} (no valid order 3)", fontsize=11)
            ax.axis('off')
            continue

        for i, p in enumerate(orders):
            if p == 3:
                continue

            vsfp, errp, logp, _ = order_data[p]
            if vsfp is None:
                continue

            aligned = align_orders_for_vs3(log3, vsf3, err3, logp, vsfp, errp)
            if aligned[0] is None:
                continue
            x, y, xerr, yerr = aligned

            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt='o',
                capsize=3,
                color=colors[i],
                ms=3,
                alpha=0.55,
                elinewidth=0.9,
                capthick=0.9,
                label=f'Order {p} vs 3'
            )

            if len(x) > 3:
                slope, intercept, slope_err = fit_loglog_with_xy_errors(x, y, xerr, yerr, n_mc=500)
                if np.isfinite(slope) and np.isfinite(intercept):
                    xfit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 200)
                    yfit = 10 ** (slope * np.log10(xfit) + intercept)
                    ax.plot(
                        xfit,
                        yfit,
                        '--',
                        color=colors[i],
                        #label=f'Order {p} fit: slope={slope:.2f}±{slope_err:.2f}'
                    )
                    slope_results.append({
                        'region': fitsfile,
                        'order': p,
                        'slope': slope,
                        'slope_err': slope_err,
                    })

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$S_3$ (km$^3$ s$^{-3}$)", fontsize=13)
        ax.set_title(f"{fitsfile.split('.')[0].split('LVM_')[-1].replace('_', ' ')}", fontsize=12)
        ax.grid(True, which='both', ls='--', alpha=0.5)
        if idx == 0:
            ax.set_ylabel(r"$S_p$ (km$^p$ s$^{-p}$)", fontsize=13)
            ax.legend(fontsize=8)

    # Hide unused subplots.
    for idx in range(n_regions, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')

    os.makedirs('Figures', exist_ok=True)

    csv_path = 'Figures/vsf_vs3_slopes_by_region.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['region', 'order', 'slope', 'slope_err']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in slope_results:
            writer.writerow(row)
    print(f"✓ Slopes and uncertainties saved to {csv_path}")

    plt.tight_layout()
    out_dir = 'Figures/manuscript/06_vsf_orders'
    #out_dir = "example_vsf_vs3"
    os.makedirs(out_dir, exist_ok=True)
    out_plot = os.path.join(out_dir, 'VSF_orders1to5_vs_order3_allregions.png')
    plt.savefig(out_plot, dpi=150)
    plt.show()
    print(f"✓ Plot saved as {out_plot}")


if __name__ == '__main__':
    main()
