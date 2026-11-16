"""Extended self-similarity: VSF of order p = 1..5 against the VSF of order 3, for every region.

S_p(l) = <|dv|^p> over all fibre pairs at separation l, full field of view (no subwindow median).
Data and cuts are the same as the full-FOV VSFs (section 2) and kurtosis_plot.py, via load_line_data:
flux S/N > --snr-min (default 20), tile-overlap fibres removed, 10-sigma velocity clip, large-scale
velocity gradient subtracted, catalogue distance. Separations in pc, shell binning (20 bins/dex)
above a 37-arcsec floor; bins with fewer than MIN_PAIRS pairs are dropped.

All orders come from the SAME pairs (vsf_orders_block_jackknife), with 5x5 spatial block-jackknife
errors. Each leave-one-block-out replicate drops the same block for every order, so the ESS slope
zeta_p/zeta_3 (OLS of log10 S_p on log10 S_3) is refitted on every replicate and its jackknife
spread is the slope error -- this carries the strong correlation between S_p and S_3 and between
neighbouring bins, which independent x/y error bars do not. The plain OLS error is also written.

Outputs:
  Figures/manuscript/06_vsf_orders/VSF_orders1to5_vs_order3_allregions.png
  Figures/vsf_vs3_slopes_by_region.csv     (read by vsf_vs3_slope_vs_order_plot.py)
  NPZ/VSFORD_<region>_<LINE>.npz           S_p curves, errors and jackknife replicates

Run from the repo root:
    python manuscript_pipeline/06_vsf_orders/vsf_order_vs3_allregions.py
    python manuscript_pipeline/06_vsf_orders/vsf_order_vs3_allregions.py --line sii --snr-min 5
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import LINE_LABELS, VSF2D, load_line_data, spatial_bin_edges, vsf_orders_block_jackknife  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--line", default="nii", choices=("nii", "sii", "siii"))
parser.add_argument("--snr-min", type=float, default=20.0, help="Flux S/N cut, as for the _snr20 VSFs.")
args = parser.parse_args()
LINE = args.line

plt.style.use('./custom_plot.mplstyle')

ORDERS = [1, 2, 3, 4, 5]
REFERENCE_ORDER = 3
COLORS = {1: 'b', 2: 'g', 4: 'm', 5: 'c'}
MIN_SEPARATION_ARCSEC = 37.0
BIN_RESOLUTION = 20
MIN_PAIRS = 1000       # same bin cut as kurtosis_plot.py (4000 stored counts = 1000 real pairs)
MIN_FIT_BINS = 4

# Orion (~4e5 fibres) has its own fast pipeline in 01_summary_regions.
fits_files = sorted(f for f in glob.glob('LVM_*.fits') if 'Orion' not in f)


def ols_line(x, y):
    """Slope, intercept, slope standard error."""
    X = np.column_stack((x, np.ones_like(x)))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / max(len(x) - 2, 1)
    return coef[0], coef[1], np.sqrt(cov[0, 0])


def ess_fit(s_ref, s_p, reps_ref, reps_p, use):
    """ESS slope of log S_p vs log S_ref on bins `use`, with block-jackknife slope error."""
    with np.errstate(divide='ignore', invalid='ignore'):
        lx, ly = np.log10(s_ref[use]), np.log10(s_p[use])
        slope, intercept, slope_err_ols = ols_line(lx, ly)
        rep_slopes = np.array([ols_line(np.log10(rx[use]), np.log10(ry[use]))[0]
                               for rx, ry in zip(reps_ref, reps_p)
                               if np.all(np.isfinite(rx[use]) & (rx[use] > 0) & np.isfinite(ry[use]) & (ry[use] > 0))])
    n_rep = len(rep_slopes)
    slope_err = np.sqrt((n_rep - 1) / n_rep * np.sum((rep_slopes - rep_slopes.mean()) ** 2)) if n_rep >= 4 else np.nan
    return slope, intercept, slope_err, slope_err_ols, n_rep


n_cols = 3
n_rows = int(np.ceil(len(fits_files) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 4 * n_rows), squeeze=False)
slope_results = []
os.makedirs('NPZ', exist_ok=True)

for idx, fitsfile in enumerate(fits_files):
    region = os.path.splitext(fitsfile)[0].replace('LVM_', '')
    region_name = region.replace('_', ' ')
    ax = axes[divmod(idx, n_cols)]
    print(f"[{idx + 1}/{len(fits_files)}] {region_name}")

    d = load_line_data(fitsfile, LINE, snr_min=args.snr_min, subtract_gradient=True)
    vsf2d = VSF2D(d['ra'], d['dec'], d['vel'], dist=d['dist'], vel_err=d['vel_err'])
    x_pc, y_pc = vsf2d.x_physical, vsf2d.y_physical
    min_separation_pc = d['dist'] * np.deg2rad(MIN_SEPARATION_ARCSEC / 3600.0)

    log_centers, bin_edges, _, _ = spatial_bin_edges(
        x_pc, y_pc, bin_resolution=BIN_RESOLUTION, binning='shell', min_separation=min_separation_pc,
    )
    S, S_err, S_reps, counts = vsf_orders_block_jackknife(x_pc, y_pc, d['vel'], bin_edges, orders=ORDERS)
    np.savez(f"NPZ/VSFORD_{region}_{LINE.upper()}.npz", orders=ORDERS, log_centers=log_centers,
             bin_edges=bin_edges, S=S, S_err=S_err, S_jackknife_replicates=S_reps, counts=counts,
             distance_pc=d['dist'], snr_min=args.snr_min, min_separation_pc=min_separation_pc)

    i_ref = ORDERS.index(REFERENCE_ORDER)
    use = (counts >= MIN_PAIRS) & np.all(np.isfinite(S) & (S > 0), axis=0)
    ax.set_title(f"{region_name} ({d['dist'] / 1e3:.2f} kpc)", fontsize=12)
    if use.sum() < MIN_FIT_BINS:
        print(f"  only {use.sum()} bins with >= {MIN_PAIRS} pairs ({len(d['vel'])} fibres); skipped")
        ax.text(0.5, 0.5, f"{len(d['vel'])} fibres:\ntoo few bins", transform=ax.transAxes, ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        continue

    for k, p in enumerate(ORDERS):
        if p == REFERENCE_ORDER:
            continue
        slope, intercept, slope_err, slope_err_ols, n_rep = ess_fit(
            S[i_ref], S[k], S_reps[:, i_ref], S_reps[:, k], use)
        ax.errorbar(S[i_ref][use], S[k][use], xerr=S_err[i_ref][use], yerr=S_err[k][use],
                    fmt='o', ms=3, capsize=2, elinewidth=0.9, capthick=0.9, alpha=0.6, color=COLORS[p],
                    label=fr"$p={p}$: ${slope:.2f}\pm{slope_err:.2f}$")
        xfit = np.logspace(np.log10(S[i_ref][use].min()), np.log10(S[i_ref][use].max()), 100)
        ax.plot(xfit, 10 ** (slope * np.log10(xfit) + intercept), '--', color=COLORS[p], lw=1.2)
        slope_results.append({
            'region': fitsfile, 'order': p, 'slope': slope, 'slope_err': slope_err,
            'slope_err_ols': slope_err_ols, 'n_bins': int(use.sum()), 'n_jackknife': n_rep,
            'l_min_pc': 10 ** log_centers[use].min(), 'l_max_pc': 10 ** log_centers[use].max(),
            'distance_pc': d['dist'], 'n_fibres': len(d['vel']), 'line': LINE, 'snr_min': args.snr_min,
        })
        print(f"  zeta_{p}/zeta_3 = {slope:.3f} +/- {slope_err:.3f} (jackknife, {n_rep} reps; "
              f"OLS {slope_err_ols:.3f}), {use.sum()} bins")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$S_3$ (km$^3$ s$^{-3}$)", fontsize=12)
    if idx % n_cols == 0:
        ax.set_ylabel(r"$S_p$ (km$^p$ s$^{-p}$)", fontsize=12)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='upper left', title=r"$\zeta_p/\zeta_3$", title_fontsize=8)

for idx in range(len(fits_files), n_rows * n_cols):
    axes[divmod(idx, n_cols)].axis('off')

fig.suptitle(f"{LINE_LABELS[LINE]}, S/N $>$ {args.snr_min:g}, full FOV, block-jackknife errors", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))

csv_path = 'Figures/vsf_vs3_slopes_by_region.csv'
with open(csv_path, 'w', newline='') as fobj:
    writer = csv.DictWriter(fobj, fieldnames=list(slope_results[0].keys()))
    writer.writeheader()
    for row in slope_results:
        writer.writerow(row)
print(f"Saved slopes to {csv_path}")

out_dir = 'Figures/manuscript/06_vsf_orders'
os.makedirs(out_dir, exist_ok=True)
out_plot = os.path.join(out_dir, 'VSF_orders1to5_vs_order3_allregions.png')
fig.savefig(out_plot, dpi=150)
plt.close(fig)
print(f"Saved {out_plot}")
