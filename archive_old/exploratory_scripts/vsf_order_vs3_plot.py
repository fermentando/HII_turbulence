import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from vsf_module import VSF2D


# Usage: python vsf_order_vs3_plot.py <fitsfile> <distance_pc>
if len(sys.argv) != 3:
    print("Usage: python vsf_order_vs3_plot.py <fitsfile> <distance_pc>")
    sys.exit(1)

fitsfile = sys.argv[1]
distance = float(sys.argv[2])

# --- Load data ---
hdul = fits.open(fitsfile)
table = hdul[1].data
filter_mask = table['flux_nii'] / table['flux_nii_err'] > 5
ra = table['ra'][filter_mask]
dec = table['dec'][filter_mask]
vel = table['vel_nii'][filter_mask]
vel_err = table['vel_nii_err'][filter_mask]


# --- Compute VSF for orders 1 to 4 and 3 ---
vsf_obj = VSF2D(ra, dec, vel, dist=distance, vel_err=vel_err)
orders = [1, 2, 3, 4, 5]
vsfs = {}
vsf_errs = {}
for p in orders:
    vsf, vsf_err, log_centers, counts = vsf_obj.generate_vsf_2d(
        order=p,
        outname='Figures/vsf_temp',
        bin_resolution=20
    )
    vsfs[p] = vsf
    vsf_errs[p] = vsf_err

# --- Plot all orders vs order 3 ---

fig, ax = plt.subplots()
colors = ['b', 'g', 'r', 'm', 'c']
for idx, p in enumerate(orders):
    if p == 3:
        continue  # skip plotting VSF3 vs VSF3
    mask = (~np.isnan(vsfs[p])) & (~np.isnan(vsfs[3])) & (vsfs[p] > 0) & (vsfs[3] > 0)
    x = vsfs[3][mask]
    y = vsfs[p][mask]
    xerr = vsf_errs[3][mask]
    yerr = vsf_errs[p][mask]
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=3, color=colors[idx], label=f'Order {p} vs 3')
    # Best-fit line in log-log space
    logx = np.log10(x)
    logy = np.log10(y)
    slope, intercept = np.polyfit(logx, logy, 1)
    fit_y = 10**(slope * logx + intercept)
    ax.plot(x, fit_y, '--', color=colors[idx], label=f'Order {p} fit: slope={slope:.2f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r"VSF$_3$ (km$^3$ s$^{-3}$)", fontsize=13)
ax.set_ylabel(r"VSF$_p$ (km$^p$ s$^{-p}$)", fontsize=13)
ax.set_title(f"VSF orders 1-5 vs order 3\n{fitsfile}")
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig('Figures/VSF_orders1to5_vs_order3.png', dpi=150)
plt.show()
print("✓ Plot saved as Figures/VSF_orders1to5_vs_order3.png")
