import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from astropy.io import fits

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import VSF2D, compute_pdf_and_statistics, compute_bins

plt.style.use('./kurtosis_plot.mplstyle')

# --- Load data for Flame Nebula ---
hdul = fits.open("LVM_Flame_nebula.fits")
table = hdul[1].data
filter_mask = table['flux_nii'] / table['flux_nii_err'] > 5
ra = table['ra'][filter_mask]
dec = table['dec'][filter_mask]
vel = table['vel_nii'][filter_mask]
vel_err = table['vel_nii_err'][filter_mask]

# --- Compute VSF (order 1) ---
vsf = VSF2D(ra, dec, vel, dist=400, vel_err=vel_err)
vsf_data, vsf_err, log_centers, counts = vsf.generate_vsf_2d(
    order=1,
    outname='Figures/Flame_NII_refactored',
    bin_resolution=20
)

# --- Compute PDF and kurtosis ---
log_centers_bins, bin_edges, dv_bins, _ = compute_bins(
    ra, dec, vel, bin_resolution=20
)
pdf, c2, c4, cnt, dv_centers, kurtosis = compute_pdf_and_statistics(
    ra, dec, vel, vel_err,
    log_centers_bins, bin_edges, dv_bins,
    order=1, n_jobs=4
)

# --- 3D Kurtosis Plot ---
r = 10**log_centers_bins
dv_centers_plot = 0.5 * (dv_bins[:-1] + dv_bins[1:])
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
cmap = cm.viridis
norm = plt.Normalize(3, 6)
x_min, x_max = -10, 10
for i in range(len(r)):
    color = cmap(norm(kurtosis[i]))
    mask = (dv_centers_plot >= x_min) & (dv_centers_plot <= x_max)
    ax.plot(
        dv_centers_plot[mask],
        np.log10(np.full(np.sum(mask), r[i])),
        pdf[i][mask],
        color=color
    )

# Improved axis labels and plot presentation
ax.set_xlabel(r"$(\Delta v)^2$ (km/s)$^2$", fontsize=14, labelpad=10)
ax.set_ylabel(r"$\log r$ (pc)", fontsize=14, labelpad=10)
ax.set_zlabel(r"$\mathcal{P}(\Delta v^2)$", fontsize=14, labelpad=10)
ax.set_xlim(x_min, x_max)
ax.set_ylim(np.log10(r[0]), np.log10(r[-1]))
ax.view_init(elev=25, azim=45)
ax.dist = 9
ax.set_zscale('log')
zmin = np.nanmin([np.nanmin(pdf[i][mask]) for i in range(len(r))])
zmax = np.nanmax([np.nanmax(pdf[i][mask]) for i in range(len(r))])
ax.set_zlim(max(zmin, 1e-3), min(zmax, 1))
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.zaxis.set_major_locator(MaxNLocator(5))
ax.tick_params(axis='both', which='major', labelsize=12)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.pane.set_edgecolor('w')
# Colorbar
mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
mappable.set_array([])
cax = fig.add_axes([0.25, 0.92, 0.5, 0.02])
cbar = plt.colorbar(mappable, cax=cax, orientation='horizontal')
cbar.set_label(r"Kurtosis", labelpad=5, fontsize=13)
cbar.ax.tick_params(labelsize=11)
# Annotate colorbar: low end = Gaussian, high end = Exponential
cbar.ax.text(3, 1.2, "Gaussian", ha='center', va='bottom', fontsize=11, color='k', transform=cbar.ax.get_xaxis_transform())
cbar.ax.text(6, 1.2, "Exponential", ha='center', va='bottom', fontsize=11, color='k', transform=cbar.ax.get_xaxis_transform())
plt.tight_layout()
figure_dir = Path('Figures/manuscript/04_flame_kurtosis_3d')
figure_dir.mkdir(parents=True, exist_ok=True)
figure_out = figure_dir / 'Flame_NII_refactored_3D.png'
plt.savefig(figure_out, dpi=200)
plt.show()
print(f"✓ 3D kurtosis plot created and saved as {figure_out}")
