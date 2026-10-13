"""3D view of the velocity-increment PDFs of one region and line ([NII], [SII] or [SIII]), coloured by kurtosis.

For each separation bin l: the PDF of |dv| = sqrt(dv^2), dv = v(r) - v(r + l), over all fibre
pairs (folded: pairs are unordered, so the sign of dv carries no information), and the kurtosis
K = <dv^4> / <dv^2>^2 over all pairs (3 = Gaussian, 6 = exponential/Laplace).

Data and cuts are the same as the full-FOV VSFs (section 2), via load_line_data:
flux S/N > --snr-min (default 20, as the _snr20 VSFs), tile-overlap fibres removed, 10-sigma
velocity clip, large-scale velocity gradient subtracted, catalogue distance. Separations are in pc (physical coordinates),
with shell binning above a 37-arcsec floor.

Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/kurtosis_plot.py              # [NII]
    python manuscript_pipeline/04_flame_kurtosis_3d/kurtosis_plot.py --line sii   # or siii
    python manuscript_pipeline/04_flame_kurtosis_3d/kurtosis_plot.py --region RCW79 --line nii
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import (  # noqa: E402
    LINE_LABELS, VSF2D, compute_bins, compute_pdf_and_statistics, get_hii_radius_pc, kurtosis_block_jackknife,
    load_line_data,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--line", default="nii", choices=("nii", "sii", "siii"))
parser.add_argument("--region", default="Flame_nebula", help="Region name, as in LVM_<region>.fits.")
parser.add_argument("--snr-min", type=float, default=20.0, help="Flux S/N cut, as for the _snr20 VSFs.")
args = parser.parse_args()
LINE = args.line
LABEL = LINE_LABELS[LINE]
REGION = args.region.replace("LVM_", "").replace(".fits", "")
REGION_NAME = REGION.replace("_", " ")
FITS_FILE = f"LVM_{REGION}.fits"

plt.style.use('./kurtosis_plot.mplstyle')

MIN_SEPARATION_ARCSEC = 37.0
MIN_COUNTS = 4000      # same bin cut as the section-3 fits (stored counts = pairs x n_jobs = 1000 real pairs)
LOG_P_FLOOR = -3.0     # z-axis floor; mplot3d does not clip lines, so values below it are not drawn

# --- Region / line, standard cuts ---
d = load_line_data(FITS_FILE, LINE, snr_min=args.snr_min, subtract_gradient=True)
vsf2d = VSF2D(d["ra"], d["dec"], d["vel"], dist=d["dist"], vel_err=d["vel_err"])
x_pc, y_pc = vsf2d.x_physical, vsf2d.y_physical
min_separation_pc = d["dist"] * np.deg2rad(MIN_SEPARATION_ARCSEC / 3600.0)
print(f"{REGION_NAME} {LABEL}: {len(x_pc):,} fibres, d = {d['dist']:.0f} pc, "
      f"min separation {MIN_SEPARATION_ARCSEC:g} arcsec = {min_separation_pc:.3f} pc")

# --- Separation / velocity bins (pc) and increment PDFs + kurtosis per bin ---
log_centers_bins, bin_edges, dv_bins, _ = compute_bins(
    x_pc, y_pc, d["vel"], bin_resolution=20, binning="shell", min_separation=min_separation_pc,
)
pdf, c2, c4, cnt, dv_centers, kurtosis = compute_pdf_and_statistics(
    x_pc, y_pc, d["vel"], d["vel_err"],
    log_centers_bins, bin_edges, dv_bins,
    order=1, n_jobs=4,
)

# --- Block-jackknife errors on K (5x5 spatial blocks, as for the VSF) ---
_, kurtosis_err, kurtosis_reps, _ = kurtosis_block_jackknife(x_pc, y_pc, d["vel"], bin_edges)

# --- Save the kurtosis curve (used by plot_kurtosis_vs_l_over_rhii.py and fit_kurtosis_slope.py) ---
r = 10**log_centers_bins
R_HII = get_hii_radius_pc(FITS_FILE)
use_bin = cnt >= MIN_COUNTS
if use_bin.sum() < 3:
    sys.exit(f"{REGION_NAME} {LABEL}: only {use_bin.sum()} separation bins with >= {MIN_COUNTS} counts; skipped")
Path("NPZ").mkdir(exist_ok=True)
np.savez(f"NPZ/KURT_{REGION}_{LINE.upper()}.npz", l_pc=r[use_bin], l_over_rhii=r[use_bin] / R_HII,
         kurtosis=kurtosis[use_bin], kurtosis_err=kurtosis_err[use_bin],
         kurtosis_jackknife_replicates=kurtosis_reps[:, use_bin],
         counts=cnt[use_bin], r_hii_pc=R_HII, distance_pc=d["dist"], line=LINE, snr_min=args.snr_min)
print(f"R_HII = {R_HII:.2f} pc; saved NPZ/KURT_{REGION}_{LINE.upper()}.npz")
# Folded PDFs per separation bin (used by batch_kurtosis_3d_all_regions.py and 05_kurtosis_scale/).
Path("NPZ_kurtosis").mkdir(exist_ok=True)
np.savez_compressed(f"NPZ_kurtosis/{REGION}_{LINE.upper()}_order1_pdfkk.npz", pdf=pdf[use_bin], dv_bins=dv_bins,
                    dv_centers=dv_centers, log_centers=log_centers_bins[use_bin], kk=kurtosis[use_bin],
                    c2=c2[use_bin], c4=c4[use_bin], cnt=cnt[use_bin], folded=True, snr_min=args.snr_min,
                    distance_pc=d["dist"], r_hii_pc=R_HII)

# --- 3D Kurtosis Plot ---
dv_centers_plot = 0.5 * (dv_bins[:-1] + dv_bins[1:])
fig = plt.figure(figsize=(7, 7))

gs = fig.add_gridspec(
    2, 1,
    height_ratios=[0.03, 1.0],
    hspace=0.18,   # enough room for y/z labels
)

cax = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], projection="3d")
cmap = cm.viridis
norm = plt.Normalize(3, 6)
x_min, x_max = 0, 10
mask = (dv_centers_plot >= x_min) & (dv_centers_plot <= x_max)
for i in np.flatnonzero(use_bin):
    with np.errstate(divide="ignore"):
        log_p = np.log10(pdf[i][mask])
    log_p[log_p < LOG_P_FLOOR] = np.nan
    ax.plot(
        dv_centers_plot[mask],
        np.log10(np.full(np.sum(mask), r[i] / R_HII)),
        log_p,
        color=cmap(norm(kurtosis[i])),
    )

# Improved axis labels and plot presentation
ax.set_xlabel(r"$\Delta v$ (km/s)", fontsize=14, labelpad=10)
ax.set_ylabel(r"$\log_{10}(\ell / R_{\rm HII})$", fontsize=14, labelpad=10)
ax.set_zlabel(r"$\log_{10}\mathcal{P}(|\Delta v|)$", fontsize=14, labelpad=10)
ax.set_xlim(x_min, x_max)
ax.set_ylim(np.log10(r[use_bin][0] / R_HII), np.log10(r[use_bin][-1] / R_HII))
ax.view_init(elev=25, azim=45)
ax.dist = 9
# mplot3d has no working log axis: plot log10 P on a linear z axis instead.
zmax = np.nanmax([np.nanmax(pdf[i][mask]) for i in np.flatnonzero(use_bin)])
ax.set_zlim(LOG_P_FLOOR, np.log10(min(zmax, 1)))
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
cbar = plt.colorbar(mappable, cax=cax, orientation='horizontal')
cbar.set_label(r"Kurtosis", labelpad=3, fontsize=13)
cbar.ax.tick_params(labelsize=11)
# Annotate colorbar: low end = Gaussian, high end = Exponential
cbar.ax.text(3, 1.2, "Gaussian", ha='center', va='bottom', fontsize=11, color='k', transform=cbar.ax.get_xaxis_transform())
cbar.ax.text(6, 1.2, "Exponential", ha='center', va='bottom', fontsize=11, color='k', transform=cbar.ax.get_xaxis_transform())
ax.text2D(0.02, 0.97, f"{REGION_NAME}, {LABEL}", transform=ax.transAxes, fontsize=15, ha="left", va="top")
plt.tight_layout()
figure_dir = Path('Figures/manuscript/04_flame_kurtosis_3d')
figure_dir.mkdir(parents=True, exist_ok=True)
figure_out = figure_dir / f'{REGION.replace("_nebula", "")}_{LINE.upper()}_refactored_3D.png'
plt.savefig(figure_out, dpi=200, bbox_inches='tight')

valid = use_bin
print("largest-separation bins (pairs x n_jobs):", [(round(float(li), 2), int(ni)) for li, ni in zip(r[valid][-4:], cnt[valid][-4:])])
print(f"{valid.sum()} separation bins, l = {r[valid][0]:.3f}-{r[valid][-1]:.2f} pc; "
      f"kurtosis {np.nanmin(kurtosis[valid]):.2f}-{np.nanmax(kurtosis[valid]):.2f}")
for li, ki, ni in list(zip(r[valid], kurtosis[valid], cnt[valid]))[::4]:
    print(f"  l = {li:7.3f} pc   K = {ki:5.2f}   pairs = {int(ni):,}")
print(f"✓ 3D kurtosis plot created and saved as {figure_out}")
