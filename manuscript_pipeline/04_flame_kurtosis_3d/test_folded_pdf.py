"""Test: signed P(dv) (current) vs folded P(|dv|) = P(sqrt(dv^2)), and the kurtosis estimators.

For one observed region/line and one synthetic projected shell, computes the pair PDF twice
(signed dv with the old compute_bins range, min-mean..max-mean, and |dv| with folded_dv_bins
over 0..max-min, now the default)
and compares:
  K_pairs        <dv^4>/<dv^2>^2 from all pairs            (observed pipeline; unchanged by folding)
  K_pdf_central  central moments of the signed binned PDF  (current synthetic-model estimator)
  K_pdf_folded   raw moments of the folded binned PDF      (estimator after folding)
Writes figures to Figures/manuscript/04_flame_kurtosis_3d/test_folded/; nothing else is overwritten.
Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/test_folded_pdf.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "00_core"))
from projected_cube_kurtosis import kurtosis_from_pdf  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402
from projected_shell_kurtosis import shell_projected_velocity  # noqa: E402
from vsf_module import (  # noqa: E402
    LINE_LABELS, VSF2D, compute_bins, compute_pdf_and_statistics, folded_dv_bins, get_hii_radius_pc,
    load_line_data,
)

OUT_DIR = Path("Figures/manuscript/04_flame_kurtosis_3d/test_folded")
MIN_COUNTS = 4000
LOG_P_FLOOR = -3.0

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--region", default="Flame_nebula")
parser.add_argument("--line", default="nii", choices=("nii", "sii", "siii"))
parser.add_argument("--shell", nargs=2, type=float, default=(0.75, 1.05), metavar=("R_IN", "R_OUT"))
args = parser.parse_args()


def raw_kurtosis_from_pdf(pdf, dv_centers):
    """Raw-moment kurtosis sum p d^4 / (sum p d^2)^2 per row (no mean subtraction)."""
    p = pdf / np.where(pdf.sum(axis=1, keepdims=True) > 0, pdf.sum(axis=1, keepdims=True), np.nan)
    m2 = np.sum(p * dv_centers**2, axis=1)
    m4 = np.sum(p * dv_centers**4, axis=1)
    return m4 / m2**2


def signed_dv_bins(vel):
    """The pre-folding compute_bins velocity range."""
    return np.linspace(np.nanmin(vel) - np.mean(vel), np.nanmax(vel) - np.mean(vel), 80)


def run_both(x, y, vel, vel_err, bin_edges, log_centers, n_jobs):
    kw = dict(order=1, n_jobs=n_jobs)
    dv_bins_signed = signed_dv_bins(vel)
    pdf_s, _, _, cnt, dvc_s, k_pairs_s = compute_pdf_and_statistics(
        x, y, vel, vel_err, log_centers, bin_edges, dv_bins_signed, fold=False, **kw)
    dv_bins_f = folded_dv_bins(vel)
    pdf_f, _, _, _, dvc_f, k_pairs_f = compute_pdf_and_statistics(
        x, y, vel, vel_err, log_centers, bin_edges, dv_bins_f, fold=True, **kw)
    # Fraction of pairs the signed bins drop: unordered pairs, so dv sign is 50/50.
    lo, hi = -dv_bins_signed[0], dv_bins_signed[-1]
    lost = 0.5 * (pdf_f[:, dvc_f > lo].sum(axis=1) + pdf_f[:, dvc_f > hi].sum(axis=1))
    return dict(r=10**log_centers, cnt=cnt, pdf_s=pdf_s, dvc_s=dvc_s, pdf_f=pdf_f, dvc_f=dvc_f,
                k_pairs_s=k_pairs_s, k_pairs_f=k_pairs_f,
                k_central=kurtosis_from_pdf(pdf_s, dvc_s), k_folded=raw_kurtosis_from_pdf(pdf_f, dvc_f),
                lost=lost, signed_range=(dv_bins_signed[0], dv_bins_signed[-1]))


def plot_3d(ax, dvc, pdf, r, kk, use, x_lim, cmap, norm):
    mask = (dvc >= x_lim[0]) & (dvc <= x_lim[1])
    for i in np.flatnonzero(use):
        with np.errstate(divide="ignore"):
            log_p = np.log10(pdf[i][mask])
        log_p[log_p < LOG_P_FLOOR] = np.nan
        ax.plot(dvc[mask], np.full(mask.sum(), np.log10(r[i])), log_p, color=cmap(norm(kk[i])))
    ax.set_xlim(*x_lim)
    ax.set_zlim(LOG_P_FLOOR, 0)
    ax.view_init(elev=25, azim=45)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False


def compare_figure(res, use, title, x_unit, out_png, xlim_signed):
    cmap, norm = cm.viridis, plt.Normalize(3, 6)
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    plot_3d(ax1, res["dvc_s"], res["pdf_s"], res["r"], res["k_pairs_s"], use, xlim_signed, cmap, norm)
    ax1.set_xlabel(r"$\Delta v$"); ax1.set_ylabel(r"$\log_{10}\ell$"); ax1.set_zlabel(r"$\log_{10}\mathcal{P}$")
    ax1.set_title(r"current: signed $\mathcal{P}(\Delta v)$")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    plot_3d(ax2, res["dvc_f"], res["pdf_f"], res["r"], res["k_pairs_f"], use, (0, xlim_signed[1]), cmap, norm)
    ax2.set_xlabel(r"$|\Delta v|$"); ax2.set_ylabel(r"$\log_{10}\ell$"); ax2.set_zlabel(r"$\log_{10}\mathcal{P}$")
    ax2.set_title(r"folded: $\mathcal{P}(\sqrt{\Delta v^2})$")

    # A few separations, overlaid: signed PDF folded by hand vs the folded PDF.
    ax3 = fig.add_subplot(2, 3, 3)
    idx = np.flatnonzero(use)
    picks = idx[np.linspace(0, len(idx) - 1, 4).astype(int)]
    for k, i in enumerate(picks):
        c = plt.cm.plasma(k / 4)
        ax3.plot(np.abs(res["dvc_s"]), res["pdf_s"][i] / np.diff(res["dvc_s"])[0], ".", ms=3, color=c, alpha=0.6)
        ax3.plot(res["dvc_f"], res["pdf_f"][i] / np.diff(res["dvc_f"])[0] / 2, "-", color=c,
                 label=fr"$\ell$={res['r'][i]:.3g} {x_unit}")
    for edge in res["signed_range"]:
        ax3.axvline(abs(edge), color="k", ls=":", lw=0.8)
    ax3.set_yscale("log"); ax3.set_ylim(1e-5, None)
    ax3.set_xlabel(r"$|\Delta v|$"); ax3.set_ylabel("density (folded / 2)")
    ax3.set_title("dots: signed bins, lines: folded; dotted = signed bin range", fontsize=9)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 3, 4)
    r = res["r"][use]
    ax4.plot(r, res["k_pairs_s"][use], "k-", lw=2.5, label=r"$K_\mathrm{pairs}$ (signed run)")
    ax4.plot(r, res["k_pairs_f"][use], "y--", lw=1.5, label=r"$K_\mathrm{pairs}$ (folded run)")
    ax4.plot(r, res["k_central"][use], "C3o-", ms=3, label=r"$K$ signed PDF, central (models now)")
    ax4.plot(r, res["k_folded"][use], "C0s-", ms=3, label=r"$K$ folded PDF, raw")
    ax4.axhline(3, color="0.5", ls=":")
    ax4.set_xscale("log"); ax4.set_xlabel(fr"$\ell$ [{x_unit}]"); ax4.set_ylabel("K"); ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(r, res["k_central"][use] - res["k_pairs_s"][use], "C3o-", ms=3, label="signed PDF central - pairs")
    ax5.plot(r, res["k_folded"][use] - res["k_pairs_s"][use], "C0s-", ms=3, label="folded PDF raw - pairs")
    ax5.axhline(0, color="0.5", ls=":")
    ax5.set_xscale("log"); ax5.set_xlabel(fr"$\ell$ [{x_unit}]"); ax5.set_ylabel(r"$\Delta K$"); ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(r, 100 * res["lost"][use], "k.-")
    ax6.set_xscale("log"); ax6.set_xlabel(fr"$\ell$ [{x_unit}]")
    ax6.set_ylabel("% pairs outside signed dv_bins")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def summary(name, res, use):
    d_pairs = np.nanmax(np.abs(res["k_pairs_f"][use] - res["k_pairs_s"][use]))
    d_c = res["k_central"][use] - res["k_pairs_s"][use]
    d_f = res["k_folded"][use] - res["k_pairs_s"][use]
    print(f"\n== {name}: {use.sum()} bins")
    print(f"  signed dv_bins range {res['signed_range'][0]:.2f} .. {res['signed_range'][1]:.2f}")
    print(f"  max |K_pairs folded - K_pairs signed|        = {d_pairs:.2e}")
    print(f"  K_pdf_central - K_pairs : median {np.nanmedian(d_c):+.3f}, range {np.nanmin(d_c):+.3f} .. {np.nanmax(d_c):+.3f}")
    print(f"  K_pdf_folded  - K_pairs : median {np.nanmedian(d_f):+.3f}, range {np.nanmin(d_f):+.3f} .. {np.nanmax(d_f):+.3f}")
    print(f"  pairs outside signed bins: max {100 * np.nanmax(res['lost'][use]):.2f}%")


OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Observed region (same data/bins as kurtosis_plot.py) ---
region = args.region.replace("LVM_", "").replace(".fits", "")
d = load_line_data(f"LVM_{region}.fits", args.line, subtract_gradient=True)
vsf2d = VSF2D(d["ra"], d["dec"], d["vel"], dist=d["dist"], vel_err=d["vel_err"])
x_pc, y_pc = vsf2d.x_physical, vsf2d.y_physical
log_c, edges, _, _ = compute_bins(x_pc, y_pc, d["vel"], bin_resolution=20, binning="shell",
                                        min_separation=d["dist"] * np.deg2rad(37.0 / 3600.0))
res = run_both(x_pc, y_pc, d["vel"], d["vel_err"], edges, log_c, n_jobs=4)
use = res["cnt"] >= MIN_COUNTS
name = f"{region.replace('_', ' ')} {LINE_LABELS[args.line]}"
summary(name, res, use)
compare_figure(res, use, name, "pc", OUT_DIR / f"compare_{region}_{args.line.upper()}.png", (-10, 10))

# --- Synthetic projected shell (same setup as projected_shell_kurtosis.py defaults) ---
r_in, r_out = args.shell
rng = np.random.default_rng(7)
vx, _, _ = kolmogorov_velocity_grid(160, 2.4, rng)
y, z, proj_v, _, _ = shell_projected_velocity(vx, 2.4, r_in, r_out)
yy, zz = np.meshgrid(y, z, indexing="ij")
ok = np.isfinite(proj_v)
ys, zs, vs = yy[ok].ravel(), zz[ok].ravel(), proj_v[ok].ravel()
log_c, edges, _, _ = compute_bins(ys, zs, vs, bin_resolution=20)
with np.errstate(invalid="ignore", divide="ignore"):
    res = run_both(ys, zs, vs, np.zeros_like(vs), edges, log_c, n_jobs=4)
use = res["cnt"] > 0
name = f"synthetic shell r_in={r_in}, r_out={r_out}"
summary(name, res, use)
half = 1.2 * max(abs(res["signed_range"][0]), res["signed_range"][1])
compare_figure(res, use, name, "box units", OUT_DIR / f"compare_shell_rin{r_in}_rout{r_out}.png", (-half, half))
print(f"\nFigures in {OUT_DIR}")
