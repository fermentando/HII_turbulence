"""Linear fit of the velocity-increment kurtosis against log separation, per region and line.

For every NPZ/KURT_<region>_<LINE>.npz (written by kurtosis_plot.py):
  * consider all separation bins (optionally only l <= --l-max-over-rhii * R_HII);
  * find the runs of consecutive bins with K > --k-min (default 3, Gaussian) and keep the run
    starting closest to the smallest separation; bins with K > --k-max (default 10) inside
    it are dropped from the fit but do not end the run;
  * fit K = a + b * log10(l / R_HII) by ordinary least squares over those bins.
b = dK/dlog10(l) (the R_HII normalisation only shifts a). Errors on a and b come from the
spatial block jackknife: the same line is fitted to every leave-one-block-out K curve (same
bins), which accounts for the strong correlation between neighbouring bins. The plain OLS
slope error (which assumes independent bins) is also written for comparison.

Outputs (Figures/manuscript/04_flame_kurtosis_3d/):
  kurtosis_slope_fits.csv                       one row per region/line
  kurtosis_vs_l_over_rhii_kcut_fits.png         grid: data (fitted bins filled) and the fitted lines
  kurtosis_slope_summary.png                    dK/dlog10 l per region, coloured by line

Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/fit_kurtosis_slope.py
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

LINES = ("NII", "SII", "SIII")
OUT_DIR = Path("Figures/manuscript/04_flame_kurtosis_3d")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--k-min", type=float, default=3.0,
                    help="Fit the run of consecutive bins with K above this that starts at the smallest l.")
parser.add_argument("--k-max", type=float, default=10.0, help="Drop bins with K above this before fitting.")
parser.add_argument("--l-max-over-rhii", type=float, default=np.inf,
                    help="Fit only bins with l <= this x R_HII (default: no limit).")
parser.add_argument("--min-points", type=int, default=5, help="Minimum bins left after the cuts to fit.")
args = parser.parse_args()


def ols_line(x, y):
    """Intercept, slope, slope standard error, intercept standard error."""
    X = np.column_stack((np.ones_like(x), x))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(len(x) - 2, 1)
    cov = np.linalg.inv(X.T @ X) * (resid @ resid) / dof
    return coef[0], coef[1], np.sqrt(cov[1, 1]), np.sqrt(cov[0, 0])


def first_run_above(k, allowed, k_min):
    """Mask of the run of consecutive allowed bins with K > k_min that starts at the smallest l.

    k is ordered by increasing l; the run ends at the first bin with K <= k_min (or not allowed).
    """
    mask = np.zeros(len(k), dtype=bool)
    above = allowed & np.isfinite(k) & (k > k_min)
    if not above.any():
        return mask
    start = int(np.argmax(above))
    stop = start
    while stop < len(k) and above[stop]:
        stop += 1
    mask[start:stop] = True
    return mask


regions = sorted({p.stem[len("KURT_"):].rsplit("_", 1)[0] for p in Path("NPZ").glob("KURT_*_*.npz")})
if not regions:
    sys.exit("No NPZ/KURT_*.npz files: run kurtosis_plot.py first")

fits, curves = [], {}
for region in regions:
    for line in LINES:
        path = Path(f"NPZ/KURT_{region}_{line}.npz")
        if not path.exists():
            continue
        with np.load(path) as d:
            x_all = np.log10(d["l_pc"] / float(d["r_hii_pc"]))
            k_all = np.asarray(d["kurtosis"], dtype=float)
            k_err = np.asarray(d["kurtosis_err"], dtype=float) if "kurtosis_err" in d.files else np.full_like(k_all, np.nan)
            k_reps = (np.asarray(d["kurtosis_jackknife_replicates"], dtype=float)
                      if "kurtosis_jackknife_replicates" in d.files else np.empty((0, len(k_all))))
            r_hii = float(d["r_hii_pc"])
        order = np.argsort(x_all)
        x_all, k_all, k_err, k_reps = x_all[order], k_all[order], k_err[order], k_reps[:, order]
        run = first_run_above(k_all, x_all <= np.log10(args.l_max_over_rhii), args.k_min)
        keep = run & (k_all <= args.k_max)
        curves[(region, line)] = (x_all, k_all, k_err, keep, r_hii)
        if keep.sum() < args.min_points:
            print(f"  {region} {line}: only {keep.sum()} bins in the first K > {args.k_min:g} run "
                  f"(l <= {args.l_max_over_rhii:g} R_HII, K <= {args.k_max:g}); not fitted")
            continue
        a, b, b_err_ols, _ = ols_line(x_all[keep], k_all[keep])
        # Jackknife: refit every leave-one-block-out curve on the same bins.
        rep_fits = np.array([ols_line(x_all[keep], rep[keep])[:2] for rep in k_reps
                             if np.all(np.isfinite(rep[keep]))])
        if len(rep_fits) >= 4:
            n_rep = len(rep_fits)
            a_err, b_err = np.sqrt((n_rep - 1) / n_rep * np.sum((rep_fits - rep_fits.mean(axis=0)) ** 2, axis=0))
        else:
            n_rep, a_err, b_err = len(rep_fits), np.nan, np.nan
        fits.append({
            "region": region, "line": line, "n_bins_used": int(keep.sum()),
            "n_bins_cut_K": int(np.sum(run & (k_all > args.k_max))),
            "l_over_rhii_min": 10 ** x_all[keep].min(), "l_over_rhii_max": 10 ** x_all[keep].max(),
            "dK_dlog10l": b, "dK_dlog10l_err": b_err, "dK_dlog10l_err_ols": b_err_ols,
            "K_at_l_eq_RHII": a, "K_at_l_eq_RHII_err": a_err, "n_jackknife": n_rep, "r_hii_pc": r_hii,
        })

OUT_DIR.mkdir(parents=True, exist_ok=True)
csv_out = OUT_DIR / "kurtosis_slope_fits.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as fobj:
    writer = csv.DictWriter(fobj, fieldnames=list(fits[0].keys()))
    writer.writeheader()
    for row in fits:
        writer.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in row.items()})
fit_map = {(f["region"], f["line"]): f for f in fits}

plt.style.use("./custom_plot.mplstyle")

# ---------------------------------------------------------------- grid: data + fits
n_cols = 3
n_rows = int(np.ceil(len(regions) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
for idx, region in enumerate(regions):
    ax = axes[divmod(idx, n_cols)]
    r_hii = None
    for line in LINES:
        if (region, line) not in curves:
            continue
        x_all, k_all, k_err, keep, r_hii = curves[(region, line)]
        color = LINE_COLORS[line]
        ax.errorbar(10 ** x_all[keep], k_all[keep], yerr=k_err[keep], fmt="o", ms=3.5, color=color,
                    alpha=0.85, elinewidth=0.9, capsize=0)
        shown_out = ~keep & np.isfinite(k_all) & (k_all <= args.k_max)
        ax.plot(10 ** x_all[shown_out], k_all[shown_out], "o", ms=3.5, mfc="none", color=color, alpha=0.35)
        f = fit_map.get((region, line))
        label = LINE_LABELS[line]
        if f is not None:
            xx = np.linspace(x_all[keep].min(), x_all[keep].max(), 50)
            ax.plot(10 ** xx, f["K_at_l_eq_RHII"] + f["dK_dlog10l"] * xx, "-", lw=2.2, color=color)
            label += f": {f['dK_dlog10l']:+.2f}$\\pm${f['dK_dlog10l_err']:.2f}"
        ax.plot([], [], "o-", color=color, label=label)
    ax.axhline(3.0, color="0.3", ls="--", lw=1.2, zorder=0)
    ax.axvline(1.0, color="0.5", ls=":", lw=1.2, zorder=0)  # l = R_HII
    ax.set_xscale("log")
    ax.set_ylim(top=args.k_max)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in (2, 5) else ""))
    ax.set_xlabel(r"$\ell / R_{\rm HII}$")
    if idx % n_cols == 0:
        ax.set_ylabel(r"$K$")
    ax.set_title(region.replace("_", " ") + (f" ($R_{{\\rm HII}} = {r_hii:.1f}$ pc)" if r_hii else ""), fontsize=16)
    ax.legend(fontsize=10, loc="upper right", title=r"${\rm d}K/{\rm d}\log_{10}\ell$", title_fontsize=10)
    ax.grid(True, alpha=0.3)
for idx in range(len(regions), n_rows * n_cols):
    axes[divmod(idx, n_cols)].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(wspace=0.18)
grid_out = OUT_DIR / "kurtosis_vs_l_over_rhii_kcut_fits.png"
fig.savefig(grid_out, dpi=170, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- summary of slopes
# Scale by 1 + exp(ln(r_min/R_HII)) = 1 + r_min/R_HII.
# r_min is the 37-arcsec lower binning cutoff, not the first bin centre.
normalized = {}
for f in fits:
    with np.load(Path('NPZ') / f"KURT_{f['region']}_{f['line']}.npz") as data:
        ratio = float(data['distance_pc']) * np.deg2rad(37.0 / 3600.0) / float(data['r_hii_pc'])
    factor = 1.0 + np.exp(np.log(ratio))
    normalized[(f['region'], f['line'])] = (
        f['dK_dlog10l'] * factor, f['dK_dlog10l_err'] * factor)
fig, (ax, ax_pdf) = plt.subplots(1, 2, figsize=(12.5, 5.5), sharey=True,
                                gridspec_kw={'width_ratios': [4, 1]},
                                layout='constrained')
offsets = {"NII": -0.2, "SII": 0.0, "SIII": 0.2}
for i, region in enumerate(regions):
    for line in LINES:
        f = fit_map.get((region, line))
        if f is None:
            continue
        slope, slope_error = normalized[(region, line)]
        ax.errorbar(i + offsets[line], slope, yerr=slope_error, fmt="o", ms=8,
                    color=LINE_COLORS[line], mec="k", mew=0.5, capsize=3,
                    label=LINE_LABELS[line] if i == 0 or not any(
                        (r, line) in fit_map for r in regions[:i]) else None)
ax.axhline(0.0, color="0.3", ls="--", lw=1.2, zorder=0)
ax.set_xticks(range(len(regions)))
ax.set_xticklabels([r.replace("_nebula", "").replace("_", " ") for r in regions], rotation=35, ha="right")
ax.set_ylabel(r"$(\mathrm{d}K/\mathrm{d}\log_{10}r)\,[1+\exp(\ln(r_{\min}/R_{\mathrm{HII}}))]$")
l_cut = (f", $\\ell \\le {args.l_max_over_rhii:g}\\,R_{{\\rm HII}}$"
         if np.isfinite(args.l_max_over_rhii) else "")
ax.set_title(f"Kurtosis slope (first run with $K > {args.k_min:g}${l_cut}; jackknife errors)", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, axis="y", alpha=0.3)
# One smoothed slope distribution per species with equal region weights.
# Gaussian widths are half the jackknife errors; plotted error bars are unchanged.
ymin, ymax = ax.get_ylim()
y_pdf = np.linspace(ymin, ymax, 4000)
for line in LINES:
    valid = [f for f in fits if f['line'] == line
             and np.isfinite(f['dK_dlog10l'])
             and np.isfinite(f['dK_dlog10l_err']) and f['dK_dlog10l_err'] > 0]
    if not valid:
        continue
    means = np.array([normalized[(f['region'], line)][0] for f in valid])
    sigmas = 0.5 * np.array([normalized[(f['region'], line)][1] for f in valid])
    density = np.mean(np.exp(-0.5 * ((y_pdf[:, None] - means) / sigmas) ** 2)
                      / (np.sqrt(2 * np.pi) * sigmas), axis=1)
    ax_pdf.plot(density, y_pdf, color=LINE_COLORS[line], lw=2, label=LINE_LABELS[line])
ax.set_yscale('linear')
# Top at 0 unless a positive slope (with its error bar) would be cut off.
ax.set_ylim(ymin, max(0.0, max(v + e for v, e in normalized.values() if np.isfinite(v + e)) + 0.2))
ax_pdf.axhline(0, color='0.3', ls='--', lw=1.2)
ax_pdf.set_xlim(left=0)
ax_pdf.set_xlabel('Probability density')
ax_pdf.set_title('Slope PDFs\n(widths: 0.5 × JK errors)', fontsize=11)
ax_pdf.tick_params(axis='y', left=False, labelleft=False)
ax_pdf.grid(alpha=0.2)
ax_pdf.legend(fontsize=10)
summary_out = OUT_DIR / "kurtosis_slope_summary.png"
fig.savefig(summary_out, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"{'region':15s} {'line':4s} {'used':>4s} {'K>max':>5s} {'l/R range':>12s} {'dK/dlog10 l (jk)':>17s} "
      f"{'(ols err)':>9s} {'K(R_HII)':>13s}")
for f in fits:
    print(f"{f['region']:15s} {f['line']:4s} {f['n_bins_used']:4d} {f['n_bins_cut_K']:5d} "
          f"{f['l_over_rhii_min']:5.2f}-{f['l_over_rhii_max']:5.2f} "
          f"{f['dK_dlog10l']:+7.2f} ± {f['dK_dlog10l_err']:5.2f}   ({f['dK_dlog10l_err_ols']:.2f}) "
          f"{f['K_at_l_eq_RHII']:6.2f} ± {f['K_at_l_eq_RHII_err']:.2f}")
print(f"Wrote {csv_out}\nWrote {grid_out}\nWrote {summary_out}")
