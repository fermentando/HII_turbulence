"""Best-fit overview of the mixing-corrected (w, v) inference for all regions.

Reads Figures/manuscript/08_shell_inference/mixing_correction/<region>_k<K>.csv and <region>_k<K>_fits.npz
(written by mixing_corrected_slopes.py) and the global model B coefficients.

Outputs (same folder):
  best_fit_summary_k<K>.png      best Delta r/R_HII and v_bulk/v_turb with 1-sigma ranges, uncorrected vs corrected
  slope_vs_floor_best_fit_k<K>.png  slopes vs floor with the best-fit model, one panel per region

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/plot_mixing_corrected_best_fits.py [k_neigh]
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "04_flame_kurtosis_3d"), str(HERE.parent / "00_core")]
from fit_global_dk_dlogr_model import RMIN_REF, model  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

K_NEIGH = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DIR = Path("Figures/manuscript/08_shell_inference/mixing_correction")
rows = [r for r in csv.DictReader(open("Figures/manuscript/04_kurtosis_3d_all_regions/global_dk_dlogr_model_fit.csv"))
        if r["model"] == "B"]
P, NAMES = [float(r["value"]) for r in rows], [r["parameter"] for r in rows]
LINES = ("nii", "sii", "siii")
VERSIONS = (("uncorrected", "uncorrected", "o", "none"), ("mixing-corrected", "mixing", "s", None))

df = pd.concat([pd.read_csv(f) for f in sorted(DIR.glob(f"*_k{K_NEIGH}.csv"))])
regions = sorted(df.region.unique())
plt.style.use("./custom_plot.mplstyle")

# ---------------------------------------------------------------- summary of best fits
fig, (axW, axV) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
off = {"nii": -0.24, "sii": 0.0, "siii": 0.24}
for _, r in df.iterrows():
    x = regions.index(r.region) + off[r.line] + (-0.05 if r.version == "uncorrected" else 0.05)
    c = LINE_COLORS[r.line]
    mk, mfc = ("o", "none") if r.version == "uncorrected" else ("s", c)
    for ax, best, lo, hi in ((axW, r.best_w, r.w_1sig_lo, r.w_1sig_hi), (axV, r.best_v, r.v_1sig_lo, r.v_1sig_hi)):
        ax.errorbar(x, best, yerr=[[max(best - lo, 0)], [max(hi - best, 0)]], fmt=mk, ms=6.5, color=c, mfc=mfc,
                    mew=1.4, capsize=2.5, lw=1.2, alpha=0.55 if r.version == "uncorrected" else 1.0)
for line in LINES:
    axW.plot([], [], "s", color=LINE_COLORS[line], label=LINE_LABELS[line])
axW.plot([], [], "o", color="0.4", mfc="none", label="uncorrected")
axW.plot([], [], "s", color="0.4", label=f"mixing-corrected ($\\sigma_{{\\rm loc}}$, {K_NEIGH} neighbours)")
axW.legend(fontsize=9, ncol=5, loc="upper left", bbox_to_anchor=(0.0, 1.17))
for ax in (axW, axV):
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    for i in range(1, len(regions)):
        ax.axvline(i - 0.5, color="0.85", lw=0.8)
axW.set_ylim(0.008, 2)
axV.set_ylim(0.08, 120)
axW.set_ylabel(r"$\Delta r_{\rm shell}/R_{\rm HII}$ (best, 1$\sigma$)")
axV.set_ylabel(r"$v_{\rm bulk}/v_{\rm turb}$ (best, 1$\sigma$)")
axV.set_xticks(range(len(regions)))
axV.set_xticklabels([r.replace("_", " ") for r in regions], rotation=25, ha="right")
axV.text(0.995, 0.03, r"bars reaching the axis limits are one-sided (grid edge: $w \in [0.01, 1.5]$, $v \in [0.1, 100]$)",
         transform=axV.transAxes, ha="right", fontsize=8.5, color="0.35")
fig.tight_layout()
fig.savefig(DIR / f"best_fit_summary_k{K_NEIGH}.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- slopes vs floor with best-fit model
n_cols = 4
n_rows = int(np.ceil(len(regions) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.8 * n_rows), squeeze=False)
for ax, region in zip(axes.flat, regions):
    path = DIR / f"{region}_k{K_NEIGH}_fits.npz"
    fits = np.load(path) if path.exists() else {}
    for line in LINES:
        c = LINE_COLORS[line]
        key = f"{line}_mixing"
        if f"{key}_floors" not in fits:
            continue
        fl, sl, er, (bv, bw) = fits[f"{key}_floors"], fits[f"{key}_slopes"], fits[f"{key}_err"], fits[f"{key}_best"]
        ax.errorbar(fl, sl, yerr=er, fmt="s", ms=5, color=c, capsize=2,
                    label=fr"{LINE_LABELS[line]}: $v$={bv:.2g}, $w$={bw:.2g}")
        if f"{line}_uncorrected_floors" in fits:
            ax.plot(fits[f"{line}_uncorrected_floors"], fits[f"{line}_uncorrected_slopes"], "o", ms=4.5, mfc="none",
                    color=c, alpha=0.5)
        fd = np.geomspace(fl.min() * 0.9, fl.max() * 1.1, 60)
        xd = np.vstack((np.full(fd.size, np.log10(bw)), np.full(fd.size, bv), np.log10(fd / RMIN_REF)))
        ax.plot(fd, model(xd, *P, names=NAMES), "-", color=c, lw=1.6)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(0, color="0.3", ls="--", lw=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.set_title(region.replace("_", " "), fontsize=11)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, loc="best")
    else:
        ax.text(0.5, 0.5, "no fit", transform=ax.transAxes, ha="center", va="center", color="0.4")
for ax in axes.flat[len(regions):]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$dK/d\log_{10}\ell$")
for ax in axes[-1, :]:
    ax.set_xlabel(r"floor $r_{\rm min}/R_{\rm HII}$")
fig.suptitle(f"Mixing-corrected slopes (filled squares) with the best-fit model (lines); open circles: uncorrected "
             f"($\\sigma_{{\\rm loc}}$ from {K_NEIGH} neighbours)", fontsize=12)
fig.tight_layout()
fig.savefig(DIR / f"slope_vs_floor_best_fit_k{K_NEIGH}.png", dpi=180)
plt.close(fig)
print(f"Wrote {DIR}/best_fit_summary_k{K_NEIGH}.png and slope_vs_floor_best_fit_k{K_NEIGH}.png")
