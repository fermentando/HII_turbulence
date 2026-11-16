"""Shell-inference Delta chi2 maps for selected regions, one panel per region, all lines overlaid.

Reads Figures/manuscript/08_shell_inference/data/<region>_<LINE>_chi2map.npz (infer_shell_parameters.py).
Shaded: Delta chi2 below the two-parameter level enclosing 38.3% (the 2-D equivalent of +/-0.5 sigma);
lines: 1 sigma (thick) and 2 sigma (thin); star: best fit.

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/plot_chi2_maps_selected.py Gum38a Rosette_nebula
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_core"))
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

DATA = Path("Figures/manuscript/08_shell_inference/data")
OUT_DIR = Path("Figures/manuscript/08_shell_inference")
REGIONS = sys.argv[1:] or ["Gum38a", "Rosette_nebula"]
LINES = ("NII", "SII", "SIII")
LEVEL_HALF = chi2.ppf(erf(0.5 / np.sqrt(2)), 2)  # 0.966
LEVEL_1, LEVEL_2 = chi2.ppf(erf(1 / np.sqrt(2)), 2), chi2.ppf(erf(2 / np.sqrt(2)), 2)  # 2.30, 6.18

plt.style.use("./custom_plot.mplstyle")
fig, axes = plt.subplots(1, len(REGIONS), figsize=(6.4 * len(REGIONS), 5.6), sharey=True, squeeze=False)
for ax, region in zip(axes[0], REGIONS):
    for line in LINES:
        path = DATA / f"{region}_{line}_chi2map.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            V, W, d = z["v_grid"], z["w_grid"], z["dchi2"]
        VV, WW = np.meshgrid(V, W, indexing="ij")
        c = LINE_COLORS[line]
        ax.contourf(VV, WW, d, levels=[0, LEVEL_HALF], colors=[c], alpha=0.35)
        ax.contour(VV, WW, d, levels=[LEVEL_1], colors=[c], linewidths=1.8)
        ax.contour(VV, WW, d, levels=[LEVEL_2], colors=[c], linewidths=0.8, linestyles="--")
        ib = np.unravel_index(np.argmin(d), d.shape)
        ax.plot(VV[ib], WW[ib], "*", ms=17, color=c, mec="black", mew=0.9, zorder=10, clip_on=False)
        print(f"{region} {line}: best v={VV[ib]:.3g}, dr/R_HII={WW[ib]:.2g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(V[0], V[-1])
    ax.set_ylim(W[0], W[-1])
    ax.set_xlabel(r"$v_{\rm bulk}/v_{\rm turb}$")
    ax.set_title(region.replace("_", " "), fontsize=12)
axes[0, 0].set_ylabel(r"$\Delta r_{\rm shell}/R_{\rm HII}$")
handles = [plt.Line2D([], [], marker="s", ls="none", ms=10, color=LINE_COLORS[line], label=LINE_LABELS[line])
           for line in LINES]
axes[0, -1].legend(handles=handles, fontsize=10, loc="lower left", framealpha=0.9)
fig.tight_layout()
out = OUT_DIR / f"shell_inference_chi2_maps_{'_'.join(REGIONS)}.png"
fig.savefig(out, dpi=200)
print(f"Wrote {out}")
