#!/usr/bin/env python3
"""ESS comparison of a synthetic expanding shell with the observed regions.

Left: S_p vs S_3 (p = 2, 4, 5) for the synthetic shell (thinnest width, WIDTHS[0], N=100,000,
filled circles + solid fits), from the cache of
04_flame_kurtosis_3d/plot_vsf_order_vs_order3_ess.py, together with the observed Flame [NII]
S_p (open squares with block-jackknife errors + dashed fits) from NPZ/VSFORD_Flame_nebula_NII.npz.
Colours mark the order, as in the random-point-count VSF plots. The two datasets are in different
velocity units (synthetic: model units, Flame: km/s); the ESS slope does not depend on that.

Right: zeta_p/zeta_3 normalised by K41 (p/3) vs p, as in section 07
(vsf_vs3_slope_vs_order_plot.py): every region from Figures/vsf_vs3_slopes_by_region.csv,
She & Leveque '94 and Boldyrev '02, plus the synthetic-shell fit.

Run from the repo root, after vsf_order_vs3_allregions.py:
    python manuscript_pipeline/06_vsf_orders/plot_vsf_order_ratio_vs_r.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "04_flame_kurtosis_3d"))

from generate_shell_thickness_grid import width_grid  # noqa: E402

SYN_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
CACHE = SYN_DIR / "_vsf_order_vs_order3_cache.npz"
N_POINTS = 100_000
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20
SYN_ORDERS = [2, 3, 4, 5]
WIDTH_INDEX = 0  # smallest width (finest r_min)

OBS_NPZ = REPO_ROOT / "NPZ" / "VSFORD_Flame_nebula_NII.npz"
OBS_CSV = REPO_ROOT / "Figures" / "vsf_vs3_slopes_by_region.csv"
MIN_PAIRS = 1000  # as in vsf_order_vs3_allregions.py
REFERENCE_ORDER = 3
COLORS = {2: "#1b9e77", 4: "#d95f02", 5: "#7570b3"}
REGION_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">"]  # as plot_breakpoint_slope_vs_hii_radius
OUT = REPO_ROOT / "Figures" / "manuscript" / "06_vsf_orders" / "vsf_order_ratio_synthetic_vs_observed.png"

for path, how in ((CACHE, "04_flame_kurtosis_3d/plot_vsf_order_vs_order3_ess.py"),
                  (OBS_NPZ, "06_vsf_orders/vsf_order_vs3_allregions.py"),
                  (OBS_CSV, "06_vsf_orders/vsf_order_vs3_allregions.py")):
    if not path.exists():
        sys.exit(f"{path.relative_to(REPO_ROOT)} not found -- run {how} first")

# --- synthetic shell: cached S_p(r) for one width ---
with np.load(CACHE, allow_pickle=True) as c:
    if not (np.allclose(c["widths"], WIDTHS) and int(c["bin_resolution"]) == BIN_RESOLUTION
            and int(c["n_points"]) == N_POINTS and list(c["orders"]) == SYN_ORDERS):
        sys.exit(f"{CACHE} doesn't match current WIDTHS/BIN_RESOLUTION/N_POINTS/ORDERS -- rerun the ESS script.")
    syn = {p: np.asarray(c[f"s{p}"][WIDTH_INDEX], dtype=float) for p in SYN_ORDERS}
width = WIDTHS[WIDTH_INDEX]

# --- observed Flame [NII]: S_p with block-jackknife errors ---
with np.load(OBS_NPZ) as d:
    obs_orders = list(d["orders"])
    use = d["counts"] >= MIN_PAIRS
    obs = {p: d["S"][k][use] for k, p in enumerate(obs_orders)}
    obs_err = {p: d["S_err"][k][use] for k, p in enumerate(obs_orders)}
obs_slopes = pd.read_csv(OBS_CSV)
flame_slope = obs_slopes[obs_slopes["region"] == "LVM_Flame_nebula.fits"].set_index("order")

style_path = REPO_ROOT / "custom_plot.mplstyle"
if style_path.exists():
    plt.style.use(str(style_path))

fig, (ax, axZ) = plt.subplots(1, 2, figsize=(13.5, 5.5))


def ess_line(x, y):
    """log-log OLS: slope, intercept."""
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return np.polyfit(np.log10(x[good]), np.log10(y[good]), 1), good


syn_zeta = {REFERENCE_ORDER: 1.0}
print(f"{'order':>5} {'synthetic':>10} {'Flame':>16} {'K41':>6}")
for p in COLORS:
    (zeta, intercept), good = ess_line(syn[REFERENCE_ORDER], syn[p])
    syn_zeta[p] = zeta
    x = np.sort(syn[REFERENCE_ORDER][good])
    ax.plot(syn[REFERENCE_ORDER][good], syn[p][good], "o", ms=4.0, color=COLORS[p], zorder=3)
    ax.plot(x, 10 ** (intercept + zeta * np.log10(x)), "-", lw=1.8, color=COLORS[p], zorder=4,
            label=fr"$p={p}$: synthetic ${zeta:.2f}$, Flame ${flame_slope.loc[p, 'slope']:.2f}\pm"
                  fr"{flame_slope.loc[p, 'slope_err']:.2f}$ (K41 ${p / 3:.2f}$)")

    # Flame: same fit as vsf_order_vs3_allregions.py (slope and jackknife error from its CSV).
    xo, yo = obs[REFERENCE_ORDER], obs[p]
    ax.errorbar(xo, yo, xerr=obs_err[REFERENCE_ORDER], yerr=obs_err[p], fmt="s", ms=4.0, mfc="none",
                mec=COLORS[p], ecolor=COLORS[p], elinewidth=0.8, capsize=1.5, alpha=0.8, zorder=3)
    (zo, io), _ = ess_line(xo, yo)
    xs = np.geomspace(xo.min(), xo.max(), 100)
    ax.plot(xs, 10 ** (io + zo * np.log10(xs)), "--", lw=1.6, color=COLORS[p], zorder=4)
    print(f"{p:>5} {zeta:>10.3f} {flame_slope.loc[p, 'slope']:>8.3f}+/-{flame_slope.loc[p, 'slope_err']:.3f} {p / 3:>6.3f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$S_3(r)$")
ax.set_ylabel(r"$S_p(r)$")
ax.grid(True, ls="--", alpha=0.4)
leg_orders = ax.legend(fontsize=8, loc="upper left", title=r"$\zeta_p/\zeta_3$", title_fontsize=8)
ax.add_artist(leg_orders)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="-", color="0.3", ms=4, label="synthetic shell (model units)"),
    Line2D([], [], marker="s", ls="--", color="0.3", mfc="none", ms=4, label=r"Flame [NII] (km s$^{-1}$)"),
], fontsize=8, loc="lower right")

# --- Right: section-07 figure (all regions, normalised by K41) + synthetic fit ---
ps = np.arange(1, 6)
k41 = ps / 3
sl94 = ps / 9 + 2 * (1 - (1 - 2 / (3 * 2)) ** (ps / 3))
bol02 = ps / 9 + 1 * (1 - (1 - 2 / (3 * 1)) ** (ps / 3))

# tab10 without its red (reserved for the synthetic shell, as in the 04 plots), plus navy.
# Region symbols: REGION_MARKERS[i] for the i-th of the alphabetically sorted LVM_*.fits regions
# (Orion included), as in plot_breakpoint_slope_vs_hii_radius, so a region keeps its symbol across figures.
ALL_REGIONS = sorted(p.name for p in REPO_ROOT.glob("LVM_*.fits"))
plot_regions = sorted(obs_slopes["region"].unique())
offsets = np.linspace(-0.15, 0.15, len(plot_regions))  # small p offsets so markers/error bars don't stack
for i_reg, region in enumerate(plot_regions):
    rows = obs_slopes[obs_slopes["region"] == region].set_index("order")
    p_r = np.array(sorted(set(rows.index) | {REFERENCE_ORDER}), dtype=float)
    z = np.array([1.0 if p == REFERENCE_ORDER else rows.loc[p, "slope"] for p in p_r])
    zerr = np.array([0.0 if p == REFERENCE_ORDER else rows.loc[p, "slope_err"] for p in p_r])
    marker = REGION_MARKERS[ALL_REGIONS.index(region) % len(REGION_MARKERS)]
    xr = p_r + np.where(p_r == REFERENCE_ORDER, 0.0, offsets[i_reg])
    axZ.plot(xr, z / (p_r / 3), "-", lw=0.5, color="0.3", alpha=0.35, zorder=2)
    axZ.errorbar(xr, z / (p_r / 3), yerr=zerr / (p_r / 3), fmt=marker, ls="none", ms=5.5, mfc="0.3", mec="k",
                 mew=0.5, ecolor="0.3", elinewidth=0.7, capsize=2, capthick=0.7, alpha=0.85, zorder=3,
                 )
axZ.plot(ps, sl94 / k41, "--", color="k", alpha=0.6, label=r"She \& Leveque '94")
axZ.plot(ps, bol02 / k41, ":", color="k", alpha=0.6, label="Boldyrev '02")
p_syn = np.array(sorted(syn_zeta), dtype=float)
axZ.plot(p_syn, np.array([syn_zeta[p] for p in p_syn]) / (p_syn / 3), "D-", ms=7, lw=2.2, color="crimson",
         zorder=5, label="synthetic shell (fit)")
axZ.axhline(1.0, color="0.4", lw=1.0, ls="-.", zorder=1, label="K41")

axZ.set_xlabel(r"$p$ (for $S_p$)")
axZ.set_ylabel(r"$\tilde{\zeta}_p / \zeta_{\rm K41}$")
axZ.set_xticks(ps)
axZ.grid(True, ls="--", alpha=0.4)
axZ.legend(fontsize=9, loc="lower left")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=220)
plt.close(fig)
print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")
