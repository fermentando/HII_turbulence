#!/usr/bin/env python3
"""Flame nebula, NII-only VSF with spatial block-jackknife error bars.

Demo/comparison script: computes the full-FOV VSF the same way
orion_fullfov_fast.py does, but with error_mode='block_jackknife' instead
of the default 'std', and overplots the old 'std' error band for contrast.
Flame has ~5,200 S/N>5 NII points, under the 10,000-point subsample cap in
compute_vsf_chunk_2d, so the point estimate itself is already using the
full dataset -- this is purely about which error bars are "proper".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import (  # noqa: E402
    VSF2D,
    fkp_vsf_error_mode_counting,
    fkp_vsf_error_effective_volume,
    pair_covariance_vsf_error,
)

style_path = REPO_ROOT / "custom_plot.mplstyle"
if style_path.exists():
    plt.style.use(str(style_path))

FITS_PATH = REPO_ROOT / "LVM_Flame_nebula.fits"
OUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "01_summary_regions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

hdul = fits.open(FITS_PATH)
table = hdul[1].data

mask = table["flux_nii"] / table["flux_nii_err"] > 5
ra = table["ra"][mask]
dec = table["dec"][mask]
vel = table["vel_nii"][mask]
vel_err = table["vel_nii_err"][mask]
hdul.close()

print(f"Flame NII: {len(ra):,} points (S/N > 5)")

vsf_std = VSF2D(ra, dec, vel, dist=1500, vel_err=vel_err)
vsf_jk = VSF2D(ra, dec, vel, dist=1500, vel_err=vel_err)

print("Computing VSF with sigma_Dp(r) = sigma_dv^p(r) / sqrt(Np(r) - 1)...")
y_std, err_std, logc_std, counts_std = vsf_std.generate_vsf_2d(
    order=1,
    outname=str(OUT_DIR / "_tmp_flame_nii_std"),
    n_jobs=1,
    min_pairs=3,
    bin_resolution=30,
    error_mode="std",
    binning="shell",
)
with np.load(str(OUT_DIR / "_tmp_flame_nii_std_order1.npz")) as d:
    pair_std = d["vsf_std"]  # sigma_dv^p(r): dispersion of |dv|^p within the bin
with np.errstate(divide="ignore", invalid="ignore"):
    err_std = pair_std / np.sqrt(np.maximum(counts_std - 1, 0))

print("Computing VSF with error_mode='block_jackknife' (new)...")
jackknife_n_blocks_per_side = 5
y_jk, err_jk, logc_jk, counts_jk = vsf_jk.generate_vsf_2d(
    order=1,
    outname=str(OUT_DIR / "_tmp_flame_nii_jackknife"),
    n_jobs=1,
    min_pairs=3,
    bin_resolution=30,
    error_mode="block_jackknife",
    jackknife_n_blocks_per_side=jackknife_n_blocks_per_side,
    jackknife_min_block_points=20,
    jackknife_min_valid_blocks=4,
    binning="shell",
)

print("Computing VSF with error_mode='bootstrap' (spaxel resampling)...")
vsf_bs = VSF2D(ra, dec, vel, dist=1500, vel_err=vel_err)
y_bs, err_bs, logc_bs, counts_bs = vsf_bs.generate_vsf_2d(
    order=1,
    outname=str(OUT_DIR / "_tmp_flame_nii_bootstrap"),
    n_jobs=1,
    min_pairs=3,
    bin_resolution=30,
    error_mode="bootstrap",
    n_bootstrap=200,
    bootstrap_sample_size=len(ra),
    binning="shell",
)

with np.load(str(OUT_DIR / "_tmp_flame_nii_jackknife_order1.npz")) as d:
    n_valid_blocks = d["jackknife_n_valid_blocks"]

valid = np.isfinite(y_jk) & np.isfinite(err_jk) & (counts_jk > 0)
r = 10**logc_jk[valid]
y = y_jk[valid]
yerr_jk = err_jk[valid]
yerr_std = err_std[valid]
yerr_bs = err_bs[valid]
n_blocks_valid = n_valid_blocks[valid]
counts_valid = counts_jk[valid]

print("Computing FKP (1994)-style error: N_eff(r) from literal Fourier mode-counting (no fitting)...")
x_phys = vsf_jk.x_physical
y_phys = vsf_jk.y_physical
area_survey = (x_phys.max() - x_phys.min()) * (y_phys.max() - y_phys.min())
n_bar = len(x_phys) / area_survey
print(f"  Survey area: {area_survey:.2f} pc^2, n_bar: {n_bar:.3f} points/pc^2")
yerr_fkp, n_eff = fkp_vsf_error_mode_counting(r, y, n_bar, area_survey)
print("  N_eff(r) = 2*pi * area_survey * d(ln r)/r^2  (uniform-density / Kaiser 1986 limit)")

# n_bar -> infinity: shot noise term (1+1/(n_bar*Dp)) -> 1, leaving only the
# pure sample/cosmic-variance floor sqrt(2/N_eff) -- "how precise could this
# ever be, even with perfect/infinite tracer density," using the same N_eff.
yerr_fkp_perfect = y * np.sqrt(2.0 / n_eff)
print(f"  Perfect-density limit: shot term drops from up to {np.nanmax(1 + 1/(n_bar*y)):.2f}x down to 1x")

print("Computing FKP (1994)-style error: TRUE effective-volume result (local S/N^2 weighting)...")
n_grid = 20
yerr_fkp_veff, n_eff_veff, a_eff = fkp_vsf_error_effective_volume(x_phys, y_phys, r, y, n_grid=n_grid)
print(f"  {n_grid}x{n_grid} density grid; A_eff(r) ranges {a_eff.min():.1f}-{a_eff.max():.1f} pc^2"
      f" (naive survey area = {area_survey:.1f} pc^2)")

print("Computing exact pair-covariance error (real-space, Gaussian, Monte Carlo over pair-pairs)...")


def make_Dfun(r_data, y_data):
    r_data = np.asarray(r_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    log_r = np.log10(r_data)

    def Dfun(rq):
        rq = np.atleast_1d(np.asarray(rq, dtype=float))
        out = np.zeros_like(rq)
        pos = rq > 0
        if np.any(pos):
            out[pos] = np.interp(np.log10(rq[pos]), log_r, y_data, left=0.0, right=y_data[-1])
        return out

    return Dfun


Dfun = make_Dfun(r, y)
sig_n = float(np.median(vel_err))
print(f"  D(r) model: log-log interpolation of the observed curve, D(0)=0; sig_n = median(vel_err) = {sig_n:.3f} km/s")

# Rebuild the pair list + bin assignment to match r/y/yerr_* exactly. NOTE:
# re-deriving bin edges via spatial_bin_edges(binning="shell") does NOT
# reproduce the same edges generate_vsf_2d used internally -- its r_min step
# samples via the unseeded global np.random.choice (the reproducibility gap
# flagged earlier in this session), so it silently returned a different bin
# count (53 vs. 57) here. Sidestep that entirely: reconstruct edges from the
# log_centers we already have (logc_jk, the actual centers y/yerr_jk/etc are
# defined on) via simple log-midpoints, rather than recomputing from scratch.
_bw = np.diff(logc_jk) / 2
_edges_log = np.empty(len(logc_jk) + 1)
_edges_log[1:-1] = logc_jk[:-1] + _bw
_edges_log[0] = logc_jk[0] - _bw[0]
_edges_log[-1] = logc_jk[-1] + _bw[-1]
bin_edges_pc = 10**_edges_log

from scipy.spatial import cKDTree as _cKDTree
_coords = np.column_stack((x_phys, y_phys))
_tree = _cKDTree(_coords)
_pairs = _tree.query_pairs(r=bin_edges_pc[-1], output_type="ndarray")
_i, _j = _pairs[:, 0], _pairs[:, 1]
_dr2 = np.sum((_coords[_i] - _coords[_j])**2, axis=1)
_bin_idx_full = np.searchsorted(bin_edges_pc**2, _dr2) - 1
_nbins_full = len(bin_edges_pc) - 1
_valid_pairs = (_bin_idx_full >= 0) & (_bin_idx_full < _nbins_full)
i_pairs, j_pairs, bin_idx_pc = _i[_valid_pairs], _j[_valid_pairs], _bin_idx_full[_valid_pairs]

import time as _time
_t0 = _time.time()
n_mc = 100_000
sigma_pc_full, n_eff_pc_full, r_used_full = pair_covariance_vsf_error(
    _coords, i_pairs, j_pairs, bin_idx_pc, Dfun, sig_n=sig_n, n_mc=n_mc, seed=1,
    n_bins=len(bin_edges_pc) - 1,
)
print(f"  {n_mc:,} MC draws x {_nbins_full} bins took {_time.time() - _t0:.1f} s")

yerr_pc = sigma_pc_full[valid]
n_eff_pc = n_eff_pc_full[valid]

# The raw pair-covariance curve hugs the VSF's own wiggle (same issue the
# capped-FKP band had) because it's a direct, bin-by-bin MC estimate -- it
# has no reason to be smooth in r. Fit a single smooth power law through it
# in log-log space (excluding the smallest bins, where N_eff_pc < 2 means
# the MC itself is noise-dominated) to get an FKP-like monotonic "opening"
# envelope, but anchored to the pair-covariance method's amplitude instead
# of an idealized geometric/mode-counting argument.
_fit_mask = np.isfinite(yerr_pc) & (yerr_pc > 0) & (n_eff_pc >= 2)
_slope_pc, _intercept_pc = np.polyfit(np.log10(r[_fit_mask]), np.log10(yerr_pc[_fit_mask]), 1)
yerr_pc_smooth = 10**(_intercept_pc + _slope_pc * np.log10(r))
print(f"  Smooth power-law fit: sigma_pc(r) = 10^{_intercept_pc:.3f} * r^{_slope_pc:.3f}"
      f" (fit to {_fit_mask.sum()}/{len(r)} bins with N_eff_pc >= 2)")

print()
print(f"{'r [pc]':>10} {'VSF':>10} {'err(Np-1)':>10} {'err(boot)':>10} {'err(jack)':>10} {'err(FKP unif)':>14} {'err(FKP Veff)':>14} {'err(pair-cov)':>14} {'N_eff(pc)':>10}")
for ri, yi, es, eb, ej, ef, efv, epc, nepc in zip(
    r, y, yerr_std, yerr_bs, yerr_jk, yerr_fkp, yerr_fkp_veff, yerr_pc, n_eff_pc
):
    print(f"{ri:10.3f} {yi:10.4f} {es:10.4f} {eb:10.4f} {ej:10.4f} {ef:14.4f} {efv:14.4f} {epc:14.4f} {nepc:10.2f}")

fig, ax = plt.subplots(figsize=(8.0, 6.0))

ax.fill_between(
    r, np.maximum(y - yerr_std, 1e-12), y + yerr_std,
    color="0.7", alpha=0.5,
    label=r"error: $\sigma_{\Delta v^p}(r)/\sqrt{N_p(r)-1}$ (independent-pairs assumption)",
)
ax.fill_between(
    r, np.maximum(y - yerr_bs, 1e-12), y + yerr_bs,
    color="#D95F0E", alpha=0.35, label="error: bootstrap (spaxel resampling)",
)
ax.fill_between(
    r, np.maximum(y - yerr_jk, 1e-12), y + yerr_jk,
    color="#2C7FB8", alpha=0.45, label="error: block jackknife",
)
ax.plot(
    r, y + yerr_fkp, "--", color="#238B45", lw=1.6,
    label=r"error: FKP, uniform-density limit (Kaiser 1986)",
)
ax.plot(r, np.maximum(y - yerr_fkp, 1e-12), "--", color="#238B45", lw=1.6)
ax.plot(
    r, y + yerr_fkp_perfect, ":", color="#74C476", lw=1.6,
    label=r"error: FKP, perfect/infinite density ($\bar{n}\to\infty$, no shot term)",
)
ax.plot(r, np.maximum(y - yerr_fkp_perfect, 1e-12), ":", color="#74C476", lw=1.6)
ax.plot(
    r, y + yerr_fkp_veff, "--", color="#B15928", lw=1.8,
    label=r"error: FKP (1994), true $V_{\rm eff}$ (local S/N$^2$ weighting)",
)
ax.plot(r, np.maximum(y - yerr_fkp_veff, 1e-12), "--", color="#B15928", lw=1.8)
ax.plot(
    r, y + yerr_pc_smooth, "--", color="#DD3497", lw=1.8,
    label="error: pair-covariance, smooth power-law fit",
)
ax.plot(r, np.maximum(y - yerr_pc_smooth, 1e-12), "--", color="#DD3497", lw=1.8)
ax.plot(r, y, "o-", color="#08306B", ms=4.5, lw=2.0, label="NII VSF (full FOV)")

ax.set_xscale("log")
ax.set_ylim(0.0, 1.15 * np.max(y + np.maximum.reduce([yerr_std, yerr_jk, yerr_bs, yerr_fkp, yerr_fkp_veff, yerr_pc_smooth])))
ax.set_xlabel(r"$l_{2D} = |r_i - r_j|$ [pc]")
ax.set_ylabel(r"$\bar{S}_1 = \langle |v(r+l)-v(r)| \rangle$")
ax.set_title("Flame nebula, NII: VSF error-bar comparison")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="best")
fig.tight_layout()

out_path = OUT_DIR / "LVM_Flame_nebula_NII_vsf_jackknife_demo.png"
fig.savefig(out_path, dpi=180)
plt.close(fig)
print(f"\nWrote {out_path.relative_to(REPO_ROOT)}")
