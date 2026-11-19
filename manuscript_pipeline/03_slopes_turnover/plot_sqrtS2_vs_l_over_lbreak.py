#!/usr/bin/env python3
"""sqrt(S2) of all regions on a common l / l_break axis, with l_break inferred by the segmented fit.

l_break is re-inferred here from NPZ/VSFORD_<region>_<LINE>.npz (full FOV, S/N > 20) with the
continuous segmented power-law fit of fit_segmented_vsf_breakpoints.py applied to log10 sqrt(S2)
(max 3 segments, >= 5 points/segment, Delta-BIC >= 6, bins with >= 1000 pairs). The first break
is used. Regions where no break is selected are drawn dashed, scaled by their largest lag
(a lower limit on l_break).

Panels: (a) sqrt(S2)/sqrt(S2)(l_break) vs l/l_break, (b) local slope vs l/l_break,
(c) same as (a) but scaled by R_HII, for comparison of how well each scale collapses the curves.
Also prints fixed-window slopes in l/l_break with block-jackknife errors, and a collapse metric.

Usage:
    python manuscript_pipeline/03_slopes_turnover/plot_sqrtS2_vs_l_over_lbreak.py [--line SII]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_segmented_vsf_breakpoints import best_fit_by_segment_count, select_fit  # noqa: E402

REPO_ROOT = HERE.parents[1]
RADIUS_CSV = REPO_ROOT / "manuscript_notes" / "adopted_hii_region_radii.csv"
NPZ_DIR = REPO_ROOT / "NPZ"
OUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "03_slopes_turnover"
STYLE = REPO_ROOT / "custom_plot.mplstyle"
MIN_PAIRS = 1000
MIN_DELTA_BIC = 6.0
MAX_SEGMENTS = 3
MIN_POINTS_PER_SEGMENT = 5
HALF_WINDOW = 2
WINDOWS = [(0.1, 0.5), (0.2, 1.0)]
COLLAPSE_RANGE = (0.1, 1.0)  # in units of the scale, for the collapse metric


def local_slope(x, y, yerr):
    out = np.full_like(x, np.nan)
    for i in range(x.size):
        sl = slice(max(i - HALF_WINDOW, 0), i + HALF_WINDOW + 1)
        if x[sl].size >= 2 * HALF_WINDOW + 1:
            out[i] = np.polyfit(x[sl], y[sl], 1, w=1 / yerr[sl])[0]
    return out


def jackknife_window_slope(lag, s2_reps, s2, sel):
    x = np.log10(lag[sel])
    fit = lambda c: float(np.polyfit(x, 0.5 * np.log10(c[sel]), 1)[0])  # noqa: E731
    reps = np.array([fit(r) for r in s2_reps if np.all(np.isfinite(r[sel]) & (r[sel] > 0))])
    n = reps.size
    err = float(np.sqrt((n - 1) / n * np.sum((reps - reps.mean()) ** 2))) if n >= 4 else np.nan
    return fit(s2), err


def collapse_rms(curves, lo, hi, n=30):
    """RMS (dex) across regions of the normalised log curves, averaged over a common grid."""
    grid = np.linspace(np.log10(lo), np.log10(hi), n)
    stack = []
    for lx, ly in curves:
        v = np.interp(grid, lx, ly, left=np.nan, right=np.nan)
        stack.append(v)
    stack = np.array(stack)
    good = np.sum(np.isfinite(stack), axis=0) >= 3
    return float(np.sqrt(np.nanmean(np.nanvar(stack[:, good], axis=0)))), int(np.min(np.sum(np.isfinite(stack[:, good]), axis=0)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--line", default="SII")
    args = p.parse_args()
    line = args.line.upper()
    if STYLE.exists():
        plt.style.use(str(STYLE))
    with RADIUS_CSV.open() as f:
        radii = {row["object"]: float(row["radius_pc"]) for row in csv.DictReader(f)}

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, 10))
    curves_break, curves_rhii, rows = [], [], []
    print(f"[{line}] segmented fit on sqrt(S2): dBIC >= {MIN_DELTA_BIC:g}, <= {MAX_SEGMENTS} segments")
    for k, path in enumerate(sorted(NPZ_DIR.glob(f"VSFORD_*_{line}.npz"))):
        region = path.stem[len("VSFORD_"):-len(f"_{line}")]
        d = np.load(path)
        i2 = list(d["orders"]).index(2)
        s2, s2_err = d["S"][i2], d["S_err"][i2]
        reps = d["S_jackknife_replicates"][:, i2]
        ok = (d["counts"] >= MIN_PAIRS) & np.isfinite(s2) & (s2 > 0) & (s2_err > 0)
        if ok.sum() < 2 * MIN_POINTS_PER_SEGMENT:
            print(f"  {region:15s} skipped: {ok.sum()} bins with >= {MIN_PAIRS} pairs")
            continue
        r_pc = 10 ** d["log_centers"]
        x = np.log10(r_pc[ok])
        y = 0.5 * np.log10(s2[ok])
        ye = 0.5 * s2_err[ok] / (s2[ok] * np.log(10))
        fit = select_fit(best_fit_by_segment_count(x, y, ye, MAX_SEGMENTS, MIN_POINTS_PER_SEGMENT), MIN_DELTA_BIC)
        has_break = fit.n_segments > 1
        lbreak = 10 ** fit.break_log_r[0] if has_break else r_pc[ok].max()
        m_seg, m_seg_err = fit.slopes[0], fit.slope_errs[0]

        lag = r_pc / lbreak
        lx, ly = np.log10(lag[ok]), y
        ref = np.interp(0.0, lx, ly)
        c = colors[k % 10]
        ls = "-" if has_break else "--"
        label = region.replace("_", " ") + ("" if has_break else r" (no break, $\ell_{\max}$)")
        axes[0].plot(lag[ok], 10 ** (ly - ref), ls, color=c, lw=1.8, marker="o", ms=3, label=label)
        axes[1].plot(lag[ok], local_slope(lx, ly, ye), ls, color=c, lw=1.8)

        lr = np.log10(r_pc[ok] / radii[region]) if region in radii else None
        if lr is not None:
            ref_r = np.interp(0.0, lr, ly, left=np.nan, right=np.nan)
            if np.isfinite(ref_r):
                axes[2].plot(10**lr, 10 ** (ly - ref_r), ls, color=c, lw=1.8, marker="o", ms=3)
                curves_rhii.append((lr, ly - ref_r))
        if has_break:
            curves_break.append((lx, ly - ref))

        win = []
        for lo, hi in WINDOWS:
            sel = ok & (lag >= lo) & (lag <= hi)
            win.append(jackknife_window_slope(lag, reps, s2, sel) if sel.sum() >= MIN_POINTS_PER_SEGMENT
                       else (np.nan, np.nan))
        rows.append((region, has_break, lbreak, lbreak / radii.get(region, np.nan), m_seg, m_seg_err, win))

    hdr = "  ".join(f"m({lo:g}-{hi:g} lb)" for lo, hi in WINDOWS)
    print(f"  {'region':15s} {'l_break[pc]':>11s} {'/R_HII':>7s} {'m_seg':>13s}   {hdr}")
    for region, hb, lb, lbr, m, me, win in rows:
        flag = " " if hb else ">"
        ws = "  ".join(f"{a:5.3f}±{b:5.3f}" for a, b in win)
        print(f"  {region:15s} {flag}{lb:10.2f} {flag}{lbr:6.2f} {m:6.3f}±{me:5.3f}   {ws}")
    for j, (lo, hi) in enumerate(WINDOWS):
        ms = np.array([r[6][j][0] for r in rows if r[1]])
        es = np.array([r[6][j][1] for r in rows if r[1]])
        g = np.isfinite(ms) & np.isfinite(es)
        ms, es = ms[g], es[g]
        w = 1 / es**2
        mu = np.sum(w * ms) / w.sum()
        print(f"  window {lo:g}-{hi:g} l_break (regions with a break, n={ms.size}): mean {mu:.3f} ± "
              f"{1 / np.sqrt(w.sum()):.3f}, scatter {ms.std(ddof=1):.3f}, median err {np.median(es):.3f}, "
              f"chi2 = {np.sum(((ms - mu) / es) ** 2):.1f}/{ms.size - 1}")
    rb, nb = collapse_rms(curves_break, *COLLAPSE_RANGE)
    rr, nr = collapse_rms([c for c, r in zip(curves_rhii, rows) if r[1]], *COLLAPSE_RANGE)
    print(f"  collapse rms over {COLLAPSE_RANGE[0]:g}-{COLLAPSE_RANGE[1]:g} x scale (regions with a break): "
          f"l_break {rb:.3f} dex (>= {nb} curves/pt), R_HII {rr:.3f} dex (>= {nr} curves/pt)")

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(1, color="0.3", ls=":", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\ell / \ell_{\rm break}$")
    axes[0].set_ylabel(r"$\sqrt{S_2}\,/\,\sqrt{S_2}(\ell_{\rm break})$")
    axes[0].set_title(rf"[{line}] normalised at the inferred $\ell_{{\rm break}}$")
    axes[0].legend(fontsize=10)
    axes[1].axhline(1 / 3, color="0.4", ls="--", lw=1)
    axes[1].axhline(0, color="0.4", lw=0.8)
    axes[1].set_ylim(-1.0, 1.0)
    axes[1].set_xlabel(r"$\ell / \ell_{\rm break}$")
    axes[1].set_ylabel(r"local slope $d\log\sqrt{S_2}/d\log\ell$")
    axes[1].set_title(f"running {2 * HALF_WINDOW + 1}-bin slope")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\ell / R_{\rm HII}$")
    axes[2].set_ylabel(r"$\sqrt{S_2}\,/\,\sqrt{S_2}(R_{\rm HII})$")
    axes[2].set_title(r"for comparison: normalised at $R_{\rm HII}$")

    out = OUT_DIR / f"{line.lower()}_sqrtS2_vs_l_over_lbreak_allregions.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
