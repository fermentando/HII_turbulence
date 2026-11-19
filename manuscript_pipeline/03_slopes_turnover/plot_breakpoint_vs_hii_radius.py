#!/usr/bin/env python3
"""l_break [pc] vs R_HII [pc] per tracer, with power-law fits l_break ~ R_HII^alpha.

Rebuild of ShareSpace/LVM/breakpoint_vs_hii_radius_sqrtS2_dBIC6_snr20.png with the breaks of
plot_vsf_fits_and_slope_vs_radius.py (VSFORD NPZs, S/N > 20, full FOV, segmented fit on sqrt(S2),
Delta-BIC >= 6, same forced fits).

Break errors: block jackknife. Each replicate is refitted with the same number of segments, letting
the break positions move; the jackknife spread of log10(l_break) gives the (asymmetric) error bar.
Filled symbols: break selected. Open symbols: no break, plotted at the largest fitted lag (a lower
limit); they are excluded from the fits and from the Spearman correlations.

alpha: weighted least squares in log-log on the break-selected points (weights 1/sigma^2 from the jackknife\nerrors of log10 l_break); alpha error from the unscaled fit covariance.

Usage:
    python manuscript_pipeline/03_slopes_turnover/plot_breakpoint_vs_hii_radius.py [--radii lvm-for-wiki]
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
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "00_core"))
from fit_segmented_vsf_breakpoints import best_fit_by_segment_count  # noqa: E402
import plot_vsf_fits_and_slope_vs_radius as comp  # noqa: E402

REPO_ROOT = HERE.parents[1]
OUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "03_slopes_turnover"
STYLE = REPO_ROOT / "custom_plot.mplstyle"
LINE_COLOR = {"NII": "#264653", "SII": "#2A9D8F", "SIII": "#E9C46A"}
SHORT = {"Flame_nebula": "Flame", "Rosette_nebula": "Rosette"}
N_BOOT = 5000


def load_radii(mode: str) -> dict[str, float]:
    with comp.RADIUS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    radii = {row["object"]: float(row["radius_pc"]) for row in rows}
    if mode != "adopted":
        from vsf_module import get_hii_radius_pc
        for row in rows:
            if mode == "lvm" or "wikipedia" in row["source_url"].lower():
                radii[row["object"]] = float(get_hii_radius_pc(f"LVM_{row['object']}.fits"))
    return radii


def jackknife_break(region: str, line: str, n_segments: int) -> tuple[float, float]:
    """Jackknife error on log10(first break [pc]), refitting each replicate with n_segments fixed."""
    d = np.load(comp.NPZ_DIR / f"VSFORD_{region}_{line}.npz")
    i2 = list(d["orders"]).index(2)
    s2, s2_err, reps = d["S"][i2], d["S_err"][i2], d["S_jackknife_replicates"][:, i2]
    ok = (d["counts"] >= comp.MIN_PAIRS) & np.isfinite(s2) & (s2 > 0) & (s2_err > 0)
    x = d["log_centers"][ok]
    ye = 0.5 * s2_err[ok] / (s2[ok] * np.log(10))
    lb = []
    for rep in reps:
        c = rep[ok]
        if not np.all(np.isfinite(c) & (c > 0)):
            continue
        fits = best_fit_by_segment_count(x, 0.5 * np.log10(c), ye, n_segments, comp.MIN_POINTS_PER_SEGMENT)
        if n_segments in fits:
            lb.append(fits[n_segments].break_log_r[0])
    lb = np.asarray(lb)
    n = lb.size
    err = float(np.sqrt((n - 1) / n * np.sum((lb - lb.mean()) ** 2)))
    # breaks sit between bin centres, so they are only known to ~half a bin: floor the error there
    half_bin = 0.5 * float(np.median(np.diff(d["log_centers"])))
    return max(err, half_bin), n


def fit_alpha(r: np.ndarray, lb: np.ndarray, log_err: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """Weighted least squares of log10(l_break) = a + alpha log10(R_HII), weights 1/sigma^2 with sigma the
    jackknife error of log10(l_break). Returns (coef [alpha, a], alpha error from the unscaled covariance,
    chi2, dof)."""
    x, y = np.log10(r), np.log10(lb)
    coef, cov = np.polyfit(x, y, 1, w=1 / log_err, cov="unscaled")
    chi2 = float(np.sum(((y - np.polyval(coef, x)) / log_err) ** 2))
    return coef, float(np.sqrt(cov[0, 0])), chi2, x.size - 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radii", choices=("adopted", "lvm-for-wiki", "lvm"), default="lvm-for-wiki")
    p.add_argument("--exclude", default="Rosette_nebula",
                   help="comma-separated regions left out of the figure and fits (default: Rosette_nebula, whose "
                        "marginal [NII]/[SII] breaks have jackknife errors spanning > 1 dex); '' keeps all")
    args = p.parse_args()
    if STYLE.exists():
        plt.style.use(str(STYLE))
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"],
                         "axes.grid": True, "grid.alpha": 0.4})
    radii = load_radii(args.radii)
    excluded = [r for r in args.exclude.split(",") if r]
    regions = [r for r in comp.REGIONS if r not in excluded]
    rng = np.random.default_rng(1)

    rows = []
    for line in comp.LINES:
        for region in regions:
            res = comp.fit_region(region, line, radii[region])
            if res is None:
                continue
            if res["has_break"]:
                err, n = jackknife_break(region, line, res["fit"].n_segments)
                lb = res["breaks_pc"][0]
            else:
                err, n, lb = np.nan, 0, res["r"].max()
            rows.append({"line": line, "region": region, "r_hii": radii[region], "lbreak": lb,
                         "log_err": err, "has_break": res["has_break"]})

    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    labels = []
    all_r, all_lb, all_e = [], [], []
    for line in comp.LINES:
        c = LINE_COLOR[line]
        sel = [r for r in rows if r["line"] == line]
        for r in sel:
            ms = 13 if comp.MARKER[r["region"]] == "*" else 10
            if r["has_break"]:
                lo = r["lbreak"] * (1 - 10 ** -r["log_err"])
                hi = r["lbreak"] * (10 ** r["log_err"] - 1)
                ax.errorbar(r["r_hii"], r["lbreak"], yerr=[[lo], [hi]], fmt="none", ecolor=c, elinewidth=1.3,
                            capsize=2.5, zorder=2)
                ax.plot(r["r_hii"], r["lbreak"], comp.MARKER[r["region"]], ms=ms, mfc=c, mec="k", mew=0.8, zorder=3)
            else:
                ax.plot(r["r_hii"], r["lbreak"], comp.MARKER[r["region"]], ms=ms, mfc="white", mec=c, mew=1.8,
                        zorder=3)
        brk = [r for r in sel if r["has_break"]]
        rr = np.array([r["r_hii"] for r in brk])
        lbb = np.array([r["lbreak"] for r in brk])
        lee = np.array([r["log_err"] for r in brk])
        all_e.append(lee)
        all_r.append(rr)
        all_lb.append(lbb)
        coef, alpha_err, chi2, dof = fit_alpha(rr, lbb, lee)
        alpha = coef[0]
        rho, pval = spearmanr(rr, lbb)
        grid = np.logspace(np.log10(rr.min()), np.log10(rr.max()), 50)
        ax.plot(grid, 10 ** np.polyval(coef, np.log10(grid)), color=c, lw=2.6, alpha=0.5, zorder=1)
        labels.append((Line2D([], [], marker="o", ls="", mfc=c, mec="k", ms=10),
                       rf"{comp.LINE_LABEL[line]}: $\alpha={alpha:.2f}\pm{alpha_err:.2f}$"))
        print(f"{line:4s} chi2/dof = {chi2:.1f}/{dof}")
        print(f"{line:4s} n={rr.size}: alpha={alpha:.2f}±{alpha_err:.2f}, rho_s={rho:+.2f} (p={pval:.3g})")
    ra, la, ea = np.concatenate(all_r), np.concatenate(all_lb), np.concatenate(all_e)
    coef, alpha_err, chi2, dof = fit_alpha(ra, la, ea)
    alpha = coef[0]
    print(f"all  chi2/dof = {chi2:.1f}/{dof}")
    rho, pval = spearmanr(ra, la)
    labels.append((Line2D([], [], ls=""), rf"all: $\alpha={alpha:.2f}\pm{alpha_err:.2f}$"))
    labels.append((Line2D([], [], marker="o", ls="", mfc="white", mec="0.4", mew=1.8, ms=10), r"no break"))
    print(f"all  n={ra.size}: alpha={alpha:.2f}±{alpha_err:.2f}, rho_s={rho:+.2f} (p={pval:.3g})")

    lim = (0.8, 120)
    ax.plot(lim, lim, ":", color="0.35", lw=1.4, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(0.8, 1500)
    for axis, ticks in ((ax.xaxis, [1, 2, 5, 10, 20, 50, 100]), (ax.yaxis, [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])):
        comp.plain_log_axis(axis, ticks)
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r"$R_{\rm HII}$ [pc]", fontsize=22)
    ax.set_ylabel(r"$\ell_{\rm break}$ [pc]", fontsize=22)
    ax.legend([h for h, _ in labels], [t for _, t in labels], loc="upper left", fontsize=10.5, framealpha=0.9,
              handletextpad=0.4, labelspacing=0.3)
    # 1:1 label just above the dotted line, rotated to the line's on-screen angle
    p0, p1 = ax.transData.transform([(10, 10), (100, 100)])
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(r"$\ell_{\rm break}=R_{\rm HII}$", xy=(40, 40), xytext=(0, 5), textcoords="offset points",
                rotation=angle, rotation_mode="anchor", ha="center", va="bottom", color="0.35", fontsize=12)
    region_handles = [Line2D([], [], marker=comp.MARKER[r], ls="", mfc="0.75", mec="k",
                             ms=9 if comp.MARKER[r] != "*" else 12, label=SHORT.get(r, r)) for r in regions]
    first = ax.get_legend()
    ax.legend(handles=region_handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=11,
              ncol=(len(region_handles) + 1) // 2, frameon=False, handletextpad=0.3, columnspacing=1.0,
              labelspacing=0.3, borderaxespad=0.2)
    ax.add_artist(first)

    suffix = {"adopted": "_radiiWISE", "lvm-for-wiki": "_radiiWISE_LVMforWiki", "lvm": "_radiiLVM"}[args.radii]
    if excluded:
        suffix += "_no" + "_".join(r.split("_")[0] for r in excluded)
    out = OUT_DIR / f"breakpoint_vs_hii_radius_sqrtS2_dBIC6_snr20_jackknife{suffix}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    out_csv = out.with_suffix(".csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out.relative_to(REPO_ROOT)}\nWrote {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
