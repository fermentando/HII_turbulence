"""Kurtosis K(l) vs l/R_HII for all regions, all tracers in one panel ([NII], [SII], [SIII]).

Curves use the current kurtosis_plot.py pipeline (pair estimator K = <dv^4>/<dv^2>^2; flux S/N > 20,
tile-overlap fibres removed, 10-sigma velocity clip, global velocity plane subtracted, catalogue
distance; shell binning at 20/dex above a 37 arcsec floor; bins with >= 4000 counts). They are
computed once per region/line and cached in
Figures/manuscript/04_flame_kurtosis_3d/kurtosis_curves_all_regions.npz.

Figure: Figures/manuscript/04_flame_kurtosis_3d/kurtosis_vs_l_over_rhii_by_tracer.png
All curves faint, y axis capped at K = 10; HIGHLIGHT regions drawn opaque and thicker, distinguished
by line style. Tracer colours are vsf_module.LINE_COLORS, as in the 08 shell-inference figures.

Run from the repo root:
    python manuscript_pipeline/04_flame_kurtosis_3d/plot_kurtosis_vs_l_over_rhii_by_tracer.py [--recompute]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))

OUT_DIR = Path("Figures/manuscript/04_flame_kurtosis_3d")
CACHE = OUT_DIR / "kurtosis_curves_all_regions.npz"
REGIONS = ("BFS65", "Flame_nebula", "Gum31", "Gum38a", "RCW52", "RCW60", "RCW79", "RCW87", "Rosette_nebula",
           "Sh2-311")  # Orion excluded: its LVM field spans ~27 R_HII
LINES = ("NII", "SII", "SIII")
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402  (tracer colours of the 08 shell-inference figures)
# Highlighted: the most shell-like curves across all three tracers (K > 3 at small l, smooth, monotonic
# decline back to K ~ 3 within ~0.07-1.4 R_HII): RCW60, Sh2-311, RCW79.
HIGHLIGHT = ("RCW60", "Sh2-311", "RCW79")
HIGHLIGHT_STYLES = ("-", "--", ":")
K_CAP = 10.0
MIN_COUNTS, MIN_SEPARATION_ARCSEC, SNR_MIN = 4000, 37.0, 20.0


def kurtosis_curve(job):
    """(region, line) -> (l/R_HII, K, counts) with the kurtosis_plot.py pipeline; None if too few bins."""
    from vsf_module import (VSF2D, compute_bins, compute_pdf_and_statistics, get_hii_radius_pc,
                            load_line_data)
    region, line = job
    fits_file = f"LVM_{region}.fits"
    d = load_line_data(fits_file, line.lower(), snr_min=SNR_MIN, subtract_gradient=True, verbose=False)
    if len(d["vel"]) < 50:
        return job, None
    v2 = VSF2D(d["ra"], d["dec"], d["vel"], dist=d["dist"], vel_err=d["vel_err"])
    x, y = v2.x_physical, v2.y_physical
    floor = d["dist"] * np.deg2rad(MIN_SEPARATION_ARCSEC / 3600.0)
    log_c, edges, dv_bins, _ = compute_bins(x, y, d["vel"], bin_resolution=20, binning="shell", min_separation=floor)
    with np.errstate(invalid="ignore", divide="ignore"):
        _pdf, _c2, _c4, cnt, _dvc, kk = compute_pdf_and_statistics(x, y, d["vel"], d["vel_err"], log_c, edges,
                                                                   dv_bins, order=1, n_jobs=4)
    use = cnt >= MIN_COUNTS
    if use.sum() < 3:
        return job, None
    r_hii = get_hii_radius_pc(fits_file)
    return job, (10 ** log_c[use] / r_hii, np.asarray(kk, float)[use], np.asarray(cnt, float)[use])


def load_or_compute(recompute: bool):
    if CACHE.exists() and not recompute:
        with np.load(CACHE, allow_pickle=True) as c:
            return c["curves"].item()
    jobs = [(r, l) for r in REGIONS for l in LINES]
    curves = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for (region, line), res in pool.map(kurtosis_curve, jobs):
            curves[(region, line)] = res
            print(f"  {region} {line}: {'no curve' if res is None else f'{len(res[0])} bins'}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, curves=np.array(curves, dtype=object))
    return curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recompute", action="store_true", help="Recompute the curves instead of using the cache.")
    args = parser.parse_args()
    curves = load_or_compute(args.recompute)

    plt.style.use("./custom_plot.mplstyle")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for line in LINES:
        for region in REGIONS:
            res = curves.get((region, line))
            if res is None or region in HIGHLIGHT:
                continue
            ax.plot(res[0], res[1], "-", color=LINE_COLORS[line], lw=1.0, alpha=0.25)
    for line in LINES:
        for region, ls in zip(HIGHLIGHT, HIGHLIGHT_STYLES):
            res = curves.get((region, line))
            if res is not None:
                ax.plot(res[0], res[1], ls, color=LINE_COLORS[line], lw=2.2, alpha=1.0, zorder=3)
    ax.axhline(3.0, color="0.3", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell/r_{\rm HII}$")
    ax.set_ylabel(r"$K(\ell)$")
    ax.set_ylim(top=K_CAP)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    handles = [plt.Line2D([], [], color=LINE_COLORS[l], lw=2.2, label=LINE_LABELS[l]) for l in LINES]
    handles += [plt.Line2D([], [], color="0.2", ls=ls, lw=2.2, label=r.replace("_", " "))
                for r, ls in zip(HIGHLIGHT, HIGHLIGHT_STYLES)]
    ax.legend(handles=handles, fontsize=10, ncol=2, loc="upper right")
    out = OUT_DIR / "kurtosis_vs_l_over_rhii_by_tracer.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
