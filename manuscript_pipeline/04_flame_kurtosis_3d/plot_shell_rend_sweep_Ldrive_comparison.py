#!/usr/bin/env python3
"""Fitted sqrt(S2) slope vs r_end for one shell size, comparing driving scales.

For each driving scale L_out (default 1.2 and 10.1 R_out, same pixel size), and a fixed
shell width, compute sqrt(S2) (FFT all pairs, 0.05-dex bins from r_min) for several seeds,
fit a straight line from r_min to every r_end, and plot the median +-1 sigma over seeds.
Left: median sqrt(S2) curves (normalised at r_min). Right: slope vs r_end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from mock_shell_segmented_breakpoints import OUT_DIR, ShellProjector, binned_s2, vx_field  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "00_core"))
MIN_PAIRS_INITIAL = 20_000  # same bin cut as plot_shell_projected_vsf2_slope_vs_thickness.load_s2


def initial_method_s2(pv, mask, coords, bin_resolution=20):
    """S2 = c2/cnt from compute_bins + compute_pdf_and_statistics (the original shell pipeline)."""
    from vsf_module import compute_bins, compute_pdf_and_statistics

    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    y, z, v = yy[mask], zz[mask], pv[mask]
    log_c, bin_edges, dv_bins, _ = compute_bins(y, z, v, bin_resolution=bin_resolution)
    with np.errstate(invalid="ignore", divide="ignore"):
        _, c2, _, cnt, _, _ = compute_pdf_and_statistics(
            y, z, v, np.zeros_like(v), log_c, bin_edges, dv_bins, order=1, n_jobs=1)
        s2 = np.where(cnt >= MIN_PAIRS_INITIAL, c2 / np.where(cnt > 0, cnt, 1), np.nan)
    return 10 ** log_c, s2

STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
PIXEL = 2.4 / 159
# (label, box [R_out], n_grid, kmin_factor) -> L_out = box / kmin_factor
CONFIGS = [
    (1.2, 2.4, 160, 2.0),
    (10.1, 671 * PIXEL, 672, 1.0),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--n-seeds", type=int, default=16)
    p.add_argument("--example-seed", type=int, default=7)
    p.add_argument("--r-min", type=float, default=None,
                   help="start of fits [R_out]; default 0.1 (fft) or smallest valid lag (initial)")
    p.add_argument("--method", choices=["fft", "initial"], default="fft",
                   help="fft: exact all-pairs; initial: compute_bins + compute_pdf_and_statistics (c2/cnt)")
    args = p.parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    if args.method == "fft":
        r_min = 0.1 if args.r_min is None else args.r_min
        edges = 10 ** np.arange(np.log10(r_min), np.log10(2.0) + 1e-9, 0.05)

    def slopes_vs_end(r, y):
        ok = np.isfinite(y) & (r >= r_start)
        lr_, y_ = np.log10(r[ok]), y[ok]
        return r[ok][4:], np.array([np.polyfit(lr_[: k + 1], y_[: k + 1], 1)[0] for k in range(4, ok.sum())])

    colors = {1.2: "C0", 10.1: "C3"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for l_out, box, n_grid, kmin in CONFIGS:
        proj = ShellProjector(n_grid, box)
        seeds = [args.example_seed] + [1000 + s for s in range(args.n_seeds)]
        coords = np.linspace(-box / 2, box / 2, n_grid)[proj.sl]
        ys = []
        for s in seeds:
            v = vx_field(n_grid, box, np.random.default_rng(s), kmin)
            pv, mask = proj(v, 1.0 - args.width)
            if args.method == "fft":
                r = np.sqrt(edges[:-1] * edges[1:])
                s2, _ = binned_s2(pv, mask, PIXEL, edges)
            else:
                r, s2 = initial_method_s2(pv, mask, coords)
            ys.append(0.5 * np.log10(s2))
            print(f"  L={l_out} seed {s} done", flush=True)
        ys = np.array(ys)
        if args.method == "fft":
            r_start = r_min
        else:
            r_start = r[np.all(np.isfinite(ys), axis=0)].min() if args.r_min is None else args.r_min
        ex, ens = ys[0], ys[1:]
        re, _ = slopes_vs_end(r, ens[0])
        sl = np.array([slopes_vs_end(r, y)[1] for y in ens])
        c = colors[l_out]
        lab = rf"$L_{{\rm drive}}={l_out:g}\,R_{{\rm out}}$"

        ax = axes[0]
        keep = np.all(np.isfinite(ens), axis=0) & (r >= r_start)
        norm = ens[:, keep] - ens[:, keep][:, :1]
        med = np.median(norm, axis=0)
        lo, hi = np.percentile(norm, [16, 84], axis=0)
        ax.fill_between(r[keep], 10 ** lo, 10 ** hi, color=c, alpha=0.2, lw=0)
        ax.plot(r[keep], 10 ** med, color=c, lw=2, label=lab + f" (median of {args.n_seeds})")

        ax = axes[1]
        med = np.median(sl, axis=0)
        lo, hi = np.percentile(sl, [16, 84], axis=0)
        ax.fill_between(re, lo, hi, color=c, alpha=0.2, lw=0)
        ax.plot(re, med, color=c, lw=2.2, label=lab + r": median $\pm1\sigma$")
        ax.plot(re, slopes_vs_end(r, ex)[1], color=c, lw=1, ls="--", label=lab + f": seed {args.example_seed}")

        print(f"L_drive={l_out}: median slope at r_end=0.2/0.45/1.0/1.5 = "
              + " / ".join(f"{np.interp(np.log10(x), np.log10(re), med):.2f}" for x in (0.2, 0.45, 1.0, 1.5))
              + f";  1sigma at r_end=1: {np.interp(0.0, np.log10(re), (hi - lo) / 2):.2f}")

    ax = axes[0]
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell/R_{\rm out}$")
    ax.set_ylabel(r"$\sqrt{S_2}(\ell)\,/\,\sqrt{S_2}(r_{\min})$")
    ax.set_title(rf"Projected $\sqrt{{S_2}}$, shell width $w={args.width:g}\,R_{{\rm out}}$")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    ax.axhline(1 / 3, color="gray", ls=":", lw=1)
    ax.axhline(5 / 6, color="gray", ls=":", lw=1)
    ax.text(0.15, 5 / 6 - 0.03, "5/6 (fully projected)", fontsize=8, color="gray")
    ax.text(0.15, 1 / 3 + 0.01, "1/3 (3D Kolmogorov)", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r_{\rm end}/R_{\rm out}$")
    ax.set_ylabel(rf"slope of line from $r_{{\min}}={r_start:.3f}$ to $r_{{\rm end}}$")
    meth = "FFT all pairs" if args.method == "fft" else "initial method: c2/cnt, compute_pdf_and_statistics"
    ax.set_title(r"Fitted $\sqrt{S_2}$ slope vs end of fit" + f"\n({meth})", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")

    suffix = "" if args.method == "fft" else "_initialmethod"
    out = OUT_DIR / ("shell_rend_sweep_Ldrive_comparison_w" + f"{args.width:.3f}".replace(".", "p") + f"{suffix}.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
