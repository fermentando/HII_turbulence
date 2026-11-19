#!/usr/bin/env python3
"""Illustrate how the slope-vs-r_end curve is built for one synthetic shell.

Left:  sqrt(S2) of one shell (seed 7, FFT all-pairs, 0.05-dex bins from r_min) with
       straight-line fits from r_min to several r_end (one colour each), plus the
       continuous segmented fit (Delta-BIC selection, errors = seed-to-seed scatter).
Right: fitted slope vs r_end for the same shell (black curve); the coloured markers are
       the fits drawn on the left; the star is the segmented fit (first slope, first break).
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
sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "03_slopes_turnover"))

from fit_segmented_vsf_breakpoints import (  # noqa: E402
    best_fit_by_segment_count, evaluate_model, select_fit,
)
from mock_shell_segmented_breakpoints import (  # noqa: E402
    BOX_SIZE, N_GRID, OUT_DIR, ShellProjector, binned_s2, vx_field,
)

STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"


def log_sqrt_s2(seed, width, edges, pixel, box, n_grid, kmin_factor, projector):
    vx = vx_field(n_grid, box, np.random.default_rng(seed), kmin_factor)
    proj, mask = projector(vx, 1.0 - width)
    s2, _ = binned_s2(proj, mask, pixel, edges)
    return 0.5 * np.log10(s2)


def pipeline_log_sqrt_s2(seed, width, r_min, box, n_grid, kmin_factor, projector, scratch):
    """sqrt(S2) exactly as the observational pipeline computes it (VSF2D.generate_vsf_2d, bootstrap errors)."""
    sys.path.insert(0, str(REPO_ROOT / "manuscript_pipeline" / "00_core"))
    from vsf_module import VSF2D

    vx = vx_field(n_grid, box, np.random.default_rng(seed), kmin_factor)
    proj, mask = projector(vx, 1.0 - width)
    c = np.linspace(-box / 2, box / 2, n_grid)[projector.sl]
    yy, zz = np.meshgrid(c, c, indexing="ij")
    vsf, err, log_c, _ = VSF2D(yy[mask], zz[mask], proj[mask], dist=None, coords="physical").generate_vsf_2d(
        order=2, outname=str(scratch), n_jobs=4, min_pairs=3, bin_resolution=20,
        error_mode="bootstrap", min_separation=r_min)
    y = 0.5 * np.log10(vsf)
    y_err = 0.5 * err / (vsf * np.log(10.0))
    return 10 ** log_c, y, y_err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=float, default=0.1158)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-error-seeds", type=int, default=16)
    p.add_argument("--r-min", type=float, default=0.1)
    p.add_argument("--min-delta-bic", type=float, default=6.0)
    p.add_argument("--r-ends", type=float, nargs="+", default=[0.2, 0.3, 0.45, 0.7, 1.0, 1.5])
    p.add_argument("--box-size", type=float, default=BOX_SIZE, help="box side [R_out]")
    p.add_argument("--n-grid", type=int, default=N_GRID)
    p.add_argument("--kmin-factor", type=float, default=2.0, help="L_out = box / kmin_factor")
    p.add_argument("--tag", default="")
    p.add_argument("--pipeline-vsf", action="store_true",
                   help="compute sqrt(S2) and its errors with the observational VSF2D pipeline (bootstrap)")
    args = p.parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    pixel = args.box_size / (args.n_grid - 1)
    projector = ShellProjector(args.n_grid, args.box_size)
    l_out = args.box_size / args.kmin_factor
    gen = (args.box_size, args.n_grid, args.kmin_factor, projector)
    if args.pipeline_vsf:
        scratch = OUT_DIR / "_tmp_pipeline_vsf"
        r, y, y_err = pipeline_log_sqrt_s2(args.seed, args.width, args.r_min, *gen, scratch)
        Path(str(scratch) + "_order2.npz").unlink(missing_ok=True)
        err_label = "pipeline VSF2D, bootstrap errors"
    else:
        edges = 10 ** np.arange(np.log10(args.r_min), np.log10(2.0) + 1e-9, 0.05)
        r = np.sqrt(edges[:-1] * edges[1:])
        y = log_sqrt_s2(args.seed, args.width, edges, pixel, *gen)
        others = np.array([log_sqrt_s2(1000 + s, args.width, edges, pixel, *gen)
                           for s in range(args.n_error_seeds)])
        y_err = np.nanstd(others, axis=0, ddof=1)
        err_label = f"FFT all pairs, errors = scatter of {args.n_error_seeds} seeds"
    lr = np.log10(r)
    ok = np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)

    def line_to(r_end):
        sel = ok & (r <= r_end)
        return np.polyfit(lr[sel], y[sel], 1), sel

    fits = best_fit_by_segment_count(lr[ok], y[ok], y_err[ok], 3, 5)
    seg = select_fit(fits, args.min_delta_bic)

    cols = plt.colormaps["plasma"](np.linspace(0.05, 0.85, len(args.r_ends)))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.errorbar(r[ok], 10 ** y[ok], yerr=[10 ** y[ok] - 10 ** (y[ok] - y_err[ok]),
                10 ** (y[ok] + y_err[ok]) - 10 ** y[ok]], fmt="o", color="k", ms=4,
                capsize=0, lw=0.8, zorder=3, label=rf"$\sqrt{{S_2}}$, $w={args.width:.3f}$, $L_{{\rm out}}={l_out:.1f}$ (seed {args.seed})")
    for r_end, c in zip(args.r_ends, cols):
        coef, sel = line_to(r_end)
        rr = r[sel]
        ax.plot(rr, 10 ** np.polyval(coef, np.log10(rr)), "-", color=c, lw=2,
                label=rf"$r_{{\rm end}}={r_end:g}$: slope {coef[0]:.2f}")
        ax.axvline(r_end, color=c, ls=":", lw=1)
    xs = np.linspace(lr[ok].min(), lr[ok].max(), 300)
    ax.plot(10 ** xs, 10 ** evaluate_model(seg, xs), "--", color="tab:green", lw=2.2, zorder=4,
            label=f"segmented ({seg.n_segments} seg.): first slope {seg.slopes[0]:.2f}")
    for b in seg.break_log_r:
        ax.axvline(10 ** b, color="tab:green", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell/R_{\rm out}$")
    ax.set_ylabel(r"$\sqrt{S_2}$")
    ax.set_title(rf"Lines from $r_{{\min}}={args.r_min:g}$ to each $r_{{\rm end}}$" + f"\n({err_label})", fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    ends = r[ok][r[ok] >= r[ok][4]]
    ax.plot(ends, [line_to(e)[0][0] for e in ends], "k-", lw=1.5, label="slope vs $r_{\\rm end}$ (every bin)")
    for r_end, c in zip(args.r_ends, cols):
        ax.plot(r_end, line_to(r_end)[0][0], "o", color=c, mec="k", ms=10, zorder=3)
    if seg.n_segments > 1:
        ax.plot(10 ** seg.break_log_r[0], seg.slopes[0], "*", color="tab:green", mec="k", ms=18, zorder=4,
                label=rf"segmented: $(\ell_{{\rm break}}, m)$")
    else:
        ax.plot(r[ok].max(), seg.slopes[0], ">", color="tab:green", mec="k", ms=14, zorder=4,
                label=r"segmented: no break (1 segment, at end of range)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r_{\rm end}/R_{\rm out}$  (or $\ell_{\rm break}$)")
    ax.set_ylabel(r"fitted slope of $\sqrt{S_2}$")
    ax.set_title("Each coloured line on the left = one point here")
    ax.legend(fontsize=8)

    tag = f"_{args.tag}" if args.tag else ""
    out = OUT_DIR / ("shell_rend_sweep_example_w" + f"{args.width:.3f}".replace(".", "p") + f"{tag}.png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    print(f"segmented: {seg.n_segments} segments, breaks {[round(10**b, 3) for b in seg.break_log_r]}, slopes {np.round(seg.slopes, 3)}")


if __name__ == "__main__":
    main()
