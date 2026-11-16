"""(Delta, lambda) maps from the analytic pair-kurtosis formula, fitted to the observed K(l) on a grid.

Only the parameters of the formula (manuscript_notes/kurtosis_kolmogorov_lambda_derivation.md):
    K(l; Delta, lam, L_out) = [3<D^2> + 6 lam^2 <D dmu^2> + lam^4 <dmu^4>] / [<D> + lam^2 <dmu^2>]^2
    Delta = Delta r_shell / r_HII,  lam = v_bulk / v_turb  (free, on a grid)
    L_out = turbulence outer scale = FOV size of the region (equal-area diameter of the convex hull of the
            fibres used), in units of r_HII (fixed per region/line)
    l_in  = 0.005 r_HII (fixed; irrelevant above the 37" floor)
No noise, mixing or r_HII-scale terms. Tables: kurtosis_formula_tables.py.

Data: K(l) per l/r_HII bin (6 per dex from the 37" floor to 0.8 r_HII), from the curves measured by
fit_kurtosis_formula.py measure (data/<region>_<LINE>_formula_data.npz).
Covariance: block-jackknife covariance of K, shrunk toward its diagonal (Schafer & Strimmer target D),
+ realisation scatter of the ensemble-mean formula (8-seed table; correlation exp(-|dlog10 l|)) +
model error SIG_MOD_K on the diagonal. Because the realisation term depends on (Delta, lam), the map is
    -2 ln L = r^T C^-1 r + ln det C,   Delta(-2 ln L) relative to its minimum on the grid.
Contours: Delta(-2 ln L) = 2.30 (1 sigma, filled) and 6.18 (2 sigma), as in shell_inference_chi2_maps.png.

Outputs (Figures/manuscript/08_shell_inference/formula_grid/):
  formula_grid_maps.png            per region: 1/2-sigma regions and best fit per tracer
  formula_grid_curves.png          observed K(l) with the best-fit formula
  formula_grid_summary.csv         best fit, profile 1-sigma ranges, chi2, L_out used
  <region>_<LINE>_map.npz          Delta(-2 ln L) on the grid

--radius flux: l and L_out are divided by s = R_shell / R_HII(catalogue) from the surface-brightness fit
(fit_shell_from_flux.py, same region and line), when s is well determined (1-sigma range within
S_MAX_RATIO and the FOV reaches beyond the fitted edge); otherwise the catalogue radius is kept (flagged).
Bins with l/s > L_MAX_MODEL are dropped (few model pairs there). Outputs go to formula_grid_fluxradius/.

Run from the repo root (after fit_kurtosis_formula.py measure [and fit_shell_from_flux.py]):
    python manuscript_pipeline/08_shell_inference/fit_kurtosis_formula_grid.py [--radius catalogue|flux]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from fit_kurtosis_formula import DATA_DIR, MIN_PAIRS, RHO_DEX, SIG_MOD_K, all_targets, shrink_to_diagonal  # noqa: E402
from kurtosis_formula_tables import FormulaModel  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

OUT_DIR = Path("Figures/manuscript/08_shell_inference/formula_grid")
FLUX_DIR = Path("Figures/manuscript/08_shell_inference/flux_profile")
S_MAX_RATIO, L_MAX_MODEL = 1.3, 1.6
D_GRID = np.geomspace(0.005, 1.0, 120)
LAM_GRID = np.geomspace(0.03, 300.0, 140)
LINES = ("NII", "SII", "SIII")
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}
DD, LL = np.meshgrid(D_GRID, LAM_GRID, indexing="ij")


def radius_scale(region, line):
    """(s, note): R_shell / R_HII(catalogue) from the flux fit if well determined, else (1, reason)."""
    p = FLUX_DIR / f"{region}_{line}_flux.npz"
    if not p.exists():
        return 1.0, "no flux fit: catalogue radius"
    f = np.load(p)
    s, (lo, hi) = float(f["best_s"]), f["s_1sig"]
    if hi / lo > S_MAX_RATIO or f["rho"].max() < 1.1 * s:
        return 1.0, f"flux radius poorly determined (s={s:.2f} [{lo:.2f},{hi:.2f}]): catalogue radius"
    return s, ""


def load_K(region, line, s=1.0):
    p = DATA_DIR / f"{region}_{line}_formula_data.npz"
    if not p.exists():
        return None
    d = np.load(p)
    e = d["edges_over_rhii"]
    with np.errstate(invalid="ignore", divide="ignore"):
        K = d["S4"] / d["S2"] ** 2
        Kr = d["rep_S4"] / d["rep_S2"] ** 2
    ell = np.sqrt(e[:-1] * e[1:]) / s
    use = (d["counts"] >= MIN_PAIRS) & np.isfinite(K) & np.all(np.isfinite(Kr), 0) & (ell <= L_MAX_MODEL)
    if use.sum() < 3 or Kr.shape[0] < 3:
        return None
    C, shrink = shrink_to_diagonal(Kr[:, use] - Kr[:, use].mean(0))
    return {"ell": ell[use], "K": K[use], "C_jk": C, "shrink": shrink,
            "n_rep": Kr.shape[0], "L_out": float(d["fov_over_rhii"]) / s, "n_fibres": int(d["n_fibres"])}


def m2lnL_map(data, model):
    """-2 ln L on (D_GRID, LAM_GRID) and the chi2 part alone."""
    ell = data["ell"]
    n = len(ell)
    lx = np.log10(ell)
    rho = np.exp(-np.abs(lx[:, None] - lx[None, :]) / RHO_DEX)
    C0 = data["C_jk"] + SIG_MOD_K ** 2 * np.eye(n)
    e3, d3, l3 = ell[None, None, :], DD[..., None], LL[..., None]
    _, K = model.predict(e3, d3, l3, data["L_out"])
    sK, _ = model.realisation_sigma(e3, d3, l3)
    C = C0 + sK[..., :, None] * sK[..., None, :] * rho
    Lc = np.linalg.cholesky(C)
    z = np.linalg.solve(Lc, (data["K"] - K)[..., None])[..., 0]
    chi2 = (z ** 2).sum(-1)
    return chi2 + 2 * np.log(np.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1), chi2


def profile_range(dm, axis, grid, level=1.0):
    inside = grid[dm.min(axis=axis) < level]
    return inside.min(), inside.max(), bool(inside.min() > grid[0] and inside.max() < grid[-1])


def main():
    global OUT_DIR
    use_flux = "--radius" in sys.argv and sys.argv[sys.argv.index("--radius") + 1] == "flux"
    if use_flux:
        OUT_DIR = OUT_DIR.parent / "formula_grid_fluxradius"
    model = FormulaModel()
    Lmin, Lmax = model.bounds["l_out"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for region, line in all_targets():
        s, s_note = radius_scale(region, line) if use_flux else (1.0, "")
        data = load_K(region, line, s)
        if data is None:
            print(f"{region:15s} {line:4s} no usable data")
            continue
        flags = [s_note] if s_note else []
        if not Lmin <= data["L_out"] <= Lmax:
            flags.append(f"L_out {data['L_out']:.2f} clipped to table")
        m2, chi2 = m2lnL_map(data, model)
        dm = m2 - m2.min()
        ib = np.unravel_index(np.argmin(m2), m2.shape)
        d_lo, d_hi, d_ok = profile_range(dm, 1, D_GRID)
        l_lo, l_hi, l_ok = profile_range(dm, 0, LAM_GRID)
        n = len(data["ell"])
        if not d_ok:
            flags.append("Delta 1-sigma hits grid edge")
        if not l_ok:
            flags.append("lambda 1-sigma hits grid edge")
        if chi2[ib] > n + 3 * np.sqrt(2 * n):
            flags.append("poor fit")
        res = {"region": region, "line": line, "n_fibres": data["n_fibres"], "radius_scale_s": s,
               "L_out_over_rhii": data["L_out"],
               "n_bins": n, "l_min": data["ell"].min(), "l_max": data["ell"].max(), "n_jackknife": data["n_rep"],
               "shrinkage": data["shrink"], "best_delta": DD[ib], "best_lambda": LL[ib], "chi2_min": chi2[ib],
               "dof": n - 2, "delta_1sig_lo": d_lo, "delta_1sig_hi": d_hi, "lambda_1sig_lo": l_lo,
               "lambda_1sig_hi": l_hi, "flags": "; ".join(flags), "_dm": dm, "_data": data}
        results.append(res)
        np.savez(OUT_DIR / f"{region}_{line}_map.npz", delta_grid=D_GRID, lambda_grid=LAM_GRID, dm2lnL=dm,
                 chi2=chi2, L_out=data["L_out"])
        print(f"{region:15s} {line:4s} s={s:.2f} L_out={data['L_out']:5.2f} n={n:2d} best Delta={DD[ib]:.3f} "
              f"[{d_lo:.3f},{d_hi:.3f}] lam={LL[ib]:.2f} [{l_lo:.2f},{l_hi:.2f}] chi2={chi2[ib]:.1f}/{n - 2} "
              f"{'; '.join(flags)}", flush=True)

    cols = [c for c in results[0] if not c.startswith("_")]
    with open(OUT_DIR / "formula_grid_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])

    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    by_key = {(r["region"], r["line"]): r for r in results}
    regions = sorted({r["region"] for r in results})
    n_cols = 4
    n_rows = int(np.ceil(len(regions) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.9 * n_rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, region in zip(axes.flat, regions):
        for line in LINES:
            r = by_key.get((region, line))
            if r is None:
                continue
            c = LINE_COLORS[line]
            ax.contourf(DD, LL, r["_dm"], levels=[0, 2.30], colors=[c], alpha=0.25)
            ax.contour(DD, LL, r["_dm"], levels=[2.30, 6.18], colors=[c], linestyles=[LINESTYLES[line]],
                       linewidths=[1.5, 0.8])
            ax.plot(r["best_delta"], r["best_lambda"], "*", ms=11, color=c, mec="white", mew=0.6,
                    label=fr"{LINE_LABELS[line]} ($L_{{\rm out}}$={r['L_out_over_rhii']:.1f}, $s$={r['radius_scale_s']:.2f})")
        ax.set(xscale="log", yscale="log")
        ax.set_title(region.replace("_", " "), fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$v_{\rm bulk}/v_{\rm turb}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    fig.suptitle((r"[$r_{\rm HII}$ from flux profile] " if use_flux else "") +
                 r"Formula fit to $K(\ell)$, $L_{\rm out}$ = FOV: filled = $\Delta(-2\ln L)<2.3$ (1$\sigma$), "
                 r"thin line = 6.2 (2$\sigma$), star = best fit", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "formula_grid_maps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows), squeeze=False)
    ell_d = np.geomspace(0.005, 0.8, 80)
    for ax, region in zip(axes.flat, regions):
        for line in LINES:
            r = by_key.get((region, line))
            if r is None:
                continue
            c, d = LINE_COLORS[line], r["_data"]
            ax.errorbar(d["ell"], d["K"], np.sqrt(np.diag(d["C_jk"])), fmt="o", ms=4, color=c, capsize=2,
                        label=fr"{LINE_LABELS[line]}: $\Delta$={r['best_delta']:.2g}, $\lambda$={r['best_lambda']:.2g}")
            e = ell_d[(ell_d >= d["ell"].min() * 0.8) & (ell_d <= d["ell"].max() * 1.2)]
            _, K = model.predict(e, r["best_delta"], r["best_lambda"], d["L_out"])
            ax.plot(e, K, LINESTYLES[line], color=c, lw=1.5)
        ax.set_xscale("log")
        ax.axhline(3, color="0.5", ls="--", lw=0.8)
        ax.set_title(region.replace("_", " "), fontsize=11)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$K(\ell)$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\ell / r_{\rm HII}$")
    fig.suptitle("Observed K (jackknife errors) and best-fit formula", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "formula_grid_curves.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
