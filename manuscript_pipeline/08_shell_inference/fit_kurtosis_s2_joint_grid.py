"""Joint (K, ln S2) grid fit of the pair-kurtosis formula across [NII], [SII], [SIII] with a shared v_turb.

K alone barely depends on Delta above the 37" floor (fit_kurtosis_formula_grid.py: Delta unconstrained).
The S2 amplitude does: at fixed (v_turb, lam), S2(l) / v_turb^2 at small l grows with Delta (more
line-of-sight depth, more pairs at large 3D separation). If the lines share v_turb, the S2 ratio
between lines at small l measures their difference in Delta.

Model per line (formula and tables: kurtosis_formula_tables.py), l in units of r_HII(catalogue):
    S2_obs = v_turb^2 s2(l; Delta_X, lam_X, L_out) + N2_X
    K_obs  = [K s2^2 v^4 + 6 v^2 s2 N2 + 3 N2^2] / S2_obs^2
    N2_X   = NOISE_C[X]^2 x (catalogue 2<vel_err^2>)  -- fixed; the catalogue errors are ~3-4x too large
             (tile-overlap duplicate pairs in Orion and Rosette: c = 0.25/0.28 [NII], 0.35/0.35 [SII],
             0.46/0.23 [SIII]; adopted 0.26, 0.35, 0.35), --noise none sets 0
Parameters of a region:
    --share v      v_turb shared; Delta_X and lam_X per line (v_bulk free per line)   [default]
    --share v_lam  v_turb and lam shared (one v_bulk); Delta_X per line
Data: the saved curves of fit_kurtosis_formula.py measure (formula_fit/data/*_formula_data.npz), nothing
recomputed. y_X = [K(l_i), ln S2(l_i)], joint block-jackknife covariance shrunk to its diagonal, +
realisation scatter (8-seed table, corr exp(-|dlog l|)) + model error (SIG_MOD_K, SIG_MOD_LNS2).
Lines are treated as independent (cross-line covariance ignored: the realisation term is partly common,
so the Delta differences are, if anything, conservative).

Maps: for line X on (Delta_X, lam_X),
    -2 ln L_X(Delta, lam) = min over v [ m2_X(Delta, lam, v) + sum_{Y != X} min over own params m2_Y(., v) ]
(with --share v_lam the other lines are profiled at the same lam), relative to the global minimum.

Outputs: Figures/manuscript/08_shell_inference/formula_grid_joint_<share>/
  joint_maps.png, joint_curves.png (K and S2 with best fit), joint_delta_profiles.png, joint_summary.csv

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/fit_kurtosis_s2_joint_grid.py [--share v|v_lam] [--noise overlap|none]
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
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from fit_kurtosis_formula import DATA_DIR, MIN_PAIRS, RHO_DEX, SIG_MOD_K, SIG_MOD_LNS2, shrink_to_diagonal  # noqa: E402
from kurtosis_formula_tables import FormulaModel  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

OUT_BASE = Path("Figures/manuscript/08_shell_inference")
LINES = ("NII", "SII", "SIII")
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}
NOISE_C = {"NII": 0.26, "SII": 0.35, "SIII": 0.35}  # overlap-pair calibration (Orion, Rosette)
D_GRID = np.geomspace(0.005, 1.0, 80)
LAM_GRID = np.geomspace(0.03, 30.0, 90)
LNV2_GRID = np.log(np.geomspace(0.2, 40.0, 140) ** 2)  # ln v_turb^2, v in km/s
DD, LL = np.meshgrid(D_GRID, LAM_GRID, indexing="ij")


def load(region, line, noise):
    p = DATA_DIR / f"{region}_{line}_formula_data.npz"
    if not p.exists():
        return None
    d = np.load(p)
    e = d["edges_over_rhii"]
    ell = np.sqrt(e[:-1] * e[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        K, Kr = d["S4"] / d["S2"] ** 2, d["rep_S4"] / d["rep_S2"] ** 2
        lS, lSr = np.log(d["S2"]), np.log(d["rep_S2"])
    use = ((d["counts"] >= MIN_PAIRS) & np.isfinite(K) & np.all(np.isfinite(Kr), 0)
           & np.all(np.isfinite(lSr), 0))
    if use.sum() < 3 or Kr.shape[0] < 3:
        return None
    reps = np.hstack((Kr[:, use], lSr[:, use]))
    C, shrink = shrink_to_diagonal(reps - reps.mean(0))
    n = int(use.sum())
    C = C + np.diag(np.r_[np.full(n, SIG_MOD_K ** 2), np.full(n, SIG_MOD_LNS2 ** 2)])
    return {"ell": ell[use], "K": K[use], "lnS2": lS[use], "S2": d["S2"][use], "C0": C, "shrink": shrink,
            "n_rep": Kr.shape[0], "L_out": float(d["fov_over_rhii"]), "n_fibres": int(d["n_fibres"]),
            "N2": NOISE_C[line] ** 2 * float(d["N2"]) if noise == "overlap" else 0.0,
            "sdK": np.sqrt(np.diag(C)[:n]), "sdlS": np.sqrt(np.diag(C)[n:])}


def predict(model, ell, delta, lam, L_out, lnv2, N2):
    """Observed (K, ln S2) for arrays broadcast over (..., l)."""
    s2u, Ku = model.predict(ell, delta, lam, L_out)
    S2s = np.exp(lnv2) * s2u
    S2o = S2s + N2
    Ko = (Ku * S2s ** 2 + 6 * S2s * N2 + 3 * N2 ** 2) / S2o ** 2
    return Ko, np.log(S2o)


def m2lnL_cube(data, model):
    """-2 ln L on (D_GRID, LAM_GRID, LNV2_GRID) and the chi2 part. Covariance depends on (Delta, lam) only."""
    ell, n = data["ell"], len(data["ell"])
    lx = np.log10(ell)
    rho = np.exp(-np.abs(lx[:, None] - lx[None, :]) / RHO_DEX)
    e3, d3, l3 = ell[None, None, :], DD[..., None], LL[..., None]
    sK, sS = model.realisation_sigma(e3, d3, l3)
    C = np.broadcast_to(data["C0"], DD.shape + data["C0"].shape).copy()
    C[..., :n, :n] += sK[..., :, None] * sK[..., None, :] * rho
    C[..., n:, n:] += sS[..., :, None] * sS[..., None, :] * rho
    Lc = np.linalg.cholesky(C)
    Linv = np.linalg.inv(Lc)
    logdet = 2 * np.log(np.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1)
    y = np.r_[data["K"], data["lnS2"]]
    chi2 = np.empty(DD.shape + (len(LNV2_GRID),))
    for k, u in enumerate(LNV2_GRID):
        Ko, lS = predict(model, e3, d3, l3, data["L_out"], u, data["N2"])
        r = y - np.concatenate((Ko, lS), axis=-1)
        z = np.einsum("...ij,...j->...i", Linv, r)
        chi2[..., k] = (z ** 2).sum(-1)
    return chi2 + logdet[..., None], chi2


def joint(cubes, share):
    """Per-line profiled maps on (Delta, lam), relative to the joint minimum, and the joint best fit."""
    lines = list(cubes)
    if share == "v":
        own = {X: cubes[X].min(axis=(0, 1)) for X in lines}  # (v,)
        tot = {X: cubes[X] + sum(own[Y] for Y in lines if Y != X)[None, None, :] for X in lines}
    else:  # v_lam: others profiled over their Delta at the same (lam, v)
        own = {X: cubes[X].min(axis=0) for X in lines}  # (lam, v)
        tot = {X: cubes[X] + sum(own[Y] for Y in lines if Y != X)[None] for X in lines}
    gmin = min(t.min() for t in tot.values())
    maps = {X: tot[X].min(axis=2) - gmin for X in lines}
    best = {}
    for X in lines:
        ib = np.unravel_index(np.argmin(tot[X]), tot[X].shape)
        best[X] = {"delta": D_GRID[ib[0]], "lam": LAM_GRID[ib[1]], "lnv2": LNV2_GRID[ib[2]]}
    # joint best v: the same for all lines up to grid ties
    v_prof = (sum(own[Y] for Y in lines) if share == "v" else sum(own[Y] for Y in lines).min(0))
    vp = v_prof - v_prof.min()
    return maps, best, vp


def prange(prof, grid, level=1.0):
    inside = grid[prof < level]
    return inside.min(), inside.max(), bool(inside.min() > grid[0] and inside.max() < grid[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", choices=("v", "v_lam"), default="v")
    ap.add_argument("--noise", choices=("overlap", "none"), default="overlap")
    ap.add_argument("--region", default=None)
    args = ap.parse_args()
    out_dir = OUT_BASE / f"formula_grid_joint_{args.share}" / ("" if args.noise == "overlap" else "no_noise")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = FormulaModel()
    regions = sorted({p.name.rsplit("_", 3)[0] for p in DATA_DIR.glob("*_formula_data.npz")})
    if args.region:
        regions = [args.region]

    results, store = [], {}
    for region in regions:
        datas = {X: load(region, X, args.noise) for X in LINES}
        datas = {X: d for X, d in datas.items() if d is not None}
        if len(datas) < 2:
            print(f"{region}: fewer than 2 lines usable; skipped")
            continue
        cubes, chis = {}, {}
        for X, d in datas.items():
            cubes[X], chis[X] = m2lnL_cube(d, model)
        maps, best, vp = joint(cubes, args.share)
        v_lo, v_hi, _ = prange(vp, np.sqrt(np.exp(LNV2_GRID)))
        store[region] = (datas, maps, best)
        for X, d in datas.items():
            b = best[X]
            iD, iL = np.argmin(np.abs(D_GRID - b["delta"])), np.argmin(np.abs(LAM_GRID - b["lam"]))
            iv = np.argmin(np.abs(LNV2_GRID - b["lnv2"]))
            n = len(d["ell"])
            d_lo, d_hi, d_ok = prange(maps[X].min(1), D_GRID)
            l_lo, l_hi, l_ok = prange(maps[X].min(0), LAM_GRID)
            flags = [f for f, ok in (("Delta 1-sigma hits grid edge", d_ok), ("lambda 1-sigma hits grid edge", l_ok))
                     if not ok]
            c2 = chis[X][iD, iL, iv]
            if c2 > 2 * n + 3 * np.sqrt(4 * n):
                flags.append("poor fit")
            r = {"region": region, "line": X, "share": args.share, "noise_N2": d["N2"], "n_bins": n,
                 "l_min": d["ell"].min(), "L_out_over_rhii": d["L_out"], "v_turb": np.exp(b["lnv2"] / 2),
                 "v_turb_1sig_lo": v_lo, "v_turb_1sig_hi": v_hi, "best_delta": b["delta"],
                 "delta_1sig_lo": d_lo, "delta_1sig_hi": d_hi, "best_lambda": b["lam"],
                 "lambda_1sig_lo": l_lo, "lambda_1sig_hi": l_hi, "v_bulk": b["lam"] * np.exp(b["lnv2"] / 2),
                 "chi2_line": c2, "n_data_line": 2 * n, "flags": "; ".join(flags)}
            results.append(r)
            print(f"{region:15s} {X:4s} v_turb={r['v_turb']:5.2f} [{v_lo:.2f},{v_hi:.2f}]  "
                  f"Delta={b['delta']:.3f} [{d_lo:.3f},{d_hi:.3f}]  lam={b['lam']:.2f} [{l_lo:.2f},{l_hi:.2f}]  "
                  f"chi2={c2:.1f}/{2 * n}  {r['flags']}", flush=True)
            np.savez(out_dir / f"{region}_{X}_joint_map.npz", delta_grid=D_GRID, lambda_grid=LAM_GRID,
                     dm2lnL=maps[X], share=args.share)

    cols = list(results[0])
    with open(out_dir / "joint_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])
    plot(store, model, out_dir, args)
    print(f"wrote {out_dir}")


def plot(store, model, out_dir, args):
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    regions = list(store)
    nc = 4
    nr = int(np.ceil(len(regions) / nc))
    tag = (r"shared $v_{\rm turb}$, $\lambda$ per line" if args.share == "v" else
           r"shared $v_{\rm turb}$ and $\lambda$") + (", noise floor fixed" if args.noise == "overlap" else ", no noise")

    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.9 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, region in zip(axes.flat, regions):
        datas, maps, best = store[region]
        for X in datas:
            c = LINE_COLORS[X]
            ax.contourf(DD, LL, maps[X], levels=[0, 2.30], colors=[c], alpha=0.25)
            ax.contour(DD, LL, maps[X], levels=[2.30, 6.18], colors=[c], linestyles=[LINESTYLES[X]],
                       linewidths=[1.5, 0.8])
            ax.plot(best[X]["delta"], best[X]["lam"], "*", ms=11, color=c, mec="white", mew=0.6,
                    label=fr"{LINE_LABELS[X]}: $\Delta$={best[X]['delta']:.2g}")
        v = np.exp(best[next(iter(datas))]["lnv2"] / 2)
        ax.set(xscale="log", yscale="log")
        ax.set_title(fr"{region.replace('_', ' ')}  ($v_{{\rm turb}}$={v:.1f} km/s)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$v_{\rm bulk}/v_{\rm turb}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    fig.suptitle(fr"Joint fit of $K(\ell)$ and $S_2(\ell)$ ({tag}): filled = 1$\sigma$, thin = 2$\sigma$", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_maps.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.6 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, region in zip(axes.flat, regions):
        datas, maps, _ = store[region]
        for X in datas:
            ax.plot(D_GRID, maps[X].min(1), LINESTYLES[X], color=LINE_COLORS[X], lw=1.8, label=LINE_LABELS[X])
        ax.axhline(1, color="0.5", lw=0.7, ls="--")
        ax.set(xscale="log", ylim=(0, 12))
        ax.set_title(region.replace("_", " "), fontsize=10)
        ax.legend(fontsize=7.5)
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta(-2\ln L)$ profiled")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    fig.suptitle(fr"Profile likelihood of $\Delta$ per line ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_delta_profiles.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2 * nr, nc, figsize=(4.4 * nc, 3.0 * 2 * nr), squeeze=False)
    ell_d = np.geomspace(0.005, 1.0, 80)
    for i, region in enumerate(regions):
        aK, aS = axes[2 * (i // nc), i % nc], axes[2 * (i // nc) + 1, i % nc]
        datas, _, best = store[region]
        for X, d in datas.items():
            c, b = LINE_COLORS[X], best[X]
            e = ell_d[(ell_d >= d["ell"].min() * 0.8) & (ell_d <= d["ell"].max() * 1.2)]
            Ko, lS = predict(model, e, b["delta"], b["lam"], d["L_out"], b["lnv2"], d["N2"])
            aK.errorbar(d["ell"], d["K"], d["sdK"], fmt="o", ms=3.5, color=c, capsize=2)
            aK.plot(e, Ko, LINESTYLES[X], color=c, lw=1.4)
            aS.errorbar(d["ell"], d["S2"], d["S2"] * d["sdlS"], fmt="o", ms=3.5, color=c, capsize=2,
                        label=fr"{LINE_LABELS[X]} $\Delta$={b['delta']:.2g} $\lambda$={b['lam']:.2g}")
            aS.plot(e, np.exp(lS), LINESTYLES[X], color=c, lw=1.4)
        aK.set(xscale="log", title=region.replace("_", " "), ylabel=r"$K$")
        aK.axhline(3, color="0.5", ls="--", lw=0.8)
        aS.set(xscale="log", yscale="log", xlabel=r"$\ell/r_{\rm HII}$", ylabel=r"$S_2$ [km$^2$ s$^{-2}$]")
        aS.legend(fontsize=6.5)
    for j in range(len(regions), nr * nc):
        axes[2 * (j // nc), j % nc].axis("off")
        axes[2 * (j // nc) + 1, j % nc].axis("off")
    fig.suptitle(fr"Observed $K$ and $S_2$ with the joint best fit ({tag})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
