"""Two-step shell inference: lambda from the slope of K(l), then Delta from S2(l) at that lambda.

Step 1 (lambda from the K slope). Statistic: the least-squares slope of ln K against log10 l over all usable bins,
    s = d ln K / d log10 l.
ln K rather than K so that a constant mixing factor on S4 (S4 -> M S4, dispersion varying across the field)
cancels. The same statistic is computed from the centroid model of fit_centroid_joint_grid.py (fibre = emission-
weighted line-of-sight centroid; back-side weight a = A_BACK) on the region's own l bins, with the same noise
model as the joint fits (N2 = NOISE_C^2 x catalogue N2, K_obs = (K S2_sig^2 + 6 S2_sig N2 + 3 N2^2) / S2_obs^2) and
v_turb set at each (Delta, lambda) so that the model's mean ln S2 equals the observed one. This gives the
calibration s(lambda; Delta). chi2_s(Delta, lambda) = (s_obs - s_mod)^2 / sigma_s^2 with sigma_s from the shrunk
jackknife covariance of K (+ model error), propagated to ln K. lambda: minimum and 1-sigma range of chi2_s profiled
over Delta (Delta in D_GRID, i.e. not fixed), so the Delta dependence of the calibration widens the lambda range.

Step 2 (Delta from S2). At the best lambda of step 1, fit ln S2(l) with the centroid model:
S2_obs = v_turb^2 s2(l; Delta, lambda) + N2, chi2 from the S2 block of the covariance, v_turb profiled.

Reference: the same slope for the analytic point-sample model (fit_analytic_point_K.py; thin shell, no sightline
averaging), drawn as a grey line in the calibration panels.

Outputs (Figures/manuscript/08_shell_inference/slope_lambda/):
  slope_lambda_summary.csv
  slope_lambda_calibration.png  per region: s_mod(lambda) for each line (band = range over Delta), observed slope
                                (horizontal band), inferred lambda (star)
  slope_lambda_vs_joint.png     lambda and Delta from this method vs the joint centroid fit (--joint-run)

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/fit_slope_lambda.py [--a 0] [--joint-run centroid_joint_v_a0_lammin1]
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
import fit_analytic_point_K as apk  # noqa: E402
from fit_centroid_joint_grid import A_MOM, D_GRID, LINES, LINESTYLES, LNV2_GRID, OUT_BASE, get_moments, model_s2_K  # noqa: E402
from fit_kurtosis_formula import DATA_DIR  # noqa: E402
from fit_kurtosis_s2_joint_grid import load, prange  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

LAM_GRID = np.geomspace(0.03, 30.0, 70)
OUT_DIR = OUT_BASE / "slope_lambda"
CACHE = OUT_BASE / "centroid_moments"
BAND_DELTA = (0.01, 0.3)  # Delta range drawn as the calibration band


def slope_weights(ell):
    x = np.log10(ell)
    return (x - x.mean()) / ((x - x.mean()) ** 2).sum()


def observed_K(Ku, s2u, N2, lnS2_obs):
    """K_obs and S2_obs on (..., l) with v_turb matched to the mean observed ln S2 at each leading index."""
    target = np.mean(lnS2_obs)
    mis = np.stack([np.abs(np.mean(np.log(np.exp(u) * s2u + N2), -1) - target) for u in LNV2_GRID])
    v2 = np.exp(LNV2_GRID[mis.argmin(0)])[..., None]
    S2s = v2 * s2u
    S2o = S2s + N2
    return (Ku * S2s ** 2 + 6 * S2s * N2 + 3 * N2 ** 2) / S2o ** 2, S2o


def fit_line(region, X, a):
    d = load(region, X, "overlap")
    if d is None:
        return None
    n = len(d["ell"])
    w = slope_weights(d["ell"])
    lnK = np.log(d["K"])
    C_lnK = d["C0"][:n, :n] / np.outer(d["K"], d["K"])
    s_obs, sig_s = float(w @ lnK), float(np.sqrt(w @ C_lnK @ w))

    # centroid calibration s_mod(Delta, lambda)
    mom = get_moments(region, X, d, CACHE, 0.0)[..., int(np.flatnonzero(np.isclose(A_MOM, a))[0])]  # (5, l, D)
    s2u, Ku = model_s2_K(np.moveaxis(mom, 1, -1), LAM_GRID)  # (D, l, lam)
    s2u, Ku = np.swapaxes(s2u, -1, -2), np.swapaxes(Ku, -1, -2)  # (D, lam, l)
    Ko, _ = observed_K(Ku, s2u, d["N2"], d["lnS2"])
    s_mod = np.log(Ko) @ w  # (D, lam)
    chi_s = (s_obs - s_mod) ** 2 / sig_s ** 2
    prof = chi_s.min(0)
    iL = int(prof.argmin())
    l_lo, l_hi, l_ok = prange(prof - prof.min(), LAM_GRID)

    # step 2: Delta from S2 at lambda_best (v_turb profiled)
    Linv = np.linalg.inv(np.linalg.cholesky(d["C0"][n:, n:]))
    s2l = s2u[:, iL]  # (D, l)
    chi_S = np.min([(((d["lnS2"] - np.log(np.exp(u) * s2l + d["N2"])) @ Linv.T) ** 2).sum(-1) for u in LNV2_GRID], 0)
    iD = int(chi_S.argmin())
    d_lo, d_hi, d_ok = prange(chi_S - chi_S.min(), D_GRID)

    # point-sample reference (analytic, noise as above), median over its Delta grid
    Kpt, s2pt = apk.model_grid(d["ell"], 2 * np.pi / d["L_out"])
    Kpo, _ = observed_K(Kpt, s2pt, d["N2"], d["lnS2"])
    s_pt = np.median(np.log(Kpo) @ w, 0)  # on apk.LAM_GRID

    flags = []
    if not l_ok:
        flags.append("lambda 1-sigma hits grid edge")
    if prof.min() > 1:
        flags.append("observed slope outside model range")
    if not d_ok:
        flags.append("Delta 1-sigma hits grid edge")
    band = (D_GRID >= BAND_DELTA[0]) & (D_GRID <= BAND_DELTA[1])
    row = {"region": region, "line": X, "n_bins": n, "a": a, "slope_obs": s_obs, "slope_err": sig_s,
           "lambda_slope": LAM_GRID[iL], "lambda_lo": l_lo, "lambda_hi": l_hi, "chi2_slope_min": prof.min(),
           "slope_model_min": s_mod.min(), "slope_model_max": s_mod.max(),
           "delta_S2": D_GRID[iD], "delta_lo": d_lo, "delta_hi": d_hi, "chi2_S2": chi_S.min(), "n_S2": n,
           "flags": "; ".join(flags)}
    plot = dict(s_band=(s_mod[band].min(0), s_mod[band].max(0)), s_pt=s_pt)
    return row, plot


def plot_calibration(rows, plots):
    regions = list(dict.fromkeys(r["region"] for r in rows))
    nc = 4
    nr = int(np.ceil(len(regions) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.8 * nr), squeeze=False, sharex=True)
    for ax, region in zip(axes.flat, regions):
        for r in [r for r in rows if r["region"] == region]:
            X, p, c = r["line"], plots[(region, r["line"])], LINE_COLORS[r["line"]]
            ax.fill_between(LAM_GRID, *p["s_band"], color=c, alpha=0.18, lw=0)
            ax.plot(LAM_GRID, 0.5 * (p["s_band"][0] + p["s_band"][1]), LINESTYLES[X], color=c, lw=1.4)
            ax.axhspan(r["slope_obs"] - r["slope_err"], r["slope_obs"] + r["slope_err"], color=c, alpha=0.12)
            ax.axhline(r["slope_obs"], color=c, ls=LINESTYLES[X], lw=0.9)
            ax.errorbar(r["lambda_slope"], r["slope_obs"],
                        xerr=[[r["lambda_slope"] - r["lambda_lo"]], [r["lambda_hi"] - r["lambda_slope"]]],
                        fmt="*", ms=13, color=c, mec="white", mew=0.6, capsize=3,
                        label=fr"{LINE_LABELS[X]}: $\lambda$={r['lambda_slope']:.2g} [{r['lambda_lo']:.2g}, {r['lambda_hi']:.2g}]")
            ax.plot(apk.LAM_GRID, p["s_pt"], color="0.55", lw=0.9, ls="-." if X == "NII" else "none")
        ax.set(xscale="log", xlim=(LAM_GRID[0], LAM_GRID[-1]), title=region.replace("_", " "))
        ax.axhline(0, color="0.6", lw=0.6)
        ax.legend(fontsize=7, loc="lower left")
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"slope $d\ln K/d\log_{10}\ell$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\lambda_{\rm eff}=v_{\rm bulk}/v_{\rm turb}$")
    fig.suptitle(r"$\lambda$ from the slope of $K(\ell)$: centroid-model calibration (band: $\Delta$ = "
                 fr"{BAND_DELTA[0]:g}-{BAND_DELTA[1]:g}, $a$ = {rows[0]['a']:g}) vs observed slope (horizontal band, 1$\sigma$); "
                 r"grey dash-dot: point-sample model ([NII])", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slope_lambda_calibration.png", dpi=160)
    plt.close(fig)


def plot_vs_joint(rows, joint_run):
    path = OUT_BASE / joint_run / "joint_summary.csv"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        J = {(r["region"], r["line"]): r for r in csv.DictReader(f)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    for r in rows:
        j = J.get((r["region"], r["line"]))
        if j is None:
            continue
        c = LINE_COLORS[r["line"]]
        kw = dict(fmt="o", ms=5, color=c, capsize=2, alpha=0.85)
        a1.errorbar(float(j["best_lambda_eff"]), r["lambda_slope"],
                    xerr=[[float(j["best_lambda_eff"]) - float(j["lambda_1sig_lo"])], [float(j["lambda_1sig_hi"]) - float(j["best_lambda_eff"])]],
                    yerr=[[r["lambda_slope"] - r["lambda_lo"]], [r["lambda_hi"] - r["lambda_slope"]]], **kw)
        a2.errorbar(float(j["best_delta"]), r["delta_S2"],
                    xerr=[[float(j["best_delta"]) - float(j["delta_1sig_lo"])], [float(j["delta_1sig_hi"]) - float(j["best_delta"])]],
                    yerr=[[r["delta_S2"] - r["delta_lo"]], [r["delta_hi"] - r["delta_S2"]]], **kw)
        a1.annotate(r["region"].replace("_nebula", "")[:6], (float(j["best_lambda_eff"]), r["lambda_slope"]), fontsize=6, color=c,
                    xytext=(3, 3), textcoords="offset points")
    for ax, lo, hi, lab in ((a1, 0.03, 30, r"\lambda"), (a2, 0.005, 1, r"\Delta")):
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               xlabel=fr"${lab}$ joint centroid fit ({joint_run.removeprefix('centroid_joint_v_')})",
               ylabel=fr"${lab}$ two-step (slope, then $S_2$)")
    for X in LINES:
        a1.plot([], [], "o", color=LINE_COLORS[X], label=LINE_LABELS[X])
    a1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slope_lambda_vs_joint.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.0, choices=[float(x) for x in A_MOM])
    ap.add_argument("--joint-run", default="centroid_joint_v_a0_lammin1")
    ap.add_argument("--region", default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    regions = sorted({p.name.rsplit("_", 3)[0] for p in DATA_DIR.glob("*_formula_data.npz")})
    if args.region:
        regions = [args.region]
    rows, plots = [], {}
    for region in regions:
        for X in LINES:
            out = fit_line(region, X, args.a)
            if out is None:
                continue
            row, p = out
            rows.append(row)
            plots[(region, X)] = p
            print(f"{region:15s} {X:4s} slope={row['slope_obs']:+.2f}±{row['slope_err']:.2f} "
                  f"(model {row['slope_model_min']:+.2f}..{row['slope_model_max']:+.2f})  "
                  f"lam={row['lambda_slope']:.2f} [{row['lambda_lo']:.2f},{row['lambda_hi']:.2f}]  "
                  f"Delta(S2)={row['delta_S2']:.3f} [{row['delta_lo']:.3f},{row['delta_hi']:.3f}] "
                  f"chi2_S2={row['chi2_S2']:.1f}/{row['n_S2']}  {row['flags']}", flush=True)
    suffix = "" if args.a == 0 else f"_a{args.a:g}"
    with open(OUT_DIR / f"slope_lambda_summary{suffix}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(list(rows[0]))
        for r in rows:
            wr.writerow([f"{v:.5g}" if isinstance(v, (float, np.floating)) else v for v in r.values()])
    if args.region is None:
        plot_calibration(rows, plots)
        plot_vs_joint(rows, args.joint_run)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
