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
  slope_lambda_probability.png  per region: P(lambda) from the slope (profiled over Delta) and P(Delta) from S2 at
                                lambda_best, P = exp(-dchi2/2) normalised to peak 1; thin lines: the joint centroid
                                fit's profiles of the same parameter (--joint-run)
  slope_lambda_maps_physical.png  (Delta r_shell [pc], v_bulk [km/s]) from chi2_slope + chi2_S2 on (Delta, lambda),
                                v_bulk = lambda x v_turb with v_turb profiled from S2 at each point; grey: joint fit
  <region>_<LINE>_profiles.npz  dchi2(lambda), dchi2(Delta), dchi2(Delta, lambda), v_turb(Delta, lambda)

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

    # 2D map: slope chi2 + S2 chi2 on (Delta, lambda), v_turb profiled per point (for v_bulk = lambda v_turb)
    cS = np.stack([(((d["lnS2"] - np.log(np.exp(u) * s2u + d["N2"])) @ Linv.T) ** 2).sum(-1) for u in LNV2_GRID])
    v_map = np.exp(LNV2_GRID[cS.argmin(0)] / 2)  # (D, lam), km/s
    chi2d = chi_s + cS.min(0)
    jD, jL = np.unravel_index(chi2d.argmin(), chi2d.shape)

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
    plot = dict(s_band=(s_mod[band].min(0), s_mod[band].max(0)), s_pt=s_pt, prof_lam=prof - prof.min(),
                prof_delta=chi_S - chi_S.min(), map2d=chi2d - chi2d.min(), v_map=v_map,
                best2d=(D_GRID[jD], LAM_GRID[jL] * v_map[jD, jL]))
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


def plot_physical(rows, plots, joint_run):
    """(Delta r_shell [pc], v_bulk [km/s]) maps from slope + S2 for every region; v_bulk = lambda x v_turb(Delta, lambda).
    Thin grey contours (1 sigma): the joint centroid fit, converted with its shared v_turb."""
    from plot_centroid_joint_selected_physical import load_adopted_radii
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    adopted = load_adopted_radii()
    run = OUT_BASE / joint_run
    with open(run / "joint_summary.csv", encoding="utf-8") as f:
        J = {(r["region"], r["line"]): r for r in csv.DictReader(f)}
    regions = list(dict.fromkeys(r["region"] for r in rows))
    nc = 4
    nr = int(np.ceil(len(regions) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.7 * nc, 4.1 * nr), squeeze=False)
    for ax, region in zip(axes.flat, regions):
        lines = [r["line"] for r in rows if r["region"] == region]
        r_hii = adopted.get(region) or float(np.load(DATA_DIR / f"{region}_{lines[0]}_formula_data.npz")["r_hii_pc"])
        xs, ys = [r_hii], []
        for X in lines:
            p, c = plots[(region, X)], LINE_COLORS[X]
            xx = (D_GRID * r_hii)[:, None] * np.ones(len(LAM_GRID))[None]
            yy = LAM_GRID[None, :] * p["v_map"]
            m = p["map2d"]
            inside = m <= 6.18
            xs += [xx[inside].min(), xx[inside].max()]
            ys += [yy[inside].min(), yy[inside].max()]
            ax.contourf(xx, yy, m, levels=[0, 2.30], colors=[c], alpha=0.25)
            ax.contour(xx, yy, m, levels=[2.30, 6.18], colors=[c], linestyles=[LINESTYLES[X]], linewidths=[1.8, 0.9])
            ax.plot(p["best2d"][0] * r_hii, p["best2d"][1], "*", ms=13, color=c, mec="white", mew=0.7)
            ax.plot([], [], color=c, ls=LINESTYLES[X], lw=1.8, label=LINE_LABELS[X])
            jm = run / f"{region}_{X}_map.npz"
            if jm.exists() and (region, X) in J:
                jj = np.load(jm)
                v = float(J[(region, X)]["v_turb"])
                gx, gy = np.meshgrid(jj["delta_grid"] * r_hii, jj["lambda_grid"] * v, indexing="ij")
                ax.contour(gx, gy, jj["dm2lnL"], levels=[2.30], colors=["0.35"], linestyles=[LINESTYLES[X]], linewidths=0.8)
        ax.axvline(r_hii, color="0.3", ls="--", lw=1.0)
        ax.set(xscale="log", yscale="log", title=region.replace("_", " "))
        ax.set_xlim(min(xs) / 1.3, max(xs) * 1.3)
        ylo, yhi = min(ys) / 1.2, max(ys) * 1.2
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(FixedLocator([t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300) if ylo <= t <= yhi]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8, loc="lower left")
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$v_{\rm bulk}$ (km/s)")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}$ [pc]")
    fig.suptitle(r"Slope of $\ln K$ + $S_2(\ell)$, centroid model: filled 1$\sigma$, thin 2$\sigma$, star = best; "
                 r"grey: joint centroid fit 1$\sigma$ (" + joint_run.removeprefix("centroid_joint_v_") + r"); dashed: $r_{\rm HII}$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slope_lambda_maps_physical.png", dpi=160)
    plt.close(fig)


def plot_probability(rows, plots, joint_run):
    run = OUT_BASE / joint_run
    regions = list(dict.fromkeys(r["region"] for r in rows))
    nc = 4
    nr = int(np.ceil(len(regions) / nc))
    fig, axes = plt.subplots(2 * nr, nc, figsize=(4.5 * nc, 2.9 * 2 * nr), squeeze=False)
    for i, region in enumerate(regions):
        aL, aD = axes[2 * (i // nc), i % nc], axes[2 * (i // nc) + 1, i % nc]
        for r in [r for r in rows if r["region"] == region]:
            X, p, c = r["line"], plots[(region, r["line"])], LINE_COLORS[r["line"]]
            aL.plot(LAM_GRID, np.exp(-0.5 * p["prof_lam"]), LINESTYLES[X], color=c, lw=2, label=LINE_LABELS[X])
            aD.plot(D_GRID, np.exp(-0.5 * p["prof_delta"]), LINESTYLES[X], color=c, lw=2)
            jm = run / f"{region}_{X}_map.npz"
            if jm.exists():
                m = np.load(jm)
                aL.plot(m["lambda_grid"], np.exp(-0.5 * m["dm2lnL"].min(0)), LINESTYLES[X], color=c, lw=0.9, alpha=0.7)
                aD.plot(m["delta_grid"], np.exp(-0.5 * m["dm2lnL"].min(1)), LINESTYLES[X], color=c, lw=0.9, alpha=0.7)
        for ax in (aL, aD):
            ax.set(xscale="log", ylim=(0, 1.08))
            ax.axhline(np.exp(-0.5), color="0.6", lw=0.6, ls=":")
        aL.set(xlim=(LAM_GRID[0], LAM_GRID[-1]), title=region.replace("_", " "), xlabel=r"$\lambda_{\rm eff}$")
        aD.set(xlim=(D_GRID[0], D_GRID[-1]), xlabel=r"$\Delta r_{\rm shell}/r_{\rm HII}$")
        if i % nc == 0:
            aL.set_ylabel(r"$P(\lambda)$ from slope")
            aD.set_ylabel(r"$P(\Delta)$ from $S_2$")
        if i == 0:
            aL.plot([], [], color="0.4", lw=2, label="two-step")
            aL.plot([], [], color="0.4", lw=0.9, label="joint fit")
            aL.legend(fontsize=7.5, loc="upper left")
    for j in range(len(regions), nr * nc):
        axes[2 * (j // nc), j % nc].axis("off")
        axes[2 * (j // nc) + 1, j % nc].axis("off")
    fig.suptitle(r"Profile probabilities $P=e^{-\Delta\chi^2/2}$ (peak 1; dotted: $1\sigma$ level). Thick: two-step "
                 r"($\lambda$ from the $\ln K$ slope, $\Delta$ from $S_2$ at $\lambda_{\rm best}$); thin: joint centroid fit "
                 fr"({joint_run.removeprefix('centroid_joint_v_')}, profiled)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "slope_lambda_probability.png", dpi=150)
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
            np.savez(OUT_DIR / f"{region}_{X}_profiles.npz", lambda_grid=LAM_GRID, dchi2_lambda=p["prof_lam"],
                     delta_grid=D_GRID, dchi2_delta=p["prof_delta"], lambda_best=row["lambda_slope"],
                     dchi2_2d=p["map2d"], v_turb_2d=p["v_map"])
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
        plot_probability(rows, plots, args.joint_run)
        plot_physical(rows, plots, args.joint_run)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
