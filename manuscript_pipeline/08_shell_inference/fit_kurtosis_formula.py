"""Fit the analytic pair-kurtosis formula to the observed K(l) and S2(l) of every region/line.

Replaces the slope-vs-floor inference of infer_shell_parameters.py (which matched dK/dlog l at a few
separation floors to the empirical model B) by a direct fit of the whole binned curves with the formula
of manuscript_notes/kurtosis_kolmogorov_lambda_derivation.md (tables: kurtosis_formula_tables.py).

Data vector per region/line, l = sky separation / r_HII(catalogue), bins at N_PER_DEX above the 37" floor:
    y = [K(l_i), ln S2(l_i)]       S_p = <|dv|^p> over all pairs of the (<= 10k-fibre) subsample
Same cuts as compute_multifloor_slopes.py (SNR >= 20, plane subtracted, catalogue distance and radius).
Pairs, S2 and S4 come from vsf_orders_block_jackknife (5x5 blocks), whose replicates are paired across
orders, so the jackknife covariance of (K, ln S2) is joint; it is shrunk toward its diagonal
(Schafer & Strimmer 2005 target D) because the replicates (12-21) are about as many as the data.

Model, with parameters theta = (log Delta, log lam, log L_out, log v_turb, M, log s_r, f_N):
    l_model = l / s_r                                   (true r_HII = s_r x catalogue radius)
    S2_sig  = v_turb^2 s2(l_model; Delta, lam, L_out)
    S4_sig  = M K(l_model; Delta, lam, L_out) S2_sig^2  (M: pooled-patch mixing factor, see below)
    S2_obs  = S2_sig + N2,   S4_obs = S4_sig + 6 S2_sig N2 + N4      (independent white Gaussian fibre
              noise, N2 = f_N S2_obs(l_min), N4 = 3 N2^2; f_N is free because the catalogue vel_err are
              not usable: their median (~2 km/s) exceeds the total velocity scatter of some fields)
Covariance: jackknife (shrunk) + realisation scatter of the ensemble-mean formula (8-seed table,
correlation exp(-|dlog10 l| / RHO_DEX) across bins) + model error (SIG_MOD_K, SIG_MOD_LNS2) on the diagonal.
Likelihood Gaussian with the theta-dependent covariance (log-det included).

Priors: log Delta U[log 0.005, 0]; log lam U[-1.5, 2.5] (default; --lambda-prior f instead makes f0 =
lam^2/(lam^2 + 1.7^2), the bulk share of the variance, uniform, which peaks at lam = 1.7 in log lam); log L_out U[log 0.05, log 20];
log v_turb U[-1, 2.5] (km/s); M U[1, M_measured] (--mixing marginal, default), M = 1 (none) or
M = M_measured (measured); log s_r ~ N(0, 0.2 dex) truncated at +-0.4 dex; f_N U[0, 0.95].

Sampler: Goodman & Weare (2010) affine-invariant stretch move (emcee's algorithm, no dependency).

Outputs (Figures/manuscript/08_shell_inference/formula_fit/):
  formula_fit_summary.csv          posterior medians and 16-84% ranges, chi2 at the MAP, flags
  data/<region>_<LINE>_formula_data.npz   measured curves + replicates (cache, step "measure")
  <region>_<LINE>_formula_chain.npz       posterior samples (formula_fit_K_only/ with --observables K)
  formula_fit_curves_K.png, formula_fit_curves_S2.png   data with posterior-predictive bands
  formula_fit_probability_maps.png (Delta, lam) 68/95% regions and most probable point per tracer
  formula_fit_vs_slope_method.png  Delta and lam vs the old slope-method fits

POINT-SAMPLE MODEL: the formula is for point samples; an LVM fibre is a line-of-sight centroid. Treat
these fits as a test of the machinery until the pixelised formula exists.

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/fit_kurtosis_formula.py measure [--region R --line nii] [--jobs 4]
    python manuscript_pipeline/08_shell_inference/fit_kurtosis_formula.py fit [--region R] [--mixing marginal]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from kurtosis_formula_tables import FormulaModel  # noqa: E402

OUT_DIR = Path("Figures/manuscript/08_shell_inference/formula_fit")
DATA_DIR = OUT_DIR / "data"
MULTIFLOOR_DIR = Path("Figures/manuscript/08_shell_inference/data")
OLD_SUMMARY = Path("Figures/manuscript/08_shell_inference/shell_inference_summary.csv")
LINES = ("NII", "SII", "SIII")

BASE_FLOOR_ARCSEC, SNR_MIN = 37.0, 20.0
N_PER_DEX, L_MAX_OBS = 6, 0.8
MIN_PAIRS = 1000
RHO_DEX = 1.0
SIG_MOD_K, SIG_MOD_LNS2 = 0.10, 0.05
LAMBDA_C = 1.7
LAMBDA_PRIOR = "loglam"
PRIOR = {"logD": (np.log10(0.005), 0.0), "logLam": (-1.5, 2.5), "logL": (np.log10(0.05), np.log10(20.0)),
         "logV": (-1.0, 2.5), "logS": (-0.4, 0.4), "fN": (0.0, 0.95)}
SIG_LOGS = 0.2
PNAMES = ["logD", "logLam", "logL", "logV", "M", "logS", "fN"]


# ===================================================================== measurement
def measure_one(region: str, line: str, force: bool = False) -> str:
    from vsf_module import VSF2D, get_hii_radius_pc, load_line_data, vsf_orders_block_jackknife

    out = DATA_DIR / f"{region}_{line.upper()}_formula_data.npz"
    if out.exists() and not force:
        return f"{out} exists"
    fits_file = f"LVM_{region}.fits"
    d = load_line_data(fits_file, line.lower(), snr_min=SNR_MIN, subtract_gradient=True)
    v2 = VSF2D(d["ra"], d["dec"], d["vel"], dist=d["dist"], vel_err=d["vel_err"])
    x, y = v2.x_physical, v2.y_physical
    r_hii = get_hii_radius_pc(fits_file)
    floor = d["dist"] * np.deg2rad(BASE_FLOOR_ARCSEC / 3600.0) / r_hii
    if floor >= L_MAX_OBS:
        return f"{region} {line}: floor {floor:.3f} r_HII >= {L_MAX_OBS}; skipped"
    n_bins = max(int(np.ceil(N_PER_DEX * np.log10(L_MAX_OBS / floor))), 1)
    edges = floor * (L_MAX_OBS / floor) ** (np.arange(n_bins + 1) / n_bins) * r_hii
    S, _err, reps, counts = vsf_orders_block_jackknife(x, y, d["vel"], edges, orders=(2, 4))
    from scipy.spatial import ConvexHull

    fov = 2 * np.sqrt(ConvexHull(np.column_stack((x, y))).volume / np.pi) / r_hii  # equal-area diameter
    sig2 = np.asarray(d["vel_err"], float) ** 2
    N2 = 2 * sig2.mean()
    N4 = 3 * (2 * np.mean(sig2 ** 2) + 2 * sig2.mean() ** 2)
    mf = MULTIFLOOR_DIR / f"{region}_{line.upper()}_multifloor.npz"
    M = float(np.load(mf)["mixing_factor"]) if mf.exists() else np.nan
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out, region=region, line=line.upper(), edges_over_rhii=edges / r_hii, S2=S[0], S4=S[1],
             counts=counts, rep_S2=reps[:, 0], rep_S4=reps[:, 1], N2=N2, N4=N4, mixing_factor=M,
             r_hii_pc=r_hii, fov_over_rhii=fov, distance_pc=d["dist"], floor_over_rhii=floor, n_fibres=len(x), snr_min=SNR_MIN)
    return f"{region} {line}: {len(x)} fibres, {n_bins} bins from l/R_HII={floor:.3f}, {reps.shape[0]} reps -> {out}"


def all_targets():
    """(region, LINE) for every <region>_<LINE>_multifloor.npz of compute_multifloor_slopes.py."""
    return sorted(tuple(p.stem.rsplit("_", 2)[:2]) for p in MULTIFLOOR_DIR.glob("*_multifloor.npz"))


# ===================================================================== data vector and covariance
def shrink_to_diagonal(dev):
    """Schafer-Strimmer target-D shrinkage of the jackknife covariance from replicate deviations (n x p)."""
    n = dev.shape[0]
    C = (n - 1) / n * dev.T @ dev
    sd = np.sqrt(np.diag(C))
    z = dev / sd
    w = z[:, :, None] * z[:, None, :]
    r = w.mean(0) * n / (n - 1) if n > 1 else w.mean(0)
    var_r = n / (n - 1) ** 3 * ((w - w.mean(0)) ** 2).sum(0)
    off = ~np.eye(dev.shape[1], dtype=bool)
    lam = float(np.clip(var_r[off].sum() / max((r[off] ** 2).sum(), 1e-300), 0, 1))
    R = np.where(off, (1 - lam) * np.clip(r, -1, 1), 1.0)
    return R * np.outer(sd, sd), lam


def load_data(region, line):
    p = DATA_DIR / f"{region}_{line}_formula_data.npz"
    if not p.exists():
        return None
    d = dict(np.load(p))
    e = d["edges_over_rhii"]
    ell = np.sqrt(e[:-1] * e[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        K = d["S4"] / d["S2"] ** 2
        Kr = d["rep_S4"] / d["rep_S2"] ** 2
        lS, lSr = np.log(d["S2"]), np.log(d["rep_S2"])
    use = (d["counts"] >= MIN_PAIRS) & np.isfinite(K) & np.all(np.isfinite(Kr), 0) & np.all(np.isfinite(lSr), 0)
    if use.sum() < 3:
        return None
    reps = np.hstack((Kr[:, use], lSr[:, use]))
    C, shrink = shrink_to_diagonal(reps - reps.mean(0))
    return {"region": region, "line": line, "ell": ell[use], "K": K[use], "lnS2": lS[use],
            "y": np.concatenate((K[use], lS[use])), "C_jk": C, "shrink": shrink, "n_rep": reps.shape[0],
            "N2": float(d["N2"]), "N4": float(d["N4"]), "M_meas": float(d["mixing_factor"]),
            "r_hii_pc": float(d["r_hii_pc"]), "floor": float(d["floor_over_rhii"]), "n_fibres": int(d["n_fibres"])}


# ===================================================================== likelihood
class Posterior:
    def __init__(self, data, model: FormulaModel, mixing="marginal", realisation=True, use_s2=True):
        self.d, self.m, self.realisation, self.use_s2 = data, model, realisation, use_s2
        self.n = len(data["ell"])
        self.keep = np.arange(2 * self.n if use_s2 else self.n)  # K only: drop the ln S2 half of y
        lx = np.log10(data["ell"])
        self.rho = np.exp(-np.abs(lx[:, None] - lx[None, :]) / RHO_DEX)
        M = data["M_meas"] if np.isfinite(data["M_meas"]) else 1.0
        self.M_range = {"marginal": (1.0, max(M, 1.0 + 1e-6)), "none": (1.0, 1.0), "measured": (M, M)}[mixing]
        self.C0 = data["C_jk"] + np.diag(np.r_[np.full(self.n, SIG_MOD_K ** 2), np.full(self.n, SIG_MOD_LNS2 ** 2)])
        self.C0 = self.C0[np.ix_(self.keep, self.keep)]
        self.y = data["y"][self.keep]

    def unpack(self, th):
        th = np.atleast_2d(th)
        return {k: th[:, i] for i, k in enumerate(PNAMES)}

    def log_prior(self, th):
        p = self.unpack(th)
        ok = np.ones(len(th), bool)
        for k, (lo, hi) in PRIOR.items():
            ok &= (p[k] >= lo) & (p[k] <= hi)
        Mlo, Mhi = self.M_range
        ok &= (p["M"] >= Mlo - 1e-12) & (p["M"] <= Mhi + 1e-12)
        lp = -0.5 * (p["logS"] / SIG_LOGS) ** 2
        if LAMBDA_PRIOR == "f":  # uniform in f0 = lam^2 / (lam^2 + lam_c^2): peaks at lam_c in log lam
            lam2 = 10 ** (2 * p["logLam"])
            f0 = lam2 / (lam2 + LAMBDA_C ** 2)
            lp = lp + np.log(np.clip(f0 * (1 - f0), 1e-300, None))
        return np.where(ok, lp, -np.inf)

    def predict(self, th):
        p = self.unpack(th)
        delta, lam, L = (10 ** p[k][:, None] for k in ("logD", "logLam", "logL"))
        ell = self.d["ell"][None, :] / 10 ** p["logS"][:, None]
        s2u, Ku = self.m.predict(ell, delta, lam, L)
        S2s = 10 ** (2 * p["logV"][:, None]) * s2u
        S4s = p["M"][:, None] * Ku * S2s ** 2
        N2 = p["fN"][:, None] * np.exp(self.d["lnS2"][0])
        N4 = 3 * N2 ** 2
        S2o = S2s + N2
        Ko = (S4s + 6 * S2s * N2 + N4) / S2o ** 2
        return Ko, np.log(S2o), (ell, delta, lam)

    def log_like(self, th):
        Ko, lS, (ell, delta, lam) = self.predict(th)
        r = self.y[None, :] - np.hstack((Ko, lS))[:, self.keep]
        C = np.broadcast_to(self.C0, (len(r),) + self.C0.shape).copy()
        if self.realisation:
            sK, sS = self.m.realisation_sigma(ell, delta, lam)
            n = self.n
            C[:, :n, :n] += sK[:, :, None] * sK[:, None, :] * self.rho
            if self.use_s2:
                C[:, n:, n:] += sS[:, :, None] * sS[:, None, :] * self.rho
        L = np.linalg.cholesky(C)
        z = np.linalg.solve(L, r[:, :, None])[:, :, 0]
        return -0.5 * (z ** 2).sum(1) - np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)

    def log_post(self, th):
        lp = self.log_prior(th)
        out = np.full(len(lp), -np.inf)
        ok = np.isfinite(lp)
        if ok.any():
            out[ok] = lp[ok] + self.log_like(th[ok])
        return out

    def chi2(self, th):
        """r^T C^-1 r at th (one point), and the data dimension."""
        Ko, lS, (ell, delta, lam) = self.predict(th)
        r = self.y - np.r_[Ko[0], lS[0]][self.keep]
        C = self.C0.copy()
        if self.realisation:
            sK, sS = self.m.realisation_sigma(ell, delta, lam)
            n = self.n
            C[:n, :n] += np.outer(sK[0], sK[0]) * self.rho
            if self.use_s2:
                C[n:, n:] += np.outer(sS[0], sS[0]) * self.rho
        return float(r @ np.linalg.solve(C, r)), len(r)

    def prior_draws(self, n, rng):
        th = np.empty((n, len(PNAMES)))
        for i, k in enumerate(PNAMES):
            lo, hi = self.M_range if k == "M" else PRIOR[k]
            th[:, i] = rng.uniform(lo, hi, n)
        return th


def stretch_sampler(log_post, p0, n_steps, rng, a=2.0):
    """Goodman & Weare affine-invariant ensemble sampler, vectorised over half-ensembles."""
    nw, nd = p0.shape
    pos, lp = p0.copy(), log_post(p0)
    chain = np.empty((n_steps, nw, nd))
    lnp = np.empty((n_steps, nw))
    acc = 0
    half = [np.arange(0, nw // 2), np.arange(nw // 2, nw)]
    for t in range(n_steps):
        for s in (0, 1):
            act, oth = half[s], half[1 - s]
            z = ((a - 1) * rng.random(len(act)) + 1) ** 2 / a
            partner = pos[rng.choice(oth, len(act))]
            prop = partner + z[:, None] * (pos[act] - partner)
            # fixed parameters (zero prior width) stay fixed
            lp_new = log_post(prop)
            accept = np.log(rng.random(len(act))) < (nd - 1) * np.log(z) + lp_new - lp[act]
            pos[act[accept]], lp[act[accept]] = prop[accept], lp_new[accept]
            acc += accept.sum()
        chain[t], lnp[t] = pos, lp
    return chain, lnp, acc / (n_steps * nw)


def fit_one(data, model, mixing, realisation, use_s2=True, n_walkers=64, n_steps=4000, burn=2000, seed=0):
    post = Posterior(data, model, mixing, realisation, use_s2)
    rng = np.random.default_rng(seed)
    draws = post.prior_draws(20000, rng)
    lp = post.log_post(draws)
    start = draws[np.argsort(lp)[::-1][:n_walkers]]
    fixed_M = post.M_range[0] == post.M_range[1]
    if fixed_M:  # sample the remaining parameters only
        iM = PNAMES.index("M")
        free = [i for i in range(len(PNAMES)) if i != iM]

        def lpf(t):
            full = np.insert(t, iM, post.M_range[0], axis=1)
            return post.log_post(full)

        ch, lnp, acc = stretch_sampler(lpf, start[:, free], n_steps, rng)
        ch = np.insert(ch, iM, post.M_range[0], axis=2)
    else:
        ch, lnp, acc = stretch_sampler(post.log_post, start, n_steps, rng)
    samples = ch[burn::10].reshape(-1, len(PNAMES))
    lps = lnp[burn::10].ravel()
    best = ch.reshape(-1, len(PNAMES))[np.argmax(lnp)]
    return post, samples, lps, best, acc


# ===================================================================== summary
def summarise(data, post, samples, best, acc):
    q = lambda v: np.percentile(v, [16, 50, 84])  # noqa: E731
    p = {k: samples[:, i] for i, k in enumerate(PNAMES)}
    derived = {"delta": 10 ** p["logD"], "lam": 10 ** p["logLam"], "l_out": 10 ** p["logL"],
               "v_turb": 10 ** p["logV"], "v_bulk": 10 ** (p["logV"] + p["logLam"]), "M": p["M"],
               "r_hii_scale": 10 ** p["logS"], "noise_frac": p["fN"]}
    chi2, n = post.chi2(best)
    res = {"region": data["region"], "line": data["line"], "n_fibres": data["n_fibres"], "n_bins": len(data["ell"]),
           "l_min": data["ell"].min(), "l_max": data["ell"].max(), "n_jackknife": data["n_rep"],
           "shrinkage": data["shrink"], "M_meas": data["M_meas"], "catalogue_noise_frac": data["N2"] / np.exp(data["lnS2"][0]),
           "chi2_map": chi2, "n_data": n, "acceptance": acc}
    for k, v in derived.items():
        lo, med, hi = q(v)
        res[f"{k}_p16"], res[f"{k}_p50"], res[f"{k}_p84"] = lo, med, hi
    flags = []
    for k, key in (("logD", "Delta"), ("logLam", "lambda"), ("logL", "L_out")):
        lo, hi = PRIOR[k]
        p16, p84 = np.percentile(p[k], [16, 84])
        span = hi - lo
        if p84 > hi - 0.05 * span and p16 > lo + 0.25 * span:
            flags.append(f"{key}: lower limit only")
        elif p16 < lo + 0.05 * span and p84 < hi - 0.25 * span:
            flags.append(f"{key}: upper limit only")
        elif p84 - p16 > 0.6 * span:
            flags.append(f"{key}: unconstrained")
    if chi2 > n + 3 * np.sqrt(2 * n):
        flags.append("poor fit (chi2 > n + 3 sqrt(2n))")
    res["flags"] = "; ".join(flags)
    return res


# ===================================================================== figures
def figures(results, store):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vsf_module import LINE_COLORS, LINE_LABELS

    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    regions = sorted({r["region"] for r in results})
    n_cols = 4
    n_rows = int(np.ceil(len(regions) / n_cols))

    for which in ("K", "S2"):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows), squeeze=False)
        for ax, region in zip(axes.flat, regions):
            for line in LINES:
                if (region, line) not in store:
                    continue
                data, post, samples = store[(region, line)]
                c = LINE_COLORS[line]
                n = len(data["ell"])
                err = np.sqrt(np.diag(data["C_jk"]))
                sub = samples[np.random.default_rng(0).choice(len(samples), min(300, len(samples)), replace=False)]
                Ko, lS, _ = post.predict(sub)
                if which == "K":
                    ax.errorbar(data["ell"], data["K"], err[:n], fmt="o", ms=4, color=c, capsize=2,
                                label=LINE_LABELS[line])
                    band = np.percentile(Ko, [16, 50, 84], axis=0)
                else:
                    ax.errorbar(data["ell"], np.exp(data["lnS2"]), np.exp(data["lnS2"]) * err[n:], fmt="o", ms=4,
                                color=c, capsize=2, label=LINE_LABELS[line])
                    band = np.percentile(np.exp(lS), [16, 50, 84], axis=0)
                ax.fill_between(data["ell"], band[0], band[2], color=c, alpha=0.25, lw=0)
                ax.plot(data["ell"], band[1], color=c, lw=1.3)
            ax.set_xscale("log")
            if which == "S2":
                ax.set_yscale("log")
            else:
                ax.axhline(3, color="0.5", ls="--", lw=0.8)
            ax.set_title(region.replace("_", " "), fontsize=11)
            ax.legend(fontsize=7.5)
        for ax in axes.flat[len(regions):]:
            ax.axis("off")
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\ell / r_{\rm HII}$ (catalogue)")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$K(\ell)$" if which == "K" else r"$S_2(\ell)$ [km$^2$ s$^{-2}$]")
        fig.suptitle(f"Observed {which} with formula posterior-predictive median and 16-84\\% band "
                     "(error bars: jackknife only)", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"formula_fit_curves_{which}.png", dpi=160)
        plt.close(fig)

    probability_maps({k: v[2] for k, v in store.items()})

    if OLD_SUMMARY.exists():
        old = {(r["region"], r["line"]): r for r in csv.DictReader(open(OLD_SUMMARY)) if r["status"] == "ok"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for res in results:
            o = old.get((res["region"], res["line"]))
            if o is None:
                continue
            c = LINE_COLORS[res["line"]]
            for ax, new, oldk in ((axes[0], "delta", "w"), (axes[1], "lam", "v")):
                x, xl, xh = float(o[f"best_{oldk}"]), float(o[f"{oldk}_1sig_lo"]), float(o[f"{oldk}_1sig_hi"])
                y, yl, yh = res[f"{new}_p50"], res[f"{new}_p16"], res[f"{new}_p84"]
                ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt="o", color=c, ms=5,
                            capsize=2, alpha=0.8)
        for ax, lab in zip(axes, (r"$\Delta r_{\rm shell}/r_{\rm HII}$", r"$v_{\rm bulk}/v_{\rm turb}$")):
            ax.set(xscale="log", yscale="log", xlabel=f"slope method (best, $\\Delta\\chi^2<1$): {lab}",
                   ylabel=f"formula fit (median, 16-84%): {lab}")
            lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lim, lim, "k--", lw=0.8)
        for line in LINES:
            axes[0].plot([], [], "o", color=LINE_COLORS[line], label=LINE_LABELS[line])
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "formula_fit_vs_slope_method.png", dpi=160)
        plt.close(fig)


MAP_BINS, MAP_SMOOTH = 60, 1.5
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}


def hpd_levels(p, fracs=(0.68, 0.95)):
    """Density thresholds enclosing the given probability fractions of a normalised 2D map."""
    s = np.sort(p.ravel())[::-1]
    c = np.cumsum(s)
    return [s[min(np.searchsorted(c, f), len(s) - 1)] for f in fracs]


def probability_maps(chains):
    """Per region: joint (Delta, lam) posterior of each tracer, marginalised over the nuisances.
    Filled = 68% highest-density region, thin line = 95%, star = most probable point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from vsf_module import LINE_COLORS, LINE_LABELS

    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    iD, iL = PNAMES.index("logD"), PNAMES.index("logLam")
    xe = np.linspace(*PRIOR["logD"], MAP_BINS + 1)
    ye = np.linspace(*PRIOR["logLam"], MAP_BINS + 1)
    X, Y = np.meshgrid(10 ** (0.5 * (xe[1:] + xe[:-1])), 10 ** (0.5 * (ye[1:] + ye[:-1])), indexing="ij")
    regions = sorted({r for r, _ in chains})
    n_cols = 4
    n_rows = int(np.ceil(len(regions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.9 * n_rows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, region in zip(axes.flat, regions):
        for line in LINES:
            s = chains.get((region, line))
            if s is None:
                continue
            H, _, _ = np.histogram2d(s[:, iD], s[:, iL], bins=(xe, ye))
            P = gaussian_filter(H, MAP_SMOOTH, mode="nearest")
            P /= P.sum()
            l68, l95 = hpd_levels(P)
            c, ls = LINE_COLORS[line], LINESTYLES[line]
            ax.contourf(X, Y, P, levels=[l68, P.max() * 1.01], colors=[c], alpha=0.25)
            ax.contour(X, Y, P, levels=[l95, l68], colors=[c], linestyles=[ls], linewidths=[0.8, 1.5])
            ib = np.unravel_index(np.argmax(P), P.shape)
            ax.plot(X[ib], Y[ib], "*", ms=11, color=c, mec="white", mew=0.6, label=LINE_LABELS[line])
        ax.set(xscale="log", yscale="log", xlim=(10 ** xe[0], 10 ** xe[-1]), ylim=(10 ** ye[0], 10 ** ye[-1]))
        ax.set_title(region.replace("_", " "), fontsize=11)
        ax.legend(fontsize=7.5, loc="upper right")
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$v_{\rm bulk}/v_{\rm turb}$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    fig.suptitle(r"Formula fit: filled = 68\% highest-probability region, thin line = 95\%, star = most probable "
                 r"(marginalised over $L_{\rm out}$, $v_{\rm turb}$, $M$, $r_{\rm HII}$ scale, noise)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "formula_fit_probability_maps.png", dpi=180)
    plt.close(fig)


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("measure", "fit", "maps"),
                    help="maps: redraw the probability maps from the saved chains (no refit)")
    ap.add_argument("--lambda-prior", choices=("loglam", "f"), default="loglam",
                    help="loglam: flat in log lam (maps follow the likelihood); f: flat in the bulk variance share")
    ap.add_argument("--region")
    ap.add_argument("--line", help="nii, sii or siii (default: all available)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mixing", choices=("marginal", "none", "measured"), default="marginal")
    ap.add_argument("--no-realisation", action="store_true", help="omit the field-seed realisation scatter")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--observables", choices=("K+S2", "K"), default="K+S2",
                    help="K only: v_turb and f_N then only enter through the noise term of K")
    args = ap.parse_args()

    targets = [(r, l) for r, l in all_targets()
               if (args.region is None or r == args.region) and (args.line is None or l == args.line.upper())]
    if args.step == "measure":
        from joblib import Parallel, delayed

        for msg in Parallel(n_jobs=args.jobs)(delayed(measure_one)(r, l, args.force) for r, l in targets):
            print(msg, flush=True)
        return

    global OUT_DIR, LAMBDA_PRIOR
    LAMBDA_PRIOR = args.lambda_prior
    if args.observables == "K":
        OUT_DIR = OUT_DIR.parent / "formula_fit_K_only"
    if args.step == "maps":
        chains = {(r, l): np.load(p)["samples"] for r, l in targets
                  if (p := OUT_DIR / f"{r}_{l}_formula_chain.npz").exists()}
        probability_maps(chains)
        print(f"wrote {OUT_DIR / 'formula_fit_probability_maps.png'}")
        return
    model = FormulaModel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, store = [], {}
    for region, line in targets:
        data = load_data(region, line)
        if data is None:
            print(f"{region:15s} {line:4s} no usable data")
            continue
        post, samples, lps, best, acc = fit_one(data, model, args.mixing, not args.no_realisation,
                                                args.observables == "K+S2", n_steps=args.steps,
                                                burn=args.steps // 2)
        res = summarise(data, post, samples, best, acc)
        results.append(res)
        store[(region, line)] = (data, post, samples)
        np.savez(OUT_DIR / f"{region}_{line}_formula_chain.npz", samples=samples, log_post=lps, names=PNAMES,
                 best=best, mixing=args.mixing, realisation=not args.no_realisation, lambda_prior=LAMBDA_PRIOR)
        print(f"{region:15s} {line:4s} n={res['n_bins']:2d} Delta={res['delta_p50']:.3f} "
              f"[{res['delta_p16']:.3f},{res['delta_p84']:.3f}] lam={res['lam_p50']:.2f} "
              f"[{res['lam_p16']:.2f},{res['lam_p84']:.2f}] L_out={res['l_out_p50']:.2f} "
              f"v_turb={res['v_turb_p50']:.1f} chi2={res['chi2_map']:.1f}/{res['n_data']} acc={acc:.2f} "
              f"{res['flags']}", flush=True)
    if not results:
        return
    suffix = "" if (args.region is None and args.line is None) else f"_{args.region or 'all'}_{args.line or 'all'}"
    cols = list(results[0])
    with open(OUT_DIR / f"formula_fit_summary{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])
    if not suffix:
        figures(results, store)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
