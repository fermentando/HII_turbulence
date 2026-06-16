from __future__ import annotations

import json
from pathlib import Path
import re
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import (
    fit_bpl,
    fit_lognormal,
    fit_pl_with_posterior,
    histogram_counts,
    infer_xmin_pl,
    log_edges,
    loglike_multinomial,
    pdf_bpl,
    pdf_lognormal,
    pdf_pl,
    pearson_reduced_chi2,
    pi_bpl,
    pi_lognormal,
    pi_pl,
    tail_bin_view,
    weighted_quantile,
)


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _aic_bic(logl: float, k: int, n: int) -> tuple[float, float]:
    aic = 2.0 * k - 2.0 * logl
    bic = k * np.log(max(n, 1)) - 2.0 * logl
    return float(aic), float(bic)


def _snapshot_number(key: str, fallback: int) -> int:
    m = re.search(r"step(\d+)", key)
    if m:
        return int(m.group(1))
    return fallback


def _p_err(d: dict[str, float]) -> tuple[float, float, float]:
    p16 = float(d["p16"])
    p50 = float(d["p50"])
    p84 = float(d["p84"])
    return p50, p50 - p16, p84 - p50


def _plot_overview_params(results: list[dict[str, Any]], out: Path) -> Path:
    order = np.argsort([r["snapshot"] for r in results])
    x = np.array([results[i]["snapshot"] for i in order], dtype=float)

    xmin_mid = []
    xmin_lo = []
    xmin_hi = []
    for i in order:
        xsel = float(results[i]["xmin"]["selected"])
        xp16 = float(results[i]["xmin"]["p16"])
        xp84 = float(results[i]["xmin"]["p84"])
        xmin_mid.append(xsel)
        xmin_lo.append(max(xsel - xp16, 0.0))
        xmin_hi.append(max(xp84 - xsel, 0.0))
    pl_mid, pl_lo, pl_hi = zip(*[_p_err(results[i]["PL"]["alpha"]) for i in order])
    a1_mid, a1_lo, a1_hi = zip(*[_p_err(results[i]["BPL"]["alpha1"]) for i in order])
    a2_mid, a2_lo, a2_hi = zip(*[_p_err(results[i]["BPL"]["alpha2"]) for i in order])
    xb_mid, xb_lo, xb_hi = zip(*[_p_err(results[i]["BPL"]["xb"]) for i in order])
    mu_mid, mu_lo, mu_hi = zip(*[_p_err(results[i]["Lognormal"]["mu"]) for i in order])
    sg_mid, sg_lo, sg_hi = zip(*[_p_err(results[i]["Lognormal"]["sigma"]) for i in order])

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    ax = axes.ravel()
    panels = [
        ("xmin", xmin_mid, xmin_lo, xmin_hi, True),
        ("PL alpha", pl_mid, pl_lo, pl_hi, False),
        ("BPL alpha1", a1_mid, a1_lo, a1_hi, False),
        ("BPL alpha2", a2_mid, a2_lo, a2_hi, False),
        ("BPL xb", xb_mid, xb_lo, xb_hi, True),
        ("LN mu", mu_mid, mu_lo, mu_hi, False),
        ("LN sigma", sg_mid, sg_lo, sg_hi, False),
    ]
    for i, (name, y, yl, yu, logy) in enumerate(panels):
        ax[i].errorbar(x, y, yerr=np.vstack([yl, yu]), fmt="o-", ms=3, lw=1.2, capsize=2)
        if logy:
            ax[i].set_yscale("log")
        ax[i].set_ylabel(name)
        ax[i].grid(True, alpha=0.25)
    ax[7].axis("off")
    for i in [6]:
        ax[i].set_xlabel("snapshot")
    fig.suptitle("Best-fit parameters with 16/84% intervals vs snapshot", y=0.995)
    fig.tight_layout()
    out_path = out / "overview_bestfit_vs_snapshot.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_overview_delta_aic(results: list[dict[str, Any]], out: Path) -> Path:
    order = np.argsort([r["snapshot"] for r in results])
    x = np.array([results[i]["snapshot"] for i in order], dtype=float)
    daic_bpl = np.array([results[i]["BPL"]["AIC"] - results[i]["PL"]["AIC"] for i in order], dtype=float)
    daic_ln = np.array([results[i]["Lognormal"]["AIC"] - results[i]["PL"]["AIC"] for i in order], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, daic_bpl, "o-", label="BPL - PL")
    ax.plot(x, daic_ln, "o-", label="Lognormal - PL")
    ax.axhline(0.0, color="k", lw=1.0, ls="--")
    ax.set_xlabel("snapshot")
    ax.set_ylabel("Delta AIC (vs PL)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = out / "overview_delta_aic_vs_pl_vs_snapshot.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _weighted_qdict(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "p16": float(weighted_quantile(values, weights, 0.16)),
        "p50": float(weighted_quantile(values, weights, 0.50)),
        "p84": float(weighted_quantile(values, weights, 0.84)),
    }


def fit_one_dataset(
    key: str,
    values: np.ndarray,
    output_dir: Path,
    edges: np.ndarray,
    snapshot: int,
    xmin_mode: str = "smallest_good",
    xmin_softcap: float = 200.0,
    xmin_gamma: float = 2.0,
    xmin_max: float | None = None,
    xmin_pthresh: float = 0.1,
    xmin_candidates: int = 50,
    xmin_min_tail_count: int = 200,
    xmin_lambda_discard: float = 0.5,
    xmin_bootstrap_sims: int = 40,
    propagate_xmin_uncertainty: bool = True,
    bpl_xb_prior_min_ratio: float = 1.0,
    bpl_xb_prior_strength: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    if verbose:
        print(f"[tailfit] fitting {key} (snapshot={snapshot})")
    counts = histogram_counts(values.astype(int), edges)
    n_total = int(counts.sum())

    xmin_post = infer_xmin_pl(
        edges,
        counts,
        mode=xmin_mode,
        soft_cap=xmin_softcap,
        gamma=xmin_gamma,
        xmin_max=xmin_max,
        pthresh=xmin_pthresh,
        xmin_candidates=xmin_candidates,
        min_tail_count=xmin_min_tail_count,
        lambda_discard=xmin_lambda_discard,
        bootstrap_sims=xmin_bootstrap_sims,
    )
    pl_post = fit_pl_with_posterior(xmin_post)
    xmin = float(pl_post["xmin"]["selected"])
    tail_l, tail_u, tail_k, tail_i0 = tail_bin_view(edges, counts, xmin)
    n_tail = int(tail_k.sum())

    alpha_sel = float(pl_post["alpha"]["selected_conditional"])
    pi_model_pl = pi_pl(alpha_sel, xmin, tail_l, tail_u)
    logl_pl = loglike_multinomial(tail_k, pi_model_pl)
    aic_pl, bic_pl = _aic_bic(logl_pl, k=1, n=n_tail)
    rchi2_pl = pearson_reduced_chi2(tail_k, pi_model_pl, n_params=1)

    bpl = fit_bpl(
        xmin,
        tail_l,
        tail_u,
        tail_k,
        xb_prior_min_ratio=bpl_xb_prior_min_ratio,
        xb_prior_strength=bpl_xb_prior_strength,
    )
    a1 = float(bpl["params"]["alpha1"])
    a2 = float(bpl["params"]["alpha2"])
    xb = float(bpl["params"]["xb"])
    pi_model_bpl = pi_bpl(a1, a2, xb, xmin, tail_l, tail_u)
    logl_bpl = float(bpl["logL"])
    aic_bpl, bic_bpl = _aic_bic(logl_bpl, k=3, n=n_tail)
    rchi2_bpl = pearson_reduced_chi2(tail_k, pi_model_bpl, n_params=3)
    ln = fit_lognormal(xmin, tail_l, tail_u, tail_k)
    mu = float(ln["params"]["mu"])
    sigma = float(ln["params"]["sigma"])
    pi_model_ln = pi_lognormal(mu, sigma, xmin, tail_l, tail_u)
    logl_ln = float(ln["logL"])
    aic_ln, bic_ln = _aic_bic(logl_ln, k=2, n=n_tail)
    rchi2_ln = pearson_reduced_chi2(tail_k, pi_model_ln, n_params=2)

    # In ks_min mode, propagate xmin uncertainty into BPL/LN parameter intervals.
    if xmin_mode == "ks_min" and propagate_xmin_uncertainty:
        cand = np.array(pl_post["xmin"]["candidates"], dtype=float)
        w = np.array(pl_post["xmin"]["weights"], dtype=float)
        idx_sorted = np.argsort(w)[::-1]
        idx_keep = []
        w_keep = []
        csum = 0.0
        for i in idx_sorted:
            if w[i] <= 0.0:
                continue
            idx_keep.append(int(i))
            w_keep.append(float(w[i]))
            csum += float(w[i])
            if len(idx_keep) >= 4 or csum >= 0.85:
                break
        if len(idx_keep) >= 2:
            wv = np.array(w_keep, dtype=float)
            wv = wv / wv.sum()
            a1v, a2v, xbv, muv, sigv, w_used = [], [], [], [], [], []
            for j, wi in zip(idx_keep, wv):
                xm = float(cand[j])
                tlj, tuj, tkj, _ = tail_bin_view(edges, counts, xm)
                if int(tkj.sum()) < xmin_min_tail_count:
                    continue
                bj = fit_bpl(
                    xm,
                    tlj,
                    tuj,
                    tkj,
                    n_draws=120,
                    xb_prior_min_ratio=bpl_xb_prior_min_ratio,
                    xb_prior_strength=bpl_xb_prior_strength,
                )
                lj = fit_lognormal(xm, tlj, tuj, tkj, n_draws=120)
                a1v.append(float(bj["params"]["alpha1"]))
                a2v.append(float(bj["params"]["alpha2"]))
                xbv.append(float(bj["params"]["xb"]))
                muv.append(float(lj["params"]["mu"]))
                sigv.append(float(lj["params"]["sigma"]))
                w_used.append(float(wi))
            if len(a1v) >= 2:
                avw = np.array(w_used, dtype=float)
                avw = avw / avw.sum()
                bpl["uncertainty"]["alpha1"] = _weighted_qdict(np.array(a1v, dtype=float), avw)
                bpl["uncertainty"]["alpha2"] = _weighted_qdict(np.array(a2v, dtype=float), avw)
                bpl["uncertainty"]["xb"] = _weighted_qdict(np.array(xbv, dtype=float), avw)
                ln["uncertainty"]["mu"] = _weighted_qdict(np.array(muv, dtype=float), avw)
                ln["uncertainty"]["sigma"] = _weighted_qdict(np.array(sigv, dtype=float), avw)

    widths = tail_u.copy()
    widths[-1] = edges[-1]
    widths = widths - tail_l
    pdf_hat_tail = tail_k / max(n_tail, 1) / widths
    c_tail = np.sqrt(tail_l * np.where(np.isfinite(tail_u), tail_u, edges[-1]))

    dataset_dir = output_dir / _safe_key(key)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    xg = np.logspace(np.log10(max(xmin, 1.0)), np.log10(edges[-1]), 500)
    ypl = xg * xg * pdf_pl(xg, alpha_sel, xmin)
    ybpl = xg * xg * pdf_bpl(xg, a1, a2, xb, xmin)
    yln = xg * xg * pdf_lognormal(xg, mu, sigma, xmin)

    width_all = edges[1:] - edges[:-1]
    # Use the same N_tail normalization as tail bins so both overlays share height scale.
    pdf_all = counts / max(n_tail, 1) / width_all
    c_all = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(c_all, c_all * c_all * pdf_all, s=10, c="0.75", label="all bins")
    ax.scatter(c_tail, c_tail * c_tail * pdf_hat_tail, s=18, c="k", label="tail bins")
    ax.plot(xg, ypl, label=f"PL AIC={aic_pl:.1f} BIC={bic_pl:.1f}")
    ax.plot(xg, ybpl, label=f"BPL AIC={aic_bpl:.1f} BIC={bic_bpl:.1f}")
    ax.plot(xg, yln, label=f"LN AIC={aic_ln:.1f} BIC={bic_ln:.1f}")
    ax.axvline(xmin, color="red", ls="--", lw=1.2, label=f"xmin={xmin:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$x^2 p(x)$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    pdf_plot = dataset_dir / "pdf_x2p.png"
    fig.savefig(pdf_plot, dpi=140)
    plt.close(fig)

    model_names = ["PL", "BPL", "Lognormal"]
    aics = np.array([aic_pl, aic_bpl, aic_ln], dtype=float)
    bics = np.array([bic_pl, bic_bpl, bic_ln], dtype=float)
    daic = aics - np.min(aics)
    dbic = bics - np.min(bics)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    axes[0].bar(model_names, daic)
    axes[0].set_title("Delta AIC")
    axes[1].bar(model_names, dbic)
    axes[1].set_title("Delta BIC")
    for a in axes:
        a.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    delta_plot = dataset_dir / "delta_aic_bic.png"
    fig.savefig(delta_plot, dpi=140)
    plt.close(fig)

    ks_plot = None
    if xmin_mode == "ks_min":
        ks_vals = np.array(pl_post["xmin"]["ks_values"], dtype=float)
        x_cand = np.array(pl_post["xmin"]["candidates"], dtype=float)
        i_sel = int(pl_post["xmin"]["selected_index"])
        m = np.isfinite(ks_vals)
        if np.any(m):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x_cand[m], ks_vals[m], "o-", ms=3, lw=1.2, label="KS D(xmin)")
            ax.axvline(xmin, color="red", ls="--", lw=1.2, label=f"selected xmin={xmin:.2f}")
            if 0 <= i_sel < x_cand.size and np.isfinite(ks_vals[i_sel]):
                ax.scatter([x_cand[i_sel]], [ks_vals[i_sel]], c="red", s=35, zorder=4)
            ax.set_xscale("log")
            ax.set_xlabel("xmin candidate")
            ax.set_ylabel("KS distance D")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            ks_plot = dataset_dir / "ks_d_curve.png"
            fig.savefig(ks_plot, dpi=140)
            plt.close(fig)

    result = {
        "key": key,
        "snapshot": snapshot,
        "n_total": n_total,
        "n_tail": n_tail,
        "xmin": pl_post["xmin"],
        "tail": {"bin_start_index": tail_i0, "lower_edges": tail_l.tolist(), "upper_edges": [float(u) for u in tail_u.tolist()]},
        "PL": {
            "alpha": {"p16": pl_post["alpha"]["p16"], "p50": pl_post["alpha"]["p50"], "p84": pl_post["alpha"]["p84"]},
            "params": {"alpha": alpha_sel},
            "logL": logl_pl,
            "AIC": aic_pl,
            "BIC": bic_pl,
            "reduced_chi2": rchi2_pl,
        },
        "BPL": {
            "alpha1": bpl["uncertainty"]["alpha1"],
            "alpha2": bpl["uncertainty"]["alpha2"],
            "xb": bpl["uncertainty"]["xb"],
            "params": {"alpha1": a1, "alpha2": a2, "xb": xb},
            "logL": logl_bpl,
            "AIC": aic_bpl,
            "BIC": bic_bpl,
            "reduced_chi2": rchi2_bpl,
        },
        "Lognormal": {
            "mu": ln["uncertainty"]["mu"],
            "sigma": ln["uncertainty"]["sigma"],
            "params": {"mu": mu, "sigma": sigma},
            "logL": logl_ln,
            "AIC": aic_ln,
            "BIC": bic_ln,
            "reduced_chi2": rchi2_ln,
        },
        "binning": {"nbins": int(edges.size - 1)},
        "xmin_config": {
            "mode": xmin_mode,
            "softcap": xmin_softcap,
            "gamma": xmin_gamma,
            "max": xmin_max,
            "pthresh": xmin_pthresh,
            "candidates": xmin_candidates,
            "min_tail_count": xmin_min_tail_count,
            "lambda_discard": xmin_lambda_discard,
            "bootstrap_sims": xmin_bootstrap_sims,
        },
        "uncertainty_config": {
            "propagate_xmin_uncertainty": propagate_xmin_uncertainty,
        },
        "bpl_config": {
            "xb_prior_min_ratio": bpl_xb_prior_min_ratio,
            "xb_prior_strength": bpl_xb_prior_strength,
        },
        "diagnostics": {
            "tail_counts": tail_k.tolist(),
            "pdf_hat_integral": float(np.sum(pdf_hat_tail * widths)),
            "plots": {
                "pdf_x2p": str(pdf_plot),
                "delta_aic_bic": str(delta_plot),
                "ks_d_curve": (str(ks_plot) if ks_plot is not None else None),
            },
        },
    }

    out_json = output_dir / f"dataset_{_safe_key(key)}.json"
    out_json.write_text(json.dumps(result, indent=2))
    if verbose:
        print(
            "[tailfit] done",
            key,
            f"xmin={xmin:.3g}",
            f"PL(alpha={alpha_sel:.3f}, AIC={aic_pl:.2f})",
            f"BPL(a1={a1:.3f}, a2={a2:.3f}, xb={xb:.3g}, AIC={aic_bpl:.2f})",
            f"LN(mu={mu:.3f}, sigma={sigma:.3f}, AIC={aic_ln:.2f})",
        )
    return result


def run_pipeline(
    input_npz: Path,
    output_dir: Path,
    nbins: int = 50,
    xmin_mode: str = "smallest_good",
    xmin_softcap: float = 200.0,
    xmin_gamma: float = 2.0,
    xmin_max: float | None = None,
    xmin_pthresh: float = 0.1,
    xmin_candidates: int = 50,
    xmin_min_tail_count: int = 200,
    xmin_lambda_discard: float = 0.5,
    xmin_bootstrap_sims: int = 40,
    propagate_xmin_uncertainty: bool = True,
    bpl_xb_prior_min_ratio: float = 1.0,
    bpl_xb_prior_strength: float = 0.0,
    continue_on_error: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = log_edges(1.0, 1.0e8, nbins=nbins)
    results = []
    failures: list[dict[str, Any]] = []
    if verbose:
        print(f"[tailfit] input={input_npz}")
        print(f"[tailfit] output={output_dir}")
    with np.load(input_npz) as data:
        keys = sorted([k for k in data.files if k.startswith("counts__")])
        if verbose:
            print(f"[tailfit] found {len(keys)} counts__ datasets")
        for i, key in enumerate(keys):
            snap = _snapshot_number(key, i)
            try:
                results.append(
                    fit_one_dataset(
                        key,
                        np.asarray(data[key]),
                        output_dir,
                        edges,
                        snapshot=snap,
                        xmin_mode=xmin_mode,
                        xmin_softcap=xmin_softcap,
                        xmin_gamma=xmin_gamma,
                        xmin_max=xmin_max,
                        xmin_pthresh=xmin_pthresh,
                        xmin_candidates=xmin_candidates,
                        xmin_min_tail_count=xmin_min_tail_count,
                        xmin_lambda_discard=xmin_lambda_discard,
                        xmin_bootstrap_sims=xmin_bootstrap_sims,
                        propagate_xmin_uncertainty=propagate_xmin_uncertainty,
                        bpl_xb_prior_min_ratio=bpl_xb_prior_min_ratio,
                        bpl_xb_prior_strength=bpl_xb_prior_strength,
                        verbose=verbose,
                    )
                )
            except Exception as exc:
                fail = {
                    "key": key,
                    "snapshot": snap,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
                failures.append(fail)
                if verbose:
                    print(f"[tailfit][error] key={key} snapshot={snap}: {exc}")
                if not continue_on_error:
                    raise
    overview_fit = _plot_overview_params(results, output_dir)
    overview_daic = _plot_overview_delta_aic(results, output_dir)
    agg = {
        "input": str(input_npz),
        "n_datasets": len(results),
        "datasets": results,
        "failures": failures,
        "n_failed": len(failures),
        "overview_plots": {
            "bestfit_vs_snapshot": str(overview_fit),
            "delta_aic_vs_pl_vs_snapshot": str(overview_daic),
        },
    }
    (output_dir / "fit_summary.json").write_text(json.dumps(agg, indent=2))
    if verbose:
        print(f"[tailfit] wrote summary: {output_dir / 'fit_summary.json'}")
    return agg
