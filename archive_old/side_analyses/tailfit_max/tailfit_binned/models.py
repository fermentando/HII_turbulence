from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erf, logsumexp


EPS = 1e-300


def log_edges(xmin: float = 1.0, xmax: float = 1.0e8, nbins: int = 50) -> NDArray[np.float64]:
    return np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)


def histogram_counts(values: NDArray[np.int64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    v = v[v > 0]
    if v.size == 0:
        return np.zeros(edges.size - 1, dtype=int)
    return np.histogram(v, bins=edges)[0].astype(int)


def tail_bin_view(
    edges: NDArray[np.float64], counts: NDArray[np.int64], xmin: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int]:
    lower = edges[:-1]
    i0 = int(np.searchsorted(lower, xmin, side="left"))
    tail_l = lower[i0:].copy()
    tail_u = edges[1:][i0:].copy()
    tail_u[-1] = np.inf
    tail_k = counts[i0:].copy()
    return tail_l, tail_u, tail_k, i0


def weighted_quantile(values: NDArray[np.float64], weights: NDArray[np.float64], q: float) -> float:
    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    c = np.cumsum(w)
    c /= c[-1]
    return float(np.interp(q, c, v))


def _int_power(l: NDArray[np.float64], u: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    out = np.empty_like(l, dtype=float)
    finite = np.isfinite(u)
    out[finite] = (np.power(l[finite], 1.0 - alpha) - np.power(u[finite], 1.0 - alpha)) / (alpha - 1.0)
    out[~finite] = np.power(l[~finite], 1.0 - alpha) / (alpha - 1.0)
    return out


def pi_pl(alpha: float, xmin: float, lower: NDArray[np.float64], upper: NDArray[np.float64]) -> NDArray[np.float64]:
    probs = np.power(lower / xmin, 1.0 - alpha) - np.where(
        np.isfinite(upper), np.power(upper / xmin, 1.0 - alpha), 0.0
    )
    probs = np.maximum(probs, 0.0)
    return _clip_probs(probs)


def pdf_pl(x: NDArray[np.float64], alpha: float, xmin: float) -> NDArray[np.float64]:
    p = np.zeros_like(x, dtype=float)
    m = x >= xmin
    p[m] = (alpha - 1.0) * np.power(xmin, alpha - 1.0) * np.power(x[m], -alpha)
    return p


def _phi(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def pi_lognormal(
    mu: float, sigma: float, xmin: float, lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    zl = (np.log(lower) - mu) / sigma
    zu = np.where(np.isfinite(upper), (np.log(upper) - mu) / sigma, np.inf)
    cdf_l = _phi(zl)
    cdf_u = np.where(np.isfinite(zu), _phi(zu), 1.0)
    zmin = (math.log(xmin) - mu) / sigma
    z = max(1.0 - float(_phi(np.array([zmin]))[0]), 1e-16)
    probs = (cdf_u - cdf_l) / z
    probs = np.maximum(probs, 0.0)
    return _clip_probs(probs)


def pdf_lognormal(x: NDArray[np.float64], mu: float, sigma: float, xmin: float) -> NDArray[np.float64]:
    p = np.zeros_like(x, dtype=float)
    m = x >= xmin
    xm = x[m]
    pref = 1.0 / (xm * sigma * math.sqrt(2.0 * math.pi))
    expo = np.exp(-0.5 * ((np.log(xm) - mu) / sigma) ** 2)
    zmin = (math.log(xmin) - mu) / sigma
    z = max(1.0 - float(_phi(np.array([zmin]))[0]), 1e-16)
    p[m] = (pref * expo) / z
    return p


def _bpl_norm(alpha1: float, alpha2: float, xmin: float, xb: float) -> float:
    i1 = (np.power(xmin, 1.0 - alpha1) - np.power(xb, 1.0 - alpha1)) / max(alpha1 - 1.0, 1e-12)
    i2 = np.power(xb, 1.0 - alpha1) / max(alpha2 - 1.0, 1e-12)
    den = max(i1 + i2, 1e-300)
    return 1.0 / den


def _bpl_bin_int(alpha1: float, alpha2: float, xmin: float, xb: float, l: float, u: float) -> float:
    c = _bpl_norm(alpha1, alpha2, xmin, xb)
    if u <= xb:
        return c * _int_power(np.array([l]), np.array([u]), alpha1)[0]
    if l >= xb:
        return c * np.power(xb, alpha2 - alpha1) * _int_power(np.array([l]), np.array([u]), alpha2)[0]
    i_left = c * _int_power(np.array([l]), np.array([xb]), alpha1)[0]
    i_right = c * np.power(xb, alpha2 - alpha1) * _int_power(np.array([xb]), np.array([u]), alpha2)[0]
    return i_left + i_right


def pi_bpl(
    alpha1: float, alpha2: float, xb: float, xmin: float, lower: NDArray[np.float64], upper: NDArray[np.float64]
) -> NDArray[np.float64]:
    probs = np.array([_bpl_bin_int(alpha1, alpha2, xmin, xb, l, u) for l, u in zip(lower, upper)], dtype=float)
    probs = np.maximum(probs, 0.0)
    return _clip_probs(probs)


def pdf_bpl(x: NDArray[np.float64], alpha1: float, alpha2: float, xb: float, xmin: float) -> NDArray[np.float64]:
    c = _bpl_norm(alpha1, alpha2, xmin, xb)
    p = np.zeros_like(x, dtype=float)
    m1 = (x >= xmin) & (x < xb)
    m2 = x >= xb
    p[m1] = c * np.power(x[m1], -alpha1)
    p[m2] = c * np.power(xb, alpha2 - alpha1) * np.power(x[m2], -alpha2)
    return p


def loglike_multinomial(k: NDArray[np.int64], pi: NDArray[np.float64]) -> float:
    p = _clip_probs(pi)
    return float(np.sum(k * np.log(p)))


def pearson_reduced_chi2(k: NDArray[np.int64], pi: NDArray[np.float64], n_params: int) -> float:
    n = int(k.sum())
    # Pearson chi2 is unstable for tiny expected counts; use unit floor per-bin.
    lam = np.maximum(n * pi, 1.0)
    chi2 = float(np.sum((k - lam) ** 2 / lam))
    dof = max(1, k.size - n_params)
    return chi2 / dof


def _curvature_var_1d(f: Callable[[float], float], x0: float, h: float = 1e-3) -> float:
    f0 = f(x0)
    fp = f(x0 + h)
    fm = f(x0 - h)
    sec = (fp - 2.0 * f0 + fm) / (h * h)
    if sec >= 0.0:
        return 1.0
    return max(1e-8, -1.0 / sec)


def _finite_diff_hessian(f: Callable[[NDArray[np.float64]], float], x: NDArray[np.float64], h: float = 1e-4):
    n = x.size
    hess = np.zeros((n, n), dtype=float)
    fx = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h
        fip = f(x + ei)
        fim = f(x - ei)
        hess[i, i] = (fip - 2.0 * fx + fim) / (h * h)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h
            fpp = f(x + ei + ej)
            fpm = f(x + ei - ej)
            fmp = f(x - ei + ej)
            fmm = f(x - ei - ej)
            val = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
            hess[i, j] = val
            hess[j, i] = val
    return hess


def _cov_psd(h: NDArray[np.float64]) -> NDArray[np.float64]:
    h_reg = h + np.eye(h.shape[0]) * 1e-8
    cov = np.linalg.pinv(h_reg)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    return vecs @ np.diag(vals) @ vecs.T


@dataclass
class XminPosterior:
    candidates: NDArray[np.float64]
    weights: NDArray[np.float64]
    alpha_hat: NDArray[np.float64]
    alpha_var: NDArray[np.float64]
    logl_hat: NDArray[np.float64]
    logprior: NDArray[np.float64]
    n_tail: NDArray[np.int64]
    p_values: NDArray[np.float64]
    ks_values: NDArray[np.float64]
    selected_index: int
    mode: str


def _clip_probs(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.clip(np.asarray(pi, dtype=float), EPS, 1.0)
    s = p.sum()
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(p, 1.0 / p.size)
    return p / s


def _fit_pl_alpha(
    xmin: float, tail_l: NDArray[np.float64], tail_u: NDArray[np.float64], tail_k: NDArray[np.int64]
) -> tuple[float, float, float]:
    def nll(a: float) -> float:
        if a <= 1.0:
            return np.inf
        pi = _clip_probs(pi_pl(a, xmin, tail_l, tail_u))
        return -loglike_multinomial(tail_k, pi)

    opt = minimize_scalar(nll, bounds=(1.01, 8.0), method="bounded", options={"xatol": 1e-4})
    ah = float(opt.x)
    ll = -float(opt.fun)
    var = _curvature_var_1d(lambda a: -nll(a), ah, h=2e-3)
    return ah, ll, var


def _pl_bootstrap_pvalue(
    xmin: float,
    tail_l: NDArray[np.float64],
    tail_u: NDArray[np.float64],
    tail_k: NDArray[np.int64],
    alpha_hat: float,
    n_bootstrap: int,
    seed: int,
) -> float:
    n_tail = int(tail_k.sum())
    if n_tail <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    pi_obs = _clip_probs(pi_pl(alpha_hat, xmin, tail_l, tail_u))
    lam_obs = np.maximum(n_tail * pi_obs, 1e-12)
    stat_obs = float(np.sum((tail_k - lam_obs) ** 2 / lam_obs))
    stats = []
    for _ in range(n_bootstrap):
        k_sim = rng.multinomial(n_tail, pi_obs)
        a_sim, _, _ = _fit_pl_alpha(xmin, tail_l, tail_u, k_sim)
        pi_sim = _clip_probs(pi_pl(a_sim, xmin, tail_l, tail_u))
        lam_sim = np.maximum(n_tail * pi_sim, 1e-12)
        stat_sim = float(np.sum((k_sim - lam_sim) ** 2 / lam_sim))
        stats.append(stat_sim)
    stat_arr = np.asarray(stats, dtype=float)
    return float(np.mean(stat_arr >= stat_obs))


def _pl_tail_cdf(x: NDArray[np.float64], xmin: float, alpha: float) -> NDArray[np.float64]:
    cdf = 1.0 - np.power(np.maximum(x / xmin, 1.0), -(alpha - 1.0))
    return np.clip(cdf, 0.0, 1.0)


def _ks_distance_binned_pl(
    xmin: float,
    alpha: float,
    tail_u: NDArray[np.float64],
    tail_k: NDArray[np.int64],
) -> float:
    n_tail = int(tail_k.sum())
    if n_tail <= 0:
        return np.inf
    emp = np.cumsum(tail_k, dtype=float) / float(n_tail)
    x_eval = np.where(np.isfinite(tail_u), tail_u, xmin * 1.0e12)
    model = _pl_tail_cdf(x_eval, xmin, alpha)
    model[-1] = 1.0
    return float(np.max(np.abs(emp - model)))


def infer_xmin_pl(
    edges: NDArray[np.float64],
    counts: NDArray[np.int64],
    mode: str = "smallest_good",
    soft_cap: float = 200.0,
    gamma: float = 2.0,
    xmin_max: float | None = None,
    pthresh: float = 0.1,
    xmin_candidates: int = 50,
    min_tail_count: int = 200,
    lambda_discard: float = 0.5,
    bootstrap_sims: int = 40,
    posterior_floor: float = 2e-3,
    seed: int = 7,
) -> XminPosterior:
    lowers = edges[:-1]
    if xmin_max is None:
        auto_ceiling = max(1000.0, 5.0 * soft_cap)
        candidate_idx = np.where(lowers <= auto_ceiling)[0]
    else:
        candidate_idx = np.where(lowers <= xmin_max)[0]
    if candidate_idx.size == 0:
        raise RuntimeError("No xmin candidates satisfy the hard cap.")

    if candidate_idx.size > xmin_candidates:
        pick = np.linspace(0, candidate_idx.size - 1, xmin_candidates).round().astype(int)
        candidate_idx = candidate_idx[np.unique(pick)]

    kept = []
    alpha_hat = []
    alpha_var = []
    logl_hat = []
    logprior = []
    n_tails = []
    p_values = []
    ks_values = []
    for i0 in candidate_idx:
        tail_l = lowers[i0:]
        tail_u = edges[1:][i0:].copy()
        tail_u[-1] = np.inf
        tail_k = counts[i0:]
        n_tail = int(tail_k.sum())
        if n_tail < min_tail_count:
            continue

        xmin = float(lowers[i0])
        ah, ll, var = _fit_pl_alpha(xmin, tail_l, tail_u, tail_k)
        if mode == "ks_min":
            prior = 0.0
        elif soft_cap <= 0.0:
            prior = 0.0
        else:
            prior = -((xmin / soft_cap) ** gamma)
        kept.append(xmin)
        alpha_hat.append(ah)
        alpha_var.append(var)
        logl_hat.append(ll)
        logprior.append(prior)
        n_tails.append(n_tail)
        p_values.append(np.nan)
        ks_values.append(np.nan)

    if len(kept) == 0:
        raise RuntimeError("No valid xmin candidates found for PL inference.")

    candidates = np.array(kept, dtype=float)
    logl_arr = np.array(logl_hat, dtype=float)
    prior_arr = np.array(logprior, dtype=float)
    ntail_arr = np.array(n_tails, dtype=int)
    pvals_arr = np.array(p_values, dtype=float)
    ks_arr = np.array(ks_values, dtype=float)
    scores_arr = logl_arr + prior_arr
    weights = np.exp(scores_arr - logsumexp(scores_arr))
    if posterior_floor > 0.0:
        weights = weights + posterior_floor
        weights = weights / weights.sum()

    if mode == "map":
        selected_index = int(np.argmax(weights))
    elif mode == "smallest_good":
        selected_index = -1
        for i, xmin in enumerate(candidates):
            i0 = int(np.searchsorted(lowers, xmin, side="left"))
            tail_l = lowers[i0:]
            tail_u = edges[1:][i0:].copy()
            tail_u[-1] = np.inf
            tail_k = counts[i0:]
            pval = _pl_bootstrap_pvalue(
                xmin=xmin,
                tail_l=tail_l,
                tail_u=tail_u,
                tail_k=tail_k,
                alpha_hat=float(alpha_hat[i]),
                n_bootstrap=bootstrap_sims,
                seed=seed + i,
            )
            pvals_arr[i] = pval
            if pval >= pthresh:
                selected_index = i
                break
        if selected_index < 0:
            fallback = scores_arr + lambda_discard * np.log(ntail_arr + 1.0)
            selected_index = int(np.argmax(fallback))
    elif mode == "ks_min":
        for i, xmin in enumerate(candidates):
            i0 = int(np.searchsorted(lowers, xmin, side="left"))
            tail_u = edges[1:][i0:].copy()
            tail_u[-1] = np.inf
            tail_k = counts[i0:]
            ks_arr[i] = _ks_distance_binned_pl(
                xmin=xmin,
                alpha=float(alpha_hat[i]),
                tail_u=tail_u,
                tail_k=tail_k,
            )
        selected_index = int(np.nanargmin(ks_arr))
        # ks_min selection is deterministic by min KS, but uncertainty can be read off the KS curve.
        dmin = float(np.nanmin(ks_arr))
        dd = ks_arr - dmin
        m = np.isfinite(dd)
        pos = dd[(dd > 0.0) & m]
        tau = float(np.median(pos)) if pos.size > 0 else 1e-3
        tau = max(tau, 1e-6)
        wks = np.zeros_like(weights)
        wks[m] = np.exp(-dd[m] / tau)
        wsum = float(wks.sum())
        if wsum <= 0.0 or not np.isfinite(wsum):
            wks[:] = 0.0
            wks[selected_index] = 1.0
        else:
            wks = wks / wsum
        weights = wks
    else:
        raise ValueError(f"Unknown xmin mode: {mode}")

    return XminPosterior(
        candidates=candidates,
        weights=weights,
        alpha_hat=np.array(alpha_hat, dtype=float),
        alpha_var=np.array(alpha_var, dtype=float),
        logl_hat=logl_arr,
        logprior=prior_arr,
        n_tail=ntail_arr,
        p_values=pvals_arr,
        ks_values=ks_arr,
        selected_index=selected_index,
        mode=mode,
    )


def fit_pl_with_posterior(
    xmin_post: XminPosterior,
    n_draws: int = 500,
    seed: int = 0,
) -> dict:
    xmin_vals = xmin_post.candidates
    w = xmin_post.weights
    i_map = int(np.argmax(w))
    i_sel = int(xmin_post.selected_index)
    xmin_selected = float(xmin_vals[i_sel])

    rng = np.random.default_rng(seed)
    draw_idx = rng.choice(np.arange(xmin_vals.size), size=n_draws, p=w, replace=True)
    draws = []
    for i in draw_idx:
        a = rng.normal(xmin_post.alpha_hat[i], math.sqrt(max(xmin_post.alpha_var[i], 1e-8)))
        draws.append(max(1.001, float(a)))
    ad = np.array(draws, dtype=float)
    q16 = weighted_quantile(xmin_vals, w, 0.16)
    q50 = weighted_quantile(xmin_vals, w, 0.50)
    q84 = weighted_quantile(xmin_vals, w, 0.84)
    if xmin_post.mode == "ks_min":
        ks = np.array(xmin_post.ks_values, dtype=float)
        m = np.isfinite(ks)
        if np.any(m):
            dmin = float(np.nanmin(ks))
            dd = ks - dmin
            pos = dd[(dd > 0.0) & m]
            band = float(np.median(pos)) if pos.size > 0 else 0.0
            acc = (dd <= max(band, 1e-12)) & m
            if np.any(acc):
                q16 = float(np.min(xmin_vals[acc]))
                q50 = xmin_selected
                q84 = float(np.max(xmin_vals[acc]))
        q16 = min(q16, xmin_selected)
        q84 = max(q84, xmin_selected)

    return {
        "xmin": {
            "map": float(xmin_vals[i_map]),
            "selected": xmin_selected,
            "selection_mode": xmin_post.mode,
            "selected_index": i_sel,
            "p16": q16,
            "p50": q50,
            "p84": q84,
            "mean": float(np.sum(xmin_vals * w)),
            "candidates": xmin_vals.tolist(),
            "weights": w.tolist(),
            "p_values": xmin_post.p_values.tolist(),
            "ks_values": xmin_post.ks_values.tolist(),
        },
        "alpha": {
            "p16": float(np.quantile(ad, 0.16)),
            "p50": float(np.quantile(ad, 0.50)),
            "p84": float(np.quantile(ad, 0.84)),
            "map_conditional": float(xmin_post.alpha_hat[i_map]),
            "selected_conditional": float(xmin_post.alpha_hat[i_sel]),
        },
        "xmin_map_index": i_map,
        "xmin_selected_index": i_sel,
    }


def fit_lognormal(
    xmin: float,
    tail_l: NDArray[np.float64],
    tail_u: NDArray[np.float64],
    tail_k: NDArray[np.int64],
    n_draws: int = 300,
    seed: int = 1,
) -> dict:
    mu0 = math.log(max(xmin * 2.0, 2.0))
    x0 = np.array([mu0, math.log(1.5)], dtype=float)

    def nll(z: NDArray[np.float64]) -> float:
        mu, ls = float(z[0]), float(z[1])
        sigma = math.exp(ls)
        pi = pi_lognormal(mu, sigma, xmin, tail_l, tail_u)
        return -loglike_multinomial(tail_k, pi)

    bounds = [(math.log(max(xmin, 1.0)) - 4.0, math.log(1.0e8) + 4.0), (-4.0, 4.0)]
    opt = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    zhat = opt.x.astype(float)
    ll = -float(opt.fun)
    h = _finite_diff_hessian(nll, zhat)
    cov = _cov_psd(h)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(zhat, cov, size=n_draws)
    mu_d = draws[:, 0]
    sig_d = np.exp(np.clip(draws[:, 1], -8.0, 8.0))
    mu = float(zhat[0])
    sigma = float(np.exp(np.clip(zhat[1], -8.0, 8.0)))
    return {
        "params": {"mu": mu, "sigma": sigma},
        "uncertainty": {
            "mu": {"p16": float(np.quantile(mu_d, 0.16)), "p50": float(np.quantile(mu_d, 0.50)), "p84": float(np.quantile(mu_d, 0.84))},
            "sigma": {
                "p16": float(np.quantile(sig_d, 0.16)),
                "p50": float(np.quantile(sig_d, 0.50)),
                "p84": float(np.quantile(sig_d, 0.84)),
            },
        },
        "logL": ll,
    }


def fit_bpl(
    xmin: float,
    tail_l: NDArray[np.float64],
    tail_u: NDArray[np.float64],
    tail_k: NDArray[np.int64],
    n_draws: int = 300,
    xb_prior_min_ratio: float = 1.0,
    xb_prior_strength: float = 0.0,
    xb_extra_guesses: NDArray[np.float64] | None = None,
    seed: int = 2,
) -> dict:
    x0 = np.array([math.log(1.1), math.log(2.0), math.log(2.0)], dtype=float)

    def unpack(z: NDArray[np.float64]) -> tuple[float, float, float]:
        a1 = 1.0 + math.exp(float(np.clip(z[0], -8.0, 8.0)))
        a2 = 1.0 + math.exp(float(np.clip(z[1], -8.0, 8.0)))
        xb = xmin * (1.0 + math.exp(float(np.clip(z[2], -8.0, 10.0))))
        return a1, a2, xb

    def nll(z: NDArray[np.float64]) -> float:
        a1, a2, xb = unpack(z)
        pi = pi_bpl(a1, a2, xb, xmin, tail_l, tail_u)
        nll0 = -loglike_multinomial(tail_k, pi)
        if xb_prior_strength > 0.0 and xb_prior_min_ratio > 1.0:
            ratio = max(xb / xmin, 1.0 + 1e-12)
            gap = max(math.log(xb_prior_min_ratio) - math.log(ratio), 0.0)
            nll0 += xb_prior_strength * gap * gap
        return nll0

    bounds = [(-8.0, 8.0), (-8.0, 8.0), (-8.0, 10.0)]
    starts = [
        x0,
        np.array([math.log(1.2), math.log(3.0), math.log(1.2)], dtype=float),
        np.array([math.log(2.0), math.log(2.0), math.log(4.0)], dtype=float),
        np.array([math.log(1.05), math.log(4.0), math.log(0.8)], dtype=float),
    ]
    if xb_extra_guesses is not None and xb_extra_guesses.size:
        for xbg in np.asarray(xb_extra_guesses, dtype=float):
            if not np.isfinite(xbg) or xbg <= xmin:
                continue
            z3 = math.log(max((xbg / xmin) - 1.0, math.exp(-8.0)))
            starts.extend(
                [
                    np.array([math.log(1.1), math.log(2.0), z3], dtype=float),
                    np.array([math.log(1.4), math.log(2.8), z3], dtype=float),
                    np.array([math.log(1.8), math.log(1.8), z3], dtype=float),
                ]
            )
    best_opt = None
    for s in starts:
        opt = minimize(nll, s, method="L-BFGS-B", bounds=bounds)
        if best_opt is None or float(opt.fun) < float(best_opt.fun):
            best_opt = opt
    assert best_opt is not None
    zhat = best_opt.x.astype(float)
    a1, a2, xb = unpack(zhat)
    ll = -float(best_opt.fun)
    h = _finite_diff_hessian(nll, zhat)
    cov = _cov_psd(h)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(zhat, cov, size=n_draws)
    a1d = 1.0 + np.exp(np.clip(draws[:, 0], -8.0, 8.0))
    a2d = 1.0 + np.exp(np.clip(draws[:, 1], -8.0, 8.0))
    xbd = xmin * (1.0 + np.exp(np.clip(draws[:, 2], -8.0, 10.0)))
    return {
        "params": {"alpha1": a1, "alpha2": a2, "xb": xb},
        "uncertainty": {
            "alpha1": {"p16": float(np.quantile(a1d, 0.16)), "p50": float(np.quantile(a1d, 0.50)), "p84": float(np.quantile(a1d, 0.84))},
            "alpha2": {"p16": float(np.quantile(a2d, 0.16)), "p50": float(np.quantile(a2d, 0.50)), "p84": float(np.quantile(a2d, 0.84))},
            "xb": {"p16": float(np.quantile(xbd, 0.16)), "p50": float(np.quantile(xbd, 0.50)), "p84": float(np.quantile(xbd, 0.84))},
        },
        "logL": ll,
    }
