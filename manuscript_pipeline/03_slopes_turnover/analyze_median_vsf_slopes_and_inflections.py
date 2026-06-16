"""
Estimate VSF slopes from median NPZ files using uncertainties,
and detect possible inflection points via KS tests on local slopes.
"""

import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


NPZ_DIR = "NPZ"
OUT_DIR = "SlopeResults"
FILE_PATTERN = os.path.join(NPZ_DIR, "VSF_*_median_allsubwindows.npz")
LINE_SET = {"NII", "SII", "SIII"}
LINE_ORDER = ["NII", "SII", "SIII"]
LINE_COLORS = {"NII": "tab:blue", "SII": "tab:red", "SIII": "tab:green"}

# Monte Carlo and significance settings.
N_MC = 4000
BASE_ALPHA = 0.05
MIN_POINTS_FOR_SLOPE = 4
MIN_SEGMENTS_PER_SIDE = 2
MIN_POINTS_FOR_SPLIT_SLOPE = 3
MIN_POINTS_FOR_BREAK_MODEL_SIDE = 5
EDGE_FRACTION_EXCLUDE = 0.10
MIN_SLOPE_CONTRAST_SIGMA = 3.0
MIN_DELTA_BIC = 10.0
N_BOOT_BREAK = 500
RNG_SEED = 20260504


def sanitize_filename(name):
    """Create a filesystem-safe base name for output plots."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def parse_object_and_line(npz_path):
    """Parse object and line name from NPZ file name."""
    name = os.path.basename(npz_path)
    match = re.match(r"^VSF_(.+)_([A-Za-z0-9]+)_median_allsubwindows\.npz$", name)
    if not match:
        return None, None

    object_name = match.group(1)
    line_name = match.group(2).upper()
    if line_name not in LINE_SET:
        return None, None
    return object_name, line_name


def weighted_linear_fit(x, y, y_err):
    """
    Weighted linear fit for y = a + b*x.
    Returns (slope b, intercept a).
    """
    w = 1.0 / np.square(y_err)
    sw = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x * x)
    sxy = np.sum(w * x * y)

    denom = sw * sxx - sx * sx
    if denom <= 0 or not np.isfinite(denom):
        return np.nan, np.nan

    slope = (sw * sxy - sx * sy) / denom
    intercept = (sxx * sy - sx * sxy) / denom
    return slope, intercept


def weighted_linear_fit_details(x, y, y_err):
    """
    Weighted linear fit details for y = a + b*x.
    Returns slope/intercept with slope uncertainty and chi2.
    """
    w = 1.0 / np.square(y_err)
    sw = np.sum(w)
    sx = np.sum(w * x)
    sy = np.sum(w * y)
    sxx = np.sum(w * x * x)
    sxy = np.sum(w * x * y)

    denom = sw * sxx - sx * sx
    if denom <= 0 or not np.isfinite(denom):
        return {
            "ok": False,
            "slope": np.nan,
            "intercept": np.nan,
            "slope_se": np.nan,
            "chi2": np.nan,
        }

    slope = (sw * sxy - sx * sy) / denom
    intercept = (sxx * sy - sx * sxy) / denom
    resid = y - (intercept + slope * x)
    chi2 = np.sum(np.square(resid / y_err))

    slope_var = sw / denom
    slope_se = np.sqrt(slope_var) if (slope_var > 0 and np.isfinite(slope_var)) else np.nan
    return {
        "ok": np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(chi2),
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "chi2": chi2,
    }


def compute_bic(chi2, n_points, n_params):
    """Gaussian-error BIC up to additive constant."""
    if n_points <= 0 or n_params <= 0 or not np.isfinite(chi2):
        return np.nan
    return float(chi2 + n_params * np.log(n_points))


def safe_z(values):
    """Robust z-score with zero fallback when variance is tiny."""
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(arr)
    return (arr - mean) / std


def evaluate_break_candidate(log_r, y, y_err, point_split_idx, bic_one_line, edge_fraction_exclude=EDGE_FRACTION_EXCLUDE):
    """
    Evaluate one split candidate using piecewise weighted fits.

    point_split_idx is the first point index of the right side.
    """
    n = len(log_r)
    n_left = int(point_split_idx)
    n_right = int(n - point_split_idx)

    out = {
        "point_split_idx": int(point_split_idx),
        "split_seg_idx": int(point_split_idx - 1),
        "n_left": n_left,
        "n_right": n_right,
        "model_side_ok": False,
        "edge_ok": False,
        "delta_bic": np.nan,
        "slope_contrast_sigma": np.nan,
        "break_log_r": np.nan,
        "break_r_pc": np.nan,
        "slope_left": np.nan,
        "slope_right": np.nan,
        "slope_left_se": np.nan,
        "slope_right_se": np.nan,
    }

    if n_left < MIN_POINTS_FOR_BREAK_MODEL_SIDE or n_right < MIN_POINTS_FOR_BREAK_MODEL_SIDE:
        return out

    left = weighted_linear_fit_details(log_r[:point_split_idx], y[:point_split_idx], y_err[:point_split_idx])
    right = weighted_linear_fit_details(log_r[point_split_idx:], y[point_split_idx:], y_err[point_split_idx:])
    if not left["ok"] or not right["ok"]:
        return out

    bic_two = compute_bic(left["chi2"] + right["chi2"], n, 4)
    delta_bic = bic_one_line - bic_two

    denom = np.sqrt(left["slope_se"] ** 2 + right["slope_se"] ** 2)
    slope_contrast = np.nan
    if np.isfinite(denom) and denom > 0:
        slope_contrast = abs(left["slope"] - right["slope"]) / denom

    break_log_r = 0.5 * (log_r[point_split_idx - 1] + log_r[point_split_idx])
    frac = (break_log_r - log_r[0]) / (log_r[-1] - log_r[0]) if (log_r[-1] > log_r[0]) else 0.5
    edge_ok = (frac >= edge_fraction_exclude) and (frac <= (1.0 - edge_fraction_exclude))

    out.update({
        "model_side_ok": True,
        "edge_ok": bool(edge_ok),
        "delta_bic": float(delta_bic),
        "slope_contrast_sigma": float(slope_contrast),
        "break_log_r": float(break_log_r),
        "break_r_pc": float(10.0 ** break_log_r),
        "slope_left": float(left["slope"]),
        "slope_right": float(right["slope"]),
        "slope_left_se": float(left["slope_se"]),
        "slope_right_se": float(right["slope_se"]),
    })
    return out


def bootstrap_break_stats(
    log_r,
    y,
    y_err,
    candidate_point_idxs,
    selected_point_idx,
    rng,
    n_boot=N_BOOT_BREAK,
    edge_fraction_exclude=EDGE_FRACTION_EXCLUDE,
):
    """
    Bootstrap turnover location using model-improvement and slope-contrast score.
    """
    if len(candidate_point_idxs) == 0:
        return {
            "n_boot_valid": 0,
            "break_bootstrap_p16": np.nan,
            "break_bootstrap_p50": np.nan,
            "break_bootstrap_p84": np.nan,
            "break_selection_frequency": np.nan,
        }

    selected_breaks = []
    selected_idxs = []

    for _ in range(int(n_boot)):
        yb = rng.normal(loc=y, scale=y_err)
        one = weighted_linear_fit_details(log_r, yb, y_err)
        if not one["ok"]:
            continue

        bic_one = compute_bic(one["chi2"], len(log_r), 2)
        boot_cands = []
        for point_idx in candidate_point_idxs:
            c = evaluate_break_candidate(
                log_r,
                yb,
                y_err,
                point_idx,
                bic_one,
                edge_fraction_exclude=edge_fraction_exclude,
            )
            if not c["model_side_ok"] or not c["edge_ok"]:
                continue
            if not np.isfinite(c["delta_bic"]) or not np.isfinite(c["slope_contrast_sigma"]):
                continue
            boot_cands.append(c)

        if not boot_cands:
            continue

        db = np.array([c["delta_bic"] for c in boot_cands], dtype=float)
        sc = np.array([c["slope_contrast_sigma"] for c in boot_cands], dtype=float)
        score = safe_z(db) + safe_z(sc)

        best_i = int(np.nanargmax(score))
        best = boot_cands[best_i]
        selected_breaks.append(best["break_r_pc"])
        selected_idxs.append(best["point_split_idx"])

    if len(selected_breaks) == 0:
        return {
            "n_boot_valid": 0,
            "break_bootstrap_p16": np.nan,
            "break_bootstrap_p50": np.nan,
            "break_bootstrap_p84": np.nan,
            "break_selection_frequency": np.nan,
        }

    selected_breaks = np.asarray(selected_breaks, dtype=float)
    selected_idxs = np.asarray(selected_idxs, dtype=int)
    freq = np.mean(selected_idxs == int(selected_point_idx)) if selected_point_idx is not None else np.nan

    p16, p50, p84 = np.percentile(selected_breaks, [16, 50, 84])
    return {
        "n_boot_valid": int(len(selected_breaks)),
        "break_bootstrap_p16": float(p16),
        "break_bootstrap_p50": float(p50),
        "break_bootstrap_p84": float(p84),
        "break_selection_frequency": float(freq),
    }


def slope_with_uncertainty(log_r, vsf, vsf_err, n_mc=N_MC, rng=None):
    """
    Estimate global slope in log-log space with Monte Carlo uncertainties.

    Model: log10(VSF) = intercept + slope * log10(r)
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    # Convert to log-space and propagate y errors.
    y = np.log10(vsf)
    y_err = vsf_err / (vsf * np.log(10.0))

    slope_wls, intercept_wls = weighted_linear_fit(log_r, y, y_err)

    # Monte Carlo from measurement uncertainties in y.
    mc_y = rng.normal(loc=y[None, :], scale=y_err[None, :], size=(n_mc, len(y)))
    slope_samples = np.full(n_mc, np.nan)

    for i in range(n_mc):
        s, _ = weighted_linear_fit(log_r, mc_y[i], y_err)
        slope_samples[i] = s

    slope_samples = slope_samples[np.isfinite(slope_samples)]
    if slope_samples.size == 0:
        return {
            "slope_wls": slope_wls,
            "intercept_wls": intercept_wls,
            "slope_median": np.nan,
            "slope_p16": np.nan,
            "slope_p84": np.nan,
            "slope_err_half68": np.nan,
            "n_mc_valid": 0,
        }

    p16, p50, p84 = np.percentile(slope_samples, [16, 50, 84])
    return {
        "slope_wls": slope_wls,
        "intercept_wls": intercept_wls,
        "slope_median": p50,
        "slope_p16": p16,
        "slope_p84": p84,
        "slope_err_half68": 0.5 * (p84 - p16),
        "n_mc_valid": int(slope_samples.size),
    }


def ks_inflection_test(log_r, vsf, vsf_err, n_mc=N_MC, alpha=BASE_ALPHA, rng=None, object_name=None):
    """
    KS-based inflection detection using local slope distributions.

    Steps:
    1) Draw MC realizations of log10(VSF) from uncertainties.
    2) Compute local slopes between adjacent radial bins for each realization.
     3) For each candidate split, compare left vs right local-slope samples via KS.
     4) For each candidate, compute two-line improvement (delta BIC) and slope contrast.
     5) Keep candidates that satisfy significance and quality criteria, then pick the
         clearest turnover by composite score.
     6) Bootstrap turnover location for uncertainty and selection stability.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    y = np.log10(vsf)
    y_err = vsf_err / (vsf * np.log(10.0))

    # RCW79 requires stronger edge rejection; otherwise the selected break
    # is biased to the largest-scale tail.
    edge_fraction_exclude = EDGE_FRACTION_EXCLUDE
    if object_name == "RCW79":
        edge_fraction_exclude = 0.20

    # Local slopes are defined on segments between bins.
    dx = np.diff(log_r)
    if np.any(~np.isfinite(dx)) or np.any(dx <= 0):
        return {
            "tested": False,
            "reason": "non-monotonic or invalid log_r",
        }

    n_seg = len(dx)
    if n_seg < (2 * MIN_SEGMENTS_PER_SIDE):
        return {
            "tested": False,
            "reason": "not enough segments for KS split test",
        }

    mc_y = rng.normal(loc=y[None, :], scale=y_err[None, :], size=(n_mc, len(y)))
    local_slopes = (mc_y[:, 1:] - mc_y[:, :-1]) / dx[None, :]

    one_line = weighted_linear_fit_details(log_r, y, y_err)
    if not one_line["ok"]:
        return {
            "tested": False,
            "reason": "global weighted fit failed",
        }
    bic_one = compute_bic(one_line["chi2"], len(log_r), 2)

    tested_candidates = []

    candidate_idxs = range(MIN_SEGMENTS_PER_SIDE, n_seg - MIN_SEGMENTS_PER_SIDE + 1)
    n_tests = len(list(candidate_idxs))
    if n_tests <= 0:
        return {
            "tested": False,
            "reason": "no valid split candidates",
        }

    # Recreate iterator because list(...) above consumed it.
    candidate_idxs = range(MIN_SEGMENTS_PER_SIDE, n_seg - MIN_SEGMENTS_PER_SIDE + 1)

    for k in candidate_idxs:
        # Use one representative slope per realization for each side to avoid
        # artificially inflating sample size by pooling all segments.
        left = np.nanmedian(local_slopes[:, :k], axis=1)
        right = np.nanmedian(local_slopes[:, k:], axis=1)

        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if left.size < 30 or right.size < 30:
            continue

        ks = ks_2samp(left, right, alternative="two-sided", method="auto")
        if not np.isfinite(ks.pvalue):
            continue

        fit_eval = evaluate_break_candidate(
            log_r,
            y,
            y_err,
            k,
            bic_one,
            edge_fraction_exclude=edge_fraction_exclude,
        )

        tested_candidates.append({
            "split_k": int(k),
            "split_seg_idx": int(k - 1),
            "point_split_idx": int(k),
            "ks_stat": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "left_slope_median": float(np.nanmedian(left)),
            "right_slope_median": float(np.nanmedian(right)),
            "delta_bic": fit_eval["delta_bic"],
            "slope_contrast_sigma": fit_eval["slope_contrast_sigma"],
            "edge_ok": fit_eval["edge_ok"],
            "model_side_ok": fit_eval["model_side_ok"],
            "break_log_r": fit_eval["break_log_r"],
            "break_r_pc": fit_eval["break_r_pc"],
        })

    if not tested_candidates:
        return {
            "tested": False,
            "reason": "all split tests invalid",
        }

    alpha_corrected = alpha / n_tests

    # Build candidate-wise normalized clarity terms.
    pvals = np.array([c["p_value"] for c in tested_candidates], dtype=float)
    dbic = np.array([c["delta_bic"] for c in tested_candidates], dtype=float)
    scon = np.array([c["slope_contrast_sigma"] for c in tested_candidates], dtype=float)
    p_proxy = -np.log10(np.clip(pvals, 1e-300, 1.0))
    z_dbic = safe_z(dbic)
    z_scon = safe_z(scon)
    z_p = safe_z(p_proxy)
    clarity = z_dbic + z_scon + z_p

    for i, c in enumerate(tested_candidates):
        c["clarity_score"] = float(clarity[i])
        c["ks_significant"] = bool(c["p_value"] < alpha_corrected)
        c["passes_hard_criteria"] = bool(
            c["ks_significant"]
            and c["model_side_ok"]
            and c["edge_ok"]
            and np.isfinite(c["delta_bic"]) and (c["delta_bic"] >= MIN_DELTA_BIC)
            and np.isfinite(c["slope_contrast_sigma"]) and (c["slope_contrast_sigma"] >= MIN_SLOPE_CONTRAST_SIGMA)
        )

    hard_pass = [c for c in tested_candidates if c["passes_hard_criteria"]]
    if hard_pass:
        # Primary selection: clearest candidate by composite score.
        # Tie-breaker: prefer larger-scale candidate (larger split_k).
        selected = sorted(hard_pass, key=lambda c: (c["clarity_score"], c["split_k"]), reverse=True)[0]
        selection_rule = "max clarity among hard-pass candidates"
    else:
        significant = [c for c in tested_candidates if c["ks_significant"]]
        if significant:
            selected = sorted(significant, key=lambda c: c["split_k"], reverse=True)[0]
            selection_rule = "fallback: right-to-left among KS-significant"
        else:
            selected = min(tested_candidates, key=lambda c: c["p_value"])
            selection_rule = "fallback: minimum p-value (not significant)"

    boot = bootstrap_break_stats(
        log_r=log_r,
        y=y,
        y_err=y_err,
        candidate_point_idxs=[c["point_split_idx"] for c in tested_candidates],
        selected_point_idx=selected.get("point_split_idx", None),
        rng=rng,
        n_boot=N_BOOT_BREAK,
        edge_fraction_exclude=edge_fraction_exclude,
    )

    selected.update({
        "tested": True,
        "n_tests": n_tests,
        "alpha": alpha,
        "alpha_bonferroni": alpha_corrected,
        "significant": bool(selected["p_value"] < alpha_corrected),
        "n_hard_pass": int(len(hard_pass)),
        "selection_rule": selection_rule,
        "min_delta_bic": float(MIN_DELTA_BIC),
        "min_slope_contrast_sigma": float(MIN_SLOPE_CONTRAST_SIGMA),
        "edge_fraction_exclude": float(edge_fraction_exclude),
        "break_bootstrap_p16": boot["break_bootstrap_p16"],
        "break_bootstrap_p50": boot["break_bootstrap_p50"],
        "break_bootstrap_p84": boot["break_bootstrap_p84"],
        "break_selection_frequency": boot["break_selection_frequency"],
        "n_boot_valid": boot["n_boot_valid"],
    })
    return selected


def load_clean_vsf(npz_path):
    """Load median VSF arrays and apply validity filtering."""
    with np.load(npz_path) as d:
        vsf = d["vsf"].astype(float)
        vsf_err = d["vsf_err"].astype(float)
        log_r = d["log_centers"].astype(float)

    mask = np.isfinite(log_r) & np.isfinite(vsf) & np.isfinite(vsf_err)
    mask &= (vsf > 0) & (vsf_err > 0)

    log_r = log_r[mask]
    vsf = vsf[mask]
    vsf_err = vsf_err[mask]

    # Sort by log_r if needed.
    order = np.argsort(log_r)
    log_r = log_r[order]
    vsf = vsf[order]
    vsf_err = vsf_err[order]

    return log_r, vsf, vsf_err


def compute_piecewise_slopes_if_inflected(log_r, vsf, vsf_err, ks_info, rng=None):
    """Compute independent slopes below and above a significant inflection."""
    empty = {
        "split_used": False,
        "split_idx": np.nan,
        "n_points_low": 0,
        "n_points_high": 0,
        "slope_low_median": np.nan,
        "slope_low_err_half68": np.nan,
        "slope_low_wls": np.nan,
        "intercept_low_wls": np.nan,
        "slope_high_median": np.nan,
        "slope_high_err_half68": np.nan,
        "slope_high_wls": np.nan,
        "intercept_high_wls": np.nan,
        "split_notes": "",
    }

    if not ks_info.get("tested", False):
        empty["split_notes"] = "ks not tested"
        return empty

    if not ks_info.get("significant", False):
        empty["split_notes"] = "inflection not significant"
        return empty

    split_idx = ks_info.get("split_seg_idx", None)
    if split_idx is None or not np.isfinite(split_idx):
        empty["split_notes"] = "missing split index"
        return empty

    split_idx = int(split_idx)

    # Split is between points split_idx and split_idx+1.
    low_slice = slice(0, split_idx + 1)
    high_slice = slice(split_idx + 1, None)

    log_r_low = log_r[low_slice]
    vsf_low = vsf[low_slice]
    vsf_err_low = vsf_err[low_slice]

    log_r_high = log_r[high_slice]
    vsf_high = vsf[high_slice]
    vsf_err_high = vsf_err[high_slice]

    n_low = len(log_r_low)
    n_high = len(log_r_high)

    if n_low < MIN_POINTS_FOR_SPLIT_SLOPE or n_high < MIN_POINTS_FOR_SPLIT_SLOPE:
        empty.update({
            "split_idx": split_idx,
            "n_points_low": n_low,
            "n_points_high": n_high,
            "split_notes": "insufficient points in one split side",
        })
        return empty

    low_info = slope_with_uncertainty(log_r_low, vsf_low, vsf_err_low, n_mc=N_MC, rng=rng)
    high_info = slope_with_uncertainty(log_r_high, vsf_high, vsf_err_high, n_mc=N_MC, rng=rng)

    out = {
        "split_used": True,
        "split_idx": split_idx,
        "n_points_low": n_low,
        "n_points_high": n_high,
        "slope_low_median": low_info["slope_median"],
        "slope_low_err_half68": low_info["slope_err_half68"],
        "slope_low_wls": low_info["slope_wls"],
        "intercept_low_wls": low_info["intercept_wls"],
        "slope_high_median": high_info["slope_median"],
        "slope_high_err_half68": high_info["slope_err_half68"],
        "slope_high_wls": high_info["slope_wls"],
        "intercept_high_wls": high_info["intercept_wls"],
        "split_notes": "",
    }
    return out


def plot_best_fit_overlays(plot_rows):
    """Plot median VSFs in two composite figures, one row per region."""
    plots_dir = os.path.join("Figures", "manuscript", "03_slopes_turnover")
    os.makedirs(plots_dir, exist_ok=True)

    # Group by object.
    by_object = {}
    for row in plot_rows:
        by_object.setdefault(row["object"], {})[row["line"]] = row

    object_names = sorted(by_object)
    if not object_names:
        return

    split_point = int(np.ceil(len(object_names) / 2.0))
    object_groups = [object_names[:split_point], object_names[split_point:]]

    for part_idx, object_group in enumerate(object_groups, start=1):
        if not object_group:
            continue

        n_rows = len(object_group)
        fig, axes = plt.subplots(n_rows, len(LINE_ORDER), figsize=(18, 4.6 * n_rows), squeeze=False)

        for row_idx, object_name in enumerate(object_group):
            line_map = by_object[object_name]

            for col_idx, line_name in enumerate(LINE_ORDER):
                ax = axes[row_idx, col_idx]
                payload = line_map.get(line_name)

                if payload is None:
                    ax.text(0.5, 0.5, f"{line_name}: no data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                log_r = payload["log_r"]
                r = 10.0 ** log_r
                vsf = payload["vsf"]
                vsf_err = payload["vsf_err"]
                color = LINE_COLORS.get(line_name, "black")

                ax.errorbar(
                    r,
                    vsf,
                    yerr=vsf_err,
                    fmt='o',
                    ms=4,
                    color=color,
                    ecolor=color,
                    alpha=0.8,
                    capsize=2,
                    label='Median VSF'
                )

                slope = payload["slope_wls"]
                intercept = payload["intercept_wls"]
                if np.isfinite(slope) and np.isfinite(intercept):
                    y_fit = 10.0 ** (intercept + slope * log_r)
                    ax.plot(
                        r,
                        y_fit,
                        linestyle='--',
                        linewidth=2.0,
                        color='black',
                        label=f"Best fit (m={slope:.3f})"
                    )

                # If a significant inflection split was used, report and plot
                # turnover and the independent below/above fits.
                split_used = bool(payload.get("split_used", False))
                break_r_pc = payload.get("break_r_pc", np.nan)
                if split_used and np.isfinite(break_r_pc):
                    ax.axvline(
                        break_r_pc,
                        linestyle=':',
                        linewidth=1.6,
                        color='dimgray',
                        alpha=0.9,
                        label=f"Turnover = {break_r_pc:.3f} pc"
                    )

                    split_idx = int(payload.get("split_idx", -1))
                    if split_idx >= 0 and split_idx < len(log_r) - 1:
                        # Below-turnover fit.
                        slope_low = payload.get("slope_low_wls", np.nan)
                        intercept_low = payload.get("intercept_low_wls", np.nan)
                        if np.isfinite(slope_low) and np.isfinite(intercept_low):
                            log_r_low = log_r[:split_idx + 1]
                            r_low = 10.0 ** log_r_low
                            y_fit_low = 10.0 ** (intercept_low + slope_low * log_r_low)
                            ax.plot(
                                r_low,
                                y_fit_low,
                                linestyle='-.',
                                linewidth=2.1,
                                color='purple',
                                alpha=0.95,
                                label=f"Below turnover m={slope_low:.3f}"
                            )

                        # Above-turnover fit.
                        slope_high = payload.get("slope_high_wls", np.nan)
                        intercept_high = payload.get("intercept_high_wls", np.nan)
                        if np.isfinite(slope_high) and np.isfinite(intercept_high):
                            log_r_high = log_r[split_idx + 1:]
                            r_high = 10.0 ** log_r_high
                            y_fit_high = 10.0 ** (intercept_high + slope_high * log_r_high)
                            ax.plot(
                                r_high,
                                y_fit_high,
                                linestyle='--',
                                linewidth=2.1,
                                color='orange',
                                alpha=0.95,
                                label=f"Above turnover m={slope_high:.3f}"
                            )

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('r [pc]')
                if col_idx == 0:
                    ax.set_ylabel('Median VSF [(km/s)^2]')
                ax.set_title(f"{object_name} {line_name}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='best')

        fig.tight_layout()
        out_png = os.path.join(plots_dir, f"median_vsf_bestfit_rows_part{part_idx}.png")
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Wrote plot: {out_png}")


def main():
    npz_files = sorted(glob.glob(FILE_PATTERN))
    if not npz_files:
        print(f"No files found for pattern: {FILE_PATTERN}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    plot_rows = []

    for npz_path in npz_files:
        object_name, line_name = parse_object_and_line(npz_path)
        if object_name is None:
            continue

        log_r, vsf, vsf_err = load_clean_vsf(npz_path)
        if len(log_r) < MIN_POINTS_FOR_SLOPE:
            rows.append({
                "object": object_name,
                "line": line_name,
                "npz_file": npz_path,
                "n_points": len(log_r),
                "slope_median": np.nan,
                "slope_err_half68": np.nan,
                "slope_p16": np.nan,
                "slope_p84": np.nan,
                "slope_wls": np.nan,
                "ks_tested": False,
                "ks_significant": False,
                "ks_pvalue": np.nan,
                "ks_stat": np.nan,
                "ks_n_tests": 0,
                "alpha_bonferroni": np.nan,
                "break_r_pc": np.nan,
                "delta_bic_break": np.nan,
                "slope_contrast_sigma": np.nan,
                "clarity_score": np.nan,
                "selection_rule": "",
                "n_hard_pass": 0,
                "break_bootstrap_p16": np.nan,
                "break_bootstrap_p50": np.nan,
                "break_bootstrap_p84": np.nan,
                "break_selection_frequency": np.nan,
                "n_boot_valid": 0,
                "left_slope_median": np.nan,
                "right_slope_median": np.nan,
                "notes": "insufficient points for slope fit",
            })
            continue

        slope_info = slope_with_uncertainty(log_r, vsf, vsf_err, n_mc=N_MC, rng=rng)
        ks_info = ks_inflection_test(
            log_r,
            vsf,
            vsf_err,
            n_mc=N_MC,
            alpha=BASE_ALPHA,
            rng=rng,
            object_name=object_name,
        )
        split_info = compute_piecewise_slopes_if_inflected(log_r, vsf, vsf_err, ks_info, rng=rng)

        row = {
            "object": object_name,
            "line": line_name,
            "npz_file": npz_path,
            "n_points": len(log_r),
            "slope_median": slope_info["slope_median"],
            "slope_err_half68": slope_info["slope_err_half68"],
            "slope_p16": slope_info["slope_p16"],
            "slope_p84": slope_info["slope_p84"],
            "slope_wls": slope_info["slope_wls"],
            "intercept_wls": slope_info["intercept_wls"],
            "ks_tested": bool(ks_info.get("tested", False)),
            "ks_significant": bool(ks_info.get("significant", False)),
            "ks_pvalue": ks_info.get("p_value", np.nan),
            "ks_stat": ks_info.get("ks_stat", np.nan),
            "ks_n_tests": ks_info.get("n_tests", 0),
            "alpha_bonferroni": ks_info.get("alpha_bonferroni", np.nan),
            "break_r_pc": ks_info.get("break_r_pc", np.nan),
            "delta_bic_break": ks_info.get("delta_bic", np.nan),
            "slope_contrast_sigma": ks_info.get("slope_contrast_sigma", np.nan),
            "clarity_score": ks_info.get("clarity_score", np.nan),
            "selection_rule": ks_info.get("selection_rule", ""),
            "n_hard_pass": ks_info.get("n_hard_pass", 0),
            "break_bootstrap_p16": ks_info.get("break_bootstrap_p16", np.nan),
            "break_bootstrap_p50": ks_info.get("break_bootstrap_p50", np.nan),
            "break_bootstrap_p84": ks_info.get("break_bootstrap_p84", np.nan),
            "break_selection_frequency": ks_info.get("break_selection_frequency", np.nan),
            "n_boot_valid": ks_info.get("n_boot_valid", 0),
            "left_slope_median": ks_info.get("left_slope_median", np.nan),
            "right_slope_median": ks_info.get("right_slope_median", np.nan),
            "split_used": split_info["split_used"],
            "split_idx": split_info["split_idx"],
            "n_points_low": split_info["n_points_low"],
            "n_points_high": split_info["n_points_high"],
            "slope_low_median": split_info["slope_low_median"],
            "slope_low_err_half68": split_info["slope_low_err_half68"],
            "slope_low_wls": split_info["slope_low_wls"],
            "intercept_low_wls": split_info["intercept_low_wls"],
            "slope_high_median": split_info["slope_high_median"],
            "slope_high_err_half68": split_info["slope_high_err_half68"],
            "slope_high_wls": split_info["slope_high_wls"],
            "intercept_high_wls": split_info["intercept_high_wls"],
            "notes": ks_info.get("reason", ""),
            "split_notes": split_info["split_notes"],
        }
        rows.append(row)
        plot_rows.append({
            "object": object_name,
            "line": line_name,
            "log_r": log_r,
            "vsf": vsf,
            "vsf_err": vsf_err,
            "slope_wls": slope_info["slope_wls"],
            "intercept_wls": slope_info["intercept_wls"],
            "split_used": split_info["split_used"],
            "split_idx": split_info["split_idx"],
            "break_r_pc": ks_info.get("break_r_pc", np.nan),
            "slope_low_wls": split_info["slope_low_wls"],
            "intercept_low_wls": split_info["intercept_low_wls"],
            "slope_high_wls": split_info["slope_high_wls"],
            "intercept_high_wls": split_info["intercept_high_wls"],
        })

    # Stable order in output table.
    rows.sort(key=lambda r: (r["object"], r["line"]))

    out_csv = os.path.join(OUT_DIR, "median_vsf_slope_ks_summary.csv")
    fields = [
        "object",
        "line",
        "npz_file",
        "n_points",
        "slope_median",
        "slope_err_half68",
        "slope_p16",
        "slope_p84",
        "slope_wls",
        "intercept_wls",
        "ks_tested",
        "ks_significant",
        "ks_pvalue",
        "ks_stat",
        "ks_n_tests",
        "alpha_bonferroni",
        "break_r_pc",
        "delta_bic_break",
        "slope_contrast_sigma",
        "clarity_score",
        "selection_rule",
        "n_hard_pass",
        "break_bootstrap_p16",
        "break_bootstrap_p50",
        "break_bootstrap_p84",
        "break_selection_frequency",
        "n_boot_valid",
        "left_slope_median",
        "right_slope_median",
        "split_used",
        "split_idx",
        "n_points_low",
        "n_points_high",
        "slope_low_median",
        "slope_low_err_half68",
        "slope_low_wls",
        "intercept_low_wls",
        "slope_high_median",
        "slope_high_err_half68",
        "slope_high_wls",
        "intercept_high_wls",
        "notes",
        "split_notes",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {out_csv}")
    plot_best_fit_overlays(plot_rows)

    # Human-readable summary.
    print("\nSlope summary (median VSF, uncertainty-aware):")
    for r in rows:
        slope_txt = f"{r['slope_median']:.3f} +/- {r['slope_err_half68']:.3f}"
        if not np.isfinite(r["slope_median"]) or not np.isfinite(r["slope_err_half68"]):
            slope_txt = "nan"

        ks_txt = "not tested"
        if r["ks_tested"]:
            sig = "YES" if r["ks_significant"] else "no"
            pval = r["ks_pvalue"]
            br = r["break_r_pc"]
            ks_txt = f"p={pval:.3e}, inflection={sig}, break~{br:.3f} pc"

        split_txt = "piecewise not used"
        if r.get("split_used", False):
            split_txt = (
                f"below={r['slope_low_median']:.3f} +/- {r['slope_low_err_half68']:.3f}, "
                f"above={r['slope_high_median']:.3f} +/- {r['slope_high_err_half68']:.3f}"
            )
        elif r.get("split_notes", ""):
            split_txt = r["split_notes"]

        print(f"  {r['object']} {r['line']}: slope={slope_txt}; KS: {ks_txt}; split: {split_txt}")


if __name__ == "__main__":
    main()
