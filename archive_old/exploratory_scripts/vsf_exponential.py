"""vsf_exponential.py
====================
KS-optimal exponential fitting for pre-binned velocity-structure-function
(VSF) data stored in NPZ files.

Expected NPZ layout
-------------------
    r        = data['log_centers']   # 1-D array of lag scales (bin centres, log-spaced)
    vsf      = data['vsf']           # 1-D array of VSF values  (same length as r)
    vsf_err  = data['vsf_err']       # 1-D array of uncertainties (same length as r)

The module fits  ``f(r) = A * exp(-r / lam)``  on a window [xmin, xmax] by
scanning a grid of candidate (xmin, xmax) pairs and selecting the pair that
minimises the KS-like statistic between the normalised empirical CDF of the
VSF and the model CDF.

Usage (CLI)
-----------
::

    python vsf_exponential.py  input.npz  output_dir/  [options]

Or import directly::

    from vsf_exponential import run_vsf_pipeline
    result = run_vsf_pipeline(Path("data.npz"), Path("out/"))
"""

from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

_EPS = 1e-300


# ===========================================================================
# Exponential model  f(r) = A * exp(-r / lam)
# ===========================================================================

def _exp_model(r: NDArray[np.float64], A: float, lam: float) -> NDArray[np.float64]:
    return A * np.exp(-r / lam)


def _exp_cdf_truncated(
    r: NDArray[np.float64], lam: float, xmin: float, xmax: float
) -> NDArray[np.float64]:
    """
    CDF of the truncated exponential on [xmin, xmax]:
        P(R <= r | xmin <= R <= xmax)
    Used to build the model CDF for the KS statistic.
    """
    z_lo = math.exp(-xmin / lam)
    z_hi = math.exp(-xmax / lam)
    denom = max(z_lo - z_hi, _EPS)
    r_clipped = np.clip(r, xmin, xmax)
    return np.clip((z_lo - np.exp(-r_clipped / lam)) / denom, 0.0, 1.0)


# ===========================================================================
# Weighted chi-squared fit for (A, lam) in a window
# ===========================================================================

def _fit_exp_window(
    r: NDArray[np.float64],
    vsf: NDArray[np.float64],
    vsf_err: NDArray[np.float64],
    xmin: float,
    xmax: float,
    n_starts: int = 6,
) -> tuple[float, float, float]:
    """
    Fit  f(r) = A * exp(-r / lam)  by weighted least squares (chi-squared)
    on the window [xmin, xmax].

    Weights are  w_i = 1 / vsf_err_i^2.  Points with non-positive or
    non-finite error are given weight zero.

    Returns
    -------
    A_hat, lam_hat, chi2_red
        Best-fit amplitude, decay length, and reduced chi-squared.
    """
    mask = (
        (r >= xmin) & (r <= xmax)
        & np.isfinite(vsf) & np.isfinite(vsf_err)
        & (vsf_err > 0) & (vsf > 0)
    )
    r_w   = r[mask]
    v_w   = vsf[mask]
    e_w   = vsf_err[mask]
    n_pts = int(mask.sum())

    if n_pts < 2:
        return np.nan, np.nan, np.inf

    w = 1.0 / e_w ** 2

    # ---- Linear regression in log-space for a fast initial guess ----------
    # log(vsf) = log(A) - r/lam  with propagated weights w_log = w * vsf^2
    log_v = np.log(v_w)
    w_log = w * v_w ** 2
    X = np.column_stack([np.ones(n_pts), -r_w])
    XtW = X.T * w_log
    try:
        coeffs = np.linalg.solve(XtW @ X, XtW @ log_v)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.inf

    logA_hat, inv_lam_hat = float(coeffs[0]), float(coeffs[1])
    lam0 = 1.0 / inv_lam_hat if inv_lam_hat > 0 else (xmax - xmin)

    # ---- Nonlinear WLS refinement in original space -----------------------
    span = xmax - xmin
    lam_lo = span / 500.0
    lam_hi = span * 200.0

    def chi2_of_lam(lam_try: float) -> float:
        pred = np.exp(-r_w / lam_try)
        A_opt = float(np.sum(w * v_w * pred) / max(np.sum(w * pred ** 2), _EPS))
        if A_opt <= 0:
            return np.inf
        res = v_w - A_opt * pred
        return float(np.sum(w * res ** 2))

    starts = np.geomspace(
        max(lam_lo, lam0 / 10.0),
        min(lam_hi, lam0 * 10.0),
        n_starts,
    )
    best_c2, best_lam = np.inf, lam0
    for s in starts:
        try:
            opt = minimize_scalar(
                chi2_of_lam,
                bounds=(lam_lo, lam_hi),
                method="bounded",
                options={"xatol": lam_lo * 1e-4},
            )
            if opt.fun < best_c2:
                best_c2, best_lam = float(opt.fun), float(opt.x)
        except Exception:
            pass

    # Recompute A at the best lam
    pred    = np.exp(-r_w / best_lam)
    A_hat   = float(np.sum(w * v_w * pred) / max(np.sum(w * pred ** 2), _EPS))
    if A_hat <= 0:
        return np.nan, np.nan, np.inf

    dof      = max(n_pts - 2, 1)
    res      = v_w - A_hat * pred
    chi2_red = float(np.sum(w * res ** 2)) / dof

    return A_hat, best_lam, chi2_red


# ===========================================================================
# KS statistic between VSF-weighted empirical CDF and model CDF
# ===========================================================================

def _ks_distance(
    r: NDArray[np.float64],
    vsf: NDArray[np.float64],
    A: float,
    lam: float,
    xmin: float,
    xmax: float,
) -> float:
    """
    KS distance between the empirical CDF (VSF values used as weights,
    treating the curve as a density) and the model exponential CDF,
    evaluated at all r-points inside [xmin, xmax].

    Mirrors ``_ks_distance_binned_pl`` in models.py.
    """
    mask = (r >= xmin) & (r <= xmax) & np.isfinite(vsf) & (vsf > 0)
    r_w  = r[mask]
    v_w  = vsf[mask]
    if r_w.size < 2:
        return np.inf

    # Empirical CDF: cumulative sum of vsf weights
    emp           = np.cumsum(v_w) / max(float(v_w.sum()), _EPS)
    model         = _exp_cdf_truncated(r_w, lam, xmin, xmax)
    model[-1]     = 1.0   # force closure at the right edge

    return float(np.max(np.abs(emp - model)))


# ===========================================================================
# Uncertainty via Hessian curvature + posterior draws
# ===========================================================================

def _hessian_1d(f: Any, x0: float, h: float = 1e-4) -> float:
    return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h)


def _parameter_uncertainty(
    r: NDArray[np.float64],
    vsf: NDArray[np.float64],
    vsf_err: NDArray[np.float64],
    xmin: float,
    xmax: float,
    lam_hat: float,
    A_hat: float,
    n_draws: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Gaussian uncertainty for lam via curvature of chi² in log(lam) space,
    then Monte-Carlo draws for 16/50/84 percentile intervals on both lam and A.
    Identical strategy to models.py's Hessian-based uncertainty blocks.
    """
    mask = (
        (r >= xmin) & (r <= xmax)
        & np.isfinite(vsf) & np.isfinite(vsf_err)
        & (vsf_err > 0) & (vsf > 0)
    )
    r_w, v_w, e_w = r[mask], vsf[mask], vsf_err[mask]
    w = 1.0 / e_w ** 2

    def nchi2_z(z: float) -> float:
        lam = math.exp(z)
        pred = np.exp(-r_w / lam)
        A = float(np.sum(w * v_w * pred) / max(np.sum(w * pred ** 2), _EPS))
        if A <= 0:
            return 1e30
        return float(np.sum(w * (v_w - A * pred) ** 2))

    z0     = math.log(lam_hat)
    h_step = max(abs(z0) * 0.02, 1e-3)
    sec    = _hessian_1d(nchi2_z, z0, h=h_step)
    var_z  = 1.0 / max(sec, 1e-8) if sec > 0 else 1.0

    rng      = np.random.default_rng(seed)
    z_draws  = rng.normal(z0, math.sqrt(var_z), size=n_draws)
    lam_draws = np.exp(z_draws)

    # For each lam draw, recompute the analytically optimal A
    A_draws = []
    for ld in lam_draws:
        pred  = np.exp(-r_w / ld)
        denom = float(np.sum(w * pred ** 2))
        A_d   = float(np.sum(w * v_w * pred) / max(denom, _EPS))
        A_draws.append(max(A_d, _EPS))
    A_draws = np.array(A_draws, dtype=float)

    return {
        "lam": {
            "p16": float(np.quantile(lam_draws, 0.16)),
            "p50": float(np.quantile(lam_draws, 0.50)),
            "p84": float(np.quantile(lam_draws, 0.84)),
        },
        "A": {
            "p16": float(np.quantile(A_draws, 0.16)),
            "p50": float(np.quantile(A_draws, 0.50)),
            "p84": float(np.quantile(A_draws, 0.84)),
        },
        "lam_log_var": float(var_z),
        "n_draws": n_draws,
    }


# ===========================================================================
# Result dataclass
# ===========================================================================

@dataclass
class ExpFitResult:
    xmin:     float
    xmax:     float
    lam:      float       # exponential decay length
    A:        float       # amplitude
    chi2_red: float       # reduced chi-squared of the best fit
    ks:       float       # KS distance at the best (xmin, xmax)
    n_points: int         # r-points inside the best window
    uncertainty: dict[str, Any] = field(default_factory=dict)
    ks_grid:   NDArray[np.float64] | None = None
    xmin_grid: NDArray[np.float64] | None = None
    xmax_grid: NDArray[np.float64] | None = None


# ===========================================================================
# Core: KS-optimal (xmin, xmax) scan
# ===========================================================================

def fit_vsf_exponential(
    r: NDArray[np.float64],
    vsf: NDArray[np.float64],
    vsf_err: NDArray[np.float64],
    n_xmin_candidates: int = 12,
    n_xmax_candidates: int = 12,
    min_window_points: int = 4,
    global_xmin: float | None = None,
    global_xmax: float | None = None,
    n_draws: int = 400,
    seed: int = 0,
    verbose: bool = True,
) -> ExpFitResult:
    """
    Fit  f(r) = A * exp(-r / lam)  to a pre-binned VSF curve.

    Parameters
    ----------
    r, vsf, vsf_err :
        Pre-binned VSF data — ``data['log_centers']``, ``data['vsf']``,
        ``data['vsf_err']`` from the NPZ.
    n_xmin_candidates, n_xmax_candidates :
        Grid resolution for the (xmin, xmax) KS scan.  Increase for finer
        coverage at the cost of runtime.
    min_window_points :
        Minimum number of valid r-points required inside a candidate window.
    global_xmin, global_xmax :
        Hard limits on the scan range.  Defaults to the min/max of finite r.
    n_draws :
        Posterior draws for uncertainty estimation.
    seed :
        RNG seed.
    verbose :
        Print progress.

    Returns
    -------
    ExpFitResult
        Best-fit xmin, xmax, A, lam, KS distance, chi2/dof, and uncertainties.
    """
    r       = np.asarray(r,       dtype=float)
    vsf     = np.asarray(vsf,     dtype=float)
    vsf_err = np.asarray(vsf_err, dtype=float)

    valid = (
        np.isfinite(r) & np.isfinite(vsf) & np.isfinite(vsf_err)
        & (vsf > 0) & (vsf_err > 0)
    )
    if valid.sum() < min_window_points:
        raise ValueError(
            f"Only {valid.sum()} valid data points — need at least {min_window_points}."
        )

    r_v   = r[valid]
    vsf_v = vsf[valid]
    err_v = vsf_err[valid]

    g_lo = float(global_xmin) if global_xmin is not None else float(r_v.min())
    g_hi = float(global_xmax) if global_xmax is not None else float(r_v.max())

    xmin_cands = np.geomspace(g_lo,        g_hi * 0.5, n_xmin_candidates)
    xmax_cands = np.geomspace(g_lo * 2.0,  g_hi,       n_xmax_candidates)

    if verbose:
        print(
            f"  [vsf_exp] r range [{g_lo:.4g}, {g_hi:.4g}], "
            f"grid {n_xmin_candidates}×{n_xmax_candidates}"
        )

    grid_xmin: list[float] = []
    grid_xmax: list[float] = []
    grid_ks:   list[float] = []

    best_ks = np.inf
    best: dict[str, Any] = {}

    for xmin_c in xmin_cands:
        for xmax_c in xmax_cands:
            if xmax_c <= xmin_c * 1.2:
                continue

            mask = (r_v >= xmin_c) & (r_v <= xmax_c)
            if mask.sum() < min_window_points:
                continue

            A, lam, chi2_red = _fit_exp_window(r_v, vsf_v, err_v, xmin_c, xmax_c)
            if not (np.isfinite(A) and np.isfinite(lam) and A > 0 and lam > 0):
                continue

            ks = _ks_distance(r_v, vsf_v, A, lam, xmin_c, xmax_c)

            grid_xmin.append(xmin_c)
            grid_xmax.append(xmax_c)
            grid_ks.append(ks)

            if ks < best_ks:
                best_ks = ks
                best = {
                    "xmin": xmin_c, "xmax": xmax_c,
                    "A": A, "lam": lam,
                    "chi2_red": chi2_red,
                    "n_points": int(mask.sum()),
                }

    if not best:
        raise RuntimeError(
            "No valid (xmin, xmax) window found. "
            "Try lowering --min-window-points or adjusting --global-xmin/xmax."
        )

    unc = _parameter_uncertainty(
        r_v, vsf_v, err_v,
        xmin=best["xmin"], xmax=best["xmax"],
        lam_hat=best["lam"], A_hat=best["A"],
        n_draws=n_draws, seed=seed,
    )

    if verbose:
        print(
            f"  [vsf_exp] best  xmin={best['xmin']:.4g}  xmax={best['xmax']:.4g}  "
            f"lam={best['lam']:.4g}  A={best['A']:.4g}  "
            f"KS={best_ks:.4f}  chi2/dof={best['chi2_red']:.3f}"
        )

    return ExpFitResult(
        xmin=best["xmin"],
        xmax=best["xmax"],
        lam=best["lam"],
        A=best["A"],
        chi2_red=best["chi2_red"],
        ks=best_ks,
        n_points=best["n_points"],
        uncertainty=unc,
        ks_grid=np.array(grid_ks,   dtype=float) if grid_ks   else None,
        xmin_grid=np.array(grid_xmin, dtype=float) if grid_xmin else None,
        xmax_grid=np.array(grid_xmax, dtype=float) if grid_xmax else None,
    )


# ===========================================================================
# Pipeline: handles 1-D (single snapshot) and 2-D (multiple snapshots) arrays
# ===========================================================================

def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


def _snapshot_number(key: str, fallback: int) -> int:
    m = re.search(r"(\d+)", key)
    if m:
        return int(m.group(1))
    return fallback


def _result_to_dict(key: str, snap: int, res: ExpFitResult) -> dict[str, Any]:
    return {
        "key": key,
        "snapshot": snap,
        "xmin": res.xmin,
        "xmax": res.xmax,
        "params": {"A": res.A, "lam": res.lam},
        "chi2_red": res.chi2_red,
        "ks": res.ks,
        "n_points": res.n_points,
        "uncertainty": res.uncertainty,
        "ks_scan": {
            "xmin_grid": res.xmin_grid.tolist() if res.xmin_grid is not None else [],
            "xmax_grid": res.xmax_grid.tolist() if res.xmax_grid is not None else [],
            "ks_grid":   res.ks_grid.tolist()   if res.ks_grid   is not None else [],
        },
    }


def run_vsf_pipeline(
    input_npz: Path,
    output_dir: Path,
    r_key: str = "log_centers",
    vsf_key: str = "vsf",
    vsf_err_key: str = "vsf_err",
    n_xmin_candidates: int = 12,
    n_xmax_candidates: int = 12,
    min_window_points: int = 4,
    global_xmin: float | None = None,
    global_xmax: float | None = None,
    n_draws: int = 400,
    continue_on_error: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Load ``(r, vsf, vsf_err)`` from *input_npz* and fit the exponential model.

    - **1-D arrays** → single fit, result saved as ``vsf_snapshot_0.json``.
    - **2-D arrays** (shape ``n_snapshots × n_bins``) → one fit per row;
      r is shared if 1-D, or per-row if also 2-D.

    All results are aggregated into ``vsf_fit_summary.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results:  list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if verbose:
        print(f"[vsf_exp] input  = {input_npz}")
        print(f"[vsf_exp] output = {output_dir}")

    with np.load(input_npz) as data:
        r_raw       = np.asarray(data[r_key],       dtype=float)
        vsf_raw     = np.asarray(data[vsf_key],     dtype=float)
        vsf_err_raw = np.asarray(data[vsf_err_key], dtype=float)

    # Build list of (key, snapshot_index, r, vsf, vsf_err) tuples
    if vsf_raw.ndim == 1:
        datasets = [("snapshot_0", 0, r_raw.ravel(), vsf_raw, vsf_err_raw)]
    elif vsf_raw.ndim == 2:
        n_snap = vsf_raw.shape[0]
        if verbose:
            print(f"[vsf_exp] found {n_snap} snapshots (2-D array)")
        datasets = []
        for i in range(n_snap):
            r_i = r_raw[i] if r_raw.ndim == 2 else r_raw
            datasets.append((f"snapshot_{i}", i, r_i, vsf_raw[i], vsf_err_raw[i]))
    else:
        raise ValueError(f"Unexpected vsf array shape: {vsf_raw.shape}")

    for key, snap, r, vsf, vsf_err in datasets:
        if verbose:
            print(f"[vsf_exp] fitting {key}  (snapshot={snap})")
        try:
            res = fit_vsf_exponential(
                r, vsf, vsf_err,
                n_xmin_candidates=n_xmin_candidates,
                n_xmax_candidates=n_xmax_candidates,
                min_window_points=min_window_points,
                global_xmin=global_xmin,
                global_xmax=global_xmax,
                n_draws=n_draws,
                seed=snap,
                verbose=verbose,
            )
            d = _result_to_dict(key, snap, res)
            results.append(d)
            out_json = output_dir / f"vsf_{_safe_key(key)}.json"
            out_json.write_text(json.dumps(d, indent=2))
        except Exception as exc:
            fail = {
                "key": key, "snapshot": snap,
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            failures.append(fail)
            if verbose:
                print(f"[vsf_exp][error] key={key}: {exc}")
            if not continue_on_error:
                raise

    summary: dict[str, Any] = {
        "input": str(input_npz),
        "n_datasets": len(results),
        "datasets": results,
        "failures": failures,
        "n_failed": len(failures),
    }
    summary_path = output_dir / "vsf_fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"[vsf_exp] summary written to {summary_path}")
    return summary


# ===========================================================================
# CLI
# ===========================================================================

def _main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fit f(r) = A * exp(-r / lam) to pre-binned VSF data in an NPZ.\n"
            "Finds the (xmin, xmax) window that minimises the KS distance between\n"
            "the empirical VSF CDF and the fitted exponential CDF."
        )
    )
    p.add_argument("input_npz",  type=Path, help="Input .npz file.")
    p.add_argument("output_dir", type=Path, help="Output directory.")
    p.add_argument("--r-key",       default="log_centers",
                   help="NPZ key for the r (lag) array (default: log_centers).")
    p.add_argument("--vsf-key",     default="vsf",
                   help="NPZ key for VSF values (default: vsf).")
    p.add_argument("--vsf-err-key", default="vsf_err",
                   help="NPZ key for VSF uncertainties (default: vsf_err).")
    p.add_argument("--n-xmin-candidates", type=int, default=12,
                   help="xmin grid points (default: 12).")
    p.add_argument("--n-xmax-candidates", type=int, default=12,
                   help="xmax grid points (default: 12).")
    p.add_argument("--min-window-points", type=int, default=4,
                   help="Min r-points inside a candidate window (default: 4).")
    p.add_argument("--global-xmin", type=float, default=None,
                   help="Hard lower bound on xmin search space.")
    p.add_argument("--global-xmax", type=float, default=None,
                   help="Hard upper bound on xmax search space.")
    p.add_argument("--n-draws",     type=int,   default=400,
                   help="Posterior draws for uncertainty estimation (default: 400).")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Skip failing snapshots rather than raising.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress output.")
    args = p.parse_args()

    run_vsf_pipeline(
        input_npz=args.input_npz,
        output_dir=args.output_dir,
        r_key=args.r_key,
        vsf_key=args.vsf_key,
        vsf_err_key=args.vsf_err_key,
        n_xmin_candidates=args.n_xmin_candidates,
        n_xmax_candidates=args.n_xmax_candidates,
        min_window_points=args.min_window_points,
        global_xmin=args.global_xmin,
        global_xmax=args.global_xmax,
        n_draws=args.n_draws,
        continue_on_error=args.continue_on_error,
        verbose=not args.quiet,
    )

    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # Load a single fitted snapshot JSON
    save_path = f"{args.output_dir}/vsf_summary.png"
    with open(f"{args.output_dir}/vsf_snapshot_0.json") as f:
        data = json.load(f)

    xmin, xmax = data['xmin'], data['xmax']
    A, lam = data['params']['A'], data['params']['lam']
    r_fit = np.linspace(xmin*0.8, xmax*1.2, 200)
    vsf_fit = A * np.exp(-r_fit / lam)
    
    # Posterior draws
    lam_unc = data['uncertainty']['lam']
    lam_samples = np.random.normal(lam_unc['p50'], (lam_unc['p84']-lam_unc['p16'])/2, 400)
    
    # KS scan
    ks_grid = np.array(data['ks_scan']['ks_grid'])
    xmin_grid = np.array(data['ks_scan']['xmin_grid'])
    xmax_grid = np.array(data['ks_scan']['xmax_grid'])
    
    # Create figure
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    
    # Top-left: VSF + fit
    ax = axes[0,0]
    ax.plot(r_fit, vsf_fit, 'r-', label=f"Fit: A={A:.3g}, λ={lam:.3g}")
    ax.axvspan(xmin, xmax, color='red', alpha=0.1, label='Fit window')
    ax.set_xlabel("Lag r")
    ax.set_ylabel("VSF")
    ax.set_title("VSF Exponential Fit")
    ax.legend()
    
    # Top-right: residuals (if VSF data available)
    if 'r' in data and data['r']:
        r_data = np.array(data['r'])
        vsf_data = np.array(data['vsf'])
        vsf_fit_interp = A * np.exp(-r_data / lam)
        residuals = vsf_data - vsf_fit_interp
        ax = axes[0,1]
        ax.plot(r_data, residuals, 'ko-')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("Lag r")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals")
    else:
        axes[0,1].text(0.5,0.5,"No VSF data for residuals", ha='center', va='center')
        axes[0,1].axis('off')
    
    # Bottom-left: KS heatmap
    ax = axes[1,0]
    sc = ax.scatter(xmin_grid, xmax_grid, c=ks_grid, cmap='viridis', marker='s')
    plt.colorbar(sc, ax=ax, label='KS distance')
    ax.set_xlabel("xmin")
    ax.set_ylabel("xmax")
    ax.set_title("KS Scan Heatmap")
    
    # Bottom-right: λ posterior
    ax = axes[1,1]
    ax.hist(lam_samples, bins=20, alpha=0.7, color='purple')
    ax.set_xlabel("λ")
    ax.set_ylabel("Count")
    ax.set_title("λ Posterior Samples")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved summary figure to {save_path}")
if __name__ == "__main__":
    _main()