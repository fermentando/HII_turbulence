"""Fit continuous segmented VSF power laws and select breakpoints with BIC.

This is a cleaner parallel implementation for testing line/breakpoint choices.
It reads median all-subwindow VSF curves, fits 1-, 2-, and 3-segment continuous
models in log-log space, and writes a summary table plus a diagnostic grid plot.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import re
from dataclasses import dataclass
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.stats import kstest, ks_2samp


LINE_ORDER = ["NII", "SII", "SIII"]
LINE_COLORS = {"NII": "tab:blue", "SII": "tab:red", "SIII": "tab:green"}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))

style_path = REPO_ROOT / "custom_plot.mplstyle"
if style_path.exists():
    plt.style.use(str(style_path))

MIN_SEPARATION_ARCSEC = 37.0
SPECIAL_DISTANCES_PC = {
    "Flame_nebula": 400.0,
    "Orion": 400.0,
    "Rosette_nebula": 1500.0,
}

@dataclass
class SegmentedFit:
    n_segments: int
    break_indices: tuple[int, ...]
    break_log_r: tuple[float, ...]
    params: np.ndarray
    covariance: np.ndarray
    slopes: tuple[float, ...]
    slope_errs: tuple[float, ...]
    intercept: float
    chi2: float
    bic: float
    n_points: int
    n_params: int
    residual_ks_pvalue: float
    adjacent_residual_ks_min_pvalue: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit continuous segmented VSF models and select breakpoints with BIC."
    )
    parser.add_argument(
        "--npz-glob",
        default="NPZ/VSF_*_selected_subregions_logflux2.8.npz",
        help="Input NPZ glob. Default reads selected subregions VSFs.",
    )
    parser.add_argument(
        "--out-csv",
        default="Figures/manuscript/03_slopes_turnover/segmented_vsf_breakpoint_fits_selected_subregions_logflux2.8.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-fig",
        default="Figures/manuscript/03_slopes_turnover/segmented_vsf_bestfits_selected_subregions_logflux2.8.png",
        help="Output diagnostic plot path.",
    )
    parser.add_argument("--max-segments", type=int, default=3)
    parser.add_argument("--min-points-per-segment", type=int, default=5)
    parser.add_argument("--min-delta-bic", type=float, default=10.0)
    parser.add_argument("--min-counts", type=float, default=1.0)
    parser.add_argument(
        "--min-separation-arcsec",
        type=float,
        default=MIN_SEPARATION_ARCSEC,
        help="Drop VSF bins below this angular separation before fitting.",
    )
    return parser.parse_args()


def parse_object_line_and_dataset(path: str) -> tuple[str, str, str] | tuple[None, None, None]:
    name = os.path.basename(path)
    match = re.match(
        r"^VSF_(.+)_([A-Za-z0-9]+)_(median_allsubwindows|mean_allsubwindows|total_fullfov|total_fullfov_fast|selected_subregions_logflux[0-9.]+)\.npz$",
        name,
    )
    if not match:
        return None, None, None
    return match.group(1), match.group(2).upper(), match.group(3)


def arcsec_to_pc(angle_arcsec: float, distance_pc: float) -> float:
    return distance_pc * np.deg2rad(angle_arcsec / 3600.0)


def get_heliocentric_distance(object_name: str) -> float:
    if object_name in SPECIAL_DISTANCES_PC:
        return SPECIAL_DISTANCES_PC[object_name]

    lookup_name = {
        "Flame_nebula": "Flame",
        "Rosette_nebula": "Rosette",
    }.get(object_name, object_name)

    table_path = REPO_ROOT / "extra_HII_regions.fits"
    with fits.open(table_path) as hdul:
        dist_data = hdul[1].data
        try:
            index = list(dist_data["HII Region"]).index(lookup_name)
        except ValueError as exc:
            raise ValueError(f"Distance not found for {object_name}") from exc
        return float(dist_data["Heliocentric_distance"][index]) * 1e3


def npz_min_separation_pc(path: str) -> float | None:
    with np.load(path) as data:
        if "min_separation_pc" not in data.files:
            return None
        value = float(np.asarray(data["min_separation_pc"]).item())
        return value if np.isfinite(value) and value > 0 else None


def load_curve(
    path: str,
    min_counts: float,
    min_log_r: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    vsf = np.asarray(data["vsf"], dtype=float)
    vsf_err = np.asarray(data["vsf_err"], dtype=float)
    log_r = np.asarray(data["log_centers"], dtype=float)
    counts = np.asarray(data["counts"], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log10(vsf)
        y_err = vsf_err / (vsf * np.log(10.0))

    valid = np.isfinite(log_r) & np.isfinite(y) & np.isfinite(y_err)
    valid &= (vsf > 0) & (vsf_err > 0) & (counts >= min_counts)
    if min_log_r is not None:
        valid &= log_r >= min_log_r
    return log_r[valid], y[valid], y_err[valid], counts[valid]


def break_log_positions(x: np.ndarray, break_indices: tuple[int, ...]) -> tuple[float, ...]:
    return tuple(float(0.5 * (x[idx - 1] + x[idx])) for idx in break_indices)


def design_matrix(x: np.ndarray, taus: tuple[float, ...]) -> np.ndarray:
    columns = [np.ones_like(x), x]
    for tau in taus:
        columns.append(np.maximum(0.0, x - tau))
    return np.column_stack(columns)


def segment_slices(n_points: int, break_indices: tuple[int, ...]) -> list[slice]:
    edges = (0,) + tuple(break_indices) + (n_points,)
    return [slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def weighted_least_squares(x: np.ndarray, y: np.ndarray, y_err: np.ndarray, taus: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    X = design_matrix(x, taus)
    w_sqrt = 1.0 / y_err
    Xw = X * w_sqrt[:, None]
    yw = y * w_sqrt
    params, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ params
    resid = y - pred
    chi2 = float(np.sum(np.square(resid / y_err)))

    xtwx = Xw.T @ Xw
    covariance = np.linalg.pinv(xtwx)
    return params, covariance, chi2, resid


def slope_values_and_errors(params: np.ndarray, covariance: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    slopes = []
    slope_errs = []
    coeff = np.zeros(len(params), dtype=float)
    coeff[1] = 1.0
    for seg_idx in range(len(params) - 1):
        if seg_idx > 0:
            coeff[seg_idx + 1] = 1.0
        slope = float(coeff @ params)
        var = float(coeff @ covariance @ coeff)
        slopes.append(slope)
        slope_errs.append(float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan)
    return tuple(slopes), tuple(slope_errs)


def residual_diagnostics(resid: np.ndarray, y_err: np.ndarray, break_indices: tuple[int, ...]) -> tuple[float, float]:
    z = resid / y_err
    z = z[np.isfinite(z)]
    ks_p = float(kstest(z, "norm").pvalue) if z.size >= 3 else np.nan

    segment_z = []
    for slc in segment_slices(len(resid), break_indices):
        vals = resid[slc] / y_err[slc]
        vals = vals[np.isfinite(vals)]
        segment_z.append(vals)

    adjacent_p = []
    for left, right in zip(segment_z[:-1], segment_z[1:]):
        if left.size >= 2 and right.size >= 2:
            adjacent_p.append(float(ks_2samp(left, right, alternative="two-sided", method="asymp").pvalue))
    min_adjacent = float(np.min(adjacent_p)) if adjacent_p else np.nan
    return ks_p, min_adjacent


def fit_for_breaks(x: np.ndarray, y: np.ndarray, y_err: np.ndarray, break_indices: tuple[int, ...]) -> SegmentedFit:
    taus = break_log_positions(x, break_indices)
    params, covariance, chi2, resid = weighted_least_squares(x, y, y_err, taus)
    slopes, slope_errs = slope_values_and_errors(params, covariance)
    n_segments = len(break_indices) + 1
    n_points = len(x)
    n_params = 2 * n_segments
    bic = float(chi2 + n_params * np.log(n_points))
    ks_p, adjacent_ks_p = residual_diagnostics(resid, y_err, break_indices)
    return SegmentedFit(
        n_segments=n_segments,
        break_indices=tuple(int(i) for i in break_indices),
        break_log_r=taus,
        params=params,
        covariance=covariance,
        slopes=slopes,
        slope_errs=slope_errs,
        intercept=float(params[0]),
        chi2=chi2,
        bic=bic,
        n_points=n_points,
        n_params=n_params,
        residual_ks_pvalue=ks_p,
        adjacent_residual_ks_min_pvalue=adjacent_ks_p,
    )


def candidate_breaks(n_points: int, n_segments: int, min_points_per_segment: int) -> list[tuple[int, ...]]:
    if n_segments == 1:
        return [()]

    valid_indices = range(min_points_per_segment, n_points - min_points_per_segment + 1)
    out = []
    for combo in itertools.combinations(valid_indices, n_segments - 1):
        edges = (0,) + combo + (n_points,)
        lengths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        if min(lengths) >= min_points_per_segment:
            out.append(tuple(int(i) for i in combo))
    return out


def best_fit_by_segment_count(
    x: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    max_segments: int,
    min_points_per_segment: int,
) -> dict[int, SegmentedFit]:
    max_allowed = min(max_segments, len(x) // min_points_per_segment)
    fits: dict[int, SegmentedFit] = {}
    for n_segments in range(1, max_allowed + 1):
        candidates = candidate_breaks(len(x), n_segments, min_points_per_segment)
        segment_fits = [fit_for_breaks(x, y, y_err, breaks) for breaks in candidates]
        if segment_fits:
            fits[n_segments] = min(segment_fits, key=lambda f: f.bic)
    return fits


def select_fit(fits: dict[int, SegmentedFit], min_delta_bic: float) -> SegmentedFit:
    selected = fits[1]
    for n_segments in sorted(k for k in fits if k > 1):
        candidate = fits[n_segments]
        simpler_best_bic = min(fits[k].bic for k in fits if k < n_segments)
        if simpler_best_bic - candidate.bic >= min_delta_bic and candidate.bic < selected.bic:
            selected = candidate
    return selected


def evaluate_model(fit: SegmentedFit, x: np.ndarray) -> np.ndarray:
    return design_matrix(x, fit.break_log_r) @ fit.params

def evaluate_model_uncertainty(fit: SegmentedFit, x: np.ndarray) -> np.ndarray:
    X = design_matrix(x, fit.break_log_r)
    var = np.einsum("ij,jk,ik->i", X, fit.covariance, X)
    return np.sqrt(np.clip(var, 0.0, np.inf))


def dense_segment_line(fit: SegmentedFit, x: np.ndarray, segment_index: int) -> tuple[np.ndarray, np.ndarray]:
    slc = segment_slices(len(x), fit.break_indices)[segment_index]
    x_seg = x[slc]
    xx = np.linspace(float(np.min(x_seg)), float(np.max(x_seg)), 80)
    yy = evaluate_model(fit, xx)
    return xx, yy


def padded(values: tuple[float, ...], length: int) -> list[float]:
    return [float(values[i]) if i < len(values) else np.nan for i in range(length)]


def make_summary_row(
    path: str,
    obj: str,
    line: str,
    dataset: str,
    fits: dict[int, SegmentedFit],
    selected: SegmentedFit,
    min_separation_pc: float,
    min_separation_arcsec: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "file": path,
        "object": obj,
        "line": line,
        "dataset": dataset,
        "min_separation_arcsec": min_separation_arcsec,
        "min_separation_pc": min_separation_pc,
        "n_points": selected.n_points,
        "selected_segments": selected.n_segments,
        "selected_bic": selected.bic,
        "selected_chi2": selected.chi2,
        "selected_residual_ks_pvalue": selected.residual_ks_pvalue,
        "selected_adjacent_residual_ks_min_pvalue": selected.adjacent_residual_ks_min_pvalue,
    }

    for n_segments in (1, 2, 3):
        fit = fits.get(n_segments)
        row[f"bic_m{n_segments}"] = fit.bic if fit is not None else np.nan
        row[f"chi2_m{n_segments}"] = fit.chi2 if fit is not None else np.nan
        row[f"breaks_m{n_segments}_pc"] = (
            ";".join(f"{10.0 ** b:.6g}" for b in fit.break_log_r) if fit is not None else ""
        )

    break_pcs = tuple(10.0 ** b for b in selected.break_log_r)
    for idx, val in enumerate(padded(break_pcs, 2), start=1):
        row[f"break{idx}_pc"] = val
    for idx, val in enumerate(padded(selected.slopes, 3), start=1):
        row[f"slope{idx}"] = val
    for idx, val in enumerate(padded(selected.slope_errs, 3), start=1):
        row[f"slope{idx}_err"] = val

    return row


def plot_all_curves(plot_payloads: list[dict[str, object]], out_fig: str) -> None:
    objects = sorted({str(p["object"]) for p in plot_payloads})
    objects = ["Flame_nebula", "Rosette_nebula", "RCW79"]
    line_to_col = {line: i for i, line in enumerate(LINE_ORDER)}
    fig, axes = plt.subplots(
        len(objects),
        len(LINE_ORDER),
        figsize=(4.8 * len(LINE_ORDER), 3.2 * len(objects)),
        squeeze=False,
    )

    payload_map = {(p["object"], p["line"]): p for p in plot_payloads}
    for row_idx, obj in enumerate(objects):
        for line in LINE_ORDER:
            ax = axes[row_idx, line_to_col[line]]
            payload = payload_map.get((obj, line))
            if payload is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = payload["x"]
            y = payload["y"]
            y_err = payload["y_err"]
            selected = payload["selected"]
            fits = payload["fits"]

            color = LINE_COLORS.get(line, "black")
            r = 10.0 ** x
            vsf = 10.0 ** y
            vsf_err_linear = np.log(10.0) * vsf * y_err
            ax.errorbar(
                r,
                vsf,
                yerr=vsf_err_linear,
                fmt="o",
                ms=3.2,
                capsize=1.8,
                color=color,
                ecolor=color,
                alpha=0.78,
                label="VSF",
            )

            for seg_idx in range(selected.n_segments):
                xx, yy = dense_segment_line(selected, x, seg_idx)
                yy_err = 5 * evaluate_model_uncertainty(selected, xx)

                r_fit = 10.0 ** xx
                vsf_fit = 10.0 ** yy
                vsf_fit_lo = 10.0 ** (yy - yy_err)
                vsf_fit_hi = 10.0 ** (yy + yy_err)

                print(obj, line, "yy_err range:", np.nanmin(yy_err), np.nanmax(yy_err))
                print("relative band:", np.nanmax(10**yy_err - 1))

                ax.plot(r_fit, vsf_fit, color="black", lw=2.0, alpha=0.85, linestyle="--")
                ax.fill_between(
                    r_fit,
                    vsf_fit_lo,
                    vsf_fit_hi,
                    color=color,
                    alpha=0.35,
                    linewidth=0,
                )

            for break_log in selected.break_log_r:
                ax.axvline(10.0 ** break_log, color="dimgray", lw=1.2, ls=":")

            #bic_text = ", ".join(
            #    f"BIC{m}={fits[m].bic:.1f}" for m in sorted(fits)
            #)
            slope_text = ", ".join(f"m{i + 1}={s:.2f}" for i, s in enumerate(selected.slopes))
            ax.text(
                0.04,
                0.96,
                f"{slope_text}",#\n{bic_text}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=10.0,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
           # ax.set_title(f"{obj} {line}", fontsize=9)
            if row_idx == len(objects) - 1:
                ax.set_xlabel("r [pc]")
            if line_to_col[line] == 0:
                ax.set_ylabel(r"$S_2$ [(km/s)$^2$]")

    fig.tight_layout()
    outdir = os.path.dirname(os.path.abspath(out_fig))
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_segments < 1:
        raise ValueError("--max-segments must be >= 1")

    npz_paths = sorted(glob.glob(args.npz_glob))
    if not npz_paths:
        raise FileNotFoundError(f"No files matched: {args.npz_glob}")

    rows = []
    plot_payloads = []
    for path in npz_paths:
        obj, line, dataset = parse_object_line_and_dataset(path)
        if obj is None or line is None or dataset is None:
            print(f"Skipping unexpected filename: {path}")
            continue

        min_separation_pc = npz_min_separation_pc(path)
        if min_separation_pc is None:
            min_separation_pc = arcsec_to_pc(
                args.min_separation_arcsec,
                get_heliocentric_distance(obj),
            )
        min_log_r = np.log10(min_separation_pc)

        x, y, y_err, counts = load_curve(
            path,
            min_counts=args.min_counts,
            min_log_r=min_log_r,
        )
        if len(x) < args.min_points_per_segment:
            print(f"Skipping {obj} {line}: only {len(x)} valid points")
            continue

        fits = best_fit_by_segment_count(
            x,
            y,
            y_err,
            max_segments=args.max_segments,
            min_points_per_segment=args.min_points_per_segment,
        )
        if 1 not in fits:
            print(f"Skipping {obj} {line}: no valid segmented fit")
            continue

        selected = select_fit(fits, min_delta_bic=args.min_delta_bic)
        rows.append(
            make_summary_row(
                path,
                obj,
                line,
                dataset,
                fits,
                selected,
                min_separation_pc,
                args.min_separation_arcsec,
            )
        )
        plot_payloads.append(
            {
                "object": obj,
                "line": line,
                "x": x,
                "y": y,
                "y_err": y_err,
                "counts": counts,
                "fits": fits,
                "selected": selected,
                "min_separation_pc": min_separation_pc,
            }
        )
        print(
            f"{obj:16s} {line:4s}: selected {selected.n_segments} segment(s), "
            f"BIC={selected.bic:.2f}, breaks="
            f"{[round(10.0 ** b, 3) for b in selected.break_log_r]}, "
            f"r_min={min_separation_pc:.4g} pc"
        )

    if not rows:
        raise RuntimeError("No fits completed.")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_all_curves(plot_payloads, args.out_fig)
    print(f"Wrote summary CSV: {args.out_csv}")
    print(f"Wrote diagnostic plot: {args.out_fig}")


if __name__ == "__main__":
    main()
