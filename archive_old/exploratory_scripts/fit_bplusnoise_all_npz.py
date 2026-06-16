#!/usr/bin/env python3
"""Batch-fit VSF NPZ files using the B(r) + Bnoise model.

Model:
    B(r) = 2 * sigma * (1 - 2^(-(r/r0)^m))
    VSF(r) = B(r) + Bnoise

This script reproduces the fitting setup used in the notebook cell:
- finite, positive (r, vsf) masking
- drop the largest-scale n_drop points
- bounded scipy curve_fit with free m

Outputs:
- Per-file fit plots in Figures/bplusnoise_fits/
- Summary CSV in results/bplusnoise_fit_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def B(r: np.ndarray, r0: float, m: float, sigma: float) -> np.ndarray:
    return 2.0 * sigma * (1.0 - 2.0 ** (-(r / r0) ** m))


def model_b_plus_noise(
    r: np.ndarray, r0: float, m: float, sigma: float, bnoise: float
) -> np.ndarray:
    return B(r, r0, m, sigma) + bnoise


def estimate_loglog_slope(r: np.ndarray, vsf: np.ndarray) -> tuple[float, float]:
    x = np.log10(r)
    y = np.log10(vsf)
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    slope = float(coeffs[0])
    slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    return slope, slope_err


def _unwrap_array(a: np.ndarray) -> np.ndarray:
    if isinstance(a, np.ndarray) and a.dtype == object and a.ndim == 0:
        return np.asarray(a.item())
    return np.asarray(a)


def load_r_vsf(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    if "log_centers" in data.files:
        logc = _unwrap_array(data["log_centers"]).astype(float)
        r_data = 10 ** logc
    elif "r" in data.files:
        r_data = _unwrap_array(data["r"]).astype(float)
    else:
        raise KeyError(f"No radial key in {npz_path.name}. Keys: {data.files}")

    if "vsf" in data.files:
        vsf_data = _unwrap_array(data["vsf"]).astype(float)
    elif "B" in data.files:
        vsf_data = _unwrap_array(data["B"]).astype(float)
    else:
        raise KeyError(f"No VSF key in {npz_path.name}. Keys: {data.files}")

    return np.asarray(r_data, dtype=float), np.asarray(vsf_data, dtype=float)


def fit_file(npz_path: Path, n_drop: int, maxfev: int) -> dict:
    r_data, vsf_data = load_r_vsf(npz_path)

    mask = np.isfinite(r_data) & np.isfinite(vsf_data) & (r_data > 0) & (vsf_data > 0)
    r_fit = r_data[mask]
    vsf_fit = vsf_data[mask]

    if r_fit.size <= n_drop + 5:
        raise RuntimeError(
            f"Not enough valid points ({r_fit.size}) to drop {n_drop} and fit"
        )

    if n_drop > 0:
        r_fit = r_fit[:-n_drop]
        vsf_fit = vsf_fit[:-n_drop]

    x_max = float(np.nanmax(r_fit))
    vsf_max = float(np.nanmax(vsf_fit))
    vsf_min = float(np.nanmin(vsf_fit))

    r0_lo, r0_hi = 0.01 * x_max, 2.0 * x_max
    m_lo, m_hi = 0.5, 2.0
    sigma_lo, sigma_hi = 0.25 * vsf_max, 2.0 * vsf_max
    bnoise_lo, bnoise_hi = 0.0, 3.0 * vsf_min

    p0 = [
        np.clip(np.nanmedian(r_fit), r0_lo, r0_hi),
        2.0 / 3.0,
        np.clip(np.nanpercentile(vsf_fit, 75) / 2.0, sigma_lo, sigma_hi),
        np.clip(np.nanpercentile(vsf_fit, 10), bnoise_lo, bnoise_hi),
    ]

    lower = [r0_lo, m_lo, sigma_lo, bnoise_lo]
    upper = [r0_hi, m_hi, sigma_hi, bnoise_hi]

    popt, pcov = curve_fit(
        model_b_plus_noise,
        r_fit,
        vsf_fit,
        p0=p0,
        bounds=(lower, upper),
        maxfev=maxfev,
    )
    perr = np.sqrt(np.diag(pcov))
    vsf_slope, vsf_slope_err = estimate_loglog_slope(r_fit, vsf_fit)

    return {
        "file": npz_path.name,
        "n_points": int(r_fit.size),
        "r_fit": r_fit,
        "vsf_fit": vsf_fit,
        "r0": float(popt[0]),
        "r0_err": float(perr[0]),
        "m": float(popt[1]),
        "m_err": float(perr[1]),
        "sigma": float(popt[2]),
        "sigma_err": float(perr[2]),
        "bnoise": float(popt[3]),
        "bnoise_err": float(perr[3]),
        "vsf_slope": float(vsf_slope),
        "vsf_slope_err": float(vsf_slope_err),
        "bounds": {
            "r0_lo": r0_lo,
            "r0_hi": r0_hi,
            "m_lo": m_lo,
            "m_hi": m_hi,
            "sigma_lo": sigma_lo,
            "sigma_hi": sigma_hi,
            "bnoise_lo": bnoise_lo,
            "bnoise_hi": bnoise_hi,
        },
    }


def save_plot(result: dict, out_png: Path) -> None:
    r_fit = result["r_fit"]
    vsf_fit = result["vsf_fit"]
    r0 = result["r0"]
    m = result["m"]
    sigma = result["sigma"]
    bnoise = result["bnoise"]
    m_err = result["m_err"]
    vsf_slope = result["vsf_slope"]
    vsf_slope_err = result["vsf_slope_err"]

    r_model = np.logspace(np.log10(np.nanmin(r_fit)), np.log10(np.nanmax(r_fit)), 300)
    y_model = model_b_plus_noise(r_model, r0, m, sigma, bnoise)
    y_b = B(r_model, r0, m, sigma)
    y_noise = np.full_like(r_model, bnoise)

    plt.figure(figsize=(8, 5))
    plt.loglog(r_fit, vsf_fit, "o", ms=4, alpha=0.65, label="VSF data (trimmed)")
    plt.loglog(r_model, y_model, "-", lw=2.2, label="Fit: B(r) + Bnoise")
    plt.loglog(r_model, y_b, "--", lw=2.0, label="B(r)")
    plt.loglog(
        r_model,
        y_noise,
        "-.",
        lw=1.8,
        color="darkorange",
        label=f"Bnoise = {bnoise:.3g}",
    )
    plt.axvline(r0, color="teal", ls="--", lw=1.8, label=f"r0 = {r0:.3g}")
    plt.xlabel("r")
    plt.ylabel("VSF")
    plt.title(f"B(r)+Bnoise fit: {result['file']}")
    plt.grid(True, which="both", alpha=0.25)
    slope_text = (
        f"Model slope m = {m:.3f} +/- {m_err:.3f}\n"
        f"Data log-log slope = {vsf_slope:.3f} +/- {vsf_slope_err:.3f}"
    )
    plt.text(
        0.04,
        0.96,
        slope_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.4"},
    )
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def write_summary(rows: list[dict], csv_path: Path) -> None:
    fields = [
        "file",
        "status",
        "message",
        "n_points",
        "r0",
        "r0_err",
        "m",
        "m_err",
        "sigma",
        "sigma_err",
        "bnoise",
        "bnoise_err",
        "vsf_slope",
        "vsf_slope_err",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-fit NPZ files with B(r)+Bnoise")
    p.add_argument(
        "--npz-dir",
        type=Path,
        default=Path("NPZ"),
        help="Directory containing NPZ files (default: NPZ)",
    )
    p.add_argument(
        "--pattern",
        default="*.npz",
        help="Glob pattern for NPZ selection (default: *.npz)",
    )
    p.add_argument(
        "--contains",
        default="",
        help="Only process filenames containing this substring",
    )
    p.add_argument(
        "--n-drop",
        type=int,
        default=1,
        help="Drop this many largest-scale points before fitting (default: 1)",
    )
    p.add_argument(
        "--maxfev",
        type=int,
        default=200000,
        help="Max function evaluations for curve_fit (default: 200000)",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("Figures") / "bplusnoise_fits",
        help="Directory for output fit plots",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=Path("results") / "bplusnoise_fit_summary.csv",
        help="CSV summary output path",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.npz_dir.glob(args.pattern))
    if args.contains:
        files = [f for f in files if args.contains in f.name]

    if not files:
        print("No NPZ files matched the selection.")
        return

    print(f"Found {len(files)} NPZ files to process")

    rows: list[dict] = []
    ok = 0

    for npz_path in files:
        try:
            result = fit_file(npz_path, n_drop=args.n_drop, maxfev=args.maxfev)
            out_png = args.plots_dir / f"{npz_path.stem}_bplusnoise_fit.png"
            save_plot(result, out_png)

            rows.append(
                {
                    "file": result["file"],
                    "status": "OK",
                    "message": "",
                    "n_points": result["n_points"],
                    "r0": result["r0"],
                    "r0_err": result["r0_err"],
                    "m": result["m"],
                    "m_err": result["m_err"],
                    "sigma": result["sigma"],
                    "sigma_err": result["sigma_err"],
                    "bnoise": result["bnoise"],
                    "bnoise_err": result["bnoise_err"],
                    "vsf_slope": result["vsf_slope"],
                    "vsf_slope_err": result["vsf_slope_err"],
                }
            )
            ok += 1
            print(
                f"OK   {npz_path.name}: m={result['m']:.3g}, r0={result['r0']:.3g}, "
                f"sigma={result['sigma']:.3g}, Bnoise={result['bnoise']:.3g}, "
                f"slope={result['vsf_slope']:.3g}"
            )
        except Exception as exc:
            rows.append(
                {
                    "file": npz_path.name,
                    "status": "FAIL",
                    "message": str(exc),
                }
            )
            print(f"FAIL {npz_path.name}: {exc}")

    write_summary(rows, args.summary)
    print(f"\nDone: {ok}/{len(files)} fits succeeded")
    print(f"Summary written to: {args.summary}")
    print(f"Plots written to:   {args.plots_dir}")


if __name__ == "__main__":
    main()
