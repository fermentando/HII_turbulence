#!/usr/bin/env python3
"""All regions' [NII] sqrt(S2) on a common l / R_HII axis.

Reads NPZ/VSFORD_<region>_<LINE>.npz (full FOV, S/N > 20, block-jackknife errors, written by
06_vsf_orders/vsf_order_vs3_allregions.py) and the adopted radii in
manuscript_notes/adopted_hii_region_radii.csv.

Panels:
  (a) sqrt(S2) [km/s] vs l / R_HII
  (b) sqrt(S2) normalised at l = R_HII (shape comparison)
  (c) local logarithmic slope d log sqrt(S2) / d log l (running 5-bin weighted fit)

Usage:
    python manuscript_pipeline/03_slopes_turnover/plot_nii_sqrtS2_vs_l_over_rhii.py [--line NII]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RADIUS_CSV = REPO_ROOT / "manuscript_notes" / "adopted_hii_region_radii.csv"
NPZ_DIR = REPO_ROOT / "NPZ"
OUT_DIR = REPO_ROOT / "Figures" / "manuscript" / "03_slopes_turnover"
STYLE = REPO_ROOT / "custom_plot.mplstyle"
MIN_PAIRS = 1000  # same bin cut as vsf_order_vs3_allregions.py
HALF_WINDOW = 2


def load_radii() -> dict[str, float]:
    with RADIUS_CSV.open() as f:
        return {row["object"]: float(row["radius_pc"]) for row in csv.DictReader(f)}


def local_slope(x: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for i in range(x.size):
        sl = slice(max(i - HALF_WINDOW, 0), i + HALF_WINDOW + 1)
        if x[sl].size >= 2 * HALF_WINDOW + 1:
            out[i] = np.polyfit(x[sl], y[sl], 1, w=1 / yerr[sl])[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--line", default="NII")
    args = p.parse_args()
    line = args.line.upper()
    if STYLE.exists():
        plt.style.use(str(STYLE))

    radii = load_radii()
    paths = sorted(NPZ_DIR.glob(f"VSFORD_*_{line}.npz"))
    if not paths:
        raise SystemExit(f"No NPZ/VSFORD_*_{line}.npz files")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), constrained_layout=True)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, 10))
    for k, path in enumerate(paths):
        region = path.stem[len("VSFORD_"):-len(f"_{line}")]
        if region not in radii:
            print(f"skip {region}: no adopted radius")
            continue
        d = np.load(path)
        i2 = list(d["orders"]).index(2)
        s2, s2_err, counts = d["S"][i2], d["S_err"][i2], d["counts"]
        ok = (counts >= MIN_PAIRS) & np.isfinite(s2) & (s2 > 0) & (s2_err > 0)
        if ok.sum() < 2 * HALF_WINDOW + 1:
            print(f"skip {region}: only {ok.sum()} bins with >= {MIN_PAIRS} pairs")
            continue
        lag = 10 ** d["log_centers"][ok] / radii[region]
        sig = np.sqrt(s2[ok])
        sig_err = s2_err[ok] / (2 * sig)
        c = colors[k % 10]
        label = region.replace("_", " ")

        axes[0].errorbar(lag, sig, yerr=sig_err, color=c, lw=1.8, marker="o", ms=3, capsize=0, label=label)

        ref = np.interp(0.0, np.log10(lag), np.log10(sig), left=np.nan, right=np.nan)
        if np.isfinite(ref):
            axes[1].plot(lag, sig / 10**ref, color=c, lw=1.8, marker="o", ms=3, label=label)

        lx, ly = np.log10(lag), np.log10(sig)
        lye = sig_err / (sig * np.log(10))
        axes[2].plot(lag, local_slope(lx, ly, lye), color=c, lw=1.8, label=label)

        print(f"{label:16s} R_HII={radii[region]:6.2f} pc  l/R_HII range {lag.min():.3f}-{lag.max():.2f}")

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(1, color="0.3", ls=":", lw=1.2)
        ax.axvline(2, color="0.3", ls="--", lw=1.0)
        ax.set_xlabel(r"$\ell / R_{\rm HII}$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\sqrt{S_2}$ [km/s]")
    axes[0].set_title(rf"[{line}], S/N $>$ 20, full FOV")
    axes[0].legend(fontsize=11, ncol=2)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\sqrt{S_2}\,/\,\sqrt{S_2}(\ell=R_{\rm HII})$")
    axes[1].set_title("normalised at $\\ell = R_{\\rm HII}$")
    axes[2].axhline(1 / 3, color="0.4", ls="--", lw=1)
    axes[2].axhline(0, color="0.4", lw=0.8)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].set_ylabel(r"local slope $d\log\sqrt{S_2}/d\log\ell$")
    axes[2].set_title(f"running {2 * HALF_WINDOW + 1}-bin slope")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{line.lower()}_sqrtS2_vs_l_over_rhii_allregions.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
