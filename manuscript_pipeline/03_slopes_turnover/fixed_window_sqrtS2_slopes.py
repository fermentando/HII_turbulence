#!/usr/bin/env python3
"""Fixed-window sqrt(S2) slopes in l / R_HII, per region and tracer, with block-jackknife errors.

Tests whether the first-segment slopes of the segmented fit differ between regions only because each
is fitted over a different window (r_min -> l_break). Every region is refitted over the same
l / R_HII windows; errors come from the S2 jackknife replicates in NPZ/VSFORD_<region>_<LINE>.npz
(written by 06_vsf_orders/vsf_order_vs3_allregions.py, S/N > 20, full FOV).

For each line and window it reports the weighted mean, the region-to-region scatter and the chi2
of a single common slope.

Writes Figures/manuscript/03_slopes_turnover/fixed_window_sqrtS2_slopes.csv

Usage:
    python manuscript_pipeline/03_slopes_turnover/fixed_window_sqrtS2_slopes.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RADIUS_CSV = REPO_ROOT / "manuscript_notes" / "adopted_hii_region_radii.csv"
NPZ_DIR = REPO_ROOT / "NPZ"
OUT_CSV = REPO_ROOT / "Figures" / "manuscript" / "03_slopes_turnover" / "fixed_window_sqrtS2_slopes.csv"
LINES = ["NII", "SII", "SIII"]
WINDOWS = [(0.1, 0.5), (0.1, 1.0)]
MIN_PAIRS = 1000
MIN_BINS = 5
EXCLUDE_FROM_COMMON = {"Rosette_nebula"}  # R_HII (20 pc) puts its sampled lags far below the others'


def window_slope(path: Path, r_hii: float, lo: float, hi: float) -> tuple[float, float, int]:
    d = np.load(path)
    i2 = list(d["orders"]).index(2)
    s2 = d["S"][i2]
    lag = 10 ** d["log_centers"] / r_hii
    sel = (d["counts"] >= MIN_PAIRS) & np.isfinite(s2) & (s2 > 0) & (lag >= lo) & (lag <= hi)
    if sel.sum() < MIN_BINS:
        return np.nan, np.nan, int(sel.sum())
    x = np.log10(lag[sel])

    def fit(curve: np.ndarray) -> float:
        return float(np.polyfit(x, 0.5 * np.log10(curve[sel]), 1)[0])

    reps = []
    for rep in d["S_jackknife_replicates"]:
        c = rep[i2]
        if np.all(np.isfinite(c[sel]) & (c[sel] > 0)):
            reps.append(fit(c))
    reps = np.asarray(reps)
    n = reps.size
    err = float(np.sqrt((n - 1) / n * np.sum((reps - reps.mean()) ** 2))) if n >= 4 else np.nan
    return fit(s2), err, int(sel.sum())


def main() -> None:
    with RADIUS_CSV.open() as f:
        radii = {row["object"]: float(row["radius_pc"]) for row in csv.DictReader(f)}

    rows = []
    for line in LINES:
        paths = sorted(NPZ_DIR.glob(f"VSFORD_*_{line}.npz"))
        if not paths:
            print(f"[{line}] no NPZ/VSFORD_*_{line}.npz files, skipped")
            continue
        for lo, hi in WINDOWS:
            print(f"\n[{line}] window {lo:g}-{hi:g} R_HII")
            ms, es = [], []
            for path in paths:
                region = path.stem[len("VSFORD_"):-len(f"_{line}")]
                if region not in radii:
                    continue
                m, e, nb = window_slope(path, radii[region], lo, hi)
                rows.append({"line": line, "region": region, "lo": lo, "hi": hi,
                             "slope": m, "slope_err": e, "n_bins": nb})
                print(f"  {region:15s} m = {m:6.3f} ± {e:5.3f}  (n_bins={nb})")
                if np.isfinite(m) and np.isfinite(e) and region not in EXCLUDE_FROM_COMMON:
                    ms.append(m)
                    es.append(e)
            if len(ms) >= 2:
                ms, es = np.asarray(ms), np.asarray(es)
                w = 1 / es**2
                mu = np.sum(w * ms) / w.sum()
                chi2 = np.sum(((ms - mu) / es) ** 2)
                print(f"  common (excl. {', '.join(sorted(EXCLUDE_FROM_COMMON))}): mean {mu:.3f} ± "
                      f"{1 / np.sqrt(w.sum()):.3f}, scatter {ms.std(ddof=1):.3f}, "
                      f"median err {np.median(es):.3f}, chi2 = {chi2:.1f}/{len(ms) - 1}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
