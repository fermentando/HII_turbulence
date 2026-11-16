"""Observed dK/dlog10 l of one region/line at several separation floors, with the jackknife covariance.

Raising the separation floor r_min turns one slope measurement into several (same shell width
and v_bulk/v_turb, different r_min), which is what lets infer_shell_parameters.py constrain
both Delta r_shell/R_HII and v_bulk/v_turb from the global synthetic model.

Data, cuts and fit criteria are those of kurtosis_plot.py and fit_kurtosis_slope.py:
flux S/N > 20, tile-overlap fibres removed, 10-sigma (MAD) velocity clip, global velocity
plane subtracted, catalogue distance; shell binning (20/dex) above the floor, bins with
>= 4000 counts; pair kurtosis K = <dv^4>/<dv^2>^2; OLS of K vs log10(l/R_HII) over the first
run of K > 3 from the smallest l, bins with K > 10 dropped, >= 5 bins; 5x5 block jackknife.
The jackknife uses the same blocks at every floor, so the slopes' covariance across floors
follows from the leave-one-block-out replicates.

Floors: from max(37 arcsec, RMIN_MODEL_MIN R_HII) -- the lower edge of the synthetic grid the
model was calibrated on -- up to RMIN_MODEL_MAX R_HII in FLOOR_STEP_DEX steps.

Also stores the width-mixing factor M = <sigma_loc^4>/<sigma_loc^2>^2 of the local velocity
dispersion (plane fit to the K_NEIGH nearest fibres): the synthetic shells have uniform
turbulence (M = 1), and a pooled field of patches with different sigma has K ~ M x K_patch.

Output: Figures/manuscript/08_shell_inference/data/<region>_<LINE>_multifloor.npz

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/compute_multifloor_slopes.py --region RCW60 --line nii
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CORE_DIR = Path(__file__).resolve().parents[1] / "00_core"
sys.path.insert(0, str(CORE_DIR))
from vsf_module import (  # noqa: E402
    VSF2D, compute_bins, compute_pdf_and_statistics, get_hii_radius_pc, kurtosis_block_jackknife, load_line_data,
)

OUT_DIR = Path("Figures/manuscript/08_shell_inference/data")
BASE_FLOOR_ARCSEC = 37.0
RMIN_MODEL_MIN, RMIN_MODEL_MAX = 0.011, 0.15
FLOOR_STEP_DEX = 0.15
MIN_COUNTS, K_MIN, K_MAX, MIN_POINTS = 4000, 3.0, 10.0, 5
N_JOBS = 4  # as kurtosis_plot.py: MIN_COUNTS assumes counts = pairs x 4
K_NEIGH = 60
M_MAX_QUERY = 20000  # fibres at which sigma_loc is evaluated (neighbours from the full set)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--region", required=True, help="Region name, as in LVM_<region>.fits.")
parser.add_argument("--line", required=True, choices=("nii", "sii", "siii"))
parser.add_argument("--snr-min", type=float, default=20.0)
parser.add_argument("--force", action="store_true", help="Recompute even if the output exists.")
args = parser.parse_args()

REGION, LINE = args.region, args.line
FITS_FILE = f"LVM_{REGION}.fits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"{REGION}_{LINE.upper()}_multifloor.npz"
if out_path.exists() and not args.force:
    sys.exit(f"{out_path} exists (use --force to recompute)")


def first_run_above(k, k_min):
    """Mask of the run of consecutive bins with K > k_min that starts at the smallest l."""
    mask = np.zeros(len(k), dtype=bool)
    above = np.isfinite(k) & (k > k_min)
    if not above.any():
        return mask
    start = int(np.argmax(above))
    stop = start
    while stop < len(k) and above[stop]:
        stop += 1
    mask[start:stop] = True
    return mask


def ols_slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def mixing_factor(x, y, v):
    """M = <s^4>/<s^2>^2 of the local residual dispersion (plane fit to K_NEIGH neighbours).

    Evaluated at up to M_MAX_QUERY random fibres, neighbours drawn from all fibres.
    """
    xy = np.column_stack((x, y))
    query = np.random.default_rng(0).choice(len(x), size=min(M_MAX_QUERY, len(x)), replace=False)
    _, nn = cKDTree(xy).query(xy[query], min(K_NEIGH, len(x)))
    X, Y, V = x[nn], y[nn], v[nn]
    Xc, Yc, Vc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True), V - V.mean(1, keepdims=True)
    sxx, syy, sxy = (Xc**2).sum(1), (Yc**2).sum(1), (Xc * Yc).sum(1)
    sxv, syv = (Xc * Vc).sum(1), (Yc * Vc).sum(1)
    det = sxx * syy - sxy**2
    b = (syy * sxv - sxy * syv) / det
    c = (sxx * syv - sxy * sxv) / det
    resid = Vc - b[:, None] * Xc - c[:, None] * Yc
    s = 1.4826 * np.median(np.abs(resid - np.median(resid, 1, keepdims=True)), axis=1)
    return float(np.mean(s**4) / np.mean(s**2) ** 2), float(np.median(s))


d = load_line_data(FITS_FILE, LINE, snr_min=args.snr_min, subtract_gradient=True)
v2 = VSF2D(d["ra"], d["dec"], d["vel"], dist=d["dist"], vel_err=d["vel_err"])
x_pc, y_pc = v2.x_physical, v2.y_physical
R_HII = get_hii_radius_pc(FITS_FILE)
base_floor = d["dist"] * np.deg2rad(BASE_FLOOR_ARCSEC / 3600.0) / R_HII
start = max(base_floor, RMIN_MODEL_MIN)
n_steps = int(np.floor(np.log10(RMIN_MODEL_MAX / start) / FLOOR_STEP_DEX)) + 1 if start <= RMIN_MODEL_MAX else 1
floors = start * 10 ** (FLOOR_STEP_DEX * np.arange(max(n_steps, 1)))
M, sigma_med = mixing_factor(x_pc, y_pc, d["vel"])
print(f"{REGION} {LINE}: {len(x_pc):,} fibres, R_HII={R_HII:.2f} pc, 37\" = {base_floor:.4f} R_HII, "
      f"M={M:.2f}, floors {np.round(floors, 4).tolist()}")

rows, reps_all = [], []
for f in floors:
    log_c, edges, dv_bins, _ = compute_bins(
        x_pc, y_pc, d["vel"], bin_resolution=20, binning="shell", min_separation=f * R_HII,
    )
    _pdf, _c2, _c4, cnt, _dvc, kurt = compute_pdf_and_statistics(
        x_pc, y_pc, d["vel"], d["vel_err"], log_c, edges, dv_bins, order=1, n_jobs=N_JOBS,
    )
    _, _kerr, reps, _ = kurtosis_block_jackknife(x_pc, y_pc, d["vel"], edges)
    use = cnt >= MIN_COUNTS
    lx = np.log10(10 ** log_c[use] / R_HII)
    k = np.asarray(kurt[use], dtype=float)
    reps = reps[:, use]
    keep = first_run_above(k, K_MIN) & (k <= K_MAX)
    if keep.sum() < MIN_POINTS:
        print(f"  floor {f:.4f}: only {keep.sum()} bins in the first K>3 run; dropped")
        continue
    b = ols_slope(lx[keep], k[keep])
    rep_b = np.array([ols_slope(lx[keep], r[keep]) if np.all(np.isfinite(r[keep])) else np.nan for r in reps])
    rows.append((f, b, int(keep.sum()), 10 ** lx[keep].min(), 10 ** lx[keep].max()))
    reps_all.append(rep_b)
    print(f"  floor {f:.4f}: slope {b:+.3f}, {keep.sum()} bins, l/R_HII {10 ** lx[keep].min():.3f}-"
          f"{10 ** lx[keep].max():.3f}", flush=True)

if rows:
    R = np.array(reps_all)
    ok = np.all(np.isfinite(R), axis=0)
    R = R[:, ok]
    n_rep = R.shape[1]
    dev = R - R.mean(axis=1, keepdims=True)
    cov = (n_rep - 1) / n_rep * dev @ dev.T
else:
    n_rep, cov = 0, np.empty((0, 0))
arr = np.array(rows, dtype=float).reshape(-1, 5)
np.savez(
    out_path, region=REGION, line=LINE, floors=arr[:, 0], slopes=arr[:, 1], n_bins=arr[:, 2].astype(int),
    l_fit_min=arr[:, 3], l_fit_max=arr[:, 4], cov=cov, n_jackknife=n_rep, r_hii_pc=R_HII,
    distance_pc=d["dist"], base_floor_over_rhii=base_floor, n_fibres=len(x_pc), mixing_factor=M,
    sigma_loc_median=sigma_med, snr_min=args.snr_min,
)
print(f"Saved {out_path} ({len(rows)} floors, {n_rep} jackknife replicates)")
