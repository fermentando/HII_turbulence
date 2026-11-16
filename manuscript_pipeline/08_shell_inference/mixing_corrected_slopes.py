"""Mixing-corrected multi-floor slopes and (w, v) constraints for one region.

A pooled field of patches with different velocity dispersion has K ~ M x K_patch (M = <s^4>/<s^2>^2),
while the synthetic shells have uniform turbulence. Here each pair's dv is divided by
sqrt((sigma_loc_i^2 + sigma_loc_j^2)/2), sigma_loc being the robust residual scatter about a plane fitted
to the K_NEIGH nearest fibres. Cuts, bins, floors (read from data/<region>_<LINE>_multifloor.npz), fit
criteria and the 5x5 block jackknife are those of compute_multifloor_slopes.py; the uncorrected control
computed with the same pair code reproduces the stored slopes.

Caveat: the normalisation also removes real structure below the patch scale (a dip in K near the patch
radius), and the synthetic shells are not normalised the same way, so the corrected constraints depend
on K_NEIGH (see the K_NEIGH = 60-400 tests).

Outputs (Figures/manuscript/08_shell_inference/mixing_correction/):
  <region>_k<K_NEIGH>.png  K(l) at the 37" floor (uncorrected vs corrected) and the 1-sigma (w, v) regions
  <region>_k<K_NEIGH>.csv  per line: M, slopes, best fit, 1-sigma ranges, 1-sigma area, both versions

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/mixing_corrected_slopes.py <region> [nii,sii,siii] [k_neigh]
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

REPO = Path.cwd()
sys.path[:0] = [str(REPO / "manuscript_pipeline" / "04_flame_kurtosis_3d"), str(REPO / "manuscript_pipeline" / "00_core")]
from fit_global_dk_dlogr_model import RMIN_REF, model  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS, VSF2D, compute_bins, load_line_data  # noqa: E402

REGION = sys.argv[1]
LINES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nii", "sii", "siii"]
K_NEIGH = int(sys.argv[3]) if len(sys.argv) > 3 else 200
OUT_DIR = REPO / "Figures/manuscript/08_shell_inference/mixing_correction"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(OUT_DIR / f"{REGION}_k{K_NEIGH}")
DATA = REPO / "Figures/manuscript/08_shell_inference/data"
MIN_PAIRS, K_MIN, K_MAX, MIN_POINTS, N_BLOCKS, MIN_BLOCK = 1000, 3.0, 10.0, 5, 5, 20
rows = [r for r in csv.DictReader(open(REPO / "Figures/manuscript/04_kurtosis_3d_all_regions/global_dk_dlogr_model_fit.csv"))
        if r["model"] == "B"]
P, NAMES = [float(r["value"]) for r in rows], [r["parameter"] for r in rows]
V = np.geomspace(0.1, 100, 200)
W = np.geomspace(0.01, 1.5, 250)
VV, WW = np.meshgrid(V, W, indexing="ij")


def local_sigma(x, y, v, k):
    xy = np.column_stack((x, y))
    _, nn = cKDTree(xy).query(xy, min(k, len(x)))
    X, Y, Vn = x[nn], y[nn], v[nn]
    Xc, Yc, Vc = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True), Vn - Vn.mean(1, keepdims=True)
    sxx, syy, sxy = (Xc**2).sum(1), (Yc**2).sum(1), (Xc * Yc).sum(1)
    sxv, syv = (Xc * Vc).sum(1), (Yc * Vc).sum(1)
    det = sxx * syy - sxy**2
    b, c = (syy * sxv - sxy * syv) / det, (sxx * syv - sxy * sxv) / det
    res = Vc - b[:, None] * Xc - c[:, None] * Yc
    s = 1.4826 * np.median(np.abs(res - np.median(res, 1, keepdims=True)), axis=1)
    return np.maximum(s, np.percentile(s, 5))


def block_ids(x, y):
    ix = np.clip(((x - x.min()) / np.ptp(x) * N_BLOCKS).astype(int), 0, N_BLOCKS - 1)
    iy = np.clip(((y - y.min()) / np.ptp(y) * N_BLOCKS).astype(int), 0, N_BLOCKS - 1)
    b = ix * N_BLOCKS + iy
    counts = np.bincount(b, minlength=N_BLOCKS**2)
    return b, np.flatnonzero(counts >= MIN_BLOCK)


def pair_kurtosis(x, y, v, edges, scale=None):
    """K per bin over all pairs, and leave-one-block-out replicates (blocks with >= MIN_BLOCK fibres)."""
    nb, nbl = len(edges) - 1, N_BLOCKS**2
    blk, active = block_ids(x, y)
    tot = np.zeros((3, nb))
    Ai, Aj, Asame = (np.zeros((3, nb, nbl)) for _ in range(3))
    idx = np.arange(len(x))
    for i0 in range(0, len(x), 300):
        i1 = min(i0 + 300, len(x))
        r = np.hypot(x[i0:i1, None] - x[None, :], y[i0:i1, None] - y[None, :])
        dv = v[i0:i1, None] - v[None, :]
        if scale is not None:
            dv = dv / np.sqrt(0.5 * (scale[i0:i1, None] ** 2 + scale[None, :] ** 2))
        ok = (idx[i0:i1, None] < idx[None, :])
        bin_ = np.digitize(r, edges) - 1
        ok &= (bin_ >= 0) & (bin_ < nb)
        ii, jj = np.nonzero(ok)
        bb = bin_[ok]
        d2 = dv[ok] ** 2
        w = np.vstack((np.ones_like(d2), d2, d2**2))
        bi, bj = blk[i0 + ii], blk[jj]
        for q in range(3):
            tot[q] += np.bincount(bb, w[q], nb)
            Ai[q] += np.bincount(bb * nbl + bi, w[q], nb * nbl).reshape(nb, nbl)
            Aj[q] += np.bincount(bb * nbl + bj, w[q], nb * nbl).reshape(nb, nbl)
            same = bi == bj
            Asame[q] += np.bincount(bb[same] * nbl + bi[same], w[q][same], nb * nbl).reshape(nb, nbl)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (tot[2] / tot[0]) / (tot[1] / tot[0]) ** 2
        reps = []
        for b in active:
            s = tot - (Ai[:, :, b] + Aj[:, :, b] - Asame[:, :, b])
            reps.append((s[2] / s[0]) / (s[1] / s[0]) ** 2)
    return k, np.array(reps), tot[0]


def first_run_above(k, k_min):
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


def multifloor(x, y, v, floors_pc, r_hii, scale):
    rows, reps_all, curves = [], [], []
    for fpc in floors_pc:
        log_c, edges, _dvb, _ = compute_bins(x, y, v, bin_resolution=20, binning="shell", min_separation=fpc)
        k, reps, n = pair_kurtosis(x, y, v, edges, scale)
        use = n >= MIN_PAIRS
        lx, k, reps = np.log10(10 ** log_c[use] / r_hii), k[use], reps[:, use]
        curves.append((10**lx, k))
        keep = first_run_above(k, K_MIN) & (k <= K_MAX)
        if keep.sum() < MIN_POINTS:
            rows.append((fpc / r_hii, np.nan, 0, np.nan, np.nan))
            reps_all.append(np.full(len(reps), np.nan))
            continue
        b = np.polyfit(lx[keep], k[keep], 1)[0]
        rb = np.array([np.polyfit(lx[keep], r[keep], 1)[0] if np.all(np.isfinite(r[keep])) else np.nan for r in reps])
        rows.append((fpc / r_hii, b, keep.sum(), 10 ** lx[keep].min(), 10 ** lx[keep].max()))
        reps_all.append(rb)
    return np.array(rows), np.array(reps_all), curves


def infer(rows, R):
    ok = np.isfinite(rows[:, 1])
    if not ok[0]:
        return None
    use = ok.copy()
    use[1:] &= rows[1:, 3] <= rows[0, 4]
    idx = np.flatnonzero(use)
    Rg = R[idx][:, np.all(np.isfinite(R[idx]), axis=0)]
    n = Rg.shape[1]
    idx = idx[: max(n - 3, 1)]
    Rg = R[idx][:, np.all(np.isfinite(R[idx]), axis=0)]
    dev = Rg - Rg.mean(1, keepdims=True)
    C = (n - 1) / n * dev @ dev.T
    p = len(idx)
    Ci = (n - p - 2) / (n - 1) * np.linalg.inv(C)
    preds = np.array([model(np.vstack((np.log10(WW.ravel()), VV.ravel(), np.full(WW.size, np.log10(f / RMIN_REF)))),
                            *P, names=NAMES).reshape(VV.shape) for f in rows[idx, 0]])
    r = rows[idx, 1][:, None, None] - preds
    chi = np.einsum("iab,ij,jab->ab", r, Ci, r)
    d = chi - chi.min()
    ib = np.unravel_index(np.argmin(chi), chi.shape)
    pw, pv = d.min(0), d.min(1)
    return dict(dchi=d, best=(VV[ib], WW[ib]), idx=idx, floors=rows[idx, 0], slopes=rows[idx, 1], w1=(W[pw < 1].min(), W[pw < 1].max()), v1=(V[pv < 1].min(), V[pv < 1].max()),
                area=(d < 2.30).sum() * np.log10(V[1] / V[0]) * np.log10(W[1] / W[0]), chi2=float(chi.min()), p=p,
                err=np.sqrt(np.diag(C)))


summary = []
fits = {}
plt.style.use(str(REPO / "custom_plot.mplstyle"))
fig, axes = plt.subplots(2, len(LINES), figsize=(5.6 * len(LINES), 9.5), squeeze=False)
for col, line in enumerate(LINES):
    with np.load(DATA / f"{REGION}_{line.upper()}_multifloor.npz") as z:
        floors_rel, slopes_stored, r_hii = z["floors"], z["slopes"], float(z["r_hii_pc"])
    if len(floors_rel) == 0:
        print(f"\n{REGION} {line}: no fit at the 37\" floor; skipped")
        for ax in axes[:, col]:
            ax.set_title(f"{LINE_LABELS[line]}: no fit", fontsize=11)
            ax.axis("off")
        continue
    d = load_line_data(f"LVM_{REGION}.fits", line, snr_min=20.0, subtract_gradient=True, verbose=False)
    v2 = VSF2D(d["ra"], d["dec"], d["vel"], dist=d["dist"])
    x, y, v = v2.x_physical, v2.y_physical, d["vel"]
    if len(x) > 10000:
        sel = np.random.default_rng(0).choice(len(x), 10000, replace=False)
        x, y, v = x[sel], y[sel], v[sel]
    sig = local_sigma(x, y, v, K_NEIGH)
    M = np.mean(sig**4) / np.mean(sig**2) ** 2
    floors_pc = floors_rel * r_hii
    raw, Rraw, craw = multifloor(x, y, v, floors_pc, r_hii, None)
    nor, Rnor, cnor = multifloor(x, y, v, floors_pc, r_hii, sig)
    print(f"\n{REGION} {line}: {len(x)} fibres, sigma_loc ({K_NEIGH} neighbours) median {np.median(sig):.2f} km/s, M={M:.2f}")
    print("  floor   stored   control  normalised")
    for f, s_st, a, b in zip(floors_rel, slopes_stored, raw[:, 1], nor[:, 1]):
        print(f"  {f:.4f}  {s_st:+.3f}   {a:+.3f}    {b:+.3f}")
    res = {}
    for tag, rows_, R_ in (("uncorrected", raw, Rraw), ("mixing-corrected", nor, Rnor)):
        try:
            res[tag] = infer(rows_, R_)
        except (np.linalg.LinAlgError, ValueError) as exc:
            print(f"  {tag}: inference failed ({exc})")
            res[tag] = None
        if res[tag] is None:
            print(f"  {tag}: no fit at the base floor")
            continue
        q = res[tag]
        print(f"  {tag:17s}: p={q['p']} best v={q['best'][0]:.2f} w={q['best'][1]:.3f}  1s w {q['w1'][0]:.3f}-{q['w1'][1]:.3f} "
              f"v {q['v1'][0]:.2f}-{q['v1'][1]:.2f}  area {q['area']:.2f} dex^2  chi2 {q['chi2']:.2f}  "
              f"slope errs {np.round(q['err'], 2).tolist()}")
        fits[f"{line}_{tag.split('-')[0]}_floors"] = q["floors"]
        fits[f"{line}_{tag.split('-')[0]}_slopes"] = q["slopes"]
        fits[f"{line}_{tag.split('-')[0]}_err"] = q["err"]
        fits[f"{line}_{tag.split('-')[0]}_best"] = np.array(q["best"])
        fits[f"{line}_{tag.split('-')[0]}_allfloors"] = rows_[:, 0]
        fits[f"{line}_{tag.split('-')[0]}_allslopes"] = rows_[:, 1]
        summary.append((REGION, line, K_NEIGH, round(M, 3), tag, q["p"], round(rows_[0, 1], 4), round(q["best"][0], 3),
                        round(q["best"][1], 4), round(q["w1"][0], 4), round(q["w1"][1], 4), round(q["v1"][0], 3),
                        round(q["v1"][1], 3), round(q["area"], 3), round(q["chi2"], 3)))

    ax = axes[0, col]
    for (l, k), ls, lab in ((craw[0], "-o", "uncorrected"), (cnor[0], "-s", "mixing-corrected")):
        ax.plot(l, k, ls, ms=3.5, lw=1.2, color=LINE_COLORS[line] if lab == "uncorrected" else "#2a78b5", label=lab)
    ax.axhline(3, color="0.3", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell/R_{\rm HII}$")
    ax.set_ylabel(r"$K$ (37$''$ floor)")
    ax.set_title(f"{LINE_LABELS[line]}: $M$ = {M:.2f}", fontsize=11)
    ax.legend(fontsize=8.5)
    ax = axes[1, col]
    for tag, c in (("uncorrected", LINE_COLORS[line]), ("mixing-corrected", "#2a78b5")):
        q = res[tag]
        if q is None:
            continue
        ax.contourf(VV, WW, q["dchi"], levels=[0, 2.30], colors=[c], alpha=0.2)
        ax.contour(VV, WW, q["dchi"], levels=[2.30], colors=[c], linewidths=1.8)
        ax.plot(*q["best"], "*", ms=12, color=c, mec="white", label=f"{tag}: area {q['area']:.2f} dex$^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$v_{\rm bulk}/v_{\rm turb}$")
    ax.set_ylabel(r"$\Delta r_{\rm shell}/R_{\rm HII}$")
    ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle(f"{REGION}: mixing correction (pairs normalised by local $\\sigma$, {K_NEIGH} neighbours)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}.png", dpi=170)
print(f"\nWrote {OUT}.png")

with open(f"{OUT}.csv", "w") as fh:
    fh.write("region,line,k_neigh,M,version,n_floors,slope_37arcsec,best_v,best_w,w_1sig_lo,w_1sig_hi,"
             "v_1sig_lo,v_1sig_hi,area_1sig_dex2,chi2_min\n")
    for row in summary:
        fh.write(",".join(str(v) for v in row) + "\n")
print(f"Wrote {OUT}.csv")

np.savez(f"{OUT}_fits.npz", **fits)
print(f"Wrote {OUT}_fits.npz")
