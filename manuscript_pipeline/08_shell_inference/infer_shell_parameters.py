"""Joint inference of shell width and bulk-to-turbulent velocity ratio for every region/line.

For each Figures/manuscript/08_shell_inference/data/<region>_<LINE>_multifloor.npz (written by
compute_multifloor_slopes.py) the observed slopes s_k at separation floors rho_k are compared with
the global synthetic model (model B of 04_flame_kurtosis_3d/fit_global_dk_dlogr_model.py):

    dK/dlog10 r = F(v, rho) / (1 + exp[(log10 w - x0(rho)) / s(v, rho)])

    w = Delta r_shell / R_HII,  v = v_bulk / v_turb,  rho = log10[(r_min/R_HII) / 0.01]

chi2(w, v) = r^T C^-1 r with r_k = s_k - model(w, v, rho_k) and C the block-jackknife covariance of
the slopes across floors; C^-1 carries the Hartlap factor (n - p - 2)/(n - 1) for n jackknife
replicates and p floors. Raised floors are used only when their fitted l range overlaps the 37"
floor's (otherwise the "first K > 3 run" has jumped to a different part of K(l)).

Outputs (Figures/manuscript/08_shell_inference/):
  shell_inference_summary.csv         best fit, profile 1-sigma ranges, flags, mixing factor M
  data/<region>_<LINE>_chi2map.npz    Delta chi2 on the (v, w) grid
  shell_inference_chi2_maps.png       1/2-sigma regions per region, [NII] and [SII]
  shell_inference_slope_vs_floor.png  observed slopes vs floor with the best-fit model
  shell_inference_summary.png         best w and v with 1-sigma ranges, all regions

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/infer_shell_parameters.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "04_flame_kurtosis_3d"), str(HERE.parent / "00_core")]
from fit_global_dk_dlogr_model import RMIN_REF, model  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS  # noqa: E402

OUT_DIR = Path("Figures/manuscript/08_shell_inference")
DATA_DIR = OUT_DIR / "data"
MODEL_CSV = Path("Figures/manuscript/04_kurtosis_3d_all_regions/global_dk_dlogr_model_fit.csv")
MODEL_LABEL = "B"
V_GRID = np.geomspace(0.1, 100.0, 200)
W_GRID = np.geomspace(0.01, 1.5, 250)
LINES = ("NII", "SII", "SIII")
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}

rows = [r for r in csv.DictReader(open(MODEL_CSV)) if r["model"] == MODEL_LABEL]
PARAMS = [float(r["value"]) for r in rows]
NAMES = [r["parameter"] for r in rows]
VV, WW = np.meshgrid(V_GRID, W_GRID, indexing="ij")


def model_grid(floor_over_rhii: float) -> np.ndarray:
    xd = np.vstack((np.log10(WW.ravel()), VV.ravel(), np.full(WW.size, np.log10(floor_over_rhii / RMIN_REF))))
    return model(xd, *PARAMS, names=NAMES).reshape(VV.shape)


def profile_range(dchi: np.ndarray, axis: int, grid: np.ndarray, level: float = 1.0):
    prof = dchi.min(axis=axis)
    inside = grid[prof < level]
    return inside.min(), inside.max(), bool(inside.min() > grid[0] and inside.max() < grid[-1])


results = []
for path in sorted(DATA_DIR.glob("*_multifloor.npz")):
    with np.load(path) as d:
        region, line = str(d["region"]), str(d["line"]).upper()
        floors, slopes, cov = d["floors"], d["slopes"], d["cov"]
        l_min, l_max, n_rep = d["l_fit_min"], d["l_fit_max"], int(d["n_jackknife"])
        meta = {k: float(d[k]) for k in ("r_hii_pc", "base_floor_over_rhii", "mixing_factor", "n_fibres")}
    res = {"region": region, "line": line, **meta, "n_floors_measured": len(floors)}
    if len(floors) == 0:
        res["status"] = "no fit at the 37\" floor"
        results.append(res)
        continue
    use = np.zeros(len(floors), dtype=bool)
    use[0] = True
    use[1:] = l_min[1:] <= l_max[0]
    idx = np.flatnonzero(use)
    p_max = n_rep - 3  # keeps the Hartlap factor (n - p - 2)/(n - 1) positive
    if p_max < 1:
        res["status"] = f"too few jackknife replicates ({n_rep})"
        results.append(res)
        continue
    trimmed = len(idx) > p_max
    idx = idx[:p_max]
    p = len(idx)
    C = cov[np.ix_(idx, idx)]
    hartlap = (n_rep - p - 2) / (n_rep - 1)
    Ci = hartlap * np.linalg.inv(C)
    preds = np.array([model_grid(f) for f in floors[idx]])
    r = slopes[idx][:, None, None] - preds
    chi = np.einsum("iab,ij,jab->ab", r, Ci, r)
    dchi = chi - chi.min()
    ib = np.unravel_index(np.argmin(chi), chi.shape)
    w_lo, w_hi, w_ok = profile_range(dchi, 0, W_GRID)
    v_lo, v_hi, v_ok = profile_range(dchi, 1, V_GRID)
    base_pred = preds[0]
    flags = []
    if slopes[0] < base_pred.min() - np.sqrt(C[0, 0]) or slopes[0] > base_pred.max() + np.sqrt(C[0, 0]):
        flags.append("base slope >1 sigma outside model range")
    if p == 1:
        flags.append("single floor: w-v degenerate")
    if trimmed:
        flags.append(f"floors limited to {p} by {n_rep} jackknife replicates")
    if not w_ok:
        flags.append("w 1-sigma hits grid edge")
    if not v_ok:
        flags.append("v 1-sigma hits grid edge")
    if meta["base_floor_over_rhii"] < 0.011:
        flags.append("37\" floor below model range; first floor raised to 0.011")
    res.update({
        "status": "ok", "floors_used": ";".join(f"{f:.4f}" for f in floors[idx]), "n_floors_used": p,
        "slope_base": slopes[0], "slope_base_err": float(np.sqrt(C[0, 0])), "n_jackknife": n_rep,
        "best_v": VV[ib], "best_w": WW[ib], "chi2_min": float(chi.min()), "dof": p - 2,
        "w_1sig_lo": w_lo, "w_1sig_hi": w_hi, "v_1sig_lo": v_lo, "v_1sig_hi": v_hi,
        "w_constrained": w_ok, "v_constrained": v_ok, "flags": "; ".join(flags),
        "_dchi": dchi, "_floors": floors, "_slopes": slopes, "_err": np.sqrt(np.diag(cov)), "_use": use,
    })
    np.savez(DATA_DIR / f"{region}_{line}_chi2map.npz", v_grid=V_GRID, w_grid=W_GRID, dchi2=dchi,
             floors_used=floors[idx], hartlap=hartlap)
    results.append(res)
    print(f"{region:15s} {line:4s} p={p} best v={VV[ib]:6.2f} w={WW[ib]:.3f} chi2={chi.min():5.2f} "
          f"w1s=[{w_lo:.3f},{w_hi:.3f}] v1s=[{v_lo:.2f},{v_hi:.2f}] M={meta['mixing_factor']:.2f} {'; '.join(flags)}")

# ---------------------------------------------------------------- summary table
cols = ["region", "line", "status", "n_fibres", "r_hii_pc", "base_floor_over_rhii", "mixing_factor",
        "n_floors_measured", "n_floors_used", "floors_used", "slope_base", "slope_base_err", "n_jackknife",
        "best_v", "best_w", "chi2_min", "dof", "w_1sig_lo", "w_1sig_hi", "v_1sig_lo", "v_1sig_hi",
        "w_constrained", "v_constrained", "flags"]
with open(OUT_DIR / "shell_inference_summary.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(cols)
    for res in results:
        wr.writerow([f"{res[c]:.5g}" if isinstance(res.get(c), float) else res.get(c, "") for c in cols])
print(f"Wrote {OUT_DIR / 'shell_inference_summary.csv'}")

plt.style.use("./custom_plot.mplstyle")
ok = [r for r in results if r["status"] == "ok"]
regions = sorted({r["region"] for r in results})
by_key = {(r["region"], r["line"]): r for r in ok}
n_cols = 4
n_rows = int(np.ceil(len(regions) / n_cols))

# ---------------------------------------------------------------- chi2 maps
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.9 * n_rows), sharex=True, sharey=True,
                         squeeze=False)
for ax, region in zip(axes.flat, regions):
    for line in LINES:
        res = by_key.get((region, line))
        if res is None:
            continue
        c, ls = LINE_COLORS[line], LINESTYLES[line]
        ax.contourf(VV, WW, res["_dchi"], levels=[0, 2.30], colors=[c], alpha=0.25)
        ax.contour(VV, WW, res["_dchi"], levels=[2.30, 6.18], colors=[c], linestyles=[ls], linewidths=[1.5, 0.8])
        ax.plot(res["best_v"], res["best_w"], "*", ms=11, color=c, mec="white", mew=0.6,
                label=f"{LINE_LABELS[line]} ($p$={res['n_floors_used']}, $M$={res['mixing_factor']:.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(region.replace("_", " "), fontsize=11)
    ax.legend(fontsize=7.5, loc="lower right")
for ax in axes.flat[len(regions):]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
for ax in axes[-1, :]:
    ax.set_xlabel(r"$v_{\rm bulk}/v_{\rm turb}$")
fig.suptitle(r"Joint constraints: filled = $\Delta\chi^2<2.3$ (1$\sigma$), thin line = $\Delta\chi^2<6.2$ (2$\sigma$)",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "shell_inference_chi2_maps.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- slopes vs floor
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows), squeeze=False)
f_dense = np.geomspace(0.011, 0.15, 60)
for ax, region in zip(axes.flat, regions):
    for line in LINES:
        res = by_key.get((region, line))
        if res is None:
            continue
        c = LINE_COLORS[line]
        fl, sl, er, use = res["_floors"], res["_slopes"], res["_err"], res["_use"]
        ax.errorbar(fl[use], sl[use], yerr=er[use], fmt="o", ms=5, color=c, capsize=2,
                    label=fr"{LINE_LABELS[line]}: $v$={res['best_v']:.2g}, $w$={res['best_w']:.2f}")
        if (~use).any():
            ax.errorbar(fl[~use], sl[~use], yerr=er[~use], fmt="o", ms=5, mfc="none", color=c, capsize=2, alpha=0.5)
        xd = np.vstack((np.full(f_dense.size, np.log10(res["best_w"])), np.full(f_dense.size, res["best_v"]),
                        np.log10(f_dense / RMIN_REF)))
        fmask = (f_dense >= fl[use].min() * 0.9) & (f_dense <= fl[use].max() * 1.1)
        ax.plot(f_dense[fmask], model(xd, *PARAMS, names=NAMES)[fmask], LINESTYLES[line], color=c, lw=1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(0, color="0.3", ls="--", lw=0.9)
    ax.set_title(region.replace("_", " "), fontsize=11)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend(fontsize=7.5)
for ax in axes.flat[len(regions):]:
    ax.axis("off")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$dK/d\log_{10}\ell$")
for ax in axes[-1, :]:
    ax.set_xlabel(r"floor $r_{\rm min}/r_{\rm HII}$")
fig.suptitle("Observed slope vs separation floor (filled: used; open: fitted run detached) and best-fit model",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "shell_inference_slope_vs_floor.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- summary, with per-tracer marginal PDFs
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], wspace=0.03, hspace=0.08)
axW = fig.add_subplot(gs[0, 0])
axV = fig.add_subplot(gs[1, 0], sharex=axW)
mW = fig.add_subplot(gs[0, 1], sharey=axW)
mV = fig.add_subplot(gs[1, 1], sharey=axV)
xpos = {reg: i for i, reg in enumerate(regions)}
offsets = {"NII": -0.2, "SII": 0.0, "SIII": 0.2}
DLW, DLV = np.log10(W_GRID[1] / W_GRID[0]), np.log10(V_GRID[1] / V_GRID[0])


def marginal_pdfs(dchi):
    """Marginal posteriors per dex in log10 w and log10 v (flat priors in both logs), P ~ exp(-Delta chi2 / 2)."""
    post = np.exp(-0.5 * dchi)
    pw, pv = post.sum(axis=0), post.sum(axis=1)
    return pw / (pw.sum() * DLW), pv / (pv.sum() * DLV)


stacked = {line: [] for line in LINES}
for line in LINES:
    for res in (r for r in ok if r["line"] == line):
        x = xpos[res["region"]] + offsets[line]
        c = LINE_COLORS[line]
        good = not any(k in res["flags"] for k in ("outside model range", "single floor"))
        mfc = c if good else "none"
        for ax, key, lo, hi in ((axW, "best_w", "w_1sig_lo", "w_1sig_hi"), (axV, "best_v", "v_1sig_lo", "v_1sig_hi")):
            yerr = [[res[key] - res[lo]], [res[hi] - res[key]]]
            ax.errorbar(x, res[key], yerr=yerr, fmt="o", ms=7, color=c, mfc=mfc, capsize=3, lw=1.5)
        if good:
            stacked[line].append(marginal_pdfs(res["_dchi"]))

for line in LINES:
    if not stacked[line]:
        continue
    c = LINE_COLORS[line]
    pw = np.mean([p[0] for p in stacked[line]], axis=0)
    pv = np.mean([p[1] for p in stacked[line]], axis=0)
    for ax, grid, pdf in ((mW, W_GRID, pw), (mV, V_GRID, pv)):
        ax.plot(pdf, grid, color=c, lw=2.0, label=f"{LINE_LABELS[line]} ({len(stacked[line])})")

for line in LINES:
    if any(r["line"] == line for r in ok):
        axW.plot([], [], "o", color=LINE_COLORS[line], label=LINE_LABELS[line])
axW.plot([], [], "o", color="0.4", mfc="none", label="flagged (outside model range / single floor)")
axW.legend(fontsize=9, ncol=4, loc="upper left")
for ax in (axW, axV):
    ax.set_yscale("log")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
for ax in (mW, mV):
    ax.set_xlim(left=0)
    ax.tick_params(axis="y", labelleft=False)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
mV.legend(fontsize=8.5, loc="upper right", title="tracer (N fits)", title_fontsize=8.5)
mV.set_xlabel(r"PDF [dex$^{-1}$]")
mW.set_title("non-flagged fits", fontsize=10)
axW.set_ylabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$ (best, 1$\sigma$)")
axV.set_ylabel(r"$v_{\rm bulk}/v_{\rm turb}$ (best, 1$\sigma$)")
axW.set_ylim(W_GRID[0], W_GRID[-1])
axV.set_ylim(V_GRID[0], V_GRID[-1])
axV.set_xticks(range(len(regions)))
axV.set_xticklabels([r.replace("_", " ") for r in regions], rotation=30, ha="right")
axW.tick_params(axis="x", labelbottom=False)
fig.savefig(OUT_DIR / "shell_inference_summary.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote figures to {OUT_DIR}")
