"""Shell thickness Delta (and outer radius) from the line surface-brightness profile of each region/line.

For optically thin emission from a shell 1 - Delta <= r/R_shell <= 1 of uniform emissivity, the surface
brightness is proportional to the line-of-sight path length through the shell (the column n of the
kurtosis note, sections 5 and 8):
    n(u; Delta) = 2 [sqrt(1 - u^2) - sqrt(max(a^2 - u^2, 0))],   u = rho / s,  a = 1 - Delta
so thin shells are limb-brightened (limb/centre ~ sqrt(2/Delta)) and Delta = 1 is a filled sphere.
Model of the azimuthally averaged profile:
    I(rho) = A n(rho / s; Delta) + B
rho = angular distance from the (fitted) centre / catalogue radius; s = R_shell / R_HII(catalogue);
B = uniform foreground/background; the model is smoothed by the fibre footprint (35.3" top hat -> Gaussian
sigma = 35.3"/4) and averaged over each annulus. A, B are solved by weighted least squares at every
(Delta, s) grid point; Delta(-2 ln L) is profiled over s.

Data: all fibres of the region (no S/N cut, so faint cavity fibres are kept; tile-overlap fibres dropped),
flux_<line> from LVM_<region>.fits. Annuli of width DRHO out to RHO_MAX; each annulus is split into
N_SECTORS azimuthal sectors; the annulus value is the mean of the sector medians (medians: robust to
failed line fits, stars and bright knots) and its error is their standard error, so clumps and asymmetry
enter the uncertainty. Annuli need >= MIN_SECTORS sectors with >= MIN_FIB fibres (the FOV
need not cover the whole region). If chi2_min / dof > 1, Delta chi2 is divided by it (errors inflated).
The centre is free: the profile is rebinned around every offset on a +-0.35 r_HII grid (step 0.05) and
Delta(-2 ln L) is profiled over centre and s; every trial centre must cover the annuli covered at the
catalogue centre, so chi2 values are comparable.

Outputs (Figures/manuscript/08_shell_inference/flux_profile/):
  flux_profile_fits.png      profiles with the best fit and Delta = 0.02 / 0.1 / 0.3 / 1 shapes
  flux_profile_delta.png     profile likelihood of Delta per region and line
  flux_profile_summary.csv   best Delta, s, 1-sigma ranges, chi2, error inflation
  <region>_<LINE>_flux.npz   Delta(-2 ln L)(Delta, s) on the grid (used by fit_kurtosis_formula_grid.py)

Run from the repo root:
    python manuscript_pipeline/08_shell_inference/fit_shell_from_flux.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "00_core")]
from fit_kurtosis_formula import all_targets  # noqa: E402
from fit_kurtosis_formula_grid import D_GRID  # noqa: E402
from vsf_module import LINE_COLORS, LINE_LABELS, flag_overlap_fibres, get_heliocentric_distance, \
    get_hii_center_and_radius  # noqa: E402

OUT_DIR = Path("Figures/manuscript/08_shell_inference/flux_profile")
S_GRID = np.geomspace(0.4, 2.5, 90)
DRHO, RHO_MAX = 0.05, 2.0
N_SECTORS, MIN_SECTORS, MIN_FIB = 8, 4, 3
FIBRE_DIAM_ARCSEC = 35.3
LINES = ("NII", "SII", "SIII")
LINESTYLES = {"NII": "-", "SII": "--", "SIII": ":"}
U_FINE = np.linspace(0, RHO_MAX / S_GRID[0] + 0.1, 4000)


def column(u, delta):
    a = 1.0 - delta
    return 2 * (np.sqrt(np.clip(1 - u ** 2, 0, None)) - np.sqrt(np.clip(a ** 2 - u ** 2, 0, None)))


C_GRID = np.round(np.arange(-0.35, 0.351, 0.05), 3)  # centre offsets in units of the catalogue radius
EDGES = np.arange(0, RHO_MAX + DRHO / 2, DRHO)
RHO_C = 0.5 * (EDGES[1:] + EDGES[:-1])


def load_fibres(region, line):
    """All fibres (no S/N cut, overlaps dropped) in catalogue-centred coordinates, units of the catalogue radius."""
    fn = f"LVM_{region}.fits"
    with fits.open(fn) as h:
        t = h[1].data
        ra, dec = np.asarray(t["ra"], float), np.asarray(t["dec"], float)
        flux = np.asarray(t[f"flux_{line.lower()}"], float)
    keep = np.isfinite(flux) & ~flag_overlap_fibres(ra, dec, get_heliocentric_distance(fn))
    ra0, dec0, rad_deg = get_hii_center_and_radius(fn)
    x = (ra[keep] - ra0) * np.cos(np.deg2rad(dec0)) / rad_deg
    y = (dec[keep] - dec0) / rad_deg
    return {"x": x, "y": y, "f": flux[keep], "sigma_rho": FIBRE_DIAM_ARCSEC / 4 / (rad_deg * 3600),
            "radius_arcsec": rad_deg * 3600}


def bin_profile(fib, x0, y0):
    """Annulus values (mean of sector medians) and errors on EDGES around (x0, y0); NaN where under-covered."""
    dx, dy = fib["x"] - x0, fib["y"] - y0
    rho, phi = np.hypot(dx, dy), np.arctan2(dy, dx)
    ib = np.digitize(rho, EDGES) - 1
    sec = ((phi + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    I = np.full(len(RHO_C), np.nan)
    err = np.full(len(RHO_C), np.nan)
    order = np.lexsort((sec, ib))
    ib_s, sec_s, f_s = ib[order], sec[order], fib["f"][order]
    key = ib_s * N_SECTORS + sec_s
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts, stops = np.r_[0, bounds], np.r_[bounds, len(key)]
    med = {}
    for a, b in zip(starts, stops):
        if b - a >= MIN_FIB and 0 <= ib_s[a] < len(RHO_C):
            med.setdefault(ib_s[a], []).append(np.median(f_s[a:b]))
    for k, v in med.items():
        if len(v) >= MIN_SECTORS:
            I[k], err[k] = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
    return I, err


def model_cube(sigma_rho):
    """Shell column at every annulus centre for every (Delta, s): shape (n_Delta, n_s, n_annuli)."""
    return np.array([model_profiles(RHO_C, d, S_GRID, sigma_rho) for d in D_GRID])


def model_profiles(rho, delta, s, sigma_rho):
    """Shell column at the annulus centres for every s (annulus-averaged and fibre-smoothed)."""
    du = U_FINE[1] - U_FINE[0]
    n = column(U_FINE, delta)
    out = np.empty((len(s), len(rho)))
    sub = rho[:, None] + DRHO * np.linspace(-0.4, 0.4, 5)[None, :]
    for i, si in enumerate(s):
        ns = gaussian_filter1d(n, sigma_rho / si / du, mode="nearest") if sigma_rho / si > du else n
        out[i] = np.interp(np.abs(sub) / si, U_FINE, ns).mean(1)
    return out


def chi2_cube(M, I, err):
    """Weighted LS of I = A M + B (A >= 0) for every (Delta, s) at once; returns chi2, A, B."""
    w = 1 / err ** 2
    Sw, Sy = w.sum(), (w * I).sum()
    Sm, Smm, Smy = (w * M).sum(-1), (w * M * M).sum(-1), (w * M * I).sum(-1)
    det = Sw * Smm - Sm ** 2
    A = np.clip(np.where(det > 0, (Sw * Smy - Sm * Sy) / np.where(det > 0, det, 1), 0.0), 0, None)
    B = (Sy - A * Sm) / Sw
    r = I - A[..., None] * M - B[..., None]
    return (w * r ** 2).sum(-1), A, B


def fit_region_line(region, line):
    """Profile likelihood of (Delta, s) with the centre free on C_GRID x C_GRID."""
    fib = load_fibres(region, line)
    I0, e0 = bin_profile(fib, 0.0, 0.0)
    need = np.isfinite(I0) & (e0 > 0)  # annuli every trial centre must cover (fixed data dimension)
    if need.sum() < 8:
        return None
    Mall = model_cube(fib["sigma_rho"])[..., need]
    best = None
    chi_cat = None
    for cx in C_GRID:
        for cy in C_GRID:
            I, e = bin_profile(fib, cx, cy)
            I, e = I[need], e[need]
            if not (np.all(np.isfinite(I)) and np.all(e > 0)):
                continue
            chi, A, B = chi2_cube(Mall, I, e)
            if cx == 0 and cy == 0:
                chi_cat = chi.min()
            if best is None:
                best = {"chi": chi.copy(), "cmin": chi.min(), "c": (cx, cy), "A": A, "B": B, "I": I, "e": e}
                best["envelope"] = chi.copy()
            else:
                best["envelope"] = np.minimum(best["envelope"], chi)
                if chi.min() < best["cmin"]:
                    best.update({"chi": chi.copy(), "cmin": chi.min(), "c": (cx, cy), "A": A, "B": B, "I": I, "e": e})
    best["rho"] = RHO_C[need]
    best["sigma_rho"] = fib["sigma_rho"]
    best["chi_catalogue_centre"] = chi_cat
    return best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("./custom_plot.mplstyle")
    except OSError:
        pass
    results, store = [], {}
    for region, line in all_targets():
        fitres = fit_region_line(region, line)
        if fitres is None:
            print(f"{region:15s} {line:4s} too few annuli; skipped")
            continue
        chi = fitres["envelope"]  # profiled over the centre
        n = len(fitres["rho"])
        dof = n - 6
        infl = max(1.0, chi.min() / max(dof, 1))
        dm = (chi - chi.min()) / infl
        ia, js = np.unravel_index(np.argmin(fitres["chi"]), fitres["chi"].shape)
        p = {"rho": fitres["rho"], "I": fitres["I"], "err": fitres["e"], "sigma_rho": fitres["sigma_rho"]}
        amp, bkg = fitres["A"], fitres["B"]
        profD = dm.min(1)
        profS = dm.min(0)
        inD, inS = D_GRID[profD < 1], S_GRID[profS < 1]
        flags = []
        if inD.min() <= D_GRID[0] or inD.max() >= D_GRID[-1]:
            flags.append("Delta 1-sigma hits grid edge")
        if inS.min() <= S_GRID[0] or inS.max() >= S_GRID[-1]:
            flags.append("s 1-sigma hits grid edge")
        if p["rho"].max() < 1.1 * S_GRID[js]:
            flags.append("FOV does not reach beyond the fitted shell edge")
        cx, cy = fitres["c"]
        if max(abs(cx), abs(cy)) >= C_GRID.max() - 1e-9:
            flags.append("centre at the edge of the search box")
        res = {"region": region, "line": line, "n_annuli": n, "rho_max_covered": p["rho"].max(),
               "centre_dx": cx, "centre_dy": cy, "chi2_catalogue_centre": fitres["chi_catalogue_centre"],
               "best_delta": D_GRID[ia], "delta_1sig_lo": inD.min(), "delta_1sig_hi": inD.max(),
               "best_s": S_GRID[js], "s_1sig_lo": inS.min(), "s_1sig_hi": inS.max(), "chi2_min": chi.min(),
               "dof": dof, "error_inflation": infl, "bkg_over_peak": bkg[ia, js] / p["I"].max(),
               "flags": "; ".join(flags)}
        results.append(res)
        store[(region, line)] = (p, res, profD, amp[ia, js], bkg[ia, js])
        np.savez(OUT_DIR / f"{region}_{line}_flux.npz", delta_grid=D_GRID, s_grid=S_GRID, dm2lnL=dm,
                 rho=p["rho"], I=p["I"], err=p["err"], best_delta=D_GRID[ia], best_s=S_GRID[js],
                 s_1sig=(inS.min(), inS.max()), centre=(cx, cy))
        print(f"{region:15s} {line:4s} {n:2d} annuli, centre ({cx:+.2f},{cy:+.2f}): Delta={D_GRID[ia]:.3f} "
              f"[{inD.min():.3f},{inD.max():.3f}] s={S_GRID[js]:.2f} [{inS.min():.2f},{inS.max():.2f}] "
              f"chi2={chi.min():.1f}/{dof} (catalogue centre {fitres['chi_catalogue_centre']:.1f}) "
              f"{'; '.join(flags)}", flush=True)

    cols = list(results[0])
    with open(OUT_DIR / "flux_profile_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], (float, np.floating)) else r[c] for c in cols])

    regions = sorted({r for r, _ in store})
    n_cols, n_rows = 4, int(np.ceil(len(regions) / 4))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.6 * n_rows), squeeze=False)
    rr = np.linspace(0, RHO_MAX, 300)
    for ax, region in zip(axes.flat, regions):
        for line in LINES:
            if (region, line) not in store:
                continue
            p, res, _, A, B = store[(region, line)]
            c = LINE_COLORS[line]
            norm = p["I"].max()
            ax.errorbar(p["rho"], p["I"] / norm, p["err"] / norm, fmt="o", ms=3, color=c, capsize=1.5,
                        label=fr"{LINE_LABELS[line]}: $\Delta$={res['best_delta']:.2g}, $s$={res['best_s']:.2f}")
            M = model_profiles(rr, res["best_delta"], np.array([res["best_s"]]), p["sigma_rho"])[0]
            ax.plot(rr, (A * M + B) / norm, LINESTYLES[line], color=c, lw=1.4)
        ax.axvline(1, color="0.5", lw=0.8, ls="--")
        ax.set_title(region.replace("_", " "), fontsize=11)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\rho / r_{\rm HII}$ (fitted centre, catalogue radius)")
    for ax in axes[:, 0]:
        ax.set_ylabel("azimuthal mean flux / max")
    fig.suptitle(r"Line surface-brightness profiles and best-fit projected shell $A\,n(\rho/s;\Delta)+B$ "
                 "(errors: azimuthal sector scatter)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flux_profile_fits.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.2 * n_rows), sharex=True, squeeze=False)
    for ax, region in zip(axes.flat, regions):
        for line in LINES:
            if (region, line) not in store:
                continue
            ax.plot(D_GRID, store[(region, line)][2], LINESTYLES[line], color=LINE_COLORS[line], lw=1.5,
                    label=LINE_LABELS[line])
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ax.set(xscale="log", ylim=(0, 10))
        ax.set_title(region.replace("_", " "), fontsize=11)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(regions):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta(-2\ln L)$, profiled over $s$")
    fig.suptitle("Shell thickness from the surface-brightness profile", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "flux_profile_delta.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
