#!/usr/bin/env python3
"""N=5,000 dK/dlogr shell-width curves vs v_bulk/v_turb, with logistic fits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from generate_random_point_expanding_shell_grid import write_random_shell_npz  # noqa: E402
from generate_shell_thickness_grid import R_OUTER_FIXED, shell_output_path, width_grid  # noqa: E402
from plot_shell_dk_dlogr_aperture_comparison import fit_k_vs_logr_above_floor  # noqa: E402
from plot_shell_kurtosis_slope_vs_thickness import KURTOSIS_FLOOR  # noqa: E402
from simulate_kolmogorov_sphere import kolmogorov_velocity_grid  # noqa: E402

BASE_DIR = REPO_ROOT / "Figures" / "manuscript" / "04_kurtosis_3d_all_regions"
N_POINTS = 5_000
SIGMA_TURB = 1.0
VBULK_RATIOS = np.concatenate(([0.0], np.geomspace(0.1, 100.0, 13)))
WIDTHS = width_grid(0.02, 1.0, 14)
BIN_RESOLUTION = 20
MIN_FIT_POINTS_FOR_LINE = 3
KK_KEY = "kk_pairs"  # pair estimator <dv^4>/<dv^2>^2 (older shell files store the PDF kurtosis in "kk")
X0_FIXED = -1.34  # logistic centre pinned: free-x0 fits gave x0 ~ -1.34 + 0.07 log10(v), i.e. ~flat
FIT_BOUNDS = ([-5.0, 0.03], [0.0, 5.0])  # (floor, scale); floor is fitted, not fixed
FIELD_SEED = 7
POINT_SEED = 1234
N_GRID = 160
BOX_SIZE = 2.4
R_OUTER = R_OUTER_FIXED

CACHE = BASE_DIR / "_vbulk_ratio_sweep_n5000_logistic_cache.npz"


def ratio_tag(ratio: float) -> str:
    return f"{ratio:.6g}".replace("-", "m").replace(".", "p")


def run_dir(ratio: float) -> Path:
    if np.isclose(ratio, 0.0):
        return BASE_DIR / "synthetic_projected_shell_random_points_nopixel_nobulk_n5000"
    if np.isclose(ratio, 10.0):
        return BASE_DIR / "synthetic_projected_shell_random_points_nopixel_n5000"
    return BASE_DIR / f"synthetic_projected_shell_random_points_nopixel_vratio{ratio_tag(ratio)}_n5000"


def metadata_matches(path: Path, ratio: float) -> bool:
    metadata_path = path / "random_point_expanding_shell_metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        int(metadata.get("n_points_per_shell", -1)) == N_POINTS
        and bool(metadata.get("no_pixelize", False))
        and np.isclose(float(metadata.get("v_bulk", np.nan)), ratio * SIGMA_TURB)
        and np.isclose(float(metadata.get("sigma_vx_turb_grid", np.nan)), SIGMA_TURB)
    )


def shell_files_exist(path: Path) -> bool:
    for width in WIDTHS:
        r_inner = R_OUTER - float(width)
        if not shell_output_path(path, r_inner, R_OUTER).exists():
            return False
    return True


def write_metadata(path: Path, ratio: float) -> None:
    metadata = {
        "description": "N=5000 no-pixelized random-point shells for v_bulk/v_turb sweep.",
        "n_grid": N_GRID,
        "box_size": BOX_SIZE,
        "seed": FIELD_SEED,
        "point_seed": POINT_SEED,
        "v_bulk": float(ratio * SIGMA_TURB),
        "v_bulk_over_sigma_turb": float(ratio),
        "r_outer": R_OUTER,
        "width_min": float(WIDTHS.min()),
        "width_max": float(WIDTHS.max()),
        "n_widths": int(len(WIDTHS)),
        "n_points_per_shell": N_POINTS,
        "bin_resolution": BIN_RESOLUTION,
        "sigma_vx_turb_grid": SIGMA_TURB,
        "no_pixelize": True,
    }
    metadata_path = path / "random_point_expanding_shell_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def ensure_shell_data() -> None:
    missing = [ratio for ratio in VBULK_RATIOS if not (metadata_matches(run_dir(ratio), ratio) and shell_files_exist(run_dir(ratio)))]
    if not missing:
        return

    print(f"Generating missing N={N_POINTS:,} no-pixelized shell grids for {len(missing)} v_bulk/v_turb values...")
    rng_field = np.random.default_rng(FIELD_SEED)
    grids = kolmogorov_velocity_grid(N_GRID, BOX_SIZE, rng_field)

    for ratio in missing:
        out_dir = run_dir(ratio)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(out_dir, ratio)
        rng_points = np.random.default_rng(POINT_SEED)
        v_bulk = float(ratio * SIGMA_TURB)
        for width in WIDTHS:
            r_inner = R_OUTER - float(width)
            out_path = shell_output_path(out_dir, r_inner, R_OUTER)
            if out_path.exists():
                continue
            stats = write_random_shell_npz(
                grids,
                BOX_SIZE,
                r_inner,
                R_OUTER,
                v_bulk,
                N_POINTS,
                n_proj_grid=51,
                out_path=out_path,
                rng=rng_points,
                bin_resolution=BIN_RESOLUTION,
                n_jobs=1,
                no_pixelize=True,
            )
            print(
                f"  ratio={ratio:g} width={width:.4f} r_min={stats['r_min']:.5f} "
                f"Kmed={stats['k_median']:.2f}"
            )


def load_shell_records(ratio: float):
    records, r_min = [], []
    d = run_dir(ratio)
    for width in WIDTHS:
        r_inner = R_OUTER - float(width)
        path = shell_output_path(d, r_inner, R_OUTER)
        with np.load(path) as data:
            r_min.append(float(data["r_min"]))
            records.append({
                "r": 10 ** np.asarray(data["log_centers"], dtype=float),
                "kk": np.asarray(data[KK_KEY], dtype=float),
            })
    return records, np.asarray(r_min)


def slope_for_record(record: dict[str, np.ndarray]) -> tuple[float, bool]:
    fit = fit_k_vs_logr_above_floor(record, floor=KURTOSIS_FLOOR)
    return fit["slope"], bool(fit["used_zero_fallback"])


def logistic(log_w: np.ndarray, floor: float, scale: float) -> np.ndarray:
    return floor / (1.0 + np.exp((log_w - X0_FIXED) / scale))


def logistic_fit(log_w: np.ndarray, y: np.ndarray, mask: np.ndarray):
    if int(np.count_nonzero(mask)) < MIN_FIT_POINTS_FOR_LINE:
        return None
    p0 = [min(float(np.nanmin(y[mask])), -0.05), 0.35]
    try:
        params, cov = curve_fit(
            logistic, log_w[mask], y[mask], p0=p0, bounds=FIT_BOUNDS, maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    dense = np.linspace(log_w[mask].min(), log_w[mask].max(), 300)
    curve = logistic(dense, *params)
    has_cov = np.all(np.isfinite(cov))
    perr = np.sqrt(np.diag(cov)) if has_cov else np.full(2, np.nan)
    return dense, curve, *[float(x) for x in params], *[float(x) for x in perr]


def bulk_switch(v: np.ndarray, p_zero: float, p_inf: float, v_c: float, n: float) -> np.ndarray:
    """p_zero at v=0 rising/falling to p_inf for v >> v_c: p_zero + (p_inf - p_zero) v^n/(v^n + v_c^n)."""
    v = np.asarray(v, dtype=float)
    s = v**n / (v**n + v_c**n)
    return p_zero + (p_inf - p_zero) * s


def fit_bulk_switch(v: np.ndarray, y: np.ndarray, y_err: np.ndarray):
    mask = np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    p0 = [float(y[mask][0]), float(y[mask][-1]), 1.0, 1.5]
    bounds = ([-np.inf, -np.inf, 1e-3, 0.2], [np.inf, np.inf, 1e3, 10.0])
    params, cov = curve_fit(
        bulk_switch, v[mask], y[mask], p0=p0, sigma=y_err[mask],
        absolute_sigma=True, bounds=bounds, maxfev=20000,
    )
    chi2 = float(np.sum(((y[mask] - bulk_switch(v[mask], *params)) / y_err[mask]) ** 2))
    dof = int(np.count_nonzero(mask)) - len(params)
    return params, np.sqrt(np.diag(cov)), chi2, dof


def fit_log_linear_trend(x: np.ndarray, y: np.ndarray):
    mask = (x > 0) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(np.log10(x[mask]), y[mask], 1)
    return float(slope), float(intercept)


def load_or_compute():
    if CACHE.exists():
        with np.load(CACHE) as c:
            if (
                np.allclose(c["vbulk_ratios"], VBULK_RATIOS)
                and np.allclose(c["widths"], WIDTHS)
                and int(c["bin_resolution"]) == BIN_RESOLUTION
                and int(c["n_points"]) == N_POINTS
                and "kk_key" in c and str(c["kk_key"]) == KK_KEY
            ):
                print(f"Loading cached slopes from {CACHE.relative_to(REPO_ROOT)}...")
                return c["slopes"], c["fallback"], c["r_min"]
        print("Cache does not match current sweep; recomputing slopes.")

    print("Computing dK/dlogr slopes for the N=5,000 v_bulk/v_turb sweep...")
    slopes = np.full((len(VBULK_RATIOS), len(WIDTHS)), np.nan)
    fallback = np.zeros_like(slopes, dtype=bool)
    r_min_used = np.full_like(slopes, np.nan)

    for j, ratio in enumerate(VBULK_RATIOS):
        records, r_min = load_shell_records(float(ratio))
        r_min_used[j] = r_min
        for i, record in enumerate(records):
            slopes[j, i], fallback[j, i] = slope_for_record(record)
        print(f"  ratio={ratio:g}: {['%.3f' % s for s in slopes[j, :5]]} ...")

    np.savez_compressed(
        CACHE,
        vbulk_ratios=VBULK_RATIOS,
        widths=WIDTHS,
        bin_resolution=BIN_RESOLUTION,
        n_points=N_POINTS,
        kk_key=KK_KEY,
        slopes=slopes,
        fallback=fallback,
        r_min=r_min_used,
    )
    print(f"Cached to {CACHE.relative_to(REPO_ROOT)}.")
    return slopes, fallback, r_min_used


def draw_panels(fig, axL, axF, axR, v_cmap="plasma", zero_color="0.2", floor_color="crimson", scale_color="#31688e"):
    """Draw the slope curves, fitted floor and fitted scale panels onto given axes.

    v_cmap (name or Colormap) colours the v_bulk/v_turb curves (zero_color for v = 0); floor_color and
    scale_color the fitted parameters.

    Returns (fit_rows, switch_fits, r_min_used) for the CSV outputs.
    """
    ensure_shell_data()
    slopes, fallback, r_min_used = load_or_compute()

    log_widths = np.log10(WIDTHS)
    fit_rows = []
    fit_curves = []
    for ratio, y_arr, fb_arr in zip(VBULK_RATIOS, slopes, fallback):
        fit = logistic_fit(log_widths, y_arr, ~fb_arr)
        if fit is None:
            fit_rows.append((ratio, np.nan, np.nan, np.nan, np.nan))
            fit_curves.append(None)
            continue
        dense, curve, floor, scale, floor_err, scale_err = fit
        fit_rows.append((ratio, floor, floor_err, scale, scale_err))
        print(f"ratio={ratio:g}: floor={floor:.3f}+/-{floor_err:.3f}, scale={scale:.3f}+/-{scale_err:.3f}")
        fit_curves.append((dense, curve))

    nonzero = VBULK_RATIOS > 0
    norm = matplotlib.colors.LogNorm(vmin=VBULK_RATIOS[nonzero].min(), vmax=VBULK_RATIOS[nonzero].max())
    cmap = plt.colormaps[v_cmap] if isinstance(v_cmap, str) else v_cmap

    for ratio, y_arr, curve_fit in zip(VBULK_RATIOS, slopes, fit_curves):
        color = zero_color if ratio == 0 else cmap(norm(ratio))
        label = r"$v_{\rm bulk}/v_{\rm turb}=0$" if ratio == 0 else None
        axL.plot(WIDTHS, y_arr, "o", ms=4.5, color=color, alpha=0.9, zorder=3, label=label)
        if curve_fit is not None:
            dense, curve = curve_fit
            axL.plot(10**dense, curve, "-", color=color, lw=1.5, alpha=0.95, zorder=4)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axL)
    cbar.set_label(r"$v_{\rm bulk}/v_{\rm turb}$")
    axL.axvline(0.2, color="0.25", ls=":", lw=1.1)
    axL.axhline(0, color="0.25", ls="--", lw=1.1)
    axL.set_xscale("log")
    axL.set_xlabel(r"$\Delta r_{\rm shell}/r_{\rm HII}$")
    axL.set_ylabel(r"$dK/d\log_{10} r$")
    axL.set_title(r"$N=5{,}000$: $v_{\rm bulk}/v_{\rm turb}$ sweep", fontsize=11)
    axL.grid(True, axis="y", ls="--", alpha=0.4)
    axL.legend(fontsize=8, loc="lower right")

    fit_arr = np.asarray(fit_rows, dtype=float)
    good = np.isfinite(fit_arr[:, 1])
    x_plot = VBULK_RATIOS.copy()
    x_plot[0] = 0.03

    axF.errorbar(
        x_plot[good], fit_arr[good, 1], yerr=fit_arr[good, 2],
        fmt="o", ms=5, lw=1.2, capsize=3, color=floor_color,
    )
    switch_fits = {}
    v_dense = np.concatenate(([0.0], np.geomspace(0.05, VBULK_RATIOS.max(), 300)))
    x_dense = v_dense.copy()
    x_dense[0] = x_plot[0]
    for name, col, ax, color in [("floor", 1, axF, floor_color), ("scale", 3, axR, scale_color)]:
        params, perr, chi2, dof = fit_bulk_switch(VBULK_RATIOS, fit_arr[:, col], fit_arr[:, col + 1])
        switch_fits[name] = (params, perr, chi2, dof)
        p_zero, p_inf, v_c, n = params
        ax.plot(
            x_dense, bulk_switch(v_dense, *params), "-", lw=1.5, color=color, alpha=0.7, zorder=2,
            label=(
                fr"${p_zero:.2f}+({p_inf:.2f}-{p_zero:.2f})\,\frac{{v^{{{n:.2f}}}}}{{v^{{{n:.2f}}}+{v_c:.2f}^{{{n:.2f}}}}}$"
            ),
        )
        ax.legend(fontsize=9, loc="best")
        print(
            f"{name} switch: p0={p_zero:.3f}+/-{perr[0]:.3f}, p_inf={p_inf:.3f}+/-{perr[1]:.3f}, "
            f"v_c={v_c:.3f}+/-{perr[2]:.3f}, n={n:.3f}+/-{perr[3]:.3f}, chi2/dof={chi2:.1f}/{dof}"
        )
    axF.axhline(0, color="0.25", ls="--", lw=1.0)
    axF.set_xscale("log")
    axF.set_xlabel(r"$v_{\rm bulk}/v_{\rm turb}$ ($0$ shown at $0.03$)")
    axF.set_ylabel(r"logistic floor (small-width $dK/d\log_{10} r$)")
    axF.set_title(fr"Fitted logistic floor ($x_0$ fixed$={X0_FIXED:.2f}$)", fontsize=11)
    axF.set_ylim(FIT_BOUNDS[0][0] - 0.3, 0.3)  # unconstrained fits have error bars far beyond this
    axF.grid(True, axis="y", ls="--", alpha=0.4)

    axR.errorbar(
        x_plot[good], fit_arr[good, 3], yerr=fit_arr[good, 4],
        fmt="^", ms=5, lw=1.2, capsize=3, color=scale_color,
    )
    axR.set_xscale("log")
    axR.set_xlabel(r"$v_{\rm bulk}/v_{\rm turb}$ ($0$ shown at $0.03$)")
    axR.set_ylabel("logistic scale")
    axR.set_title(fr"Fitted logistic scale ($x_0$ fixed$={X0_FIXED:.2f}$)", fontsize=11)
    axR.grid(True, axis="y", ls="--", alpha=0.4)
    return fit_rows, switch_fits, r_min_used


def main() -> None:
    style_path = REPO_ROOT / "custom_plot.mplstyle"
    try:
        plt.style.use(str(style_path))
    except OSError:
        pass
    fig, (axL, axF, axR) = plt.subplots(1, 3, figsize=(21.0, 6.0))
    fit_rows, switch_fits, r_min_used = draw_panels(fig, axL, axF, axR)
    fig.tight_layout()

    out_png = BASE_DIR / "random_point_count_dK_dlogr_vbulk_ratio_sweep_n5000_logistic.png"
    out_csv = BASE_DIR / "random_point_count_dK_dlogr_vbulk_ratio_sweep_n5000_logistic.csv"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(
            "v_bulk_over_v_turb,floor,floor_err,x0_fixed,scale,scale_err,min_r_min,max_r_min\n"
        )
        for (ratio, floor, floor_err, scale, scale_err), rmins in zip(fit_rows, r_min_used):
            f.write(
                f"{ratio:.8g},{floor:.8g},{floor_err:.8g},{X0_FIXED:.8g},"
                f"{scale:.8g},{scale_err:.8g},{np.nanmin(rmins):.8g},{np.nanmax(rmins):.8g}\n"
            )

    out_switch_csv = BASE_DIR / "random_point_count_dK_dlogr_vbulk_ratio_sweep_n5000_switch_fits.csv"
    with out_switch_csv.open("w", encoding="utf-8") as f:
        f.write("parameter,x0_fixed,p_zero,p_zero_err,p_inf,p_inf_err,v_c,v_c_err,n,n_err,chi2,dof\n")
        for name, (params, perr, chi2, dof) in switch_fits.items():
            vals = ",".join(f"{a:.8g},{b:.8g}" for a, b in zip(params, perr))
            f.write(f"{name},{X0_FIXED:.8g},{vals},{chi2:.6g},{dof}\n")

    print(f"Wrote {out_png.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_switch_csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
