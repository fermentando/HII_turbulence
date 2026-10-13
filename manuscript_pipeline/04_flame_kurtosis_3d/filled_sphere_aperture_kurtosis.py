#!/usr/bin/env python3
"""Kurtosis analysis for concentric projected apertures of a filled sphere."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_DIR = REPO_ROOT / "manuscript_pipeline" / "00_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CORE_DIR))

from projected_cube_kurtosis import gaussian_from_pdf, kurtosis_from_pdf  # noqa: E402
from vsf_module import compute_bins, compute_pdf_and_statistics  # noqa: E402


DEFAULT_NPZ = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell_width_to_rout"
    / "Synthetic_x_projected_shell_rin0p0000_rout1p0000_order1_pdfkk.npz"
)


def aperture_tag(radius: float) -> str:
    return f"rproj{radius:.2f}".replace(".", "p")


def aperture_stats(
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    path_length: np.ndarray,
    radius_cut: float,
    bin_resolution: int,
    n_jobs: int,
) -> dict:
    r_proj = np.sqrt(y**2 + z**2)
    use = np.isfinite(y) & np.isfinite(z) & np.isfinite(v) & (r_proj <= radius_cut)
    y_ap = y[use]
    z_ap = z[use]
    v_ap = v[use]
    vel_err = np.zeros_like(v_ap)

    log_centers, bin_edges, dv_bins, _ = compute_bins(y_ap, z_ap, v_ap, bin_resolution=bin_resolution)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdf, c2, c4, cnt, dv_centers, kk_pairs = compute_pdf_and_statistics(
            y_ap,
            z_ap,
            v_ap,
            vel_err,
            log_centers,
            bin_edges,
            dv_bins,
            order=1,
            n_jobs=n_jobs,
        )
    kk_pdf = kurtosis_from_pdf(pdf, dv_centers)
    finite = np.isfinite(kk_pairs)
    above = finite & (kk_pairs > 3.0)
    path = path_length[use]
    return {
        "radius_cut": radius_cut,
        "y": y_ap,
        "z": z_ap,
        "v": v_ap,
        "path": path,
        "log_centers": log_centers,
        "r": 10**log_centers,
        "pdf": pdf,
        "dv_bins": dv_bins,
        "dv_centers": dv_centers,
        "kk": kk_pairs,
        "kk_pairs": kk_pairs,
        "kk_pdf": kk_pdf,
        "cnt": cnt,
        "c2": c2,
        "c4": c4,
        "n_pixels": int(v_ap.size),
        "path_min": float(np.nanmin(path)),
        "path_max": float(np.nanmax(path)),
        "path_median": float(np.nanmedian(path)),
        "path_std": float(np.nanstd(path)),
        "k_median": float(np.nanmedian(kk_pairs[finite])) if np.any(finite) else np.nan,
        "k_max": float(np.nanmax(kk_pairs[finite])) if np.any(finite) else np.nan,
        "n_above3": int(np.count_nonzero(above)),
        "first_above3_r": float(np.nanmin((10**log_centers)[above])) if np.any(above) else np.nan,
    }


def save_aperture_npz(record: dict, out_dir: Path) -> None:
    out = out_dir / f"filled_sphere_aperture_{aperture_tag(record['radius_cut'])}_order1_pdfkk.npz"
    np.savez_compressed(
        out,
        radius_cut=record["radius_cut"],
        y=record["y"],
        z=record["z"],
        projected_v=record["v"],
        path_length=record["path"],
        pdf=record["pdf"],
        dv_bins=record["dv_bins"],
        dv_centers=record["dv_centers"],
        log_centers=record["log_centers"],
        kk=record["kk"],
        kk_pairs=record["kk_pairs"],
        kk_pdf=record["kk_pdf"],
        c2=record["c2"],
        c4=record["c4"],
        cnt=record["cnt"],
    )


def plot_kurtosis_curves(records: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cmap = plt.colormaps["viridis"]
    radii = np.array([record["radius_cut"] for record in records])
    rmin, rmax = radii.min(), radii.max()
    for record in records:
        color = cmap((record["radius_cut"] - rmin) / (rmax - rmin))
        good = np.isfinite(record["r"]) & np.isfinite(record["kk"]) & (record["kk"] > 0)
        ax.plot(
            record["r"][good],
            record["kk"][good],
            "o-",
            ms=3.2,
            lw=1.2,
            color=color,
            label=fr"$R_{{ap}}\leq {record['radius_cut']:.2f}$",
        )
    ax.axhline(3.0, color="0.25", ls="--", lw=1.1, label="Gaussian K=3")
    ax.set_xscale("log")
    ax.set_xlabel(r"Projected separation, $r_{\rm proj}$")
    ax.set_ylabel("Kurtosis, $K$")
    ax.set_title("Filled sphere: kurtosis curves inside cumulative projected apertures")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_summary(records: list[dict], out: Path) -> None:
    radius = np.array([record["radius_cut"] for record in records])
    kmax = np.array([record["k_max"] for record in records])
    kmedian = np.array([record["k_median"] for record in records])
    n_above = np.array([record["n_above3"] for record in records])
    path_spread = np.array([record["path_max"] - record["path_min"] for record in records])
    path_frac = path_spread / np.array([record["path_median"] for record in records])

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.0), sharex=True)
    ax = axes[0]
    ax.plot(radius, kmax, "o-", label="max K")
    ax.plot(radius, kmedian, "s-", label="median K")
    ax.axhline(3.0, color="0.25", ls="--", lw=1.0)
    for x, y, n in zip(radius, kmax, n_above):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center", color="0.35")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Aperture growth adds chord-length variation and high-K bins")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(radius, path_spread, "o-", label="path max-min")
    ax.plot(radius, path_frac, "s-", label="(path max-min)/median")
    ax.set_xlabel(r"Projected aperture radius, $R_{\rm ap}/r_{\rm out}$")
    ax.set_ylabel("Chord-length spread")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_gaussian_examples(records: list[dict], out: Path) -> None:
    n = len(records)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.0), sharey="row")
    for col, record in enumerate(records):
        ax_map = axes[0, col]
        ax_pdf = axes[1, col]

        sc = ax_map.scatter(record["y"], record["z"], c=record["path"], s=1.3, cmap="magma", linewidths=0)
        ax_map.set_aspect("equal")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_title(fr"$R_{{ap}}\leq {record['radius_cut']:.2f}$")

        finite_high = np.flatnonzero(np.isfinite(record["kk"]) & (record["kk"] > 3.0))
        if finite_high.size:
            idx = finite_high[0]
        else:
            finite = np.flatnonzero(np.isfinite(record["kk"]))
            idx = finite[0] if finite.size else 0
        mean, sigma, model = gaussian_from_pdf(record["pdf"][idx], record["dv_centers"])
        good_pdf = np.isfinite(record["pdf"][idx]) & (record["pdf"][idx] > 0)
        good_model = np.isfinite(model) & (model > 0)
        ax_pdf.semilogy(record["dv_centers"][good_pdf], record["pdf"][idx][good_pdf], "o", ms=3, label=r"$P(|\Delta v|)$")
        ax_pdf.semilogy(record["dv_centers"][good_model], model[good_model], color="0.1", lw=1.5, label="half-normal")
        ax_pdf.set_title(fr"$r={record['r'][idx]:.3f}$, $K={record['kk'][idx]:.2f}$", fontsize=9)
        ax_pdf.set_xlabel(r"$\Delta\langle v_x\rangle_x$")
        if col == 0:
            ax_pdf.set_ylabel(r"$P(|\Delta v|)$")
            ax_pdf.legend(frameon=False, fontsize=8)
        if np.isfinite(mean) and np.isfinite(sigma):
            ax_pdf.text(0.03, 0.92, fr"$\sigma={sigma:.2f}$", transform=ax_pdf.transAxes, va="top", fontsize=8)

    fig.colorbar(sc, ax=axes[0, :], shrink=0.75, label="LOS path length")
    fig.suptitle("Filled sphere aperture test: path length maps and first high-K PDF", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--apertures", type=float, nargs="+", default=[0.20, 0.35, 0.50, 0.65, 0.80, 1.00])
    parser.add_argument("--bin-resolution", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    out_dir = args.npz.parent / "filled_sphere_aperture_kurtosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz) as data:
        y = np.asarray(data["y"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
        v = np.asarray(data["projected_v"], dtype=float)
        projected_map = np.asarray(data["projected_map"], dtype=float)
        shell_counts = np.asarray(data["shell_counts"], dtype=float)
        yy, zz = np.meshgrid(
            np.linspace(-1.2, 1.2, projected_map.shape[0]),
            np.linspace(-1.2, 1.2, projected_map.shape[1]),
            indexing="ij",
        )
        valid_map = np.isfinite(projected_map)
        path_flat = shell_counts[valid_map].ravel()

    # y,z,v in the NPZ are already flattened over valid projected pixels, so path_flat matches that order.
    records = []
    for radius in args.apertures:
        record = aperture_stats(y, z, v, path_flat, radius, args.bin_resolution, args.n_jobs)
        records.append(record)
        save_aperture_npz(record, out_dir)
        print(
            f"R_ap={radius:.2f} n={record['n_pixels']:5d} "
            f"path={record['path_min']:.0f}..{record['path_max']:.0f} "
            f"Kmed={record['k_median']:.3f} Kmax={record['k_max']:.3f} "
            f"bins(K>3)={record['n_above3']:2d} first_r(K>3)={record['first_above3_r']:.4f}"
        )

    plot_kurtosis_curves(records, out_dir / "filled_sphere_aperture_kurtosis_curves.png")
    plot_summary(records, out_dir / "filled_sphere_aperture_kurtosis_summary.png")
    plot_gaussian_examples(records, out_dir / "filled_sphere_aperture_gaussian_examples.png")
    print(f"Wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
