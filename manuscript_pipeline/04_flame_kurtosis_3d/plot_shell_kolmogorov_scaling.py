#!/usr/bin/env python3
"""Plot projected VSF scaling for all synthetic shell widths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
FNAME_RE = re.compile(r"Synthetic_x_projected_shell_rin([\dp]+)_rout([\dp]+)_order1_pdfkk\.npz")

N_BINS = 24
N_PAIRS = 1_500_000
RNG_SEED = 31415


def parse_shell_radii(path: Path) -> tuple[float, float]:
    match = FNAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected shell filename: {path.name}")
    r_inner = float(match.group(1).replace("p", "."))
    r_outer = float(match.group(2).replace("p", "."))
    return r_inner, r_outer


def binned_vsf_2d(
    coords: np.ndarray,
    velocity: np.ndarray,
    n_bins: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_points = coords.shape[0]
    i = rng.integers(0, n_points, size=n_pairs)
    j = rng.integers(0, n_points, size=n_pairs)
    good_pair = i != j
    i = i[good_pair]
    j = j[good_pair]

    dr = coords[i] - coords[j]
    dv = velocity[i] - velocity[j]
    sep = np.linalg.norm(dr, axis=1)
    dv2 = dv**2

    good = sep > 0
    sep = sep[good]
    dv2 = dv2[good]

    edges = np.geomspace(np.nanmin(sep), np.nanpercentile(sep, 85), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    s2 = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    which = np.digitize(sep, edges) - 1

    for idx_bin in range(n_bins):
        idx = np.flatnonzero(which == idx_bin)
        counts[idx_bin] = idx.size
        if idx.size:
            s2[idx_bin] = np.nanmean(dv2[idx])

    return centers, s2, counts


def load_shells(shell_dir: Path) -> list[dict]:
    shells = []
    for path in sorted(shell_dir.glob("Synthetic_x_projected_shell_rin*_order1_pdfkk.npz")):
        if FNAME_RE.match(path.name) is None:
            continue
        r_inner, r_outer = parse_shell_radii(path)
        with np.load(path) as data:
            y = np.asarray(data["y"], dtype=float)
            z = np.asarray(data["z"], dtype=float)
            velocity = np.asarray(data["projected_v"], dtype=float)
        valid = np.isfinite(y) & np.isfinite(z) & np.isfinite(velocity)
        shells.append(
            {
                "path": path,
                "r_inner": r_inner,
                "r_outer": r_outer,
                "width": r_outer - r_inner,
                "coords": np.column_stack([y[valid], z[valid]]),
                "velocity": velocity[valid],
            }
        )
    return sorted(shells, key=lambda shell: shell["width"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-dir", type=Path, default=DEFAULT_SHELL_DIR)
    args = parser.parse_args()
    shell_dir = args.shell_dir
    out_fig = shell_dir / "shell_projected_vsf_kolmogorov_scaling.png"
    out_csv = shell_dir / "shell_projected_vsf_kolmogorov_slopes.csv"

    try:
        plt.style.use(str(REPO_ROOT / "custom_plot.mplstyle"))
    except OSError:
        pass

    shells = load_shells(shell_dir)
    if not shells:
        raise FileNotFoundError(f"No shell NPZ files found in {shell_dir}")

    rng = np.random.default_rng(RNG_SEED)
    cmap = plt.colormaps["viridis"]
    widths = np.array([shell["width"] for shell in shells])
    wmin, wmax = float(np.min(widths)), float(np.max(widths))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    rows = []

    for shell in shells:
        r, s2, counts = binned_vsf_2d(shell["coords"], shell["velocity"], N_BINS, N_PAIRS, rng)
        fit_mask = np.isfinite(s2) & (s2 > 0) & (counts > 50)
        fit_mask &= (r > 0.08) & (r < 0.65)
        slope, intercept = np.polyfit(np.log10(r[fit_mask]), np.log10(s2[fit_mask]), 1)

        color = cmap(0.0 if wmax == wmin else (shell["width"] - wmin) / (wmax - wmin))
        ax.loglog(
            r,
            s2,
            "o-",
            ms=3.4,
            lw=1.25,
            color=color,
            alpha=0.88,
            label=fr"$\Delta R={shell['width']:.3f}$",
        )
        rows.append((shell["width"], shell["r_inner"], shell["r_outer"], slope, len(shell["velocity"])))
        print(f"width={shell['width']:.4f} slope={slope:.3f} n={len(shell['velocity'])}")

    ref_shell = shells[len(shells) // 2]
    r_ref, s2_ref, counts_ref = binned_vsf_2d(ref_shell["coords"], ref_shell["velocity"], N_BINS, N_PAIRS, rng)
    ref_mask = np.isfinite(s2_ref) & (s2_ref > 0) & (counts_ref > 50) & (r_ref > 0.08) & (r_ref < 0.65)
    anchor = np.nanmedian(s2_ref[ref_mask] / r_ref[ref_mask] ** (2.0 / 3.0))
    r_line = np.geomspace(np.nanmin(r_ref[ref_mask]), np.nanmax(r_ref[ref_mask]), 200)
    ax.loglog(r_line, anchor * r_line ** (2.0 / 3.0), "k--", lw=1.8, label=r"$r^{2/3}$")

    ax.set_xlabel(r"Projected separation, $r_{\rm proj}$")
    ax.set_ylabel(r"$S_2(r)=\langle(\Delta v)^2\rangle$")
    ax.set_title("Projected shell VSFs compared with Kolmogorov scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2, title="Shell width")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", encoding="utf-8") as fobj:
        fobj.write("width,r_inner,r_outer,slope,n_projected_pixels\n")
        for width, r_inner, r_outer, slope, n_pixels in rows:
            fobj.write(f"{width:.8f},{r_inner:.8f},{r_outer:.8f},{slope:.8f},{n_pixels}\n")

    print(f"Wrote {out_fig.relative_to(REPO_ROOT)}")
    print(f"Wrote {out_csv.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
