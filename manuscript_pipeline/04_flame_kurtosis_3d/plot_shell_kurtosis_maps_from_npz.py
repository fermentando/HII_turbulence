#!/usr/bin/env python3
"""Create 3-D kurtosis maps and Gaussian-slice plots for every shell NPZ."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from projected_cube_kurtosis import (  # noqa: E402
    make_kurtosis_plot,
    make_kurtosis_with_gaussian_slices_plot,
)


SHELL_DIR = (
    REPO_ROOT
    / "Figures"
    / "manuscript"
    / "04_kurtosis_3d_all_regions"
    / "synthetic_projected_shell"
)
STYLE_PATH = REPO_ROOT / "kurtosis_plot.mplstyle"
FNAME_RE = re.compile(r"(?P<prefix>Synthetic_x_projected_shell_rin[\dp]+_rout[\dp]+)_order1_pdfkk\.npz")


def main() -> None:
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))

    files = sorted(SHELL_DIR.glob("Synthetic_x_projected_shell_rin*_order1_pdfkk.npz"))
    files = [path for path in files if FNAME_RE.match(path.name)]
    if not files:
        raise FileNotFoundError(f"No shell NPZ files found in {SHELL_DIR}")

    for path in files:
        match = FNAME_RE.match(path.name)
        prefix = match.group("prefix")
        with np.load(path) as data:
            pdf = np.asarray(data["pdf"], dtype=float)
            dv_bins = np.asarray(data["dv_bins"], dtype=float)
            log_centers = np.asarray(data["log_centers"], dtype=float)
            kk = np.asarray(data["kk"], dtype=float)
            r_inner = float(data["r_inner"])
            r_outer = float(data["r_outer"])

        kurt_out = SHELL_DIR / f"{prefix}_order1_3d_kurtosis.png"
        slices_out = SHELL_DIR / f"{prefix}_order1_3d_kurtosis_gaussian_slices.png"
        make_kurtosis_plot(pdf, dv_bins, log_centers, kk, kurt_out, None, None)
        make_kurtosis_with_gaussian_slices_plot(
            pdf,
            dv_bins,
            log_centers,
            kk,
            slices_out,
            None,
            None,
            n_slices=4,
        )
        finite = kk[np.isfinite(kk) & (kk > 0)]
        print(
            f"Wrote {kurt_out.relative_to(REPO_ROOT)} and {slices_out.name} "
            f"(rin={r_inner:.4f}, rout={r_outer:.4f}, "
            f"Kmed={np.nanmedian(finite):.2f}, Kmax={np.nanmax(finite):.2f})"
        )

    print(f"Done. Plotted {len(files)} shell widths.")


if __name__ == "__main__":
    main()
