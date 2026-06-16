#!/usr/bin/env python3
"""Generate order-1 NPZ files with pdf/dv_bins/log_centers/kk for all regions.

Outputs are written to a separate folder so existing files are not overwritten.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

from vsf_module import compute_bins, compute_pdf_and_statistics


def get_heliocentric_distance(filename: str) -> float:
    special = {
        "Flame_nebula": 400.0,
        "Orion": 400.0,
        "Rosette_nebula": 1500.0,
    }

    base_name = os.path.splitext(os.path.basename(filename))[0]
    object_name = base_name.replace("LVM_", "")
    if object_name in special:
        return special[object_name]

    hdul = fits.open("extra_HII_regions.fits")
    data = hdul[1].data
    hdul.close()

    name_mapping = {
        "Flame_nebula": "Flame",
        "Rosette_nebula": "Rosette",
    }
    lookup = name_mapping.get(object_name, object_name)

    try:
        idx = list(data["HII Region"]).index(lookup)
        return float(data["Heliocentric_distance"][idx]) * 1e3
    except ValueError:
        return 1500.0


def available_lines(table_names: list[str]) -> list[str]:
    lines: list[str] = []
    for line in ("nii", "sii", "siii"):
        if f"vel_{line}" in table_names and f"flux_{line}" in table_names and f"flux_{line}_err" in table_names:
            lines.append(line)
    return lines


def main() -> None:
    out_dir = Path("NPZ_kurtosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(glob.glob("LVM_*.fits"))
    if not fits_files:
        print("No LVM_*.fits files found")
        return

    ok = 0
    skipped = 0

    for f in fits_files:
        hdul = fits.open(f)
        table = hdul[1].data
        names = list(table.names)

        obj = os.path.splitext(os.path.basename(f))[0].replace("LVM_", "")
        dist = get_heliocentric_distance(f)

        for line in available_lines(names):
            try:
                snr = table[f"flux_{line}"] / table[f"flux_{line}_err"]
                mask = np.isfinite(snr) & (snr > 5)

                ra = np.asarray(table["ra"][mask], dtype=float)
                dec = np.asarray(table["dec"][mask], dtype=float)
                vel = np.asarray(table[f"vel_{line}"][mask], dtype=float)

                vel_err_col = f"vel_{line}_err"
                if vel_err_col in names:
                    vel_err = np.asarray(table[vel_err_col][mask], dtype=float)
                else:
                    vel_err = np.zeros_like(vel)

                valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(vel) & np.isfinite(vel_err)
                ra, dec, vel, vel_err = ra[valid], dec[valid], vel[valid], vel_err[valid]

                if ra.size < 30:
                    skipped += 1
                    print(f"SKIP {obj} {line.upper()}: too few points ({ra.size})")
                    continue

                # Keep the same coordinate scaling used in VSF code path.
                x = (ra - np.mean(ra)) * np.cos(np.deg2rad(np.mean(dec))) * dist * np.pi / 180.0
                y = (dec - np.mean(dec)) * dist * np.pi / 180.0

                log_centers, bin_edges, dv_bins, _ = compute_bins(x, y, vel, bin_resolution=20)
                pdf, c2, c4, cnt, dv_centers, kk = compute_pdf_and_statistics(
                    x, y, vel, vel_err, log_centers, bin_edges, dv_bins, order=1, n_jobs=4
                )

                out = out_dir / f"{obj}_{line.upper()}_order1_pdfkk.npz"
                np.savez_compressed(
                    out,
                    pdf=pdf,
                    dv_bins=dv_bins,
                    log_centers=log_centers,
                    kk=kk,
                    c2=c2,
                    c4=c4,
                    cnt=cnt,
                    dv_centers=dv_centers,
                )
                ok += 1
                print(f"OK   {out.name}")

            except Exception as exc:
                skipped += 1
                print(f"SKIP {obj} {line.upper()}: {exc}")

        hdul.close()

    print(f"\nDone. Created: {ok}, Skipped: {skipped}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
