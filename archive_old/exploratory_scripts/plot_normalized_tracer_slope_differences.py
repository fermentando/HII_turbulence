import csv
import os

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('./custom_plot.mplstyle')

SUMMARY_CSV = 'SlopeResults/median_vsf_slope_ks_summary.csv'
OUT_FIG = 'SlopeResults/slope_by_region_and_tracer.png'

TRACERS = ['NII', 'SII', 'SIII']
COLORS = {'NII': 'C0', 'SII': 'C1', 'SIII': 'C2'}
MARKERS = {'NII': 'o', 'SII': 's', 'SIII': '^'}


def load_rows(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_sqrt_max_vsf(npz_path):
    """Return sqrt(max(vsf)) from NPZ median VSF file."""
    if not npz_path or not os.path.exists(npz_path):
        return np.nan

    with np.load(npz_path) as d:
        vsf = d['vsf'].astype(float)

    vsf = vsf[np.isfinite(vsf) & (vsf > 0)]
    if vsf.size == 0:
        return np.nan
    return float(np.sqrt(np.max(vsf)))


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f'Missing summary file: {SUMMARY_CSV}')

    rows = load_rows(SUMMARY_CSV)

    # Keep only valid global slopes.
    filtered = []
    for row in rows:
        region = row.get('object', '')
        tracer = row.get('line', '').upper()
        slope = to_float(row.get('slope_median', np.nan))
        slope_err = to_float(row.get('slope_err_half68', np.nan))
        npz_file = row.get('npz_file', '')
        sqrt_max_vsf = load_sqrt_max_vsf(npz_file)

        if not region or tracer not in TRACERS:
            continue
        if not np.isfinite(slope) or not np.isfinite(sqrt_max_vsf):
            continue
        if not np.isfinite(slope_err) or slope_err < 0:
            slope_err = np.nan

        filtered.append({
            'region': region,
            'tracer': tracer,
            'slope': slope,
            'slope_err': slope_err,
            'sqrt_max_vsf': sqrt_max_vsf,
        })

    if not filtered:
        raise RuntimeError('No valid slope entries found in summary CSV.')

    regions = sorted({row['region'] for row in filtered})

    # Build per-region NII reference for normalization.
    nii_by_region = {}
    for region in regions:
        entries = [r for r in filtered if r['region'] == region and r['tracer'] == 'NII']
        if not entries:
            continue
        nii_by_region[region] = entries[0]

    fig, ax = plt.subplots(figsize=(9, 6))

    for tracer in TRACERS:
        if tracer == 'NII':
            continue
        x_vals, y_vals, y_errs = [], [], []
        for region in regions:
            matches = [r for r in filtered if r['region'] == region and r['tracer'] == tracer]
            if not matches:
                continue

            # There should be only one entry, but take first robustly.
            entry = matches[0]

            # Normalize by NII slope in the same region and plot relative difference.
            nii_entry = nii_by_region.get(region)
            if nii_entry is None:
                continue

            m_nii = nii_entry['slope']
            e_nii = nii_entry['slope_err']
            if (not np.isfinite(m_nii)) or m_nii == 0:
                continue

            x_nii = nii_entry['sqrt_max_vsf']
            x_t = entry['sqrt_max_vsf']
            if (not np.isfinite(x_nii)) or x_nii <= 0 or (not np.isfinite(x_t)) or x_t <= 0:
                continue

            m_t = entry['slope']
            e_t = entry['slope_err']

            rel = (m_t - m_nii) / m_nii
            x_rel = x_t / x_nii

            # First-order error propagation for rel = m_t/m_nii - 1.
            if np.isfinite(e_t) and np.isfinite(e_nii):
                rel_err = np.sqrt((e_t / m_nii) ** 2 + ((m_t * e_nii) / (m_nii ** 2)) ** 2)
            else:
                rel_err = np.nan

            x_vals.append(x_rel)
            y_vals.append(rel)
            y_errs.append(rel_err)

        if not x_vals:
            continue

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        y_err_arr = np.asarray(y_errs, dtype=float)
        finite_err = np.isfinite(y_err_arr)

        # Plot propagated uncertainties (includes NII and tracer slope errors)
        # wherever both are available.
        if np.any(finite_err):
            ax.errorbar(
                x_arr[finite_err],
                y_arr[finite_err],
                yerr=y_err_arr[finite_err],
                fmt=MARKERS[tracer],
                color=COLORS[tracer],
                ecolor=COLORS[tracer],
                markersize=7,
                capsize=3,
                linewidth=1.6,
                label=tracer,
            )

        # If some points have missing uncertainty, still show the marker.
        if np.any(~finite_err):
            ax.plot(
                x_arr[~finite_err],
                y_arr[~finite_err],
                MARKERS[tracer],
                color=COLORS[tracer],
                markersize=7,
                label=None,
            )

    ax.axhline(0.0, color='gray', linestyle='--', alpha=0.5, label='NII reference (0)')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='NII norm (1)')
    ax.set_ylabel(r'$(m_{\mathrm{tracer}}-m_{\mathrm{NII}})/m_{\mathrm{NII}}$')
    ax.set_xlabel(r'$\sqrt{\max(\mathcal{S}_2)}_{\mathrm{tracer}} / \sqrt{\max(\mathcal{S}_2)}_{\mathrm{NII}}$')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {OUT_FIG}')


if __name__ == '__main__':
    main()
