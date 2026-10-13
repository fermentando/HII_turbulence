"""Plot the four small-error, near-zero [SII] kurtosis slopes against the binning cutoff.

r_min is the 37-arcsec lower separation cutoff, normalized by R_HII on a log axis.
Uses existing slope fits and verifies the cutoff against saved VSF metadata.
Run from the repository root.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Figures/manuscript/04_flame_kurtosis_3d'
REGIONS = ('BFS65', 'RCW60', 'RCW79', 'RCW87')
with (OUT / 'kurtosis_slope_fits.csv').open(newline='') as stream:
    fits = {row['region']: row for row in csv.DictReader(stream) if row['line'] == 'SII'}
rows = []
for region in REGIONS:
    fit = fits[region]
    with np.load(ROOT / f'NPZ/KURT_{region}_SII.npz') as kurt:
        rmin = float(kurt['distance_pc']) * np.deg2rad(37.0 / 3600)
        first_center = float(np.min(kurt['l_pc']))
        r_hii = float(kurt['r_hii_pc'])
        snr_min = float(kurt['snr_min']) if 'snr_min' in kurt else 5.0
    # VSF run with the same S/N cut (no suffix for the S/N > 5 default).
    suffix = '' if snr_min == 5.0 else f'_snr{snr_min:g}'
    with np.load(ROOT / f'NPZ/VSF_{region}_SII_total_fullfov{suffix}.npz') as vsf:
        assert np.isclose(rmin, float(vsf['min_separation_pc'])), region
        assert np.isclose(first_center, np.min(10 ** vsf['log_centers'])), region
    rows.append(dict(region=region, line='SII', rmin_pc=rmin,
                     r_hii_pc=r_hii, rmin_over_rhii=rmin / r_hii,
                     logistic_x0=-1.5 + 0.75 * np.log10((rmin / r_hii) / 0.01),
                     first_retained_bin_center_pc=first_center,
                     dK_dlog10r=float(fit['dK_dlog10l']),
                     jackknife_error=float(fit['dK_dlog10l_err'])))
rows.sort(key=lambda row: row['rmin_over_rhii'])
fig, ax = plt.subplots(figsize=(6.5, 4.8))
# w = Delta r_shell / R_HII; points are placed at w = r_min / R_HII.
# x0 = -1.5 + 0.75 log10[(r_min/R_HII)/0.01].
x_curve = np.geomspace(min(row['rmin_over_rhii'] for row in rows) / 1.3,
                       max(row['rmin_over_rhii'] for row in rows) * 1.3, 600)
for row in rows:
    y_curve = -2.0 / (1.0 + np.exp((np.log10(x_curve) - row['logistic_x0']) / 0.6))
    ax.plot(x_curve, y_curve, color='grey', alpha=0.3, lw=1.8, zorder=1)
for row in rows:
    ax.errorbar(row['rmin_over_rhii'], row['dK_dlog10r'], yerr=row['jackknife_error'],
                fmt='o', color='#d81159', mec='black', mew=0.5, ms=7, capsize=4)
    ax.annotate(row['region'], (row['rmin_over_rhii'], row['dK_dlog10r']),
                xytext=(9, 6), textcoords='offset points', fontsize=11)
ax.axhline(0, color='0.4', ls='--', lw=1)
ax.set_xscale('log')
ax.set_xlabel(r'$w = \Delta r_{\mathrm{shell}}/R_{\mathrm{HII}}$ (points: $r_{\min}/R_{\mathrm{HII}}$)')
ax.set_ylabel(r'$\mathrm{d}K/\mathrm{d}\log_{10}r$')
ax.set_title('[SII] kurtosis slopes versus minimum separation')
ax.grid(alpha=0.2)
ax.margins(x=0.18, y=0.2)
fig.tight_layout()
for suffix in ('png', 'pdf'):
    fig.savefig(OUT / f'sii_kurtosis_slope_vs_rmin.{suffix}', dpi=200, bbox_inches='tight')
plt.close(fig)
with (OUT / 'sii_kurtosis_slope_vs_rmin.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
