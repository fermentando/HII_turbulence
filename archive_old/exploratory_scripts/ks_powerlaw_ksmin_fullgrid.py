import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.stats import kstest

MIN_POINTS = 4
npz_files = sorted(glob.glob('VSF_*.npz'))
results = {}

for file in npz_files:
    data = np.load(file)
    vsf = data['vsf']
    log_centers = data['log_centers']
    counts = data['counts']
    valid = np.isfinite(vsf) & np.isfinite(log_centers) & (counts > 0)
    log_r = log_centers[valid]
    log_vsf = np.log10(vsf[valid])
    n = len(log_r)
    if n < MIN_POINTS:
        print(f"Skipping {file}: not enough valid points")
        continue
    best_ks = np.inf
    best_fit = None
    best_range = None
    best_params = None
    # Try all possible fitting windows
    for i in range(n - MIN_POINTS + 1):
        for j in range(i + MIN_POINTS, n + 1):
            x = log_r[i:j]
            y = log_vsf[i:j]
            if len(x) < MIN_POINTS:
                continue
            slope, intercept = np.polyfit(x, y, 1)
            y_fit = slope * x + intercept
            ks_stat, _ = kstest(y, lambda z: np.interp(z, y_fit, y_fit))
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_fit = (x.copy(), y.copy(), y_fit.copy())
                best_range = (i, j)
                best_params = (slope, intercept)
    if best_fit is None:
        print(f"No valid fit for {file}")
        continue
    slope, intercept = best_params
    x, y, y_fit = best_fit
    results[file] = {
        'slope': slope,
        'intercept': intercept,
        'ks': best_ks,
        'range': best_range,
        'log_r': x,
        'log_vsf': y,
        'fit': y_fit
    }
    print(f"{file}: slope={slope:.2f}, KS={best_ks:.3g}, range={best_range}")

# --- Plot grid ---
n_files = len(results)
n_cols = 3
n_rows = int(np.ceil(n_files / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), sharey=False)
axes = axes.flatten()
for idx, file in enumerate(results):
    ax = axes[idx]
    res = results[file]
    data = np.load(file)
    vsf = data['vsf']
    log_centers = data['log_centers']
    counts = data['counts']
    valid = np.isfinite(vsf) & np.isfinite(log_centers) & (counts > 0)
    log_r = log_centers[valid]
    log_vsf = np.log10(vsf[valid])
    ax.plot(10**log_r, 10**log_vsf, 'o', ms=4, alpha=0.5, label='VSF data')
    x = res['log_r']
    y_fit = res['fit']
    ax.plot(10**x, 10**y_fit, '-', color='red', lw=2, label=f'Best fit (slope={res["slope"]:.2f})')
    ax.axvspan(10**x[0], 10**x[-1], color='orange', alpha=0.1, zorder=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r (pc)')
    ax.set_ylabel(r'$\bar{S}_\mathrm{2} = \langle | v(r + \ell) - v(r) | \rangle ^{2}$ (km/s)$^2$', fontsize=10)
    title = os.path.basename(file).replace('VSF_', '').replace('.npz', '')
    # Extract region name (assume format: REGION_LINE)
    region = title.split('_')[0] if '_' in title else title
    ax.set_title(f'{title}\nRegion: {region}\nKS={res["ks"]:.2g}')
    ax.legend(fontsize=8)
for i in range(n_files, len(axes)):
    axes[i].set_visible(False)
plt.tight_layout()
plt.savefig('vsf_ksfit_grid.png', dpi=200)
print('Saved vsf_ksfit_grid.png')
