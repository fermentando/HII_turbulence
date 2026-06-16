import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Load results
results = np.load('vsf_ksfit_summary.npy', allow_pickle=True).item()
npz_files = sorted(results.keys())
n_files = len(npz_files)

n_cols = 3
n_rows = int(np.ceil(n_files / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), sharey=False)
axes = axes.flatten()

for idx, file in enumerate(npz_files):
    ax = axes[idx]
    res = results[file]
    # Load all data for context
    data = np.load(file)
    vsf = data['vsf']
    log_centers = data['log_centers']
    counts = data['counts']
    valid = np.isfinite(vsf) & np.isfinite(log_centers) & (counts > 0)
    log_r = log_centers[valid]
    log_vsf = np.log10(vsf[valid])
    # Plot all data
    ax.plot(10**log_r, 10**log_vsf, 'o', ms=4, alpha=0.5, label='VSF data')
    # Plot best fit
    x = res['log_r']
    y_fit = res['fit']
    ax.plot(10**x, 10**y_fit, '-', color='red', lw=2, label=f'Best fit (slope={res["slope"]:.2f})')
    # Highlight fit window
    ax.axvspan(10**x[0], 10**x[-1], color='orange', alpha=0.1, zorder=0)
    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r (pc)')
    ax.set_ylabel('VSF')
    title = os.path.basename(file).replace('VSF_', '').replace('.npz', '')
    ax.set_title(f'{title}\nKS={res["ks"]:.2g}')
    ax.legend(fontsize=8)

# Hide unused axes
for i in range(n_files, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('vsf_ksfit_grid.png', dpi=200)
print('Saved vsf_ksfit_grid.png')
