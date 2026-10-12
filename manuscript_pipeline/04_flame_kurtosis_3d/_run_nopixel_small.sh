#!/bin/bash
set -euo pipefail
cd "/Users/fernando/PhDResearch/LVM stuff"
BASE="$(pwd)/Figures/manuscript/04_kurtosis_3d_all_regions"

for N in 10000 20000 50000 100000; do
  echo "=== generating N=$N ==="
  python3 manuscript_pipeline/04_flame_kurtosis_3d/generate_random_point_expanding_shell_grid.py \
    --no-pixelize --n-points "$N" --v-bulk 10.0 \
    --output-dir "${BASE}/synthetic_projected_shell_random_points_nopixel_n${N}"
  echo "=== aperture comparison N=$N ==="
  python3 manuscript_pipeline/04_flame_kurtosis_3d/plot_shell_dk_dlogr_aperture_comparison.py \
    --shell-dir "${BASE}/synthetic_projected_shell_random_points_nopixel_n${N}"
done

echo "SMALL_SWEEP_DONE"
