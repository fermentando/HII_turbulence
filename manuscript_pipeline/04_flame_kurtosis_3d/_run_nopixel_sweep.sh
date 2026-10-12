#!/bin/bash
set -euo pipefail
cd "/Users/fernando/PhDResearch/LVM stuff"

# Wait for the already-running N=500,000 generation job to finish.
while pgrep -f "generate_random_point_expanding_shell_grid.py --no-pixelize --n-points 500000" > /dev/null; do
  sleep 5
done

BASE="$(pwd)/Figures/manuscript/04_kurtosis_3d_all_regions"

for N in 10000 20000 50000 100000; do
  echo "=== generating N=$N ==="
  python3 manuscript_pipeline/04_flame_kurtosis_3d/generate_random_point_expanding_shell_grid.py \
    --no-pixelize --n-points "$N" --v-bulk 10.0 \
    --output-dir "${BASE}/synthetic_projected_shell_random_points_nopixel_n${N}"
done

for N in 10000 20000 50000 100000 500000; do
  echo "=== aperture comparison N=$N ==="
  python3 manuscript_pipeline/04_flame_kurtosis_3d/plot_shell_dk_dlogr_aperture_comparison.py \
    --shell-dir "${BASE}/synthetic_projected_shell_random_points_nopixel_n${N}"
done

echo "=== blue band plot ==="
python3 manuscript_pipeline/04_flame_kurtosis_3d/plot_random_point_count_nopixel_blue_band.py

echo "SWEEP_DONE"
