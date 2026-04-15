#!/usr/bin/env bash
# Full ScanNet200 data preparation pipeline (Steps 2-3 from docs/DATA.md).
# Usage: bash scripts/prepare_scannet200.sh <scannet_raw_root>
#
# Example:
#   bash scripts/prepare_scannet200.sh data/scannet200
set -euo pipefail

# Ensure `conda activate` works in non-interactive shells.
if ! command -v conda >/dev/null 2>&1; then
    for p in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /apps/external/conda/2025.09; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi
conda activate core_test

SCANNET_ROOT="${1:?Usage: $0 <scannet_raw_root>}"

echo "=== Step 1: Extract point clouds + superpoints ==="
cd "${SCANNET_ROOT}"
python batch_load_scannet_data.py --scannet200
cd -

echo "=== Step 2: Create MMDet3D info files ==="
mkdir -p data/scannet200
python tools/create_data.py scannet200 \
    --root-path "${SCANNET_ROOT}" \
    --out-dir data/scannet200 \
    --extra-tag scannet200

echo "=== Done ==="
echo "Output in data/scannet200/"
