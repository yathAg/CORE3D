#!/usr/bin/env bash
# Build scene graphs from OneFormer3D predictions.
# Usage: bash scripts/run_graph_oneformer.sh <data_root> <results_dir>
#
# Example:
#   bash scripts/run_graph_oneformer.sh data/scannet200 results/oneformer3d/
set -euo pipefail

conda activate core

DATA_ROOT="${1:?Usage: $0 <data_root> <results_dir>}"
RESULTS_DIR="${2:?Usage: $0 <data_root> <results_dir>}"

python -m graph.build_graph \
    --source oneformer \
    --dataset s200 \
    --data-root "${DATA_ROOT}" \
    --results-dir "${RESULTS_DIR}"
