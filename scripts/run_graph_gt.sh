#!/usr/bin/env bash
# Build scene graphs from ground-truth annotations.
# Usage: bash scripts/run_graph_gt.sh <data_root>
#
# Example:
#   bash scripts/run_graph_gt.sh data/scannet200
set -euo pipefail

conda activate core

DATA_ROOT="${1:?Usage: $0 <data_root>}"

python -m graph.build_graph \
    --source gt \
    --dataset s200 \
    --data-root "${DATA_ROOT}"
