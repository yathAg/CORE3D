#!/usr/bin/env bash
# Run OneFormer3D inference for ScanNet200 + ScanNet++ and dump per-scene
# predictions under CORE3D/results/. Must be invoked from CORE3D root so
# the fork configs' relative `data/...` paths resolve into CORE3D/data/.
#
# Usage:
#   bash scripts/run_oneformer_inference.sh [scannet200|scannetpp|all]
#
# Modes:
#   scannet200  segmentator superpoints (matches training).
#               Dumps to results/oneformer3d_s200/preds/
#   scannetpp   segmentator superpoints, PointSample_(50000) kept.
#               Dumps include sample_indices for graph-builder use.
#               Dumps to results/oneformer3d_spp/preds/
#   all         Runs both (default).
#
# The `test_evaluator` mirrors `val_evaluator` in the config via Python
# reference. mmengine's --cfg-options doesn't propagate through the
# reference, so we set both sides explicitly.
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
    for p in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi
conda activate core_test
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

WHICH="${1:-all}"
# Fork lives as a sibling of CORE3D by convention. Override with FORK_ROOT.
FORK="${FORK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../oneformer3d-fork}"
TEST_PY="${FORK}/tools/test.py"
SEED=42

S200_CFG="oneformer3d_1xb4_scannet200_spconv.py"
S200_CKPT="checkpoints/oneformer3d_scannet200.pth"
S200_OUT="results/oneformer3d_s200"

run_s200() {
    local out="${S200_OUT}"
    mkdir -p "${out}/preds"
    python "${TEST_PY}" \
        "${FORK}/configs/${S200_CFG}" \
        "${S200_CKPT}" \
        --work-dir "${out}" \
        --cfg-options \
            randomness.seed=${SEED} \
            val_evaluator.dump_scene_preds=True \
            val_evaluator.dump_dir="${out}/preds" \
            test_evaluator.dump_scene_preds=True \
            test_evaluator.dump_dir="${out}/preds"
}

run_spp() {
    local out=results/oneformer3d_spp
    mkdir -p "${out}/preds"
    python "${TEST_PY}" \
        "${FORK}/configs/oneformer3d_1xb4_scannetpp_spconv_sdpa_ext.py" \
        checkpoints/oneformer3d_scannetpp.pth \
        --work-dir "${out}" \
        --cfg-options \
            randomness.seed=${SEED} \
            val_evaluator.dump_scene_preds=True \
            val_evaluator.dump_dir="${out}/preds" \
            test_evaluator.dump_scene_preds=True \
            test_evaluator.dump_dir="${out}/preds"
}

case "${WHICH}" in
    scannet200) run_s200 ;;
    scannetpp)  run_spp ;;
    all)        run_s200; run_spp ;;
    *) echo "Usage: $0 [scannet200|scannetpp|all]" >&2; exit 1 ;;
esac

echo "Done. Dumps:"
for d in results/oneformer3d_s200/preds results/oneformer3d_spp/preds; do
    if [ -d "$d" ]; then
        n=$(ls "$d"/*.pth 2>/dev/null | wc -l)
        printf "  %-45s %s\n" "$d" "$n"
    fi
done
