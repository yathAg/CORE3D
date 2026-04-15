#!/usr/bin/env bash
# Generate EZ-SP superpoints for ScanNet200.
# Usage: bash scripts/run_ezsp.sh <raw_root> <ckpt_path> [<dataset_out>]
#
# Example:
#   bash scripts/run_ezsp.sh data/scannet200 checkpoints/ezsp.pth data/scannet200/super_points_ezsp
#
# EZ-SP output is written next to the segmentator superpoints in a
# sibling directory (default: data/scannet200/super_points_ezsp). The
# segmentator output at data/scannet200/super_points/ is NOT overwritten;
# the OneFormer3D fork chooses which dir to read at inference time.
set -euo pipefail

# Ensure `conda activate` works in non-interactive shells.
if ! command -v conda >/dev/null 2>&1; then
    for p in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /apps/external/conda/2025.09; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi
conda activate core_test
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RAW_ROOT="${1:?Usage: $0 <raw_root> <ckpt_path> [<dataset_out>]}"
CKPT_PATH="${2:?Usage: $0 <raw_root> <ckpt_path> [<dataset_out>]}"
DATASET_OUT="${3:-data/scannet200/super_points_ezsp}"

# Generate into ezsp/output/super_points_ezsp/ (staging).
# Allow benign exit codes from interpreter shutdown (e.g. 120 from
# sys.unraisablehook on CUDA cleanup) — check "Done." in output instead.
set +e
python -m ezsp.generate_superpoints \
    --raw_root "${RAW_ROOT}" \
    --ckpt_path "${CKPT_PATH}" 2>&1 | tee /tmp/run_ezsp_last.log
PY_EXIT=${PIPESTATUS[0]}
set -e
if ! tail -5 /tmp/run_ezsp_last.log | grep -q "^Done\."; then
    echo "ERROR: EZ-SP did not reach Done. (python exit=$PY_EXIT)" >&2
    exit "$PY_EXIT"
fi

# Publish to the dataset directory alongside segmentator super_points/.
mkdir -p "${DATASET_OUT}"
cp ezsp/output/super_points_ezsp/*.bin "${DATASET_OUT}/"
echo "EZ-SP superpoints copied to ${DATASET_OUT}/ ($(ls "${DATASET_OUT}" | wc -l) files)"
