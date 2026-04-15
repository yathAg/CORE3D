#!/usr/bin/env bash
# Run LLM inference on scene graphs.
# Usage: bash scripts/run_llm_inference.sh <config> [model_path] [output_path]
#
# Example:
#   bash scripts/run_llm_inference.sh configs/llm/scanrefer.json
#   bash scripts/run_llm_inference.sh configs/llm/scanrefer.json meta-llama/Llama-3-8B results/
set -euo pipefail

conda activate core_test

CONFIG="${1:?Usage: $0 <config> [model_path] [output_path]}"
MODEL_PATH="${2:-}"
OUTPUT_PATH="${3:-}"

CMD=(python -m llm.run --config "${CONFIG}")
[[ -n "${MODEL_PATH}" ]] && CMD+=(--model "${MODEL_PATH}")
[[ -n "${OUTPUT_PATH}" ]] && CMD+=(--output_path "${OUTPUT_PATH}")

"${CMD[@]}"
