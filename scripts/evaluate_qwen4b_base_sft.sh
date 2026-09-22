#!/usr/bin/env bash
# Evaluate the unmodified Qwen3-4B base model on the held-out GIS OODA split.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
LOCAL_MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-4B}"
TEST_DATA="${TEST_DATA:-${ROOT_DIR}/data/sft/gis-concept-v1/llm_augmented/test/anonymous_ooda_en.jsonl}"
EVAL_NAME="${EVAL_NAME:-test-greedy}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/gis-concept-v1/qwen3-4b-base-ooda/evaluation/${EVAL_NAME}}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-7168}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: server environment is missing. Run: bash scripts/setup_lora_server.sh" >&2
  exit 1
fi
if [[ ! -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
  echo "ERROR: local Qwen3-4B model is missing: ${LOCAL_MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TEST_DATA}" ]]; then
  echo "ERROR: held-out test data is missing: ${TEST_DATA}" >&2
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/evaluate_sft_ooda.py \
  --base-model "${LOCAL_MODEL_DIR}" \
  --data "${TEST_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-input-length "${MAX_INPUT_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --qlora-4bit \
  --bf16 \
  --overwrite
