#!/usr/bin/env bash
# Evaluate a completed Qwen3-4B GIS Concept QLoRA adapter on held-out OODA.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
LOCAL_MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-4B}"
RUN_NAME="${RUN_NAME:-qwen3-4b-lora-ooda-v1}"
ADAPTER_DIR="${ADAPTER_DIR:-${ROOT_DIR}/runs/gis-concept-v1/${RUN_NAME}}"
TEST_DATA="${TEST_DATA:-${ROOT_DIR}/data/sft/gis-concept-v1/llm_augmented/test/anonymous_ooda_en.jsonl}"
EVAL_NAME="${EVAL_NAME:-test-greedy}"
OUTPUT_DIR="${OUTPUT_DIR:-${ADAPTER_DIR}/evaluation/${EVAL_NAME}}"
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
if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
  echo "ERROR: completed LoRA adapter is missing: ${ADAPTER_DIR}" >&2
  exit 1
fi
if [[ ! -f "${TEST_DATA}" ]]; then
  echo "ERROR: held-out test data is missing: ${TEST_DATA}" >&2
  echo "The Hugging Face training-data downloader intentionally excludes test." >&2
  echo "Transfer the privately retained test JSONL to this exact path before evaluation." >&2
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/evaluate_sft_ooda.py \
  --base-model "${LOCAL_MODEL_DIR}" \
  --adapter "${ADAPTER_DIR}" \
  --data "${TEST_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-input-length "${MAX_INPUT_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --qlora-4bit \
  --bf16 \
  --overwrite
