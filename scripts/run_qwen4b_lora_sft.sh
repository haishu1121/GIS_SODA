#!/usr/bin/env bash
# Run the primary OODA-only Qwen 4B LoRA SFT experiment on a Linux GPU server.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
LOCAL_MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-4B}"
if [[ -n "${MODEL_ID:-}" ]]; then
  MODEL_ID="${MODEL_ID}"
elif [[ -f "${LOCAL_MODEL_DIR}/config.json" ]]; then
  MODEL_ID="${LOCAL_MODEL_DIR}"
else
  MODEL_ID="Qwen/Qwen3-4B"
fi
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/sft/gis-concept-v1/llm_augmented}"
RUN_NAME="${RUN_NAME:-qwen3-4b-lora-ooda-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/gis-concept-v1/${RUN_NAME}}"
MAX_LENGTH="${MAX_LENGTH:-7168}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: server environment is missing. Run: bash scripts/setup_lora_server.sh" >&2
  exit 1
fi
if [[ -f "${MODEL_ID}/config.json" ]]; then
  echo "Using local base model: ${MODEL_ID}"
else
  echo "Using Hugging Face model ID: ${MODEL_ID}"
  echo "Tip: run scripts/download_qwen_model.sh first to make the training run fully local."
fi
for required_file in \
  "${DATA_ROOT}/train/anonymous_ooda_en.jsonl" \
  "${DATA_ROOT}/validation/anonymous_ooda_en.jsonl"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: missing required dataset: ${required_file}" >&2
    echo "Download it first: ${PYTHON_BIN} scripts/download_hf_ooda_dataset.py" >&2
    exit 1
  fi
done

cd "${ROOT_DIR}"
"${PYTHON_BIN}" scripts/train_sft.py \
  --schema messages \
  --trace-style ooda \
  --model "${MODEL_ID}" \
  --data "${DATA_ROOT}/train/anonymous_ooda_en.jsonl" \
  --validation-data "${DATA_ROOT}/validation/anonymous_ooda_en.jsonl" \
  --output "${OUTPUT_DIR}" \
  --epochs 3 \
  --learning-rate 2e-5 \
  --max-length "${MAX_LENGTH}" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --gradient-checkpointing \
  --bf16 \
  --tf32 \
  --lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --eval-steps 100 \
  --save-steps 100 \
  --save-total-limit 2 \
  --seed 42
