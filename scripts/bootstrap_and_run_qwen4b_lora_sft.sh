#!/usr/bin/env bash
# One-command server bootstrap followed by the primary Qwen 4B OODA LoRA SFT.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  bash scripts/setup_lora_server.sh
fi
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  bash scripts/download_qwen_model.sh
fi
bash scripts/run_qwen4b_lora_sft.sh
