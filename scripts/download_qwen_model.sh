#!/usr/bin/env bash
# Download the base Qwen model once, then train only from the local copy.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${VENV_DIR}/bin/python"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3-4B}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: server environment is missing. Run: bash scripts/setup_lora_server.sh" >&2
  exit 1
fi
if [[ -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model already available locally: ${MODEL_DIR}"
  exit 0
fi

export MODEL_ID MODEL_DIR
"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repository = os.environ["MODEL_ID"]
destination = os.environ["MODEL_DIR"]
snapshot_download(repo_id=repository, local_dir=destination)
print(f"Downloaded {repository} to {destination}")
PY

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "ERROR: model download completed without config.json: ${MODEL_DIR}" >&2
  exit 1
fi
