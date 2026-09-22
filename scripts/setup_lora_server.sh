#!/usr/bin/env bash
# Create a reproducible Linux server environment for Qwen 4B LoRA SFT.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA driver / nvidia-smi is unavailable." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} is unavailable. Install Python 3.10+ first." >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("ERROR: Python 3.10 or newer is required.")
PY

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
# cu128 supports RTX 4090 and current RTX 5090 Linux servers. Override
# TORCH_INDEX_URL only when the server's deployed CUDA/PyTorch policy requires it.
"${VENV_DIR}/bin/python" -m pip install --upgrade torch torchvision --index-url "${TORCH_INDEX_URL}"
"${VENV_DIR}/bin/python" -m pip install --editable "${ROOT_DIR}[train]"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import peft
assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY

echo
echo "Environment ready: ${VENV_DIR}"
echo "Next: bash scripts/run_qwen4b_lora_sft.sh"
