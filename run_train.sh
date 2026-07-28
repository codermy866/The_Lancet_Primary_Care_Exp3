#!/bin/bash
set -e

cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo ">>> OCT_traige - OCT-only triage"
echo "    data_root: ${OCT_TRAIGE_DATA_ROOT:-（使用 config 默认）}"

PYTHON_BIN=${PYTHON_BIN:-/data2/xh/xh/bin/python3.10}
"${PYTHON_BIN}" training/train_oct_traige.py "$@"
