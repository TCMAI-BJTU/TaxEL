#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
ROOT_PATH="${ROOT_PATH:-$REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${1:-}}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH or pass the model directory as the first argument}"
export TOKENIZERS_PARALLELISM=false
mkdir -p cache

export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$REPO_ROOT/records/eval/cometa_$(date +%Y%m%d_%H%M%S)_$$}"
"$PYTHON_BIN" eval.py \
    --root_path "$ROOT_PATH" \
    --model_name_or_path "$CHECKPOINT_PATH" \
    --dataset_name_or_path cometa \
    --test_dir test.txt \
    --test_dictionary_path test_dictionary.txt \
    --max_length 25 --topk 20 \
    --last_layer mean --retrieve_similarity_func dot \
    --normalize_cui_plus true
