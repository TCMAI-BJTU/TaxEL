#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${OUTPUT_PATH:-checkpoints/bc5cdr_disease_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p cache

ROOT_PATH="${ROOT_PATH:-$REPO_ROOT}"
model_name_or_path="${MODEL_NAME_OR_PATH:-$ROOT_PATH/pretrain_model/SapBERT-from-PubMedBERT-fulltext}"
dataset_name_or_path="bc5cdr-disease"

TOKENIZERS_PARALLELISM=false \
"$PYTHON_BIN" main.py \
    --root_path "$ROOT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name_or_path "$model_name_or_path" \
    --dataset_name_or_path "$dataset_name_or_path" \
    --test_dir "processed_test" \
    --test_dictionary_path "test_dictionary.txt" \
    --train_dir "processed_traindev" \
    --train_dictionary_path "train_dictionary.txt" \
    --dev_dir "processed_dev" \
    --dev_dictionary_path "dev_dictionary.txt" \
    --max_length 25 \
    --topk 20 \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 1e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 1.0 \
    --retrieve_step_ratio 1 \
    --retrieve_similarity_func dot \
    --loss_func hybrid_kl_marginal_nll \
    --tax_aware current_parent_child_sibling \
    --kl_alpha 0.9 \
    --force_positive_injection true \
    --remove_self_match true \
    --fuzzy_cui_match true \
    --normalize_cui_plus true
