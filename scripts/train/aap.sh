#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${OUTPUT_PATH:-checkpoints/aap_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p cache

ROOT_PATH="${ROOT_PATH:-$REPO_ROOT}"
model_name_or_path="${MODEL_NAME_OR_PATH:-$ROOT_PATH/pretrain_model/SapBERT-from-PubMedBERT-fulltext}"
dataset_name_or_path="AAP"

TOKENIZERS_PARALLELISM=false \
"$PYTHON_BIN" main_aap.py \
    --root_path "$ROOT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name_or_path "$model_name_or_path" \
    --dataset_name_or_path "$dataset_name_or_path" \
    --test_dir "test.txt" \
    --test_dictionary_path "test_dictionary.txt" \
    --train_dir "train.txt" \
    --dev_dir "dev.txt" \
    --train_dictionary_path "test_dictionary.txt" \
    --dev_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 20 \
    --batch_size 8 \
    --epochs 15 \
    --learning_rate 5e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 0.5 \
    --retrieve_step_ratio 0.5 \
    --retrieve_similarity_func dot \
    --train_similarity_func dot \
    --loss_func hybrid_kl_marginal_nll \
    --tax_aware current \
    --kl_alpha 0.9 \
    --force_positive_injection true \
    --remove_self_match false \
    --fuzzy_cui_match false \
    --normalize_cui_plus false
