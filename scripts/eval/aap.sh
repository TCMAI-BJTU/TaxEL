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

# CHECKPOINT_PATH is the directory containing fold0/ ... fold9/.
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best}"
case "$CHECKPOINT_KIND" in best|last) ;; *) echo "CHECKPOINT_KIND must be best or last" >&2; exit 2;; esac
RESULTS_ROOT="${EVAL_OUTPUT_DIR:-$REPO_ROOT/records/eval/aap_$(date +%Y%m%d_%H%M%S)_$$}"
for fold in {0..9}; do
    test -f "$CHECKPOINT_PATH/fold$fold/$CHECKPOINT_KIND/modules.json" || {
        echo "Missing checkpoint: $CHECKPOINT_PATH/fold$fold/$CHECKPOINT_KIND" >&2
        exit 1
    }
done
for fold in {0..9}; do
    EVAL_OUTPUT_DIR="$RESULTS_ROOT/fold$fold" "$PYTHON_BIN" eval.py \
        --root_path "$ROOT_PATH" \
        --model_name_or_path "$CHECKPOINT_PATH/fold$fold/$CHECKPOINT_KIND" \
        --dataset_name_or_path AAP \
        --test_dir "fold$fold/test.txt" \
        --test_dictionary_path test_dictionary.txt \
        --max_length 30 --topk 20 \
        --last_layer mean --retrieve_similarity_func dot \
        --normalize_cui_plus false
done
"$PYTHON_BIN" - "$RESULTS_ROOT" <<'PYCODE'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
folds = [json.loads((root / f"fold{i}" / "metrics.json").read_text()) for i in range(10)]
averages = {k: sum(f[k] for f in folds) / len(folds) for k in folds[0]}
(root / "summary.json").write_text(json.dumps({"folds": folds, "macro_average": averages}, indent=2) + "\n")
print("AAP test macro averages:", averages)
PYCODE
