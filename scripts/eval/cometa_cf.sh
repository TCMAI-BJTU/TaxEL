TOKENIZERS_PARALLELISM=false \
python eval.py \
    --model_name_or_path "TaxEL-COMETA-c" \
    --dataset_name_or_path "cometa-c" \
    --eval_dir "test.txt" \
    --eval_dictionary_path "test_dictionary.txt" \
    --max_length 25 \
    --topk 20 \
    --use_embed_parallel
