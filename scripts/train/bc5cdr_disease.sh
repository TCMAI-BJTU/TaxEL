CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python main.py \
    --root_path "TaxEL_PATH" \
    --model_name_or_path "TaxEL-BC5CDR-Disease" \
    --dataset_name_or_path "bc5cdr-disease" \
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
    --tree_ratio 0 \
    --retrieve_step_ratio 1 \
    --retrieve_similarity_func dot \
    --loss_func KL \
    --tax_aware none