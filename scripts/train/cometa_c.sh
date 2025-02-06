CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python main.py \
    --root_path "TaxEL_PATH" \
    --model_name_or_path "TaxEL-COMETA-c" \
    --dataset_name_or_path "cometa_clinical" \
    --train_dir "train.txt" \
    --dev_dir "dev.txt" \
    --train_dictionary_path "test_dictionary_clinical.txt" \
    --dev_dictionary_path "test_dictionary_clinical.txt" \
    --max_length 25 \
    --topk 20 \
    --batch_size 8 \
    --epochs 10 \
    --learning_rate 2e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 0.5 \
    --retrieve_step_ratio 0.5 \
    --retrieve_similarity_func dot \
    --loss_func KL \
    --tax_aware current