CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python main.py \
    --root_path "TaxEL_PA" \
    --model_name_or_path "TaxEL-AAP" \
    --dataset_name_or_path "AAP_Fold0" \
    --train_dir "train.txt" \
    --dev_dir "dev.txt" \
    --train_dictionary_path "test_dictionary.txt" \
    --dev_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 20 \
    --batch_size 8 \
    --epochs 10 \
    --learning_rate 5e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 0.5 \
    --retrieve_step_ratio 0.5 \
    --retrieve_similarity_func dot \
    --loss_func KL