CUDA_VISIBLE_DEVICES=7 \
TOKENIZERS_PARALLELISM=false \
python main.py \
    --root_path "/data2/newhome/huarui/pythonProject/BioSyn_Tree" \
    --model_name_or_path "/data2/newhome/huarui/pythonProject/BioSyn_Tree/pretrain_model/SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "bc5cdr-disease" \
    --train_dir "processed_traindev" \
    --train_dictionary_path "train_dictionary.txt" \
    --dev_dir "processed_test" \
    --dev_dictionary_path "test_dictionary.txt" \
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