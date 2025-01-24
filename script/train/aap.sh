CUDA_VISIBLE_DEVICES=7 \
TOKENIZERS_PARALLELISM=false \
python main_aap.py \
    --root_path "/data2/newhome/huarui/pythonProject/BioSyn_Tree" \
    --model_name_or_path "/data2/newhome/huarui/pythonProject/BioSyn_Tree/pretrain_model/SapBERT-from-PubMedBERT-fulltext" \
    --dataset_name_or_path "AAP" \
    --train_dir "train.txt" \
    --dev_dir "test.txt" \
    --train_dictionary_path "test_dictionary.txt" \
    --dev_dictionary_path "test_dictionary.txt" \
    --max_length 30 \
    --topk 20 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 6e-5 \
    --weight_decay 2e-4 \
    --tree_ratio 0.5 \
    --retrieve_step_ratio 0.5 \
    --retrieve_similarity_func dot \
    --loss_func KL