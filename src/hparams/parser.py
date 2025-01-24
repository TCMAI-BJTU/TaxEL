import argparse
import time
from transformers import set_seed
from src.logger.logger import setup_logger


def update_dictionary_paths(args):
    args.train_dictionary_path = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.train_dictionary_path}"
    args.dev_dictionary_path = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.dev_dictionary_path}"
    args.test_dictionary_path = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.test_dictionary_path}"
    if args.dataset_name_or_path in ["bc5cdr-2", "cometa", "cometa2", "AAP", "cometa_knn_data", "cometa_knn_clinical", "AAP_Fold0", "ispo"]:
        args.train_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/train.txt"
        args.dev_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/test.txt"
        args.test_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/test.txt"
    else:
        args.train_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.train_dir}"
        args.dev_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.dev_dir}"
        args.test_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.test_dir}"
    return args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default="/data2/newhome/huarui/pythonProject/BioSyn_Tree")

    parser.add_argument(
        "--model_name_or_path",
        default="/data2/newhome/huarui/pythonProject/BioSyn_Tree/pretrain_model/SapBERT-from-PubMedBERT-fulltext",
        # default="/var/lib/docker/huarui/pythonProject/BioSyn_Tree/pretrain_model/SapBERT-from-PubMedBERT-fulltext-mean-token",
        help="Directory for pretrained model",
    )

    parser.add_argument(
        "--dataset_name_or_path",
        default="ncbi-disease",
        choices=[
            "ncbi-disease",
            "bc5cdr-chemical",
            "bc5cdr-disease",
            "bc5cdr",
            "bc5cdr-2",
            "cometa",
            "AAP",
            "cometa_knn_data",
            "cometa_knn_clinical",
            "AAP_Fold0",
            "ispo",
        ],
    )

    parser.add_argument("--output_path", type=str, default="./checkpoints")

    parser.add_argument("--train_dir", type=str, default="processed_traindev")
    parser.add_argument("--train_dictionary_path", type=str, default="train_dictionary.txt")

    parser.add_argument("--dev_dir", type=str, default="processed_test")  # 记得改上面的dev.txt
    parser.add_argument("--dev_dictionary_path", type=str, default="test_dictionary.txt")

    parser.add_argument("--test_dir", type=str, default="processed_test")
    parser.add_argument("--test_dictionary_path", type=str, default="test_dictionary.txt")

    parser.add_argument("--max_length", default=30, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", help="epoch to train", default=10, type=int)
    parser.add_argument("--learning_rate", help="learning rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", help="weight decay", default=2e-4, type=float)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--use_cuda", default=True)

    parser.add_argument("--retrieve_step_ratio", default=0.5, type=float)

    parser.add_argument("--debug", default=False, type=bool)

    parser.add_argument("--last_layer", default="mean", type=str, choices=["mean", "cls"])
    parser.add_argument("--retrieve_similarity_func", default="cosine", type=str, choices=["dot", "cosine"])
    parser.add_argument("--train_similarity_func", default="cosine", type=str, choices=["dot", "cosine"])
    parser.add_argument("--loss_func", default="KL", type=str, choices=["mse", "marginal_nll", "KL"])

    parser.add_argument("--use_tree_similarity", default=True, type=bool)
    parser.add_argument("--tree_ratio", default=0.5, type=float)
    parser.add_argument("--retrieve_tree_ratio", default=0.5, type=float)
    # parser.add_argument("--only_current", action="store_true")
    parser.add_argument("--tax_aware", default="current", type=str, choices=["current", "current_parent_child", "none"])

    parser.add_argument("--use_graph_propagation", default=False, type=bool, help="是否使用图传播")
    parser.add_argument("--graph_alpha", default=0.9, type=float, help="原始得分的权重")
    parser.add_argument("--graph_threshold", default=0.9, type=float, help="相似度阈值")
    parser.add_argument("--use_schedule", default=True, type=bool)
    args = parser.parse_args()

    args = update_dictionary_paths(args)

    set_seed(args.seed)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.log_file = f"logs/{current_time}.log"

    logger = setup_logger(log_file=args.log_file)
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    # TopK	max_len	batch	epoch	last_layer	lr	loss	retrieve_sim	train_sim	weight_decay	tree_ratio
    args_str = f"{args.model_name_or_path}\t{args.topk}\t{args.max_length}\t{args.batch_size}\t{args.epochs}\t{args.last_layer}\t{args.learning_rate}\t{args.loss_func}\t{args.retrieve_similarity_func}\t{args.train_similarity_func}\t{args.weight_decay}\t{args.tree_ratio}"
    logger.info(f"args_str: {args_str}")

    return args
