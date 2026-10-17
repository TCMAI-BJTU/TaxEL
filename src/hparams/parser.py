import argparse
import time
from transformers import set_seed
from src.logger.logger import setup_logger
import os


def update_dictionary_paths(args):
    args.train_dictionary_path = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.train_dictionary_path}"
    args.dev_dictionary_path = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.dev_dictionary_path}"
    args.test_dictionary_path = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.test_dictionary_path}"
    args.train_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.train_dir}"
    args.dev_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.dev_dir}"
    args.test_dir = f"{args.root_path}/data/{args.dataset_name_or_path}/{args.test_dir}"
    return args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default="./TaxEL")

    parser.add_argument(
        "--model_name_or_path",
        default="SapBERT-from-PubMedBERT-fulltext",
        help="Directory for pretrained model",
    )

    parser.add_argument(
        "--dataset_name_or_path",
        default="ncbi-disease",
        choices=[
            "ncbi-disease",
            "bc5cdr-chemical",
            "bc5cdr-disease",
            "AAP",
            "cometa_clinical",
            "cometa_knn_data",
            "AAP_Fold0",
            "sympel",
            "cometa",
            "ispo_en",
            "mesh_disease",
            "mesh_chemical",
            "icd_en",
            "icd_zh",
        ],
    )

    parser.add_argument("--output_path", type=str, default="./checkpoints")

    parser.add_argument("--train_dir", type=str, default="processed_traindev")
    parser.add_argument("--train_dictionary_path", type=str, default="train_dictionary.txt")

    parser.add_argument("--dev_dir", type=str, default="processed_test")
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
    parser.add_argument("--use_cuda", default=True, type=lambda x: str(x).lower() == "true")

    parser.add_argument("--retrieve_step_ratio", default=1, type=float)
    parser.add_argument("--kl_alpha", default=0.9, type=float)

    parser.add_argument("--debug", default=False, type=lambda x: str(x).lower() == "true")

    parser.add_argument("--last_layer", default="mean", type=str, choices=["mean", "cls"])
    parser.add_argument("--retrieve_similarity_func", default="dot", type=str, choices=["dot", "cosine"])
    parser.add_argument("--train_similarity_func", default="cosine", type=str, choices=["dot", "cosine"])
    parser.add_argument("--loss_func", default="KL", type=str, choices=["mse", "marginal_nll", "KL", "hybrid_kl_marginal_nll"])

    parser.add_argument("--use_tree_similarity", default=True, type=lambda x: str(x).lower() == "true")
    parser.add_argument("--tree_ratio", default=0.5, type=float)
    parser.add_argument("--retrieve_tree_ratio", default=0.5, type=float)
    parser.add_argument(
        "--tax_aware", default="current", type=str, choices=["current", "current_parent_child", "current_parent_child_sibling", "none"]
    )
    parser.add_argument("--tree_code_path", default=None, type=str,
                        help="Custom path to tree code file (for perturbation experiments). Overrides default tree file.")
    parser.add_argument("--force_positive_injection", default=False, type=lambda x: str(x).lower() == "true",
                        help="If True, ensure every training query has at least one positive in its top-k candidates by injecting the nearest positive.")
    parser.add_argument("--remove_self_match", default=False, type=lambda x: str(x).lower() == "true",
                        help="If True, replace candidates whose name equals the query name with next-best non-self-matching candidates.")
    parser.add_argument("--normalize_cui_plus", default=False, type=lambda x: str(x).lower() == "true",
                        help="If True, normalize '+' to '|' when splitting composite CUIs in label computation and evaluation.")
    parser.add_argument("--fuzzy_cui_match", default=False, type=lambda x: str(x).lower() == "true",
                        help="If True, match individual CUI components (split by | and +) when collecting tree candidates, instead of exact full-string match.")
    args = parser.parse_args()

    args = update_dictionary_paths(args)

    set_seed(args.seed)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.log_file = f"logs/{args.dataset_name_or_path}/{current_time}.log"
    if not os.path.exists(f"logs/{args.dataset_name_or_path}"):
        os.makedirs(f"logs/{args.dataset_name_or_path}", exist_ok=True)

    logger = setup_logger(log_file=args.log_file)
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    return args
