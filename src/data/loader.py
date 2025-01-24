import glob
import os
import numpy as np
from torch.utils.data import DataLoader
from src.data.candidate_dataset import CandidateDataset
from src.logger.logger import setup_logger




def load_queries(data_dir, dataset_name_or_path, stage="train"):
    def process_concepts(concepts):
        for concept in concepts:
            concept = concept.split("||")
            query_type = concept[2].strip()
            mention = concept[3].strip()
            cui = concept[4].strip()

            if stage == "train":
                for m in mention.replace("+", "|").split("|"):
                    data.append((m, cui, query_type))
            else:
                data.append((mention, cui, query_type))

    data = []
    if dataset_name_or_path in ['bc5cdr-2', 'cometa']:
        with open(data_dir, "r", encoding='utf-8') as f:
            concepts = f.readlines()
        process_concepts(concepts)
    elif dataset_name_or_path in ['cometa2', "cometa_knn_data", "cometa_knn_clinical", "ispo"]:
        with open(data_dir, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                mention, cui = line.strip().split("||")
                data.append((mention, cui, "cometa2"))
    elif dataset_name_or_path in ['AAP', "AAP_Fold0",]:
        with open(data_dir, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                _, mention, cui = line.strip().split("||")
                data.append((mention, cui, "AAP"))
    elif dataset_name_or_path in ["ncbi-disease", "bc5cdr-chemical", "bc5cdr-disease"]:
        concept_files = glob.glob(os.path.join(data_dir, "*.concept"))
        for concept_file in concept_files:
            with open(concept_file, "r", encoding='utf-8') as f:
                concepts = f.readlines()

            process_concepts(concepts)
    else:
        raise ValueError("Invalid dataset name or path")
    if stage == "train":
        data = list(dict.fromkeys(data))
    data = np.array(data)

    return data


def load_dictionary(dictionary_path, dataset_name_or_path):
    data = []
    lines = open(dictionary_path, mode='r', encoding='utf-8').readlines()
    for line in lines:
        line = line.strip()
        if line == "": continue
        cui, name = line.split("||")
        data.append((name, cui))
    data = np.array(data)
    return data


def load_data(args, shared_tools):
    # data_path = '/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/BioSyn_Tree/data/ncbi-disease'
    train_queries = load_queries(
        args.train_dir,
        args.dataset_name_or_path,
        stage="train"
    )
    train_dictionaries = load_dictionary(args.train_dictionary_path, args.dataset_name_or_path)
    if args.debug:
        train_queries = train_queries[:500]
        train_dictionaries = train_dictionaries[:3000]
    logger = setup_logger(args.log_file)
    logger.info(f"train_queries:{len(train_queries)}")
    logger.info(f"train_dictionaries:{len(train_dictionaries)}")
    train_dataset = CandidateDataset(
        args=args,
        queries=train_queries,
        dicts=train_dictionaries,
        shared_tools=shared_tools,
        stage="train",
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataset, train_loader
