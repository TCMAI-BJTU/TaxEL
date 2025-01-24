import os
import pickle
import time
from collections import defaultdict
from functools import lru_cache
from typing import List
import pandas as pd

from src.data.tree_similarity import TreeCodeSimilarity


class MEDICTree:
    def __init__(self, dataset_name_or_path, root_path, tree_ratio):
        self.dataset_name_or_path = dataset_name_or_path
        self.tree_ratio = tree_ratio

        if dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease"]:
            df = pd.read_excel(f"{root_path}/data/tree_code/CTD_Disease_Tree.xlsx")
        elif dataset_name_or_path in ["bc5cdr-chemical"]:
            df = pd.read_excel(f"{root_path}/data/tree_code/CTD_Chemical_Tree.xlsx")
        elif dataset_name_or_path in ["bc5cdr", "bc5cdr-2"]:
            df1 = pd.read_excel(f"{root_path}/data/tree_code/CTD_Disease_Tree.xlsx")
            df2 = pd.read_excel(f"{root_path}/data/tree_code/CTD_Chemical_Tree.xlsx")
            df = pd.concat([df1, df2])
        elif dataset_name_or_path in [ "AAP",  "cometa_clinical", "AAP_Fold0"]:
            # df = pd.read_csv(f"{root_path}/data/tree_code/snomed_tree_code_201907.csv")
            df = pd.read_csv(f"{root_path}/data/tree_code/snomed_tree_code_au_2024.csv")

        else:
            raise ValueError("Ontology must be CTD_Disease or CTD_Chemical")

        id2tree_codes = defaultdict(list)
        tree_code2ids = defaultdict(list)
        tree_code_sets = set()
        for tup in zip(df["ID"], df["TreeNumbers"]):
            disease_id, tree_numbers = tup
            disease_id = str(disease_id)
            for tree in tree_numbers.split("|"):
                tree = tree.replace("/", ".")
                id2tree_codes[disease_id].append(tree)
                tree_code2ids[tree].append(disease_id)
                tree_code_sets.add(tree)
        self.id2tree_codes = dict(id2tree_codes)
        self.tree_code2ids = dict(tree_code2ids)

        self.tree_code_sets = tree_code_sets

        self.tree_similarities = TreeCodeSimilarity(tree_code_sets, dataset_name_or_path)

        self.error_cuis = set()

    def get_nearest_similarity(self, tree1, tree2):
        """
        :param tree1: 第一个节点树编码的列表
        :param tree2: 第二个树编码的列表
        :return: 选取距离最近的两个节点树编码计算相似度
        """
        sim = 0
        for t1 in tree1:
            for t2 in tree2:
                sim = max(sim, self.tree_similarities.compute(t1, t2))
        return sim

    def complete_cui(self, cui):
        """
        :param tree_code: 树编码
        :return: 补全树编码
        """
        cui = cui.replace("+", "|").split("|")[0]
        if not cui[0].isdigit():
            cui = "MESH:" + cui
        else:
            cui = "OMIM:" + cui
        return cui

    @lru_cache(maxsize=100000)
    def compute_similarity(self, id1, id2):
        if self.dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical"]:
            id1 = self.complete_cui(id1)
            id2 = self.complete_cui(id2)

        try:
            tree1 = self.id2tree_codes[id1]
        except:
            self.error_cuis.add(id1)
            return 0.0
        try:
            tree2 = self.id2tree_codes[id2]
        except:
            self.error_cuis.add(id2)
            return 0.0

        sim = self.get_nearest_similarity(tree1, tree2)
        return sim * self.tree_ratio

    def get_parent_ids(self, cui) -> List[str]:
        """
        :param cui: 实体的CUI
        :return: 实体的父节点的CUI
        """
        if self.dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical"]:
            cui = self.complete_cui(cui)
        if cui not in self.id2tree_codes:
            return []
        tree_codes = self.id2tree_codes[cui]
        parent_tree_codes = []
        for tree_code in tree_codes:
            parent_tree_code = self.tree_similarities.get_parent_tree_code(tree_code)
            parent_tree_codes.append(parent_tree_code)

        parent_ids = []
        for parent_tree_code in parent_tree_codes:
            parent_ids.extend(self.tree_code2ids[parent_tree_code])

        parent_ids = list(set(parent_ids))
        if self.dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical"]:
            parent_ids = [i.split(":")[-1] for i in parent_ids]
        return parent_ids

    @lru_cache(maxsize=100000)
    def get_child_ids(self, cui) -> List[str]:
        """
        :param cui: 实体的CUI
        :return: 实体的子节点的CUI
        """
        if self.dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical"]:
            cui = self.complete_cui(cui)
        if cui not in self.id2tree_codes:
            return []
        tree_codes = self.id2tree_codes[cui]
        child_tree_codes = []
        for tree_code in tree_codes:
            child_tree_codes.extend(self.tree_similarities.get_child_tree_codes(tree_code))

        child_ids = []
        for child_tree_code in child_tree_codes:
            child_ids.extend(self.tree_code2ids[child_tree_code])

        child_ids = list(set(child_ids))
        if self.dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical"]:
            child_ids = [i.split(":")[-1] for i in child_ids]
        return child_ids

    def save_error(self):
        with open("error_cuis.txt", "w") as f:
            for cui in self.error_cuis:
                f.write(cui + "\n")
