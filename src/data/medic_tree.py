from collections import defaultdict
from functools import lru_cache
from typing import List

import pandas as pd

from src.data.tree_similarity import TreeCodeSimilarity


class MEDICTree:
    def __init__(self, dataset_name_or_path, root_path, tree_ratio, tree_code_path=None):
        self.dataset_name_or_path = dataset_name_or_path
        self.tree_ratio = tree_ratio

        if tree_code_path is not None:
            if tree_code_path.endswith(".csv"):
                df = pd.read_csv(tree_code_path)
            else:
                df = pd.read_excel(tree_code_path)
            # 用文件名哈希避免缓存冲突
            import hashlib
            cache_suffix = hashlib.md5(tree_code_path.encode()).hexdigest()[:8]
            self.dataset_name_or_path = f"{dataset_name_or_path}_{cache_suffix}"
        elif dataset_name_or_path in ["bc5cdr-disease", "ncbi-disease", "mesh_disease"]:
            df = pd.read_excel(f"{root_path}/data/tree_code/CTD_Disease_Tree.xlsx")
        elif dataset_name_or_path in ["bc5cdr-chemical", "mesh_chemical"]:
            df = pd.read_excel(f"{root_path}/data/tree_code/CTD_Chemical_Tree.xlsx")
        elif dataset_name_or_path in [ "AAP",  "cometa_clinical", "AAP_Fold0", "cometa_knn_data", "cometa"]:
            # df = pd.read_csv(f"{root_path}/data/tree_code/snomed_tree_code_201907.csv")
            df = pd.read_csv(f"{root_path}/data/tree_code/snomed_tree_code_au_2024.csv")
        elif dataset_name_or_path in ["sympel", "ispo_en"]:
            df = pd.read_excel(f"{root_path}/data/tree_code/ISPO_Tree_BioSyn.xlsx")
        elif dataset_name_or_path in ["icd_en", "icd_zh"]:
            df = pd.read_excel(f"{root_path}/data/tree_code/ICD_Tree.xlsx")
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

        self.tree_similarities = TreeCodeSimilarity(tree_code_sets, self.dataset_name_or_path)

        self.error_cuis = set()
        
        self.mesh_data_list = ["bc5cdr-disease", "ncbi-disease", "bc5cdr-chemical", "mesh_disease", "mesh_chemical"]

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

    @lru_cache(maxsize=100000)
    def complete_cui(self, cui):
        """补全 MESH/OMIM 前缀。"""
        cui = cui.replace("+", "|").split("|")[0]
        if cui.startswith("MESH") or cui.startswith("OMIM"):
            return cui
        if not cui[0].isdigit():
            return "MESH:" + cui
        return "OMIM:" + cui

    @lru_cache(maxsize=100000)
    def compute_similarity(self, id1, id2):
        if self.dataset_name_or_path in self.mesh_data_list:
            id1 = self.complete_cui(id1)
            id2 = self.complete_cui(id2)

        tree1 = self.id2tree_codes.get(id1)
        if tree1 is None:
            self.error_cuis.add(id1)
            return 0.0
        tree2 = self.id2tree_codes.get(id2)
        if tree2 is None:
            self.error_cuis.add(id2)
            return 0.0

        sim = self.get_nearest_similarity(tree1, tree2)
        return sim * self.tree_ratio

    @lru_cache(maxsize=100000)
    def get_parent_ids(self, cui) -> List[str]:
        """
        :param cui: 实体的CUI
        :return: 实体的父节点的CUI
        """
        if self.dataset_name_or_path in self.mesh_data_list:
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
        if self.dataset_name_or_path in self.mesh_data_list:
            parent_ids = [i.split(":")[-1] for i in parent_ids]
        return parent_ids

    @lru_cache(maxsize=100000)
    def get_child_ids(self, cui) -> List[str]:
        """
        :param cui: 实体的CUI
        :return: 实体的子节点的CUI
        """
        if self.dataset_name_or_path in self.mesh_data_list:
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
        if self.dataset_name_or_path in self.mesh_data_list:
            child_ids = [i.split(":")[-1] for i in child_ids]
        return child_ids

    @lru_cache(maxsize=100000)
    def get_sibling_ids(self, cui) -> List[str]:
        """
        :param cui: 实体的CUI
        :return: 实体的兄弟节点的CUI（同一父节点下的其他子节点）
        """
        if self.dataset_name_or_path in self.mesh_data_list:
            cui = self.complete_cui(cui)

        # 获取父节点
        parent_ids = self.get_parent_ids(cui)
        if not parent_ids:
            return []

        sibling_ids = []
        for parent_id in parent_ids:
            # 获取父节点的所有子节点
            children_of_parent = self.get_child_ids(parent_id)

            # 排除自己，剩下的就是兄弟节点
            siblings = [child_id for child_id in children_of_parent if child_id != cui]
            sibling_ids.extend(siblings)

        # 去重
        sibling_ids = list(set(sibling_ids))
        return sibling_ids

    def save_error(self):
        with open("error_cuis.txt", "w") as f:
            for cui in self.error_cuis:
                f.write(cui + "\n")
