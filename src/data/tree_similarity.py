from collections import defaultdict
import math
import os
from functools import lru_cache
import pickle
from typing import List

import pandas as pd
from tqdm import tqdm


class TreeCodeSimilarity:
    def __init__(self, tree_code_set, dataset_name_or_path):
        '''
        :param tree_code_set: 原始的树编码集合，不需要做加前缀01.
        '''
        self.root = "01"

        tree_code2child_num_path = "./cache/{}_tree_code2child_num.pkl".format(dataset_name_or_path)
        parent2child_tree_codes_path = "./cache/{}_parent2child_tree_codes.pkl".format(dataset_name_or_path)

        if os.path.exists(tree_code2child_num_path) and os.path.exists(parent2child_tree_codes_path):
            self.tree_code2child_num = pickle.load(open(tree_code2child_num_path, "rb"))
            self.parent2child_tree_codes = pickle.load(open(parent2child_tree_codes_path, "rb"))
            print("Load tree_code2child_num from {}".format(tree_code2child_num_path))
        else:
            self.tree_code2child_num, self.parent2child_tree_codes = self.get_tree_code2child_num(tree_code_set)
            pickle.dump(self.tree_code2child_num, open(tree_code2child_num_path, "wb"))
            pickle.dump(self.parent2child_tree_codes, open(parent2child_tree_codes_path, "wb"))

    def get_parent_tree_code(self, tree_code: str) -> str:
        return tree_code.rsplit(".", 1)[0]

    def get_child_tree_codes(self, tree_code: str) -> List[str]:
        return self.parent2child_tree_codes[tree_code]

    def get_tree_code2child_num(self, tree_code_set):

        # 先对 tree_code_set 进行排序
        sorted_codes = sorted(tree_code_set)

        # 初始化字典存储子节点数量
        tree_code2child_num = {tree_code: 0 for tree_code in tree_code_set}
        parent2child_tree_codes = defaultdict(list)

        # 遍历 sorted_codes 来计算子节点数量
        for i, tree_code in tqdm(enumerate(sorted_codes), total=len(sorted_codes), desc='计算子节点'):
            for j in range(i + 1, len(sorted_codes)):
                # 通过检查排序后的下一个节点是否以当前节点为前缀来判断是否是子节点
                if sorted_codes[j].startswith(tree_code):
                    tree_code2child_num[tree_code] += 1
                    parent2child_tree_codes[tree_code].append(sorted_codes[j])
                else:
                    # 一旦不再是前缀匹配，后面的节点也不会是子节点，提前结束循环
                    break

        # 更新结果，添加 root 前缀，并调整子节点数量
        tree_code2child_num = {
            f"{self.root}." + k: max(v + 1, 1) for k, v in tree_code2child_num.items()
        }
        tree_code2child_num[self.root] = len(tree_code2child_num)

        return tree_code2child_num, parent2child_tree_codes

    def freq(self, c):
        return self.tree_code2child_num[c]

    def IC(self, c):
        return -math.log(self.freq(c) / self.freq(self.root))

    def lca(self, c1, c2):
        # 最近公共祖先
        c1 = c1.split(".")
        c2 = c2.split(".")
        res = []
        for i, num in enumerate(c1):
            if i < len(c2) and num == c2[i]:
                res.append(num)
            else:
                break
        return ".".join(res)

    def compute(self, c1, c2):
        c1 = f"{self.root}." + c1
        c2 = f"{self.root}." + c2
        top = 2 * self.IC(self.lca(c1, c2))
        down = self.IC(c1) + self.IC(c2)
        return abs(top / down)
