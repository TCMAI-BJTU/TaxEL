# -*- coding: utf-8 -*-
# @Time    : 2024/10/16 09:38
# @Author  : Rui Hua
# @Email   : 
# @File    : test.py
# @Software: PyCharm
import glob
from collections import Counter

from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/pretrain_model/SapBERT-from-PubMedBERT-fulltext')
# files = glob.glob("/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/BioSyn_Tree/data/bc5cdr-chemical/processed_test/*.concept")
# res = []
# for file in files:
#     with open(file, "r") as f:
#         lines = f.readlines()
#         lines = [line.strip().split("||")[-2] for line in lines]
#         res.extend(lines)
# tokens = model.tokenize(res)
# aaa = []
# for token in tokens['attention_mask']:
#     aaa.append(token.sum().item())
# c = Counter(aaa)
# for k, v in c.most_common():
#     print(k, v)


model = SentenceTransformer('/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/pretrain_model/SapBERT-from-PubMedBERT-fulltext')
file = "/home/huarui/pycharmProject/symptom_entity_link/症状实体链接/BioSyn_Tree/data/AAP/dev.txt"
res = []
with open(file, "r") as f:
    lines = f.readlines()
    lines = [line.strip().split("||") for line in lines]
    lines = [line[1] for line in lines]
    tokens = model.tokenize(lines)
    for token in tokens['attention_mask']:
        res.append(token.sum().item())
c = Counter(res)
for k, v in c.most_common():
    print(k, v)
