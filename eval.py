# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 16:50
# @Author  : Rui Hua
# @Email   : 
# @File    : main.py
# @Software: PyCharm
import os
import sys

sys.path.append("/home/huarui/pycharmProject/symptom_entity_link/症状实体链接")
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from src.evaluator.evaluator import Evaluator
from src.data.loader import load_data
from src.hparams.parser import parse_args
from src.model.concpet_model import ConceptModel
from src.trainer.trainer import Trainer


def main():
    args = parse_args()

    train_dataset, train_loader = load_data(args)

    model = ConceptModel(
        encoder=train_dataset.encoder,
        args=args
    )

    evaluator = Evaluator(args, train_dataset.encoder)

    trainer = Trainer(args, evaluator)

    trainer.train(
        model=model,
        train_dataset=train_dataset,
        train_loader=train_loader
    )

    for key, value in vars(args).items():
        print(f"{key}: {value}")




if __name__ == '__main__':
    main()
