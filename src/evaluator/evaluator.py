# -*- coding: utf-8 -*-
# @Time    : 2024/8/6 18:00
# @Author  : Rui Hua
# @Email   :
# @File    : evaluator.py
# @Software: PyCharm
import json
import logging
import os
from collections import defaultdict
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.data.candidate_dataset import CandidateDataset
from src.data.loader import load_queries, load_dictionary

if TYPE_CHECKING:
    from src.logger.logger import EnhancedLogger

logger = logging.getLogger(__name__)
TARGET_METRICS = ("acc1", "acc5", "acc10", "acc20")


class Evaluator:
    def __init__(self, args, shared_tools, dev_or_test):
        self._init_config(args, dev_or_test)

        self.best_result = defaultdict(int)

        self._init_tools(shared_tools)
        self._init_data(args, shared_tools)

        self.enhanced_logger: Optional["EnhancedLogger"] = None  # 将由训练器设置

    def _init_config(self, args, dev_or_test):
        """Initialize configuration."""
        self.dev_or_test = dev_or_test
        self.root_path = args.root_path
        self.dataset_name_or_path = args.dataset_name_or_path
        self.output_path = args.output_path
        self.checkpoint_dir = getattr(args, "checkpoint_dir", None)
        self.records_dir = getattr(args, "records_dir", None)
        self.log_file = args.log_file
        self.learning_rate = args.learning_rate
        self.debug = args.debug
        self.dev_dictionary_path = args.dev_dictionary_path
        self.dev_dir = args.dev_dir
        self.test_dictionary_path = args.test_dictionary_path
        self.test_dir = args.test_dir
        self.normalize_cui_plus = getattr(args, "normalize_cui_plus", False)

    def _init_tools(self, shared_tools):
        """Initialize tools."""
        self.encoder = shared_tools.encoder
        self.tree_sim = shared_tools.tree_sim
        self.tokenizer = shared_tools.tokenizer

    def _init_data(self, args, shared_tools):
        """Initialize data."""
        if self.dev_or_test == "dev":
            self.eval_dictionary = load_dictionary(self.dev_dictionary_path, self.dataset_name_or_path)
            self.eval_queries = load_queries(self.dev_dir, self.dataset_name_or_path, stage="dev")
        elif self.dev_or_test == "test":
            self.eval_dictionary = load_dictionary(self.test_dictionary_path, self.dataset_name_or_path)
            self.eval_queries = load_queries(self.test_dir, self.dataset_name_or_path, stage="test")
        else:
            raise ValueError("dev_or_test should be 'dev' or 'test'")

        if self.debug:
            self.eval_queries = self.eval_queries[:120]
            self.eval_dictionary = self.eval_dictionary[:12000]
        # 传入candidateDataset的query和评估循环遍历时候的query不一样，传入candidateDataset的需要拆分复合术语
        eval_queries = []
        for query in self.eval_queries:
            mentions, cuis, query_type = query
            mentions = mentions.split("|")
            for mention in mentions:
                eval_queries.append((mention, cuis))

        self.test_dataset = CandidateDataset(args=args, queries=eval_queries, dicts=self.eval_dictionary, shared_tools=shared_tools, stage="test")

    def save_checkpoint(self, epoch, step):
        checkpoint = self.checkpoint_dir or os.path.join(
            self.root_path, self.output_path, self.dataset_name_or_path,
            f"model_{self.learning_rate}",
        )
        os.makedirs(checkpoint, exist_ok=True)
        self.test_dataset.encoder.save_pretrained(checkpoint, create_model_card=False)
        self.test_dataset.tokenizer.save_pretrained(checkpoint, create_model_card=False)
        with open(os.path.join(checkpoint, "checkpoint_metadata.json"), "w") as f:
            json.dump({"kind": "best", "selection_split": "dev", "epoch": epoch,
                       "step": step, "metrics": dict(self.best_result)}, f, indent=2)
        logger.info("Saved BEST dev checkpoint at epoch %s step %s to %s", epoch, step, checkpoint)

    def check_label(self, predicted_cui, golden_cui):
        """
        Some composite annotation didn't consider orders
        So, set label '1' if any cui is matched within composite cui (or single cui)
        Otherwise, set label '0'
        """
        if self.normalize_cui_plus:
            predicted_cui = predicted_cui.replace("+", "|")
            golden_cui = golden_cui.replace("+", "|")
        return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)

    def evaluate(self, model, epoch, step):
        was_training = model.training
        model.eval()
        self.test_dataset.set_candidate_idxs()

        queries = []

        dict_names = np.array(self.test_dataset.dict_names)
        dict_ids = np.array(self.test_dataset.dict_ids)

        for eval_query in self.eval_queries:

            mentions = eval_query[0].replace("+", "|").split("|")
            golden_cui = eval_query[1].replace("+", "|")

            dict_mentions = []
            for mention in mentions:
                # 从 list.index 修改为了 字典
                query_idx = self.test_dataset.query_name_to_idx[mention]

                pred_candidate_idxs = self.test_dataset.candidate_idxs[query_idx].reshape(-1)  # type: ignore
                pred_candidate_scores = self.test_dataset.candidate_scores[query_idx].reshape(-1)

                # pred_candidates = self.eval_dictionary[pred_candidate_idxs]
                pred_candidate_names = dict_names[pred_candidate_idxs]
                pred_candidate_ids = dict_ids[pred_candidate_idxs]

                dict_candidates = []
                for pred_candidate in zip(pred_candidate_names, pred_candidate_ids, pred_candidate_scores):
                    dict_candidates.append(
                        {
                            "name": pred_candidate[0],
                            "cui": pred_candidate[1],
                            "label": self.check_label(pred_candidate[1], golden_cui),
                            "score": f"{pred_candidate[2]:.4f}",
                        }
                    )
                dict_mentions.append({"mention": mention, "golden_cui": golden_cui, "candidates": dict_candidates})
            queries.append({"mentions": dict_mentions})

        result = self.evaluate_topk_acc({"queries": queries}, epoch, step)

        # 记录评估结果到增强日志器
        if self.enhanced_logger:
            self.enhanced_logger.log_evaluation_results(epoch, step, result)

        if not self.best_result or result["acc1"] > self.best_result["acc1"] or (result["acc1"] == self.best_result["acc1"] and result["acc5"] > self.best_result["acc5"]):
            for i in range(len(queries[0]["mentions"][0]["candidates"])):
                self.best_result[f"acc{i + 1}"] = result[f"acc{i + 1}"]  # type: ignore
            self.best_result["epoch"] = epoch
            self.best_result["step"] = step

            if self.dev_or_test == "dev":
                self.save_checkpoint(epoch, step)

        output_str = ", ".join(f"{k}: {self.best_result[k]:.4f}" for k in TARGET_METRICS if k in self.best_result)
        logger.info(f"Best result - Epoch {self.best_result.get('epoch', 'N/A')}: {output_str}")
        model.train(was_training)
        return result

    def evaluate_topk_acc(self, data, epoch, step) -> dict[str, float]:
        """
        evaluate acc@1~acc@k
        """
        queries = data["queries"]

        total = len(queries[0]["mentions"][0]["candidates"])

        for i in range(0, total):
            hit = 0
            for query in queries:
                mentions = query["mentions"]
                mention_hit = 0
                for mention in mentions:
                    candidates = mention["candidates"][: i + 1]  # to get acc@(i+1)
                    mention_hit += np.any([candidate["label"] for candidate in candidates])

                # When all mentions in a query are predicted correctly,
                # we consider it as a hit
                if mention_hit == len(mentions):
                    hit += 1

            data["acc{}".format(i + 1)] = round(hit / len(queries), 4)

        self.save_result(data, epoch, step)

        return data

    def save_result(self, data, epoch, step):
        directory = self.records_dir or f"./records/{self.dataset_name_or_path}"
        os.makedirs(directory, exist_ok=True)
        result_file = os.path.join(directory, f"result_{epoch}_{step}.json")
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info("Result saved to %s", result_file)
