# -*- coding: utf-8 -*-
"""统一的日志模块。

- ``setup_logger``: 在程序入口调用一次，完成全局根 logger 配置。
- ``EnhancedLogger``: 在 ``setup_logger`` 之上提供训练统计、绘图与结果汇总。

其它模块直接使用 ``logging.getLogger(__name__)`` 获取 logger，
不要重复调用 ``setup_logger``。
"""
import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
TARGET_METRICS = ("acc1", "acc5", "acc10", "acc20")


def setup_logger(log_file: str) -> logging.Logger:
    """配置全局根 logger。重复调用安全（仅首次生效）。"""
    log_dir = os.path.dirname(log_file) or "logs"
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    if not getattr(root, "_taxel_configured", False):
        logging.basicConfig(
            level=logging.INFO,
            format=_LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        root._taxel_configured = True  # type: ignore[attr-defined]
    return root


class EnhancedLogger:
    """带训练统计、绘图与结果汇总能力的日志器。"""

    def __init__(self, log_file: str, experiment_name: str = "experiment"):
        self.log_file = log_file
        self.experiment_name = experiment_name

        self._create_directories()
        self.logger = setup_logger(log_file)

        self.train_stats: Dict[str, List[Any]] = defaultdict(list)
        self.eval_stats: Dict[str, List[Any]] = defaultdict(list)
        self.experiment_config: Dict[str, Any] = {}

    @staticmethod
    def _create_directories():
        for directory in ("logs", "plots", "stats", "checkpoints"):
            os.makedirs(directory, exist_ok=True)

    def log_experiment_config(self, args):
        config = vars(args)
        self.experiment_config = config

        self.logger.info("=" * 50)
        self.logger.info(f"开始实验: {self.experiment_name}")
        self.logger.info("=" * 50)

        config_file = f"stats/config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        important_params = [
            "dataset_name_or_path", "batch_size", "learning_rate",
            "epochs", "max_length", "topk",
        ]
        self.logger.info("实验配置:")
        for param in important_params:
            if hasattr(args, param):
                self.logger.info(f"  {param}: {getattr(args, param)}")

    def log_training_step(self, epoch: int, step: int, loss: float, lr: float = None):
        self.train_stats["epoch"].append(epoch)
        self.train_stats["step"].append(step)
        self.train_stats["loss"].append(loss)
        if lr:
            self.train_stats["learning_rate"].append(lr)
        self.logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss:.6f}")

    def log_evaluation_results(self, epoch: int, step: int, results: Dict[str, float]):
        self.eval_stats["epoch"].append(epoch)
        self.eval_stats["step"].append(step)

        for metric in TARGET_METRICS:
            if metric in results:
                self.eval_stats[metric].append(results[metric])

        metrics_str = ", ".join(
            f"{k}: {results[k]:.4f}" for k in TARGET_METRICS if k in results
        )
        self.logger.info(f"Epoch {epoch}, Step {step} - 评估结果: {metrics_str}")

    def save_training_plots(self):
        if not self.train_stats["loss"]:
            return

        plt.style.use("seaborn-v0_8")
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.train_stats["loss"], linewidth=2, color="red", alpha=0.8)
        plt.title("训练损失曲线", fontsize=14, fontweight="bold")
        plt.xlabel("训练步数")
        plt.ylabel("损失值")
        plt.grid(True, alpha=0.3)

        if self.train_stats["learning_rate"]:
            plt.subplot(2, 2, 2)
            plt.plot(self.train_stats["learning_rate"], linewidth=2, color="blue", alpha=0.8)
            plt.title("学习率变化", fontsize=14, fontweight="bold")
            plt.xlabel("训练步数")
            plt.ylabel("学习率")
            plt.grid(True, alpha=0.3)

        if self.eval_stats["acc1"]:
            plt.subplot(2, 2, 3)
            for metric, color in zip(("acc1", "acc5"), ("green", "orange")):
                if self.eval_stats.get(metric):
                    plt.plot(
                        self.eval_stats[metric],
                        label=metric.upper(),
                        linewidth=2,
                        color=color,
                        alpha=0.8,
                        marker="o",
                        markersize=4,
                    )
            plt.title("模型准确率变化", fontsize=14, fontweight="bold")
            plt.xlabel("评估轮数")
            plt.ylabel("准确率")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.axis("off")
        plt.text(
            0.1, 0.9, self._generate_stats_summary(),
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
        )

        plt.tight_layout()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_file = f"plots/training_summary_{self.experiment_name}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info(f"训练图表已保存至: {plot_file}")

    def _generate_stats_summary(self) -> str:
        stats: List[str] = []
        if self.train_stats["loss"]:
            stats.append(f"最低损失: {min(self.train_stats['loss']):.6f}")
            stats.append(f"平均损失: {np.mean(self.train_stats['loss']):.6f}")

        if self.eval_stats["acc1"]:
            max_acc1 = max(self.eval_stats["acc1"])
            max_acc1_epoch = self.eval_stats["epoch"][self.eval_stats["acc1"].index(max_acc1)]
            stats.append(f"最高 ACC1: {max_acc1:.4f}")
            stats.append(f"最佳轮数: {max_acc1_epoch}")

        if self.eval_stats["acc5"]:
            stats.append(f"最高 ACC5: {max(self.eval_stats['acc5']):.4f}")

        stats.append(f"总训练步数: {len(self.train_stats['loss'])}")
        stats.append(f"总评估次数: {len(self.eval_stats['acc1']) if self.eval_stats['acc1'] else 0}")
        return "\n".join(stats)

    def save_detailed_stats(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if self.train_stats["loss"]:
            train_file = f"stats/train_stats_{self.experiment_name}_{timestamp}.csv"
            pd.DataFrame(self.train_stats).to_csv(train_file, index=False, encoding="utf-8")
            self.logger.info(f"训练统计已保存至: {train_file}")

        if self.eval_stats["acc1"]:
            eval_file = f"stats/eval_stats_{self.experiment_name}_{timestamp}.csv"
            pd.DataFrame(self.eval_stats).to_csv(eval_file, index=False, encoding="utf-8")
            self.logger.info(f"评估统计已保存至: {eval_file}")

    def log_final_results(self, best_results: List[Dict]):
        self.logger.info("=" * 50)
        self.logger.info("实验完成 - 最终结果汇总")
        self.logger.info("=" * 50)

        if not best_results:
            return

        acc1_values = [r["acc1"] for r in best_results]
        acc5_values = [r["acc5"] for r in best_results]
        acc1_mean, acc1_std = float(np.mean(acc1_values)), float(np.std(acc1_values))
        acc5_mean, acc5_std = float(np.mean(acc5_values)), float(np.std(acc5_values))

        self.logger.info(f"{len(best_results)}折交叉验证结果:")
        for i, result in enumerate(best_results):
            self.logger.info(f"  Fold {i}: ACC1={result['acc1']:.4f}, ACC5={result['acc5']:.4f}")

        self.logger.info("平均结果:")
        self.logger.info(f"  ACC1: {acc1_mean:.4f} ± {acc1_std:.4f}")
        self.logger.info(f"  ACC5: {acc5_mean:.4f} ± {acc5_std:.4f}")

        final_results = {
            "experiment_name": self.experiment_name,
            "folds": best_results,
            "summary": {
                "acc1_mean": acc1_mean, "acc1_std": acc1_std,
                "acc5_mean": acc5_mean, "acc5_std": acc5_std,
            },
            "config": self.experiment_config,
        }
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"stats/final_results_{self.experiment_name}_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"最终结果已保存至: {results_file}")

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)
