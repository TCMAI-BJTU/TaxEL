"""Train with dev selection, then reload best and last for test evaluation."""
import copy
import gc
import json
import logging
from pathlib import Path

import torch

from src.data.loader import load_data
from src.evaluator.evaluator import Evaluator
from src.model.taxel import TaxEL
from src.trainer.trainer import Trainer
from src.utils.shared_tools import SharedTools

logger = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def experiment_dir(args, fold=None):
    base = Path(args.root_path) / args.output_path
    if fold is not None:
        return base / "aap" / f"fold{fold}"
    return base / args.dataset_name_or_path / f"model_{args.learning_rate}"


def evaluate_checkpoints(args, run_dir):
    """Reload each checkpoint independently; test results never select weights."""
    run_dir = Path(run_dir)
    results = {}
    for label in ("best", "last"):
        checkpoint = run_dir / label
        if not (checkpoint / "modules.json").is_file():
            raise FileNotFoundError(f"Missing saved checkpoint: {checkpoint}")
        test_args = copy.deepcopy(args)
        test_args.model_name_or_path = str(checkpoint)
        test_args.records_dir = str(run_dir / "test" / label)
        shared = SharedTools(test_args)
        evaluator = Evaluator(test_args, shared, dev_or_test="test")
        result = evaluator.evaluate(shared.encoder, epoch=0, step=0)
        metrics = {k: v for k, v in result.items() if k.startswith("acc")}
        results[label] = {
            "checkpoint": str(checkpoint),
            "num_queries": len(result["queries"]),
            "metrics": metrics,
        }
        logger.info("TEST %s checkpoint=%s: %s", label, checkpoint, metrics)
        del evaluator, shared
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(run_dir / "test_results.json", results)
    return results


def run_experiment(args, fold=None, enhanced_logger=None):
    args = copy.deepcopy(args)
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1 to save a last-epoch checkpoint")
    run_dir = experiment_dir(args, fold).resolve()
    # Do not silently evaluate an old checkpoint after an interrupted new run.
    if (run_dir / "best").exists() or (run_dir / "last").exists():
        raise FileExistsError(f"Checkpoint directory already used: {run_dir}")
    args.checkpoint_dir = str(run_dir / "best")
    args.records_dir = str(run_dir / "dev")
    for name in ("test_dir", "test_dictionary_path"):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f"{name}: {getattr(args, name)}")
    write_json(run_dir / "run_config.json", vars(args))
    shared = SharedTools(args)
    train_dataset, train_loader = load_data(args, shared)
    if len(train_loader) == 0:
        raise ValueError("Training data must contain at least one batch")
    model = TaxEL(encoder=shared.encoder, args=args)
    evaluator = Evaluator(args, shared, dev_or_test="dev")
    evaluator.enhanced_logger = enhanced_logger
    trainer = Trainer(args, evaluator)
    trainer.train(model=model, train_dataset=train_dataset, train_loader=train_loader)

    last = run_dir / "last"
    last.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(last), create_model_card=False)
    shared.tokenizer.save_pretrained(str(last), create_model_card=False)
    write_json(last / "checkpoint_metadata.json",
               {"kind": "last", "epoch": args.epochs - 1, "step": len(train_loader) - 1})
    best_dev = dict(evaluator.best_result)
    logger.info("Saved LAST checkpoint after %s epochs: %s", args.epochs, last)
    del trainer, evaluator, model, train_dataset, train_loader, shared
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test_results = evaluate_checkpoints(args, run_dir)
    summary = {"best_dev": best_dev, "test": test_results}
    write_json(run_dir / "summary.json", summary)
    return summary


def summarize_folds(results):
    """Unweighted macro average of each fold's reported test accuracy."""
    averages = {}
    for label in ("best", "last"):
        keys = results[0]["test"][label]["metrics"]
        averages[label] = {
            key: sum(row["test"][label]["metrics"][key] for row in results) / len(results)
            for key in keys
        }
    return averages
