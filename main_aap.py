import copy
import logging
from pathlib import Path

from src.hparams.parser import parse_args
from src.utils.experiment import run_experiment, summarize_folds, write_json

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    # AAP uses fold-specific sample files and the dictionaries specified by CLI.
    for fold in range(10):
        for split in ("train", "dev", "test"):
            path = Path(args.root_path) / "data" / "AAP" / f"fold{fold}" / f"{split}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
    results = []
    for fold in range(10):
        fold_args = copy.deepcopy(args)
        base = Path(args.root_path) / "data" / "AAP" / f"fold{fold}"
        fold_args.train_dir = str(base / "train.txt")
        fold_args.dev_dir = str(base / "dev.txt")
        fold_args.test_dir = str(base / "test.txt")
        logger.info("Starting AAP fold %s", fold)
        result = run_experiment(fold_args, fold=fold)
        results.append(result)
        for label, values in result["test"].items():
            print(f"Fold {fold} TEST {label}: {values['metrics']}", flush=True)
        output = Path(args.root_path) / args.output_path / "aap" / "test_summary.json"
        write_json(output, {"completed_folds": len(results), "folds": results,
                           "macro_average": summarize_folds(results)})
    print(f"AAP TEST macro averages: {summarize_folds(results)}", flush=True)


if __name__ == "__main__":
    main()
