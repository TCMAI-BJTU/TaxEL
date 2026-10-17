import logging
import time

from src.hparams.parser import parse_args
from src.logger.logger import EnhancedLogger
from src.utils.experiment import run_experiment

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    name = f"{args.dataset_name_or_path}_{args.learning_rate}_{time.strftime('%Y%m%d_%H%M%S')}"
    enhanced_logger = EnhancedLogger(args.log_file, name)
    enhanced_logger.log_experiment_config(args)
    summary = run_experiment(args, enhanced_logger=enhanced_logger)
    enhanced_logger.save_training_plots()
    enhanced_logger.save_detailed_stats()
    logger.info("BEST DEV: %s", summary["best_dev"])
    for label, result in summary["test"].items():
        print(f"TEST {label}: {result['metrics']}", flush=True)
    print(f"Log file: {args.log_file}")


if __name__ == "__main__":
    main()
