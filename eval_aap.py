import os
import numpy as np

from src.evaluator.evaluator import Evaluator
from src.hparams.parser import parse_args
from src.logger.logger import setup_logger
from src.utils.shared_tools import SharedTools

def main():
    args = parse_args()

    logger = setup_logger(args.log_file)

    original_model_path = args.model_name_or_path
    results = []
    for i in range(10):
        args.eval_dir = f"./data/aap/fold{i}/test.txt"
        args.model_name_or_path = os.path.join(original_model_path, f"fold{i}")
        
        shared_tools = SharedTools(args)
        evaluator = Evaluator(args, shared_tools, dev_or_test="dev")
        result = evaluator.evaluate(epoch=999, step=999)
        results.append(result)

    average_acc1 = np.mean([result["acc1"] for result in results])
    average_acc5 = np.mean([result["acc5"] for result in results])
    logger.info(f"Average acc1: {average_acc1}")
    logger.info(f"Average acc5: {average_acc5}")


if __name__ == "__main__":
    main()
