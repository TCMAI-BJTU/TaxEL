import os
import json
from pathlib import Path

from src.evaluator.evaluator import Evaluator
from src.hparams.parser import parse_args
from src.utils.shared_tools import SharedTools

def main():
    args = parse_args()
    if os.environ.get("EVAL_OUTPUT_DIR"):
        args.records_dir = os.environ["EVAL_OUTPUT_DIR"]

    shared_tools = SharedTools(args)

    evaluator = Evaluator(args, shared_tools, dev_or_test="test")
    
    result = evaluator.evaluate(model=shared_tools.encoder, epoch=9, step=9)
    metrics = {k: v for k, v in result.items() if k.startswith("acc")}
    print("TEST:", metrics, flush=True)
    if getattr(args, "records_dir", None):
        Path(args.records_dir, "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
