from src.evaluator.evaluator import Evaluator
from src.hparams.parser import parse_args
from src.utils.shared_tools import SharedTools

def main():
    args = parse_args()

    shared_tools = SharedTools(args)

    evaluator = Evaluator(args, shared_tools, dev_or_test="dev")
    evaluator.evaluate(epoch=9, step=9)


if __name__ == "__main__":
    main()
