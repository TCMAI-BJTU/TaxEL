from src.utils.shared_tools import SharedTools
from src.evaluator.evaluator import Evaluator
from src.data.loader import load_data
from src.hparams.parser import parse_args
from src.model.taxel import ConceptModel
from src.trainer.trainer import Trainer


def main():
    args = parse_args()

    args.train_dictionary_path = f"{args.root_path}/data/AAP/test_dictionary.txt"
    args.test_dictionary_path = f"{args.root_path}/data/AAP/test_dictionary.txt"

    best_results = []
    for fold_num in range(10):
        print(f"Fold {fold_num}")
        args.train_dir = f"{args.root_path}/data/AAP/fold{fold_num}/train.txt"
        args.dev_dir = f"{args.root_path}/data/AAP/fold{fold_num}/test.txt"
        args.test_dir = f"{args.root_path}/data/AAP/fold{fold_num}/test.txt"
        shared_tools = SharedTools(args)
        train_dataset, train_loader = load_data(args, shared_tools=shared_tools)

        model = ConceptModel(encoder=train_dataset.encoder, args=args)

        evaluator = Evaluator(args, shared_tools, dev_or_test="dev")

        trainer = Trainer(args, evaluator)

        trainer.train(model=model, train_dataset=train_dataset, train_loader=train_loader)

        best_result = trainer.evaluator.best_result
        best_results.append(dict(best_result))

    for i, result in enumerate(best_results):
        print(f"Fold {i}: {result}")

    res_acc5 = []
    res_acc1 = []
    for i, result in enumerate(best_results):
        res_acc5.append(result["acc5"])
        res_acc1.append(result["acc1"])
    print(f"Average acc1: {sum(res_acc1) / len(res_acc1)}")
    print(f"Average acc5: {sum(res_acc5) / len(res_acc5)}")



if __name__ == "__main__":
    main()
