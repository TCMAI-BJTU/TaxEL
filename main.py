from src.utils.shared_tools import SharedTools
from src.evaluator.evaluator import Evaluator
from src.data.loader import load_data
from src.hparams.parser import parse_args
from src.model.concpet_model import ConceptModel
from src.trainer.trainer import Trainer


def main():
    args = parse_args()

    shared_tools = SharedTools(args)

    train_dataset, train_loader = load_data(args, shared_tools)

    model = ConceptModel(encoder=shared_tools.encoder, args=args)

    dev_evaluator = Evaluator(args, shared_tools, dev_or_test="dev")

    trainer = Trainer(args, dev_evaluator)    

    trainer.train(model=model, train_dataset=train_dataset, train_loader=train_loader)




if __name__ == "__main__":
    main()
