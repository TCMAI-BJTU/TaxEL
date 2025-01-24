from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.candidate_dataset import CandidateDataset
from src.evaluator.evaluator import Evaluator
from src.logger.logger import setup_logger
from src.model.taxel import TaxEL


class Trainer:
    def __init__(self, args, evaluator: Evaluator):
        self._init_config(args)
        self.logger = setup_logger(args.log_file)
        self.evaluator = evaluator

    def _init_config(self, args):
        self.epochs = args.epochs
        self.use_tree_similarity = args.use_tree_similarity
        self.use_schedule = args.use_schedule
        self.retrieve_step_ratio = args.retrieve_step_ratio
        self.early_stopping = False

    def train(
        self,
        model: TaxEL,
        train_dataset: CandidateDataset,
        train_loader: DataLoader,
    ):
        if self.use_schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(model.optimizer, self.epochs * len(train_loader))

        self.evaluator.evaluate(epoch=0, step=0)

        retrieve_step = int(len(train_loader) * self.retrieve_step_ratio)

        self._set_train_candidate_idxs(train_dataset)

        for epoch in range(self.epochs):

            total_loss = 0
            model.train()
            progress_bar = tqdm(total=len(train_loader), desc="Training epoch {}".format(epoch + 1), ncols=80)

            for step, data in enumerate(train_loader):
                if step % retrieve_step == 0 and step != 0:
                    self.evaluator.evaluate(epoch, step)
                    self._set_train_candidate_idxs(train_dataset)

                model.optimizer.zero_grad()

                batch_x, batch_y = data

                batch_pred = model(batch_x)
                loss = model.compute_loss(batch_pred, batch_y)

                loss.backward()
                model.optimizer.step()
                if self.use_schedule:
                    scheduler.step()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
                total_loss += loss.item()

            progress_bar.close()

            model.eval()
            self.evaluator.evaluate(epoch, step)

            self._set_train_candidate_idxs(train_dataset)

            model.train()

            self.save_error_cuis(train_dataset.tree_sim.error_cuis)
            self.logger.info("Epoch {} average loss: {}".format(epoch + 1, total_loss / len(train_loader)))

        return model

    def _set_train_candidate_idxs(self, train_dataset):
        if train_dataset.dict_embeds is not None and self.evaluator.test_dataset.dict_embeds is not None:
            if train_dataset.dict_embeds.shape == self.evaluator.test_dataset.dict_embeds.shape:
                train_dataset.set_candidate_idxs(
                    dict_embeds=self.evaluator.test_dataset.dict_embeds,
                )
        else:
            train_dataset.set_candidate_idxs()

    def save_error_cuis(self, error_cuis):
        if self.use_tree_similarity:
            with open("error_cuis.txt", "w") as f:
                for i in error_cuis:
                    f.write(str(i) + "\n")
