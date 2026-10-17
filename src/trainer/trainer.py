# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 18:16
# @Author  : Rui Hua
# @Email   :
# @File    : trainer.py
# @Software: PyCharm
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from src.data.candidate_dataset import CandidateDataset
from src.evaluator.evaluator import Evaluator
from src.model.taxel import TaxEL

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, evaluator: Evaluator):
        self._init_config(args)
        self.evaluator = evaluator

    def _init_config(self, args):
        self.epochs = args.epochs
        self.use_tree_similarity = args.use_tree_similarity
        self.retrieve_step_ratio = args.retrieve_step_ratio
        self.early_stopping = False

    def train(
        self,
        model: TaxEL,
        train_dataset: CandidateDataset,
        train_loader: DataLoader,
    ):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model.optimizer, self.epochs * len(train_loader))

        self.evaluator.evaluate(model, epoch=0, step=0)

        retrieve_step = max(1, int(len(train_loader) * self.retrieve_step_ratio))

        self._set_train_candidate_idxs(train_dataset)

        for epoch in range(self.epochs):

            total_loss = 0
            model.train()
            progress_bar = tqdm(total=len(train_loader), desc="Training epoch {}".format(epoch + 1), ncols=80)

            for step, data in enumerate(train_loader):
                if step % retrieve_step == 0 and step != 0:
                    self.evaluator.evaluate(model, epoch, step)
                    self._set_train_candidate_idxs(train_dataset)

                model.optimizer.zero_grad()

                batch_x, batch_y = data

                batch_pred = model(batch_x)
                loss = model.compute_loss(batch_pred, batch_y)

                loss.backward()
                model.optimizer.step()
                scheduler.step()

                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item()})
                total_loss += loss.item()

            progress_bar.close()

            model.eval()
            self.evaluator.evaluate(model, epoch, step)

            self._set_train_candidate_idxs(train_dataset)

            model.train()

            if self.use_tree_similarity:
                train_dataset.tree_sim.save_error()
            logger.info("Epoch {} average loss: {:.4f}".format(epoch + 1, total_loss / len(train_loader)))

        return model

    def _set_train_candidate_idxs(self, train_dataset):
        if (train_dataset.dict_embeds is not None
                and self.evaluator.test_dataset.dict_embeds is not None
                and train_dataset.dict_embeds.shape == self.evaluator.test_dataset.dict_embeds.shape):
            train_dataset.set_candidate_idxs(
                dict_embeds=self.evaluator.test_dataset.dict_embeds,
            )
        else:
            train_dataset.set_candidate_idxs()
