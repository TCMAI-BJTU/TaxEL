import torch
from torch import nn, optim
from torch.nn import functional as F


class ConceptModel(nn.Module):
    def __init__(self, encoder, args):
        super(ConceptModel, self).__init__()
        self.max_length = args.max_length
        self.use_cuda = args.use_cuda
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.topk = args.topk
        self.encoder = encoder
        self.train_similarity_func = args.train_similarity_func
        self.loss_func = args.loss_func

        self.embed_dim = args.embed_dim

        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_cuda:
            self.to("cuda")

    def tensor2cuda(self, query_token, candidate_tokens):
        if self.use_cuda:
            query_token = query_token.to("cuda")
            candidate_tokens = candidate_tokens.to("cuda")
            self.encoder.to("cuda")
        return query_token, candidate_tokens,

    def get_query_embedding(self, query_token):
        query_token = {k: v.squeeze(1) for k, v in query_token.items()}
        # query_embeddings = self.encoder.forward(query_token)[0][:, 0]  # 16, 768
        query_embeddings = self.encoder.forward(query_token)['sentence_embedding']
        return query_embeddings

    def get_candidate_embedding(self, candidate_tokens):
        # input_ids = []
        # attention_mask = []
        # for candidate_token in candidate_tokens:
        #     input_ids.append(candidate_token["input_ids"])
        #     attention_mask.append(candidate_token["attention_mask"])
        # input_ids = torch.stack(input_ids, dim=0).view(-1, self.max_length)
        # attention_mask = torch.stack(attention_mask, dim=0).view(-1, self.max_length)
        #
        # candidate_tokens = {"input_ids": input_ids, "attention_mask": attention_mask}

        # candidate_embeddings = self.encoder.forward(
        #     input_ids=candidate_tokens["input_ids"].reshape(-1, self.max_length),
        #     attention_mask=candidate_tokens["attention_mask"].reshape(-1, self.max_length)
        # )[0][:, 0].view(-1, self.topk, self.embed_dim)
        candidate_tokens = {k: v.reshape(-1, self.max_length) for k, v in candidate_tokens.items()}
        candidate_embeddings = self.encoder.forward(candidate_tokens)
        candidate_embeddings = candidate_embeddings['sentence_embedding'].view(-1, self.topk, self.embed_dim)

        # candidate_embeddings = self.encoder(**candidate_tokens)[0][:, 0]
        # candidate_embeddings = candidate_embeddings.view(-1, self.concept_topk, self.entity_topk, self.embed_dim)
        return candidate_embeddings

    def forward(self, x):
        query_token, candidate_tokens = x

        query_token, candidate_tokens = self.tensor2cuda(query_token, candidate_tokens)

        batch_size = query_token["input_ids"].size(0)

        query_embeddings = self.get_query_embedding(query_token)  # 16, 768

        candidate_embeddings = self.get_candidate_embedding(candidate_tokens)  # (16,7,3,768)

        # dot
        if self.train_similarity_func == "dot":
            score = torch.bmm(query_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)
        elif self.train_similarity_func == "cosine":
            score = F.cosine_similarity(query_embeddings.unsqueeze(1), candidate_embeddings, dim=-1)
        else:
            raise ValueError("Unknown similarity function")
        return score

    def compute_loss(self, score, target):
        if self.use_cuda:
            target = target.to("cuda")
        # loss = F.binary_cross_entropy_with_logits( F.softmax(score, dim=-1), target)
        # loss = F.cross_entropy(score, target)
        if self.loss_func == "marginal_nll":
            loss = self.marginal_nll(score, target)
        elif self.loss_func == "mse":
            loss = self.mse_loss(score, target)
        elif self.loss_func == "cosent":
            loss = self.CoSENTLoss(score, target)
        elif self.loss_func == "marginal_nll_A":
            loss = self.marginal_nll_A(score, target)
        elif self.loss_func == "pairwise":
            loss = self.pairwise_loss(score, target)
        elif self.loss_func == "listwise":
            loss = self.listwise_loss(score, target)
        elif self.loss_func == "KL":
            loss = self.KL_loss(score, target)
        else:
            raise ValueError("Unknown loss function")
        return loss

    def KL_loss(self, model_logits, ic_similarities):
        """
        计算 KL 散度损失。

        :param model_logits: 模型对每个候选实体的预测 logits，形状为 [batch_size, num_candidates]。
        :param ic_similarities: 基于信息增益相似度的目标分布，形状为 [batch_size, num_candidates]。
        :return: KL 散度损失。
        """
        # 将 IC 相似度通过 softmax 转换为概率分布
        self.temperature = 1.0

        target_distribution = F.softmax(ic_similarities / self.temperature, dim=1)

        # 将模型的 logits 转换为概率分布
        predicted_distribution = F.softmax(model_logits, dim=1)

        # 计算 KL 散度 (batch-wise)
        kl_loss = F.kl_div(
            input=torch.log(predicted_distribution),  # 模型的 log 概率分布
            target=target_distribution,  # 目标概率分布
            reduction="batchmean"  # 计算每个样本的 KL 散度，然后取平均
        )

        return kl_loss

    def listwise_loss(self, score, target):
        """
        Listwise Loss implementation.

        Args:
            score (torch.Tensor): Predicted scores, shape (batch_size, num_candidates).
            target (torch.Tensor): Actual scores, shape (batch_size, num_candidates).

        Returns:
            torch.Tensor: Listwise loss value.
        """
        # Compute softmax probabilities for both score and target
        pred_prob = F.softmax(score, dim=1)  # Shape: (batch_size, num_candidates)
        target_prob = F.softmax(target, dim=1)  # Shape: (batch_size, num_candidates)

        # Cross-entropy loss (KL divergence-like)
        loss = -torch.sum(target_prob * torch.log(pred_prob + 1e-12), dim=1)  # Avoid log(0) by adding epsilon

        # Mean loss over the batch
        return loss.mean()

    def pairwise_loss(self, score, target):
        """
        Pairwise Loss implementation.

        Args:
            score (torch.Tensor): Predicted scores, shape (batch_size, num_candidates).
            target (torch.Tensor): Actual scores, shape (batch_size, num_candidates).

        Returns:
            torch.Tensor: Pairwise loss value.
        """
        batch_size, num_candidates = score.size()

        # Expand dimensions for pairwise comparisons
        score_diff = score.unsqueeze(2) - score.unsqueeze(1)  # Shape: (batch_size, num_candidates, num_candidates)
        target_diff = target.unsqueeze(2) - target.unsqueeze(1)  # Shape: (batch_size, num_candidates, num_candidates)

        # Mask: Keep only pairs where target_diff > 0 (i.e., correct ranking order)
        mask = (target_diff > 0).float()

        # Pairwise hinge loss
        pairwise_loss = F.relu(1 - score_diff)  # Max(0, 1 - (score_i - score_j))

        # Apply the mask and compute the mean loss
        loss = (pairwise_loss * mask).sum() / mask.sum()
        return loss

    def mse_loss(self, score, target):
        loss = F.mse_loss(score.float(), target.float())
        return loss

    def marginal_nll(self, score, target):
        """
        sum all scores among positive samples
        """
        predict = F.softmax(score, dim=-1)
        loss = predict * target
        loss = loss.sum(dim=-1)  # sum all positive scores
        loss = loss[loss > 0]  # filter sets with at least one positives
        loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
        loss = -torch.log(loss)  # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()  # will return zero loss
        else:
            loss = loss.mean()
        return loss

    def marginal_nll_A(self, score, target):
        """
        sum all scores among positive samples
        """
        # predict = F.softmax(score, dim=-1)
        loss = score * target
        loss = loss.sum(dim=-1)  # sum all positive scores
        loss = loss[loss > 0]  # filter sets with at least one positives
        # loss = torch.clamp(loss, min=1e-9, max=1)  # for numerical stability
        loss = -torch.log(loss)  # for negative log likelihood
        if len(loss) == 0:
            loss = loss.sum()  # will return zero loss
        else:
            loss = loss.mean()
        return loss

    def CoSENTLoss(self, score, target):
        scale = 20.0
        score = score * scale
        score = score[:, None] - score[None, :]

        # label matrix indicating which pairs are relevant
        target = target[:, None] < target[None, :]
        target = target.float()

        # mask out irrelevant pairs so they are negligible after exp()
        score = score - (1 - target) * 1e12

        # append a zero as e^0 = 1
        score = torch.cat((torch.zeros(1).to(score.device), score.view(-1)), dim=0)
        loss = torch.logsumexp(score, dim=0)
        torch.cuda.is_available()
        return loss
