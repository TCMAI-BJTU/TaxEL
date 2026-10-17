# -*- coding: utf-8 -*-
# @Time    : 2024/8/4 17:56
# @Author  : Rui Hua
# @Email   :
# @File    : concpet_model.py
# @Software: PyCharm
import torch
from torch import nn, optim
from torch.nn import functional as F


class TaxEL(nn.Module):
    def __init__(self, encoder, args):
        super(TaxEL, self).__init__()
        self.max_length = args.max_length
        self.use_cuda = args.use_cuda
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.topk = args.topk
        self.encoder = encoder
        self.train_similarity_func = args.train_similarity_func
        self.loss_func = args.loss_func
        self.kl_alpha = args.kl_alpha

        self.embed_dim = args.embed_dim

        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.use_cuda:
            self.to("cuda")

    def tensor2cuda(self, query_token, candidate_tokens):
        if self.use_cuda:
            query_token = {k: v.to("cuda") for k, v in query_token.items()}
            candidate_tokens = {k: v.to("cuda") for k, v in candidate_tokens.items()}
        return query_token, candidate_tokens

    def get_query_embedding(self, query_token):
        query_token = {k: v.squeeze(1) for k, v in query_token.items()}
        query_embeddings = self.encoder.forward(query_token)["sentence_embedding"]
        return query_embeddings

    def get_candidate_embedding(self, candidate_tokens):
        candidate_tokens = {k: v.reshape(-1, self.max_length) for k, v in candidate_tokens.items()}
        candidate_embeddings = self.encoder.forward(candidate_tokens)
        candidate_embeddings = candidate_embeddings["sentence_embedding"].view(-1, self.topk, self.embed_dim)
        return candidate_embeddings

    def forward(self, x):
        query_token, candidate_tokens = x
        query_token, candidate_tokens = self.tensor2cuda(query_token, candidate_tokens)
        query_embeddings = self.get_query_embedding(query_token)
        candidate_embeddings = self.get_candidate_embedding(candidate_tokens)

        if self.loss_func == "hybrid_kl_marginal_nll":
            dot_score = torch.bmm(query_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)
            cosine_score = F.cosine_similarity(query_embeddings.unsqueeze(1), candidate_embeddings, dim=-1)
            return dot_score, cosine_score

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
        if self.loss_func == "marginal_nll":
            loss = self.marginal_nll(score, target)
        elif self.loss_func == "KL":
            loss = self.KL_loss(score, target)
        elif self.loss_func == "hybrid_kl_marginal_nll":
            loss = self.hybrid_kl_marginal_nll(score, target)
        else:
            raise ValueError("Unknown loss function")
        return loss

    def KL_loss(self, model_logits, ic_similarities):
        target_distribution = F.softmax(ic_similarities, dim=1)

        # 将模型的 logits 转换为概率分布
        predicted_distribution = F.softmax(model_logits, dim=1)

        # 计算 KL 散度 (batch-wise)
        kl_loss = F.kl_div(
            input=torch.log(predicted_distribution),  # 模型的 log 概率分布
            target=target_distribution,  # 目标概率分布
            reduction="batchmean",  # 计算每个样本的 KL 散度，然后取平均
        )

        return kl_loss
    
    
    def marginal_nll(self, score, target):
        """
        sum all scores among positive samples
        """
        # Step 1: Convert soft labels to hard labels (argmax one-hot)
        if target.dtype != torch.long and target.max() <= 1.0:
            # Get index of max value (assumed to be soft label)
            hard_indices = torch.argmax(target, dim=-1, keepdim=True)
            target = torch.zeros_like(target).scatter_(-1, hard_indices, 1.0)

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

    def hybrid_kl_marginal_nll(self, score, target):
        kl_alpha = self.kl_alpha
        dot_score, cosine_score = score
        kl_loss = self.KL_loss(cosine_score, target)
        marginal_nll_loss = self.marginal_nll(dot_score, target)
        loss = kl_loss * kl_alpha + marginal_nll_loss * (1 - kl_alpha)
        return loss
