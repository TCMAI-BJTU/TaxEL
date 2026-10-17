# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# tax_aware 配置 → 参与检索的相对节点类型集合
_TAX_AWARE_RELATIONS = {
    "current": ("current",),
    "current_parent_child": ("current", "parent", "child"),
    "current_parent_child_sibling": ("current", "parent", "child", "sibling"),
    "none": (),
}


class CandidateDataset(Dataset):
    """基于编码 + 检索的候选数据集。"""

    def __init__(self, args, queries, dicts, stage, shared_tools):
        if stage not in ("train", "test"):
            raise ValueError(f"Invalid stage: {stage}")

        self.args = args
        self.stage = stage

        self.encoder = shared_tools.encoder
        self.tokenizer = shared_tools.tokenizer
        self.tree_sim = shared_tools.tree_sim


        self.query_names = [row[0] for row in queries]
        self.query_ids = [row[1] for row in queries]
        self.dict_names = [row[0] for row in dicts]
        self.dict_ids = np.array([row[1] for row in dicts])
        self.query_name_to_idx = {name: i for i, name in enumerate(self.query_names)}

        self.candidate_idxs = None
        self.candidate_scores = None
        self.dict_embeds = None

        self._tok_query = None
        self._tok_cands = None
        self._labels = None

        # ---- CUI -> dict indices cache (used by _inject_positive) ----
        self._cui_to_dict_idxs: dict = None

        logger.info("CandidateDataset initialization completed")

    # ---------- Dataset 接口 ----------
    def __len__(self):
        return len(self.query_names)

    def __getitem__(self, query_idx):
        assert self._tok_query is not None, "set_candidate_idxs() must be called first"
        query_token = {k: v[query_idx] for k, v in self._tok_query.items()}
        candidate_tokens = {k: v[query_idx] for k, v in self._tok_cands.items()}
        labels = self._labels[query_idx]
        return (query_token, candidate_tokens), labels

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            max_length=self.args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # ---------- 编码 ----------
    def embed_names(self, names, show_progress_bar=False):
        self.encoder.eval()
        if len(names) < 500_00000000:
            embeddings = self.encoder.encode(names, show_progress_bar=False, batch_size=4096)
        else:
            pool = self.encoder.start_multi_process_pool()
            embeddings = self.encoder.encode_multi_process(
                names, pool, show_progress_bar=show_progress_bar, batch_size=4096
            )
            self.encoder.stop_multi_process_pool(pool)
        return torch.tensor(embeddings)

    # ---------- 顶层入口 ----------
    def set_candidate_idxs(self, dict_embeds=None):
        """编码 + 检索，写入 ``candidate_idxs`` / ``candidate_scores``。"""
        query_embeds = self.embed_names(self.query_names)

        if dict_embeds is None:
            self.dict_embeds = self.embed_names(self.dict_names)
        else:
            self.dict_embeds = dict_embeds
            logger.info("Using pre-computed candidate embeddings")

        if self.stage == "train":
            self.candidate_idxs, self.candidate_scores = self._retrieve_with_tree(query_embeds)
            if getattr(self.args, "force_positive_injection", False):
                self._inject_positive(query_embeds)
            if getattr(self.args, "remove_self_match", False):
                self._remove_self_match_candidates(query_embeds)
        else:
            self.candidate_idxs, self.candidate_scores = self._retrieve_faiss(query_embeds)

        self._precompute_batch_data()

    # ---------- 评估阶段：FAISS ----------
    def _retrieve_faiss(self, query_embeds):
        q = np.asarray(query_embeds, dtype="float32")
        d = np.asarray(self.dict_embeds, dtype="float32")
        if self.args.retrieve_similarity_func == "cosine":
            faiss.normalize_L2(q)
            faiss.normalize_L2(d)
        index = faiss.IndexFlatIP(self.args.embed_dim)
        index.add(d)
        scores, idxs = index.search(q, self.args.topk)
        return idxs, scores

    def _precompute_batch_data(self):
        """Pre-tokenize queries + candidates and pre-compute labels.

        Called once per call to ``set_candidate_idxs()`` so that
        ``__getitem__`` reduces to pure tensor indexing.
        """
        n_queries = len(self.query_names)
        topk = self.args.topk
        max_chunk = 50000

        self._tok_query = self._tokenize(self.query_names)

        all_cand_names = []
        for i in range(n_queries):
            all_cand_names.extend(self.dict_names[j] for j in self.candidate_idxs[i])

        chunks = []
        for start in range(0, len(all_cand_names), max_chunk):
            chunks.append(self._tokenize(all_cand_names[start:start + max_chunk]))
        tok_cands = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
        self._tok_cands = {k: v.view(n_queries, topk, -1) for k, v in tok_cands.items()}

        self._labels = torch.tensor(
            [self.get_labels(i, self.candidate_idxs[i]) for i in range(n_queries)],
            dtype=torch.float32,
        )

    # ---------- 正样本注入 ----------
    def _build_cui_index(self):
        """Build mapping from single CUI to dict indices (computed once)."""
        if self._cui_to_dict_idxs is not None:
            return
        self._cui_to_dict_idxs = defaultdict(list)
        for j, dict_id in enumerate(self.dict_ids):
            for single_cui in dict_id.replace("+", "|").split("|"):
                single_cui = single_cui.strip()
                if single_cui:
                    self._cui_to_dict_idxs[single_cui].append(j)

    def _inject_positive(self, query_embeds: torch.Tensor):
        """Ensure every query has at least one positive in its top-k candidates.

        For queries whose ``candidate_idxs`` contain no positive, this method
        searches all dictionary entries sharing the same CUI, picks the most
        similar one, and injects it at the last candidate position.
        """
        self._build_cui_index()
        n_queries = len(self.query_names)
        topk = self.args.topk

        # Pre-compute positive dict indices for every query
        query_positives = []
        for i in range(n_queries):
            pos = set()
            for single_cui in self.query_ids[i].replace("+", "|").split("|"):
                single_cui = single_cui.strip()
                if single_cui and single_cui in self._cui_to_dict_idxs:
                    pos.update(self._cui_to_dict_idxs[single_cui])
            query_positives.append(list(pos))

        device = query_embeds.device
        injected = 0

        for i in range(n_queries):
            pos_idxs = query_positives[i]
            if not pos_idxs:
                continue
            current_set = set(self.candidate_idxs[i].tolist())
            if current_set & set(pos_idxs):
                continue  # already has a positive

            # Find nearest positive by embedding similarity
            q = query_embeds[i:i + 1]                         # (1, D)
            p = self.dict_embeds[pos_idxs].to(device)         # (n_pos, D)
            if self.args.retrieve_similarity_func == "cosine":
                q = torch.nn.functional.normalize(q, dim=1)
                p = torch.nn.functional.normalize(p, dim=1)
            sims = torch.mm(q, p.T)[0]                        # (n_pos,)
            best = int(torch.argmax(sims).item())

            # Replace the lowest-scoring candidate slot
            self.candidate_idxs[i, -1] = pos_idxs[best]
            self.candidate_scores[i, -1] = float(sims[best].item())
            injected += 1

        logger.info("_inject_positive: injected for %d/%d queries", injected, n_queries)

    def _remove_self_match_candidates(self, query_embeds: torch.Tensor):
        """Replace candidates whose name equals the query name with the next-best
        non-self-matching candidates from the full dictionary."""
        n_queries = len(self.query_names)
        topk = self.args.topk

        dict_names_lower = [n.lower() for n in self.dict_names]
        query_names_lower = [n.lower() for n in self.query_names]

        device = query_embeds.device
        replaced = 0

        for i in range(n_queries):
            qname = query_names_lower[i]
            self_idxs = {j for j, dn in enumerate(dict_names_lower) if dn == qname}
            if not self_idxs:
                continue

            current = self.candidate_idxs[i]
            bad_pos = [k for k, idx in enumerate(current) if idx in self_idxs]
            if not bad_pos:
                continue

            # Mask: blocked dict indices = self-matches + already in candidates
            blocked = self_idxs | set(current.tolist())

            q = query_embeds[i:i + 1]
            d = self.dict_embeds.to(device)
            if self.args.retrieve_similarity_func == "cosine":
                q = torch.nn.functional.normalize(q, dim=1)
                d = torch.nn.functional.normalize(d, dim=1)
            sims = torch.mm(q, d.T)[0]

            for b in blocked:
                sims[b] = float("-inf")

            need = len(bad_pos)
            _, repls = torch.topk(sims, need)
            for pos, repl in zip(bad_pos, repls):
                self.candidate_idxs[i, pos] = int(repl.item())
                self.candidate_scores[i, pos] = float(sims[repl].item())
                replaced += 1

        logger.info("_remove_self_match: replaced %d candidates", replaced)

    # ---------- 训练阶段：树 + 全局 ----------
    def _retrieve_with_tree(self, query_embeds):
        sims = self._compute_pairwise_sims(query_embeds)
        tree_candidates = self._collect_tree_candidates()

        topk = self.args.topk
        half_topk = int(topk * self.args.retrieve_tree_ratio)
        n_queries, _ = sims.shape

        final_idxs = np.zeros((n_queries, topk), dtype=np.int64)
        final_scores = np.zeros((n_queries, topk))
        n_tree = np.zeros(n_queries, dtype=np.int64)

        # ---- Step 1: 树候选填充。逐 query 但每次只动几十个元素，开销很小 ----
        for i in range(n_queries):
            tc = tree_candidates[i]
            if len(tc) == 0:
                continue
            tree_unique = np.unique(tc).astype(np.int64)
            tree_sims = sims[i, tree_unique]
            order = np.argsort(-tree_sims)
            tree_unique = tree_unique[order]
            tree_sims = tree_sims[order]

            k = min(len(tree_unique), half_topk)
            if k == 0:
                continue
            n_tree[i] = k
            final_idxs[i, :k] = tree_unique[:k]
            final_scores[i, :k] = tree_sims[:k]

        # ---- Step 2: 批量 top-k（GPU 优先）；输出按相似度降序 ----
        part_sorted, sims_sorted = self._batch_topk_masked(sims, final_idxs, n_tree, topk)

        # ---- Step 3: 把每行前 (topk - n_tree[i]) 个填到树槽位之后 ----
        for i in range(n_queries):
            need = topk - n_tree[i]
            if need <= 0:
                continue
            final_idxs[i, n_tree[i]:] = part_sorted[i, :need]
            final_scores[i, n_tree[i]:] = sims_sorted[i, :need]

        return final_idxs, final_scores

    def _batch_topk_masked(self, sims, masked_idxs, n_tree, topk):
        """对 sims 每行屏蔽 ``masked_idxs[i, :n_tree[i]]`` 后取 top-k。

        优先走 GPU（``args.use_cuda=True`` 且 CUDA 可用）；不可用时退回 numpy。
        返回 ``(top_idx, top_val)``，形状均为 ``(N_q, topk)``，按降序。
        """
        n_queries, n_dict = sims.shape

        if getattr(self.args, "use_cuda", False) and torch.cuda.is_available():
            device = torch.device("cuda")
            sims_g = torch.from_numpy(np.ascontiguousarray(sims)).to(device)

            rows_l, cols_l = [], []
            for i in range(n_queries):
                k = int(n_tree[i])
                if k > 0:
                    rows_l.extend([i] * k)
                    cols_l.extend(masked_idxs[i, :k].tolist())
            if rows_l:
                rows_t = torch.as_tensor(rows_l, device=device, dtype=torch.long)
                cols_t = torch.as_tensor(cols_l, device=device, dtype=torch.long)
                sims_g[rows_t, cols_t] = float("-inf")

            top_vals, top_idx = torch.topk(sims_g, topk, dim=1, largest=True, sorted=True)
            return top_idx.cpu().numpy().astype(np.int64), top_vals.cpu().numpy()

        # CPU 退路：argpartition + 行内排序
        sims_masked = sims.copy()
        for i in range(n_queries):
            k = int(n_tree[i])
            if k > 0:
                sims_masked[i, masked_idxs[i, :k]] = -np.inf

        if topk < n_dict:
            part = np.argpartition(-sims_masked, topk - 1, axis=1)[:, :topk]
        else:
            part = np.tile(np.arange(n_dict), (n_queries, 1))

        rows = np.arange(n_queries)[:, None]
        part_sims = sims_masked[rows, part]
        sort_order = np.argsort(-part_sims, axis=1)
        return (
            np.take_along_axis(part, sort_order, axis=1).astype(np.int64),
            np.take_along_axis(part_sims, sort_order, axis=1),
        )

    def _compute_pairwise_sims(self, query_embeds):
        q = np.asarray(query_embeds)
        d = np.asarray(self.dict_embeds)
        func = self.args.retrieve_similarity_func
        if func == "cosine":
            return cosine_similarity(q, d)
        if func == "dot":
            return np.dot(q, d.T)
        raise ValueError(f"Invalid similarity function: {func}")

    def _collect_tree_candidates(self):
        """对每个 query 收集所有启用关系类型的字典索引。

        返回：``list[ndarray]``，长度 = n_queries；每个元素是该 query 的字典
        索引一维数组（未去重；调用方再做 ``np.unique``）。
        """
        if self.args.tax_aware not in _TAX_AWARE_RELATIONS:
            raise ValueError(f"Invalid tax_aware: {self.args.tax_aware}")
        active = _TAX_AWARE_RELATIONS[self.args.tax_aware]

        getters = {
            "current": self._get_current_cui_ids,
        }
        if self.tree_sim is not None:
            getters["parent"] = self.tree_sim.get_parent_ids
            getters["child"] = self.tree_sim.get_child_ids
            getters["sibling"] = self.tree_sim.get_sibling_ids
        else:
            # tree_sim 不可用的占位，不会被实际调用
            getters["parent"] = lambda cui: []
            getters["child"] = lambda cui: []
            getters["sibling"] = lambda cui: []
        dict_id_to_idx = self._dict_id_to_indices()
        n_queries = len(self.query_ids)

        # 并行获取每种关系下的字典索引
        per_relation = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                name: executor.submit(self._lookup_indices, getters[name], dict_id_to_idx)
                for name in active
            }
            for name, fut in futures.items():
                per_relation[name] = fut.result()

        if self.stage == "train" and "sibling" in per_relation:
            avg = (
                float(np.mean([len(per_relation["sibling"][i]) for i in range(n_queries)]))
                if n_queries else 0.0
            )
            logger.info(f"平均每个查询找到 {avg:.2f} 个兄弟节点候选实体")

        empty = np.empty(0, dtype=np.int64)
        merged = []
        for i in range(n_queries):
            parts = [per_relation[name][i] for name in active]
            merged.append(np.concatenate(parts) if parts else empty)
        return merged

    def _lookup_indices(self, get_related_ids, dict_id_to_idx):
        """对每个 query 调用 ``get_related_ids``，把得到的 ID 映射成字典索引。"""
        result = []
        for qid in self.query_ids:
            collected = []
            for rid in get_related_ids(qid):
                idxs = dict_id_to_idx.get(rid)
                if idxs:
                    collected.extend(idxs)
            result.append(np.asarray(collected, dtype=np.int64))
        return result

    def _dict_id_to_indices(self):
        m = defaultdict(list)
        for i, did in enumerate(self.dict_ids):
            m[did].append(i)
            if getattr(self.args, "fuzzy_cui_match", False):
                for comp in did.replace("+", "|").split("|"):
                    comp = comp.strip()
                    if comp:
                        m[comp].append(i)
        return dict(m)

    def _get_current_cui_ids(self, qid):
        if getattr(self.args, "fuzzy_cui_match", False):
            return [c.strip() for c in qid.replace("+", "|").split("|") if c.strip()]
        return [qid]

    # ---------- 标签 ----------
    def get_labels(self, query_idx, candidate_idxs):
        query_id = self.query_ids[query_idx]
        candidate_ids = self.dict_ids[candidate_idxs]
        return np.array([self._check_label(query_id, c) for c in candidate_ids])

    def _check_label(self, query_id, candidate_id_set):
        if getattr(self.args, "normalize_cui_plus", False):
            query_ids = query_id.replace("+", "|").split("|")
            candidate_norm = candidate_id_set.replace("+", "|")
        else:
            query_ids = query_id.split("|")
            candidate_norm = candidate_id_set
        label = 0
        for q_id in query_ids:
            if q_id in candidate_norm:
                label = 1
                continue
            label = 0
            break

        if label == 0 and self.args.use_tree_similarity:
            candidate_ids = candidate_norm.split("|")
            for q_id in query_ids:
                for cid in candidate_ids:
                    label = max(self.tree_sim.compute_similarity(q_id, cid), label)
        return float(label)
