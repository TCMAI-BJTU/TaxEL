from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional
import faiss
import networkx as nx
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from tqdm import tqdm
from src.logger.logger import setup_logger


class CandidateDataset(Dataset):
    """Dataset class for candidate retrieval and scoring."""

    def __init__(self, args, queries, dicts, stage, shared_tools):
        """Initialize the dataset with queries and dictionary entries."""
        self.logger = setup_logger(args.log_file)

        # Configuration parameters
        self._init_config(args, stage)

        # Data initialization
        self._init_data(queries, dicts)

        # Initialize components
        self._init_tools(shared_tools)

        self.logger.info("CandidateDataset initialization completed")

    def _init_config(self, args, stage):
        """Initialize configuration parameters."""
        self.max_length = args.max_length
        self.model_name_or_path = args.model_name_or_path
        self.use_cuda = args.use_cuda
        self.topk = args.topk
        self.last_layer = args.last_layer
        self.use_tree_similarity = args.use_tree_similarity
        self.dataset_name_or_path = args.dataset_name_or_path
        self.retrieve_similarity_func = args.retrieve_similarity_func
        self.embed_dim = args.embed_dim
        self.tree_ratio = args.tree_ratio
        self.stage = stage
        self.use_graph_propagation = args.use_graph_propagation
        self.graph_alpha = args.graph_alpha
        self.graph_threshold = args.graph_threshold
        self.retrieve_tree_ratio = args.retrieve_tree_ratio
        # self.only_current = args.only_current
        self.tax_aware = args.tax_aware

    def _init_tools(self, shared_tools):
        """Initialize tools."""
        self.encoder = shared_tools.encoder
        self.tree_sim = shared_tools.tree_sim
        self.tokenizer = shared_tools.tokenizer

    def _init_data(self, queries, dicts):
        """Initialize query and dictionary data."""
        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.dict_ids = np.array(self.dict_ids)
        self.candidate_idxs = None
        self.dict_embeds = None

    # Dataset interface methods
    def __len__(self):
        return len(self.query_names)

    def __getitem__(self, query_idx):
        """Get a single item from the dataset."""
        assert self.candidate_idxs is not None

        # Prepare query
        query_name = self.query_names[query_idx]
        query_token = self._tokenize_text(query_name)

        # Prepare candidates
        topk_candidate_idxs = self.candidate_idxs[query_idx]
        assert len(topk_candidate_idxs) == self.topk

        candidate_names = [self.dict_names[i] for i in topk_candidate_idxs]
        candidate_tokens = self._tokenize_text(candidate_names)

        # Get labels
        labels = self.get_labels(query_idx, topk_candidate_idxs)

        return (query_token, candidate_tokens), labels

    def _tokenize_text(self, text):
        """Tokenize text using the tokenizer."""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    # Embedding and retrieval methods
    def embed_names(self, names, show_progress_bar=False):
        """Embed names using the encoder."""
        self.encoder.eval()
        if len(names) < 100000:
            embeddings = self.encoder.encode(names, show_progress_bar=True, batch_size=1024)
        else:
            pool = self.encoder.start_multi_process_pool()
            embeddings = self.encoder.encode_multi_process(names, pool, show_progress_bar=show_progress_bar, batch_size=4096)
            self.encoder.stop_multi_process_pool(pool)
        return torch.tensor(embeddings)

    def set_candidate_idxs(self, dict_embeds=None):
        """Set candidate indices based on embeddings."""
        query_embeds = self.embed_names(self.query_names)

        self.dict_embeds = self.embed_names(self.dict_names, show_progress_bar=True) if dict_embeds is None else dict_embeds
        if dict_embeds is not None:
            self.logger.info("Using pre-computed candidate embeddings")

        if self.stage not in ["train", "test"]:
            raise ValueError(f"Invalid stage: {self.stage}")

        # Both train and test stages use the same retrieval logic
        # if self.use_graph_propagation and self.stage == "train":
        #     candidate_idxs, candidate_scores = self.retrieve_candidate_with_graph(query_embeds, topk=self.topk)
        # else:
        #     candidate_idxs, candidate_scores = self.retrieve_candidate(query_embeds, topk=self.topk)

        # 根据本体树结构筛选候选实体，先筛选出所有上位词和下位词，再排序
        if self.stage == "train":
            candidate_idxs, candidate_scores = self.retrieve_candidate_with_tree(query_embeds, topk=self.topk)
            # old_candidate_idxs, _ = self.retrieve_candidate(query_embeds, topk=self.topk)
            # self.save_difference(old_candidate_idxs, candidate_idxs)
            # exit()
        else:
            candidate_idxs, candidate_scores = self.retrieve_candidate(query_embeds, topk=self.topk)

        self.candidate_idxs = candidate_idxs
        self.candidate_scores = candidate_scores

    # Similarity computation methods
    def retrieve_candidate(self, query_embeds, topk):
        """Retrieve candidates using FAISS."""
        query_embeds = np.array(query_embeds)
        dict_embeds = np.array(self.dict_embeds)

        if self.retrieve_similarity_func != "cosine":
            faiss.normalize_L2(query_embeds)
            faiss.normalize_L2(dict_embeds)
        if self.use_cuda:
            # 构建Faiss的GPU索引
            gpu_resources = faiss.StandardGpuResources()  # 创建GPU资源
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = False  # 如果你想进一步加速，可以将其设为True
            flat_config.device = 0  # 使用GPU 0
            index = faiss.GpuIndexFlatIP(gpu_resources, self.embed_dim, flat_config)
        else:
            index = faiss.IndexFlatIP(self.embed_dim)

        index.add(dict_embeds.astype('float32'))  # type: ignore

        # 计算每个查询向量与600万个候选向量的余弦相似度，获取前k个最相似的候选向量
        topk_sim, topk_idx = index.search(query_embeds.astype('float32'), topk)  # type: ignore
        return topk_idx, topk_sim

    def get_dict_id_to_indices(self):
        """Create a dictionary mapping dict_ids to their indices."""
        dict_id_to_indices = defaultdict(list)
        for idx, dict_id in enumerate(self.dict_ids):
            dict_id_to_indices[dict_id].append(idx)
        dict_id_to_indices = dict(dict_id_to_indices)
        return dict_id_to_indices

    def retrieve_candidate_with_tree(self, query_embeds, topk):
        """Retrieve candidates using tree-based approach."""
        query_embeds = np.array(query_embeds)

        # Get initial similarity scores for all candidates
        dict_embeds = np.array(self.dict_embeds)
        if self.retrieve_similarity_func == "cosine":
            mention_candidate_similarities = cosine_similarity(query_embeds, dict_embeds)
        elif self.retrieve_similarity_func == "dot":
            mention_candidate_similarities = np.dot(query_embeds, dict_embeds.T)
        else:
            raise ValueError(f"Invalid similarity function: {self.retrieve_similarity_func}")
        # Create dictionary mapping dict_ids to their indices
        dict_id_to_indices = self.get_dict_id_to_indices()

        # 获取父节点、当前节点、子节点的候选实体
        with ThreadPoolExecutor(max_workers=3) as executor:
            child_future = executor.submit(self.retrieve_child_candidates, query_embeds, dict_id_to_indices)
            parent_future = executor.submit(self.retrieve_parent_candidates, query_embeds, dict_id_to_indices)
            current_future = executor.submit(self.retrieve_current_candidates, query_embeds, dict_id_to_indices)

            child_idxs = child_future.result()
            parent_idxs = parent_future.result()
            current_idxs = current_future.result()

        if self.tax_aware == "current":
            child_idxs = np.ones_like(child_idxs) * -1
            parent_idxs = np.ones_like(parent_idxs) * -1
        elif self.tax_aware == "current_parent_child":
            pass
        elif self.tax_aware == "none":
            parent_idxs = np.ones_like(parent_idxs) * -1
            child_idxs = np.ones_like(child_idxs) * -1
            current_idxs = np.ones_like(current_idxs) * -1
        else:
            raise ValueError(f"Invalid tax_aware: {self.tax_aware}")

        # Initialize arrays to store final results
        final_topk_idx = np.zeros((len(query_embeds), topk), dtype=np.int64)
        final_topk_sim = np.zeros((len(query_embeds), topk))

        # Process each query
        # Process queries in parallel using ThreadPoolExecutor
        def process_single_query(i):
            # Get valid indices (not -1) from parent and child candidates
            tree_candidates = np.concatenate(
                [
                    parent_idxs[i][parent_idxs[i] != -1],
                    child_idxs[i][child_idxs[i] != -1],
                    current_idxs[i][current_idxs[i] != -1],
                ]
            )

            # 对tree_candidates进行去重，并转换为int64类型
            tree_candidates = np.unique(tree_candidates).astype(np.int64)

            # 获取tree_candidates的相似度
            tree_similarities = mention_candidate_similarities[i][tree_candidates]

            # 对tree_candidates进行排序，按照相似度从高到低
            sorted_tree_indices = np.argsort(-tree_similarities)  # Descending order
            sorted_tree_candidates = tree_candidates[sorted_tree_indices]
            sorted_tree_similarities = tree_similarities[sorted_tree_indices]

            # 按比例取topk
            half_topk = int(topk * self.retrieve_tree_ratio)
            n_tree_candidates = min(len(sorted_tree_candidates), half_topk)

            # 初始化当前查询的结果数组
            query_topk_idx = np.zeros(topk, dtype=np.int64)
            query_topk_sim = np.zeros(topk)

            # 将tree_candidates的前一半填充到结果数组中
            query_topk_idx[:n_tree_candidates] = sorted_tree_candidates[:n_tree_candidates]
            query_topk_sim[:n_tree_candidates] = sorted_tree_similarities[:n_tree_candidates]

            # 如果需要更多的候选实体来达到topk
            if n_tree_candidates < topk:
                # 排除已经选择的候选实体
                mask = np.ones(len(dict_embeds), dtype=bool)
                mask[query_topk_idx[:n_tree_candidates]] = False
                remaining_similarities = mention_candidate_similarities[i][mask]
                remaining_indices = np.arange(len(dict_embeds))[mask]

                # 对剩余的候选实体进行排序，按照相似度从高到低
                sorted_remaining_indices = np.argsort(-remaining_similarities)
                sorted_remaining_indices = sorted_remaining_indices[: topk - n_tree_candidates]

                # 将剩余的候选实体填充到结果数组中
                query_topk_idx[n_tree_candidates:] = remaining_indices[sorted_remaining_indices]
                query_topk_sim[n_tree_candidates:] = remaining_similarities[sorted_remaining_indices]

            return i, query_topk_idx, query_topk_sim

        # 使用ThreadPoolExecutor并行处理查询
        with ThreadPoolExecutor(max_workers=min(32, len(query_embeds))) as executor:
            futures = [executor.submit(process_single_query, i) for i in range(len(query_embeds))]

            # 使用tqdm显示进度
            for future in as_completed(futures):
                i, query_topk_idx, query_topk_sim = future.result()
                final_topk_idx[i] = query_topk_idx
                final_topk_sim[i] = query_topk_sim

        return final_topk_idx, final_topk_sim

    def retrieve_parent_candidates(self, query_embeds, dict_id_to_indices):
        """Retrieve parent candidates using tree-based approach."""
        parent_idxs = -1 * np.ones((len(query_embeds), len(self.dict_ids)))

        for i in range(len(query_embeds)):
            query_id = self.query_ids[i]
            # Get parent node CUIs
            parent_ids = self.tree_sim.get_parent_ids(query_id)
            cnt = 0
            for parent_id in parent_ids:
                # Get indices for parent ID from dictionary
                if parent_id in dict_id_to_indices:
                    for parent_idx in dict_id_to_indices[parent_id]:
                        parent_idxs[i, cnt] = parent_idx
                        cnt += 1
        return parent_idxs

    def retrieve_current_candidates(self, query_embeds, dict_id_to_indices):
        """Retrieve current candidates using tree-based approach."""
        current_idxs = -1 * np.ones((len(query_embeds), len(self.dict_ids)))
        for i in range(len(query_embeds)):
            query_id = self.query_ids[i]
            cnt = 0
            if query_id in dict_id_to_indices:
                for current_idx in dict_id_to_indices[query_id]:
                    current_idxs[i, cnt] = current_idx
                    cnt += 1
        return current_idxs

    def retrieve_child_candidates(self, query_embeds, dict_id_to_indices):
        """Retrieve child candidates using tree-based approach."""
        child_idxs = -1 * np.ones((len(query_embeds), len(self.dict_ids)))
        for i in range(len(query_embeds)):
            query_id = self.query_ids[i]
            child_ids = self.tree_sim.get_child_ids(query_id)
            cnt = 0
            for child_id in child_ids:
                if child_id in dict_id_to_indices:
                    for child_idx in dict_id_to_indices[child_id]:
                        child_idxs[i, cnt] = child_idx
                        cnt += 1
        return child_idxs

    def retrieve_candidate_with_graph(self, query_embeds, topk):
        """Retrieve candidates using graph-based approach."""
        dict_embeds = np.array(self.dict_embeds)
        mention_candidate_similarities = cosine_similarity(query_embeds, dict_embeds)
        mention_mention_similarities = cosine_similarity(query_embeds, query_embeds)
        # 利用图传播获取更新后的相似度矩阵
        updated_similarity_matrix = self.update_propagated_score(mention_candidate_similarities, mention_mention_similarities)

        self.compare_difference(mention_candidate_similarities, updated_similarity_matrix, topk)

        topk_idx = np.argsort(updated_similarity_matrix, axis=1)[:, -topk:][:, ::-1]
        topk_sim = np.sort(updated_similarity_matrix, axis=1)[:, -topk:][:, ::-1]
        return topk_idx, topk_sim

    def compare_difference(self, original_scores, updated_scores, topk):
        updated_topk_idx = np.argsort(updated_scores, axis=1)[:, -topk:][:, ::-1]
        original_topk_idx = np.argsort(original_scores, axis=1)[:, -topk:][:, ::-1]
        dict_names_np = np.array(self.dict_names)
        name_to_index = {name: idx for idx, name in enumerate(self.dict_names)}

        updated_topk_names = [dict_names_np[row] for row in updated_topk_idx]
        original_topk_names = [dict_names_np[row] for row in original_topk_idx]
        result_str = ""

        for i in range(len(updated_topk_names)):
            diff_names = set(updated_topk_names[i]).symmetric_difference(set(original_topk_names[i]))
            if diff_names:
                query_name = self.query_names[i]
                result_str += f"Query: {query_name}\n"

                for name in diff_names:
                    idx = name_to_index[name]
                    updated_score = updated_scores[i, idx]
                    original_score = original_scores[i, idx]
                    if name in updated_topk_names[i]:
                        score_increase = updated_score - original_score
                        result_str += (
                            f"Added: {name}, Updated Score: {updated_score:.3f}, "
                            f"Original Score: {original_score:.3f}, Increase: {score_increase:.3f}\n"
                        )
                    else:
                        score_decrease = original_score - updated_score
                        result_str += (
                            f"Removed: {name}, Updated Score: {updated_score:.3f}, "
                            f"Original Score: {original_score:.3f}, Decrease: {score_decrease:.3f}\n"
                        )
        with open("graph_propagation_result.txt", "w") as f:
            f.write(result_str)

    def update_propagated_score(self, original_scores, similarity_matrix):
        def disparity_filter(graph, alpha, gamma=2):
            """
            使用显著性阈值过滤图中的边。
            """
            filtered_graph = nx.Graph()
            for node in graph.nodes:
                neighbors = list(graph.neighbors(node))
                total_weight = sum(graph[node][neighbor]["weight"] for neighbor in neighbors)
                k = len(neighbors)
                if k <= 1:
                    continue  # 跳过孤立节点

                for neighbor in neighbors:
                    weight = graph[node][neighbor]["weight"]
                    normalized_weight = (weight / total_weight) ** gamma
                    p_value = 1 - (1 - normalized_weight) ** (k - 1)
                    if p_value <= alpha:  # 显著性阈值筛选
                        filtered_graph.add_edge(node, neighbor, weight=weight)
            return filtered_graph

        num_mentions = len(similarity_matrix)
        alpha = self.graph_alpha  # 原始得分的权重
        threshold = self.graph_threshold  # 相似度阈值

        # 确保 similarity_matrix 对称并去掉自身连接
        similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.T)
        np.fill_diagonal(similarity_matrix, 0)

        # 构建连通图
        graph = nx.Graph()
        for i in range(num_mentions):
            for j in range(i + 1, num_mentions):
                if similarity_matrix[i, j] > threshold:
                    graph.add_edge(i, j, weight=similarity_matrix[i, j])

        print("Number of nodes:", graph.number_of_nodes())
        print("Number of edges:", graph.number_of_edges())
        print("Number of connected components:", nx.number_connected_components(graph))

        # 过滤图
        filtered_graph = disparity_filter(graph, alpha=0.5)
        print("Filtered Graph: Nodes:", filtered_graph.number_of_nodes(), "Edges:", filtered_graph.number_of_edges())

        neighbors_dict = {node: list(filtered_graph.neighbors(node)) for node in filtered_graph.nodes}
        weights_dict = {}
        for node, neighbors in neighbors_dict.items():
            total_weight = sum(filtered_graph[node][neighbor]["weight"] for neighbor in neighbors)
            if total_weight > 0:
                weights_dict[node] = np.array([filtered_graph[node][neighbor]["weight"] / total_weight for neighbor in neighbors])
            else:
                weights_dict[node] = np.zeros(len(neighbors))

        # 初始化新得分矩阵
        updated_scores = original_scores.copy()

        # 遍历每个提及并更新候选实体得分
        for mention_id in range(num_mentions):
            neighbors = neighbors_dict.get(mention_id, [])
            if not neighbors:
                continue  # 如果没有邻居，则保留原始得分

            # 获取邻居相关的分数和权重
            neighbor_ids = np.array(neighbors)
            neighbor_scores = original_scores[neighbor_ids]  # 获取邻居的候选实体得分
            valid_neighbors = neighbor_scores.sum(axis=1) > 0  # 过滤掉无效邻居
            neighbor_scores = neighbor_scores[valid_neighbors]
            weights = weights_dict[mention_id][valid_neighbors]

            # 传播得分矩阵计算
            propagated_scores = np.dot(weights, neighbor_scores)  # 矩阵乘法传播得分
            updated_scores[mention_id] = alpha * original_scores[mention_id] + (1 - alpha) * propagated_scores

        return updated_scores

    # Label computation methods
    def check_label(self, query_id, candidate_id_set):
        """Check if query and candidate match."""
        query_ids = query_id.split("|")
        label = 0

        # Check direct matches
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break

        # Check tree similarity if enabled
        if label == 0 and self.use_tree_similarity:
            candidate_ids = candidate_id_set.split("|")
            for q_id in query_ids:
                for candidate_id in candidate_ids:
                    label = max(self.tree_sim.compute_similarity(q_id, candidate_id), label)
        return label

    def get_labels(self, query_idx, candidate_idxs):
        """Get labels for candidates."""
        query_id = self.query_ids[query_idx]
        candidate_ids = self.dict_ids[candidate_idxs]
        return np.array([self.check_label(query_id, candidate_id) for candidate_id in candidate_ids])

    def save_difference(self, old_candidate_idxs, new_candidate_idxs):
        results = []
        for i in range(len(self.query_names)):
            unchanged_idxs = set(old_candidate_idxs[i]) & set(new_candidate_idxs[i])
            added_idxs = set(new_candidate_idxs[i]) - set(old_candidate_idxs[i])
            removed_idxs = set(old_candidate_idxs[i]) - set(new_candidate_idxs[i])
            if not added_idxs:
                continue
            unchanged_results = [f"{self.dict_names[i]}|{self.dict_ids[i]}" for i in unchanged_idxs]
            added_results = [f"{self.dict_names[i]}|{self.dict_ids[i]}" for i in added_idxs]
            removed_results = [f"{self.dict_names[i]}|{self.dict_ids[i]}" for i in removed_idxs]
            results.append(f"Query: {self.query_names[i]}; {self.query_ids[i]}\n")
            results.append(f"Unchanged candidates: {unchanged_results}\n")
            results.append(f"Added candidates: {added_results}\n")
            results.append(f"Removed candidates: {removed_results}\n")
            results.append(f"*" * 100 + "\n")

        with open(f"candidate_differences.txt", "w") as f:
            f.write("\n".join(results))
