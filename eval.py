import numpy as np

from dataset import LetorDataset
from ranker import LinearRanker


class Evaluator():
    def __init__(self, dataset: LetorDataset, rank_cnt: int, online_discount: float):
        self._dataset = dataset
        self._rank_cnt = rank_cnt
        self._online_discount = online_discount
        self._idcg_map = {}
        self._dcg_weight = 1/np.log2(np.arange(2, rank_cnt + 2))

        # Pre-populate ideal DCG for all queries
        for qid in dataset.qids:
            _, relevance = dataset.get_data_for_qid(qid)
            relevance.sort()  # Ascending
            # Reverse and grab top k
            relevance = relevance[:-(self._rank_cnt+1):-1]
            self._idcg_map[qid] = np.sum(
                (2 ** relevance - 1) * self._dcg_weight[:self._rank_cnt])

    def calculate_average_offline_ndcg(self, ranker: LinearRanker) -> float:
        total_dcg = 0.0
        for qid in self._dataset.qids:
            if self._idcg_map[qid] == 0.0:
                continue

            X, relevance = self._dataset.get_data_for_qid(qid)
            scores = ranker.forward(X).reshape(-1)
            ranking = np.argsort(scores)[::-1]  # Sort the indices then reverse
            total_dcg += self.calculate_ndcg_for_query_ranking(
                qid, ranking, relevance)

        return total_dcg / len(self._dataset.qids)

    def calculate_average_online_ndcg(self, inputs: list[tuple[str, list[int], np.ndarray]]) -> float:
        gamma = 1.0
        ndcg = 0.0
        for qid, ranking, relevance in inputs:
            ndcg += self.calculate_ndcg_for_query_ranking(
                qid, ranking, relevance) * gamma
            gamma *= self._online_discount
        return ndcg / len(inputs)

    def calculate_ndcg_for_query_ranking(self, qid: str, ranking: list[int], relevance: np.ndarray) -> float:
        idcg = self._idcg_map[qid]
        if idcg == 0.0:
            return 0.0

        ranking = ranking[:self._rank_cnt]
        ranking_relevance = relevance[ranking]
        n = len(ranking)
        dcg = np.sum((2 ** ranking_relevance - 1) * self._dcg_weight[:n])
        return dcg / idcg
