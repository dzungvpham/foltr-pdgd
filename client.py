import itertools
import numpy as np

from dataset import LetorDataset
from ranker import LinearRanker
from typing import Optional

def create_click_pairs(ranking: list[int], clicks: list[bool]) -> list[tuple[int, int]]:
        clicked = []
        not_clicked = []

        for i in range(len(ranking)):
            if clicks[i]:
                clicked.append(ranking[i])
            else:
                not_clicked.append(ranking[i])

        return list(itertools.product(clicked, not_clicked))

class ClickModel():
    def __init__(self, prob_click: list[int], prob_stop: list[int]):
        if len(prob_click) != len(prob_stop) or len(prob_click) == 0:
            raise ValueError("Invalid input")

        self.prob_click = prob_click
        self.prob_stop = prob_stop
    
    def get_clicks(self, ranking: list[int], relevance: np.ndarray, rng: np.random.Generator) -> list[bool]:
        n = len(ranking)
        clicks = [False] * n
        for i in range(n):
            r = relevance[ranking[i]]
            clicks[i] = rng.random() < self.prob_click[r]
            if clicks[i] and rng.random() < self.prob_stop[r]:
                break

        return clicks


class Client():

    def __init__(self, ranker: LinearRanker, dataset: LetorDataset, click_model: ClickModel,
                 rng: np.random.Generator, num_queries: int = 1, learning_rate: float = 0.1):

        self.ranker = ranker
        self.dataset = dataset
        self.click_model = click_model
        self.rng = rng
        self.num_queries = num_queries
        self.learning_rate = learning_rate

    def train_model(self, parameters: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
        if parameters is not None:
            self.ranker.set_parameters(parameters)

        num_clicked = 0
        queries = self.rng.choice(self.dataset.qids, size=self.num_queries, replace=False)
        for query in queries:
            X, R = self.dataset.get_data_for_qid(query)
            ranking = self.ranker.rank_data(X)
            clicks = self.click_model.get_clicks(ranking, R, self.rng)
            click_pairs = create_click_pairs(ranking, clicks)
            if len(click_pairs) == 0:
                continue

            grad = self.ranker.calculate_gradient(X, ranking, click_pairs)
            self.ranker.add_to_parameters(self.learning_rate * grad)
            num_clicked += sum(clicks)

        return (self.ranker.get_parameters(), num_clicked)

    