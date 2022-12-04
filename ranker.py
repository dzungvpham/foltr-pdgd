import numpy as np

from copy import copy
from scipy.special import logsumexp, softmax


def flip_ranking(ranking: list[int], clicked_idx: int, not_clicked_idx: int) -> list[int]:
    clicked_ranking_idx = None
    not_clicked_ranking_idx = None
    for i in range(len(ranking)):
        if ranking[i] == clicked_idx:
            clicked_ranking_idx = i
        elif ranking[i] == not_clicked_idx:
            not_clicked_ranking_idx = i
    ranking[clicked_ranking_idx] = not_clicked_idx
    ranking[not_clicked_ranking_idx] = clicked_idx


def calculate_ranking_score(ranking: list[int], scores: np.ndarray) -> float:
    res = 0.0
    for i in range(len(ranking)):
        res += logsumexp(scores[ranking[i:]])
    return -res


class LinearRanker():
    def __init__(self, num_features: int, rank_cnt: int, rng: np.random.Generator):
        self.W = np.zeros((num_features, 1))
        self.rank_cnt = rank_cnt
        self.rng = rng

    def get_parameters(self):
        return self.W

    def set_parameters(self, params: np.ndarray):
        self.W = params.copy()

    def add_to_parameters(self, delta: np.ndarray):
        self.W += delta

    def forward(self, X: np.ndarray):
        return X @ self.W

    def rank_data(self, X: np.ndarray) -> list[int]:
        scores = self.forward(X)
        ranking = []
        candidates = np.arange(X.shape[0])
        remaining_cnt = min(self.rank_cnt, X.shape[0])

        while remaining_cnt > 0:
            scores[ranking] = np.NINF
            probs = softmax(scores).reshape(-1)
            size = min(remaining_cnt, np.sum(probs > 0.0))
            ranking += self.rng.choice(candidates, size=size,
                                       p=probs, replace=False, shuffle=False).tolist()
            remaining_cnt -= size

        return ranking

    def calculate_gradient(
            self, X: np.ndarray, ranking: list[int], click_pairs: list[tuple[int, int]]) -> np.ndarray:

        grad = np.zeros_like(self.W)  # Initialize gradient

        # Calculate Pr(Ranking | Data)
        fX = self.forward(X)
        fXe = np.exp(fX)
        score_ranking = calculate_ranking_score(ranking, fX)

        ranking_flipped = copy(ranking)
        for clicked_idx, not_clicked_idx in click_pairs:
            # Calculate pair weight
            flip_ranking(ranking_flipped, clicked_idx, not_clicked_idx)
            score_flipped = calculate_ranking_score(ranking_flipped, fX)
            # Pr of flipped ranking / (Pr of original ranking + Pr of flipped ranking)
            weight = np.exp(score_flipped - logsumexp([score_ranking, score_flipped]))

            # Calculate pair gradient
            e1 = fXe[clicked_idx].item()
            e2 = fXe[not_clicked_idx].item()
            # (e1 * e2) / ((e1 + e2) ** 2)
            weight *= np.exp(np.log(e1) + np.log(e2) - 2 * np.log(e1 + e2))
            # f'(clicked) - f'(not_clicked)
            weight *= (X[clicked_idx] - X[not_clicked_idx]).reshape((-1, 1))

            # Add to model gradient
            grad += weight

            # Unflip ranking
            flip_ranking(ranking_flipped, clicked_idx, not_clicked_idx)

        return grad
