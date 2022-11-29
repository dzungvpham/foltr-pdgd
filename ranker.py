import numpy as np

from copy import copy


def create_flipped_ranking(ranking: list[int], clicked_idx: int, not_clicked_idx: int) -> list[int]:
    ranking_switched = copy(ranking)
    clicked_ranking_idx = None
    not_clicked_ranking_idx = None
    for i in range(len(ranking)):
        if ranking[i] == clicked_idx:
            clicked_ranking_idx = i
        elif ranking[i] == not_clicked_idx:
            not_clicked_ranking_idx = i
    ranking_switched[clicked_ranking_idx] = not_clicked_idx
    ranking_switched[not_clicked_ranking_idx] = clicked_idx

    return ranking_switched


def calculate_ranking_score(ranking: list[int], values: np.ndarray, values_sum: float) -> float:
    prob_ranking_denom = 0
    for r in ranking:
        prob_ranking_denom += np.log(values_sum)
        values_sum -= values[r].item()
    return np.exp(-prob_ranking_denom)


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
        n = X.shape[0]
        scores = self.forward(X)
        remaining_cnt = min(self.rank_cnt, n)
        ranking = []
        while remaining_cnt > 0:
            scores[ranking] = np.NINF  # Mask already selected items
            scores -= np.max(scores)  # Prevent overflow
            exp_scores = np.exp(scores)
            probs = (exp_scores / np.sum(exp_scores)).reshape(-1)
            size = np.minimum(remaining_cnt, np.sum(probs > 0))
            ranking += self.rng.choice(np.arange(n), size=size,
                                       p=probs, replace=False, shuffle=False).tolist()
            remaining_cnt -= size

        return ranking

    def calculate_gradient(
            self, X: np.ndarray, ranking: list[int], click_pairs: list[tuple[int, int]]) -> np.ndarray:

        grad = np.zeros_like(self.W) # Initialize gradient

        # Calculate Pr(Ranking | Data)
        fX = self.forward(X)
        fXe = np.exp(fX)
        fXe_sum = np.sum(fXe)
        score_ranking = calculate_ranking_score(ranking, fXe, fXe_sum)

        for clicked_idx, not_clicked_idx in click_pairs:
            # Pair weight
            ranking_flipped = create_flipped_ranking(
                ranking, clicked_idx, not_clicked_idx)
            score_flipped = calculate_ranking_score(
                ranking_flipped, fXe, fXe_sum)
            weight = score_flipped / (score_ranking + score_flipped)
            
            # Pair gradient
            e1 = fXe[clicked_idx].item()
            e2 = fXe[not_clicked_idx].item()
            weight *= (e1 * e2) / ((e1 + e2) ** 2)
            
            # Model gradient
            grad += weight * (X[clicked_idx] -
                              X[not_clicked_idx]).reshape((-1, 1))

        return grad
