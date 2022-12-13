import matplotlib.pyplot as plt
import numpy as np

from client import Client, ClickModel
from dataset import LetorDataset
from eval import Evaluator
from numpy.random import default_rng
from ranker import LinearRanker
from scipy.special import softmax
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from tqdm import tqdm


class Metrics():
    def __init__(self):
        self.metrics = {}

    def add_metrics(self, name: str):
        self.metrics[name] = {
            "n_preds": [],
            "acc": [],
            "bacc": [],
            "prec": [],
            "recall": []
        }

    def update_metrics(self, name: str, target: list[bool], preds: list[bool]):
        if not name in self.metrics:
            self.add_metrics(name)

        metrics = self.metrics[name]
        metrics["n_preds"].append(np.sum(preds))
        metrics["acc"].append(accuracy_score(target, preds))
        metrics["bacc"].append(balanced_accuracy_score(target, preds))
        if np.sum(preds) > 0:
            metrics["prec"].append(precision_score(target, preds))
        metrics["recall"].append(recall_score(target, preds))

    def print_metrics(self):
        for metrics, vals in self.metrics.items():
            print("{} average # of predictions: {:.2f} | accuracy: {:.2f} | balanced acc: {:.2f} | precision: {:.2f} | recall: {:.2f}".format(
                metrics, np.mean(vals["n_preds"]), np.mean(vals["acc"]),
                np.mean(vals["bacc"]), np.mean(vals["prec"]), np.mean(vals["recall"])))


click_models = {
    "MSLR10k_perfect":
        ClickModel(prob_click=[0.0, 0.2, 0.4, 0.8, 1.0],
                   prob_stop=[0.0, 0.0, 0.0, 0.0, 0.0]),
    "MSLR10k_navigational":
        ClickModel(prob_click=[0.05, 0.3, 0.5, 0.7, 0.95],
                   prob_stop=[0.2, 0.3, 0.5, 0.7, 0.9]),
    "MSLR10k_informational":
        ClickModel(prob_click=[0.4, 0.6, 0.7, 0.8, 0.9],
                   prob_stop=[0.1, 0.2, 0.3, 0.4, 0.5]),
    "MQ2007_perfect":
        ClickModel(prob_click=[0.0, 0.5, 1.0],
                   prob_stop=[0.0, 0.0, 0.0]),
    "MQ2007_navigational":
        ClickModel(prob_click=[0.05, 0.5, 0.95],
                   prob_stop=[0.2, 0.5, 0.9]),
    "MQ2007_informational":
        ClickModel(prob_click=[0.4, 0.7, 0.9],
                   prob_stop=[0.1, 0.3, 0.5]),
}

data_path = "./data/MSLR-WEB10k/Fold1/vali.txt"
params_path = "./models/MSLR10k_perfect_nclients1000_nround100_nquery2_lr0.1_sens3_eps1.2.npy"

rng = default_rng(1)
rank_cnt = 10
num_clients = 1000
num_queries = 1
lr = 0.1
sensitivity = 3
epsilon = 1.2

dataset = LetorDataset(data_path, normalize=True)
evaluator = Evaluator(dataset, rank_cnt=rank_cnt)
num_features = dataset.get_num_features()
master_ranker = LinearRanker(
    num_features=num_features, rank_cnt=rank_cnt, rng=rng)
client_ranker = LinearRanker(
    num_features=num_features, rank_cnt=rank_cnt, rng=rng)

params = np.load(params_path)
master_ranker.set_parameters(params)

client = Client(client_ranker, dataset, click_models["MSLR10k_perfect"],
                evaluator, rng,
                num_clients=num_clients, num_queries=num_queries,
                learning_rate=lr,
                enable_dp=True, sensitivity=sensitivity, epsilon=epsilon)

metrics = Metrics()
n_queries = 0

for qid in tqdm(dataset.qids):
    X = dataset.get_data_for_qid(qid)[0]
    if (X.shape[0] < 10):
        continue

    client_params, _, _, maybe_clicks = client.train_model(
        params, [qid], sample=False, get_clicks=True)

    clicks = []
    if maybe_clicks is not None:
        clicks = maybe_clicks[0]
    if not np.any(clicks):
        continue

    master_ranker.set_parameters(params)
    original_ranking = master_ranker.rank_data(X, sample=False)
    original_scores = master_ranker.forward(X).reshape(-1)[original_ranking]
    original_probs = softmax(original_scores)

    master_ranker.set_parameters(client_params)
    X = X[original_ranking]
    new_ranking = master_ranker.rank_data(X, sample=False)
    new_scores = master_ranker.forward(X).reshape(-1)[new_ranking]
    new_probs = softmax(new_scores)

    n_queries += 1
    n = len(original_ranking)

    # Random guesses
    for k in range(1, n + 1):
        guess = np.array([False] * n)
        guess[rng.choice(n, size=k, replace=False, shuffle=False)] = True
        metrics_name = "Random guess " + str(k)
        metrics.update_metrics(metrics_name, clicks, guess.tolist())

    # Guess first position
    guess = [False] * n
    guess[0] = True
    metrics.update_metrics("FIRST_POS", clicks, guess)

    # Method 1
    guess = new_scores > original_scores
    metrics.update_metrics("HIGHER_SCORE", clicks, guess)

    # Method 1b
    guess = new_probs > original_probs
    metrics.update_metrics("HIGHER_PROB", clicks, guess)

    # Method 2
    guess = [False] * n
    guess[np.argmax(new_probs - original_probs)] = True
    metrics.update_metrics("MAX_PROB_DIFF", clicks, guess)

    # Method 3
    guess = [False] * n
    for i in range(n):
        original_diff = []
        new_diff = []
        for j in range(n):
            if i == j:
                continue
            original_diff.append(original_scores[i] - original_scores[j])
            new_diff.append(new_scores[i] - new_scores[j])

        higher_diff_cnt = 0
        for k in range(len(original_diff)):
            if new_diff[k] > original_diff[k]:
                higher_diff_cnt += 1

        if higher_diff_cnt >= (rank_cnt / 2):
            guess[i] = True
    metrics.update_metrics("DIFF_COUNT", clicks, guess)

    # Method 4
    guess = [False] * n
    reg = LinearRegression(fit_intercept=False).fit(X.T, (client_params - params).reshape(-1))
    for i in range(n):
        if reg.coef_[i] > 0.0:
            guess[i] = True
    metrics.update_metrics("REGRESSION", clicks, guess)


print("Number of eligible queries: {}".format(n_queries))

metrics.print_metrics()
