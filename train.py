import numpy as np

from client import Client, ClickModel
from dataset import LetorDataset
from numpy.random import default_rng
from ranker import LinearRanker

click_models = {
    "MQ2007_perfect":
        ClickModel(prob_click=[0.0, 0.2, 0.4, 0.8, 1.0],
                   prob_stop=[0.0, 0.0, 0.0, 0.0, 0.0]),
    "MQ2007_navigational":
        ClickModel(prob_click=[0.05, 0.3, 0.5, 0.7, 0.95],
                   prob_stop=[0.2, 0.3, 0.5, 0.7, 0.9]),
    "MQ2007_informational":
        ClickModel(prob_click=[0.4, 0.6, 0.7, 0.8, 0.9],
                   prob_stop=[0.1, 0.2, 0.3, 0.4, 0.5]),
}

config = {
    "data_path": "./data/MQ2007/",
    "num_clients": 2,
    "display_cnt": 10,
    "num_queries_per_round": 4,
    "num_rounds": 100,
    "click_model": "MQ2007_perfect",
    "learning_rate": 0.1,
}


def train(config):
    dataset = LetorDataset(config["data_path"] + "Fold1/train.txt")
    num_clients = config["num_clients"]
    clients = []
    for i in range(num_clients):
        rng = default_rng(seed=i)
        ranker = LinearRanker(num_features=dataset.get_num_features(),
                              rank_cnt=config["display_cnt"], rng=rng)
        clients.append(Client(ranker, dataset, click_models[config["click_model"]], rng,
                              num_queries=config["num_queries_per_round"],
                              learning_rate=config["learning_rate"]))

    master_model = LinearRanker(num_features=dataset.get_num_features(
    ), rank_cnt=config["display_cnt"], rng=default_rng())

    for _ in range(config["num_rounds"]):
        total_clicks = 0
        new_params = np.zeros_like(master_model.get_parameters())

        for client in clients:
            client_params, num_clicks = client.train_model(
                master_model.get_parameters())
            new_params += client_params * num_clicks
            total_clicks += num_clicks

        if total_clicks == 0:
            continue

        master_model.set_parameters(new_params / total_clicks)


if __name__ == "__main__":
    train(config)
