import matplotlib.pyplot as plt
import numpy as np

from client import Client, ClickModel
from copy import deepcopy
from dataset import LetorDataset
from eval import Evaluator
from numpy.random import default_rng
from ranker import LinearRanker

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

config = {
    "data_path": "./data/MSLR-WEB10K/",
    "save_path": "./models/",
    "plot_path": "./plots/",
    "fold": 1,
    "num_clients": 1000,
    "rank_cnt": 10,
    "num_queries_per_round": 2,
    "num_rounds": 100,
    "click_model": "MSLR10k_perfect",
    "learning_rate": 0.1,
    "online_eval_discount": 0.9995,
    "enable_dp": True,
    "sensitivity": 3,
    "epsilon": 1.2,
}


def get_base_save_name_from_config(config):
    return "{}_nclients{}_nround{}_nquery{}_lr{}_sens{}_eps{}".format(
        config["click_model"], config["num_clients"], config["num_rounds"],
        config["num_queries_per_round"], config["learning_rate"],
        config["sensitivity"], config["epsilon"]
    )


def save_plot_ndcg(test_ndcgs, online_ndcgs, rank_cnt, path, title, show=False):
    n = len(test_ndcgs)
    plt.plot(np.arange(n), test_ndcgs,
             "-r", label="Test nDCG@{}".format(rank_cnt))
    plt.plot(np.arange(n), online_ndcgs, "--b",
             label="Online nDCG@{}".format(rank_cnt))
    plt.xlabel("Round")
    plt.ylabel("nDCG@{}".format(rank_cnt))
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.clf()


def train(config, train_dataset=None, test_dataset=None, plot=True):
    num_clients = config["num_clients"]
    rank_cnt = config["rank_cnt"]
    online_eval_discount = config["online_eval_discount"]

    data_path = config["data_path"]
    fold = config["fold"]

    if train_dataset is None:
        train_dataset = LetorDataset(
            "{}/Fold{}/train.txt".format(data_path, fold), normalize=True)
        test_dataset = LetorDataset("{}/Fold{}/test.txt".format(data_path, fold), normalize=True,
                                    norm_mean=train_dataset.norm_mean, norm_std=train_dataset.norm_std)
    train_evaluator = Evaluator(train_dataset, rank_cnt=rank_cnt)
    test_evaluator = Evaluator(
        test_dataset, rank_cnt=rank_cnt, online_discount=online_eval_discount)

    clients = []
    num_features = train_dataset.get_num_features()
    for i in range(num_clients):
        rng = default_rng(seed=num_clients * fold + i)
        ranker = LinearRanker(num_features=num_features,
                              rank_cnt=rank_cnt, rng=rng)
        clients.append(Client(ranker, train_dataset, click_models[config["click_model"]],
                              train_evaluator, rng,
                              num_clients=num_clients,
                              num_queries=config["num_queries_per_round"],
                              learning_rate=config["learning_rate"],
                              enable_dp=config["enable_dp"],
                              sensitivity=config["sensitivity"],
                              epsilon=config["epsilon"]))

    master_model = LinearRanker(
        num_features=num_features, rank_cnt=rank_cnt, rng=default_rng())

    print("Round 0: Test nDCG@{} = {}".format(rank_cnt,
                                              test_evaluator.calculate_average_offline_ndcg(master_model)))
    test_ndcgs = []
    avg_online_ndcgs = []

    save_name = "fold{}_{}".format(
        fold, get_base_save_name_from_config(config))

    for round in range(config["num_rounds"]):
        total_clicks = 0
        new_params = np.zeros_like(master_model.get_parameters())
        online_ndcgs = []

        for client in clients:
            client_params, num_clicks, online_ndcg, _ = client.train_model(
                master_model.get_parameters())
            new_params += client_params * num_clicks
            total_clicks += num_clicks
            online_ndcgs.append(online_ndcg)

        if total_clicks == 0:
            continue

        master_model.set_parameters(new_params / total_clicks)

        # Eval
        test_ndcg = test_evaluator.calculate_average_offline_ndcg(master_model)
        test_ndcgs.append(test_ndcg)
        avg_online_ndcgs.append(np.mean(online_ndcgs))
        print("Round {}: Test nDCG@{} = {:.4f} | Online nDCG@{} = {:.4f}".format(
            round + 1, rank_cnt, test_ndcg, rank_cnt, np.mean(online_ndcgs)))

        # Save model
        np.save("{}/{}.npy".format(config["save_path"],
                save_name), master_model.get_parameters())

    if plot:
        save_plot_ndcg(test_ndcgs, avg_online_ndcgs, rank_cnt,
                       "{}/ndcg_{}.png".format(config["plot_path"], save_name), config["click_model"], show=True)

    return test_ndcgs, avg_online_ndcgs


def train_multiple(config):
    models = ["MSLR10k_perfect", "MSLR10k_navigational", "MSLR10k_informational"]
    folds = [1, 2, 3, 4, 5]
    data_path = config["data_path"]
    rank_cnt = config["rank_cnt"]

    all_test_ndcgs = {}
    all_online_ndcgs = {}
    for click_model in models:
        all_test_ndcgs[click_model] = {}
        all_online_ndcgs[click_model] = {}

    for fold in folds:
        print("Training on fold {}".format(fold))

        train_dataset = LetorDataset(
            "{}/Fold{}/train.txt".format(data_path, fold), normalize=True)
        test_dataset = LetorDataset("{}/Fold{}/test.txt".format(data_path, fold), normalize=True,
                                    norm_mean=train_dataset.norm_mean, norm_std=train_dataset.norm_std)

        for click_model in models:
            print(click_model)

            new_config = deepcopy(config)
            new_config["click_model"] = click_model
            new_config["fold"] = fold

            test_ndcgs, online_ndcgs = train(
                new_config, train_dataset, test_dataset, plot=False)

            all_test_ndcgs[click_model][fold] = test_ndcgs
            all_online_ndcgs[click_model][fold] = online_ndcgs

    for click_model in models:
        avg_test_ndcgs = np.array(
            list(all_test_ndcgs[click_model].values())).mean(axis=0)
        avg_online_ndcgs = np.array(
            list(all_online_ndcgs[click_model].values())).mean(axis=0)

        new_config = deepcopy(config)
        new_config["click_model"] = click_model
        base_save_name = get_base_save_name_from_config(new_config)
        save_plot_ndcg(avg_test_ndcgs, avg_online_ndcgs, rank_cnt,
                       "{}/cv_ndcg_{}.png".format(config["plot_path"], base_save_name), click_model)


if __name__ == "__main__":
    train_multiple(config)
