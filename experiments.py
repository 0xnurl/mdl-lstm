import json
import math
import pathlib

import torch
from loguru import logger

import corpora
import encoding
import golden_an_bn_net
import grid
import networks
import utils

_CORPUS_GRID = {
    "p": (0.3,),
}

GRID = {
    "corpus_params": _CORPUS_GRID,
    "training_params": grid.TRAINING_GRID,
}

EVAL_GRID = {
    "test_max_n": (1000,),
    "quantize": (
        True,
        False,
    ),
    "compress_g": (True, False),
}

_SIMULATIONS_PATH = pathlib.Path("./simulations/")


def iterate_grid_non_unique(grid):
    # This doesn't reduce the iterations to unique configs (but id will lead to correct caching).
    for kwargs in utils.iterate_grid_combinations(grid):
        if kwargs["training_params"]["regularization"] is None:
            kwargs["training_params"]["regularization_lambda"] = None
        yield kwargs


def evaluate(net: networks.LSTM, corpus: corpora.Corpus, eval_params: dict) -> dict:
    net.eval()
    if eval_params["quantize"]:
        net = encoding.get_quantized_net(net)
    g_encoding_length = encoding.get_g_encoding_length(
        net, zip_bitstring=eval_params["compress_g"]
    )
    d_given_g = encoding.get_data_given_g(net, corpus)
    accuracy = networks.get_accuracy(net, corpus)
    return {
        "G": g_encoding_length,
        "D:G": d_given_g,
        "test_accuracy": accuracy,
    }


def _init_net(params, corpus_params, corpus) -> networks.LSTM:
    initialization = params["initialization"]
    if initialization == "golden":
        return golden_an_bn_net.make_an_bn_lstm(
            p=corpus_params["p"], hidden_size=params["hidden_size"]
        )

    net = networks.LSTM(
        input_size=corpus.input_sequence.shape[-1],
        hidden_size=params["hidden_size"],
        output_size=corpus.target_sequence.shape[-1],
        dropout_rate=params["dropout"],
    )
    networks.initialize_weights(net, initialization)
    return net


def get_first_an_bn_failure(net, p, max_n=3000) -> int:
    for n in range(1, max_n + 1):
        test_corpus = corpora.make_an_bn_from_n_values(n_values=(n,), p=p)
        acc = networks.get_accuracy(net, test_corpus)
        if acc != 1.0:
            return n
    return max_n


def run_simulation(corpus_params, training_params) -> tuple[networks.LSTM, dict]:
    train_size = math.floor(
        training_params["train_size"] * (1 - training_params["validation_ratio"])
    )
    validation_size = training_params["train_size"] - train_size

    p = corpus_params["p"]

    utils.seed(training_params["corpus_seed"])
    train_corpus = corpora.make_an_bn_sampled(
        batch_size=train_size,
        p=p,
    )
    validation_corpus = corpora.get_an_bn_validation_set_from_training(
        train_corpus=train_corpus,
        validation_size=validation_size,
        p=p,
    )

    utils.seed(training_params["training_seed"])

    net = _init_net(
        params=training_params,
        corpus_params=corpus_params,
        corpus=train_corpus,
    )

    device = utils.get_device()
    net.to(device)

    train_results = networks.train(
        net=net,
        train_corpus=train_corpus,
        validation_corpus=validation_corpus,
        learning_rate=training_params["learning_rate"],
        num_epochs=training_params["num_epochs"],
        regularization=training_params["regularization"],
        regularization_lambda=training_params["regularization_lambda"],
        early_stop_patience=training_params["early_stop_patience"],
    )

    evaluation_results = []
    for eval_params in utils.iterate_grid_combinations(EVAL_GRID):
        logger.info(f"Evaluating args: {json.dumps(eval_params, indent=1)}")
        test_corpus = corpora.make_an_bn_from_n_values(
            n_values=tuple(range(1, eval_params["test_max_n"])),
            p=corpus_params["p"],
        )

        net.to("cpu")
        evaluation_result = evaluate(net, test_corpus, eval_params)
        evaluation_result["first_n_failure"] = get_first_an_bn_failure(
            net, p=corpus_params["p"]
        )
        logger.info(f"Evaluation results: {json.dumps(evaluation_result)}")
        evaluation_results.append({"params": eval_params, "results": evaluation_result})

    results = {
        "corpus_params": corpus_params,
        "training_params": training_params,
        "evaluation_results": evaluation_results,
        "train_results": train_results,
    }
    return net, results


def get_simulation_path(simulation_id):
    return _SIMULATIONS_PATH / simulation_id


def _save_results(net, data, simulation_id):
    simulation_path = get_simulation_path(simulation_id)
    simulation_path.mkdir(parents=True, exist_ok=True)
    with (simulation_path / "net.pickle").open("wb") as f:
        torch.save(net, f)
    with (simulation_path / "results.json").open("w") as f:
        json.dump(data, f)


def _simulation_exists(simulation_id):
    simulation_path = get_simulation_path(simulation_id)
    return (simulation_path / "net.pickle").exists() and (
        simulation_path / "results.json"
    ).exists()


def _run_single_simulation(kwargs):
    simulation_id = utils.get_dict_hash(kwargs)
    logger.info(f"Running simulation {simulation_id}: {json.dumps(kwargs, indent=1)}")

    if _simulation_exists(simulation_id):
        logger.info(f"Simulation {simulation_id} exists. Skipping.")
        return

    net, results = run_simulation(**kwargs)
    _save_results(net, results, simulation_id)


def run_job_idx_from_grid(idx):
    grid_combinations = tuple(iterate_grid(GRID))
    job_args = grid_combinations[idx]
    _run_single_simulation(job_args)


def iterate_grid(grid):
    seen_config_ids = set()
    for config in iterate_grid_non_unique(grid):
        config_id = utils.get_dict_hash(config)
        if config_id not in seen_config_ids:
            seen_config_ids.add(config_id)
            yield config


def run_all_serially():
    grid_combinations = tuple(iterate_grid(GRID))
    logger.info(f"Unique grid size: {len(grid_combinations)}")
    for i, config in enumerate(grid_combinations):
        logger.info(f"Running config index: {i}")
        _run_single_simulation(config)


if __name__ == "__main__":
    run_all_serially()
