import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

import corpora
import experiments
import golden_an_bn_net
import networks
import utils


def _dict_to_str(d):
    return "__".join([f"{x}_{y}" for x, y in d.items()])


def _dict_to_title(d):
    return ", ".join([f"{x}={y}" for x, y in d.items()])


def _load_results(evaluation_filter: dict) -> pd.DataFrame:
    results = []

    missing_file = open("missing.txt", "w")

    configs = tuple(experiments.iterate_grid(experiments.GRID))
    for i, kwargs in tqdm(enumerate(configs), total=len(configs)):
        simulation_id = utils.get_dict_hash(kwargs)
        results_path = experiments.get_simulation_path(simulation_id) / "results.json"
        if not results_path.exists():
            logger.warning(
                f"Simulation {simulation_id} index {i} does not exist: {json.dumps(kwargs)}"
            )
            missing_file.write(f"{i}\n")
            continue

        with results_path.open("r") as f:
            curr_results = json.load(f)
            row = {
                "simulation_id": simulation_id,
                **curr_results["corpus_params"],
                **curr_results["training_params"],
                **curr_results["train_results"],
            }

            evaluations = curr_results["evaluation_results"]
            for evaluation in evaluations:
                if all(
                    evaluation["params"][x] == evaluation_filter[x]
                    for x in evaluation_filter
                ):
                    row.update(evaluation["results"])
                    row["eval_params"] = json.dumps(evaluation["params"])
                    break
            else:
                logger.warning(
                    f"No eval result selected for simulation {simulation_id}"
                )

            results.append(row)

    missing_file.close()
    return pd.DataFrame(results)


def plot_g_against_d_g(results_df: pd.DataFrame, batch_size, x):
    results_df = results_df[~(results_df["initialization"] == "manual")].copy()
    golden_net = golden_an_bn_net.make_an_bn_lstm(
        p=0.3,
        hidden_size=3,
    )
    results_df = results_df[results_df["batch_size"] == batch_size].copy()
    test_corpus = corpora.make_an_bn_from_n_values(n_values=tuple(range(1000)), p=0.3)
    eval_params = json.loads(results_df.iloc[0]["eval_params"])

    golden_evaluation = experiments.evaluate(
        golden_net,
        test_corpus,
        eval_params=eval_params,
    )

    utils.seed(100)
    train_corpus = corpora.make_an_bn_sampled(batch_size=batch_size, p=0.3)
    golden_training_loss, *_ = networks.get_loss_for_corpus(
        golden_net, train_corpus, regularization=None, regularization_lambda=None
    )
    golden_evaluation["training_loss"] = golden_training_loss.detach().item()

    y = "G"

    results_df = results_df.fillna({"regularization": "none"})

    sns.set_theme(rc={"figure.figsize": (9, 7)})
    ax = sns.scatterplot(
        data=results_df,
        x=x,
        y=y,
        hue="regularization",
        style="dropout",
        size="hidden_size",
    )

    golden_x = golden_evaluation[x]
    golden_y = golden_evaluation[y]
    ax.scatter(x=golden_x, y=golden_y, color="red")
    ax.text(x=golden_x, y=golden_y + 70, s="Golden net", fontsize=14, color="red")

    eval_params_string = _dict_to_title(eval_params)
    ax.set_title(
        f"G vs {x}\n{eval_params_string}\nTotal nets: {len(results_df):,}",
        fontsize=13,
    )
    plt.savefig(
        f"g_vs_d_g__{_dict_to_str(eval_params)}__batch_{batch_size}.png",
        format="png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    df = _load_results(
        evaluation_filter={
            "test_max_n": 1000,
            "quantize": True,
            "compress_g": False,
        }
    )
    df.to_csv("results.csv")
