import pathlib

import numpy as np
import pandas as pd
from loguru import logger

import networks
import utils
import visualizations


def _get_best_z_and_accuracy_around(
    net,
    loss_name,
    train_corpus,
    test_corpus,
    regularization,
    regularization_lambda,
    seed,
    num_points,
    scale_range,
):
    z_func = visualizations.get_z_func(
        loss_name=loss_name,
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
    )
    center_z = z_func(net)
    logger.info(f"Center z: {center_z}")

    Z, ALPHAS, BETAS, THETAS = visualizations.get_Z_ALPHA_BETA_THETA_around_net(
        net=net,
        seed=seed,
        num_points=num_points,
        scale_range=scale_range,
        z_func=z_func,
    )

    if "acc" in loss_name:
        argbest_func = np.argmax
    else:
        argbest_func = np.argmin

    where_argbest_coords = visualizations.where_argbest(Z, argbest_func=argbest_func)
    best_net_theta = THETAS[where_argbest_coords]
    best_net = networks.net_from_param_vector(
        best_net_theta,
        input_size=net.input_size,
        output_size=net.output_size,
        hidden_size=net.hidden_size,
    )
    best_accuracy = networks.get_accuracy(best_net, test_corpus)
    best_z = Z[where_argbest_coords]
    return best_z, best_accuracy, best_net


def _calc_lambdas(net, num_points, scale_range, seed, train_corpus, test_corpus, name):
    filename = f"exhaustive_lambdas_{name}"
    if pathlib.Path(filename + ".csv").exists():
        return pd.read_csv(filename + ".csv")

    grid = {
        "regularization": (
            "l1",
            "l2",
        ),
        "regularization_lambda": (
            0.0,
            0.1,
            0.5,
            1,
        ),
        "loss_name": (
            "mdl_train",
            "ce_train_avg",
        ),
    }

    rows = []

    for kwargs in utils.iterate_grid_combinations(grid):
        best_z, best_accuracy, best_net = _get_best_z_and_accuracy_around(
            net=net,
            loss_name=kwargs["loss_name"],
            train_corpus=train_corpus,
            test_corpus=test_corpus,
            regularization=kwargs["regularization"],
            regularization_lambda=kwargs["regularization_lambda"],
            seed=seed,
            num_points=num_points,
            scale_range=scale_range,
        )

        row = {
            **kwargs,
            "best_z": best_z,
            "best_accuracy": best_accuracy,
        }
        if kwargs["regularization_lambda"] == 0.0:
            row["regularization"] = "None"

        logger.info(row)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_exhaustive_lambdas(
    net, num_points, scale_range, seed, train_corpus, test_corpus, name
):
    df = _calc_lambdas(
        net,
        num_points=num_points,
        scale_range=scale_range,
        seed=seed,
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        name=name,
    )
    df["name"] = name
    df.to_csv(f"exhaustive_lambdas_{name}.csv", index=False)
    return df


def gen_exhaustive_lambdas_table(dfs):
    tex = r"\begin{tabular}{llllllll}" + "\n" + r"\toprule" + "\n"
    tex += (
        r"\multirow{2}{*}{Regularization term} & \multirow{2}{*}{lambda} & \multicolumn{2}{c}{Golden} & \multicolumn{2}{c}{Best trained} & \multicolumn{2}{c}{Best trained, no early stopping} \\"
        + "\n"
    )

    tex += r" & & Loss & Test acc. & Loss & Test acc. \% & Loss & Test acc. \%"
    tex += r"\\" + "\n" + r"\midrule" + "\n"

    joined = pd.concat(dfs).reset_index()
    names = joined["name"].unique()
    grouped = joined.groupby(
        by=["loss_name", "regularization", "regularization_lambda"]
    )
    for group_vals, group_df in grouped:
        loss_name, reg_term, lambd = group_vals
        line = loss_name + " " + reg_term + " & " + str(lambd)
        for name in names:
            name_row = group_df[group_df.name == name].iloc[0]
            line += f" & {name_row['best_z']:.2e} & {name_row['best_accuracy']*100:.2f}"
        line += r" \\" + "\n"
        tex += line

    tex += r"\bottomrule" + "\n" + r"\end{tabular}"

    with open("exhaustive-lambdas.tex", "w") as f:
        f.write(tex)
