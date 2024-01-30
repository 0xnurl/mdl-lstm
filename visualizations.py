import functools
import math
import multiprocessing
import os
import pathlib
import pickle
from typing import Callable

import matplotlib.patches as mpatches
import numpy as np
import pandas
import pandas as pd
import plotnine
import seaborn as sns
import torch
from loguru import logger
from matplotlib import pyplot as plt
from plotnine import aes, facet_grid, geom_point, ggplot, labs
from tqdm import tqdm

import corpora
import encoding
import golden_an_bn_net
import networks
import utils

_CMAP = "magma"
_GOLDEN_COLOR = "cyan"
_BEST_Z_COLOR = "magenta"
_CENTER_COLOR = "yellow"
_AXIS_COLOR = "whitesmoke"
_AXIS_TEXT_COLOR = "lightgray"


def _get_ce_avg(net, corpus, regularization, regularization_lambda):
    # Average + regularization term, CE sum without regularization, regularization term
    ce_plus_reg, *_ = networks.get_loss_for_corpus(
        net,
        corpus,
        regularization,
        regularization_lambda,
    )
    return ce_plus_reg.item()


def _get_ce_unreduced(net, corpus, regularization, regularization_lambda):
    # Average + regularization term, CE sum without regularization, regularization term
    _, just_ce, reg_term = networks.get_loss_for_corpus(
        net,
        corpus,
        regularization,
        regularization_lambda,
    )
    return (just_ce + reg_term).item()


def _get_mdl(net, corpus):
    return encoding.get_mdl_score(
        net,
        corpus,
        compress_g=False,
    )[2]


def _plot_3d(
    ax,
    X,
    Y,
    Z,
    title,
    center_accuracy,
    golden_z,
    show_min,
    show_golden,
    is_golden_at_center,
    show_3d_axis,
    where_best_xy,
    best_point_label,
    golden_accuracy,
    best_point_accuracy,
    best_marker_offsets=(0, 0, 0),
):
    flat_plane_range = X.max()
    flat_plane_num_points = 9
    golden_mesh = np.meshgrid(
        np.linspace(-1 * flat_plane_range, flat_plane_range, flat_plane_num_points),
        np.linspace(-1 * flat_plane_range, flat_plane_range, flat_plane_num_points),
    )

    if not show_3d_axis:
        ax.set_axis_off()

    marker_size = 150
    best_z = Z[where_best_xy]

    ax.plot_surface(
        X,
        Y,
        Z,
        cmap=_CMAP,
        linewidth=0,
        alpha=1,
    )

    if show_min:
        ax.scatter(
            X[where_best_xy] + best_marker_offsets[0],
            Y[where_best_xy] + best_marker_offsets[1],
            Z[where_best_xy] + best_marker_offsets[2],
            c=_BEST_Z_COLOR,
            marker="+",
            s=marker_size,
            label=f"{best_point_label} ({best_z:.2e}, acc. {best_point_accuracy * 100:.1f}%)",
        )

    if show_golden:
        if is_golden_at_center:
            # Golden 'x' if golden is in space.
            ax.scatter(
                0,
                0,
                golden_z + best_marker_offsets[2],
                c=_GOLDEN_COLOR,
                marker="x",
                s=marker_size,
                label=f"Golden ({golden_z:.2e}, acc. {golden_accuracy * 100:.1f}%)",
            )
        else:
            width = Z.shape[0]
            center = (width // 2, width // 2)
            is_best_at_center = where_best_xy == center
            if not is_best_at_center:
                ax.scatter(
                    0,
                    0,
                    Z[center] + best_marker_offsets[2],
                    c=_CENTER_COLOR,
                    marker=".",
                    s=marker_size,
                    label=f"Trained ({Z[center]:.2e}, acc. {center_accuracy * 100:.1f}%)",
                )

            # Golden plane.
            golden_plane_z = (
                np.ones((golden_mesh[0].shape[0], golden_mesh[0].shape[0])) * golden_z
            )
            ax.plot_surface(
                golden_mesh[0],
                golden_mesh[1],
                golden_plane_z,
                alpha=0.2,
                color=_GOLDEN_COLOR,
                zorder=0,
                label=f"Golden ({golden_z:.2e}, acc. {golden_accuracy * 100:.1f}%)",
            )

    best_z_mesh = np.meshgrid(
        np.linspace(
            -1 * flat_plane_range, flat_plane_range, flat_plane_num_points // 2
        ),
        np.linspace(
            -1 * flat_plane_range, flat_plane_range, flat_plane_num_points // 2
        ),
    )

    if show_golden or show_min:
        ax.legend(fontsize=7, loc="upper center")
    ax.xaxis.line.set_color(_AXIS_COLOR)
    ax.yaxis.line.set_color(_AXIS_COLOR)
    ax.zaxis.line.set_color(_AXIS_COLOR)
    ax.grid(color=_AXIS_COLOR)
    ax.xaxis.grid(True, color=_AXIS_COLOR)
    ax.yaxis.grid(True, color=_AXIS_COLOR)
    ax.zaxis.grid(True, color=_AXIS_COLOR)
    ax.tick_params(axis="x", colors=_AXIS_TEXT_COLOR)
    ax.tick_params(axis="y", colors=_AXIS_TEXT_COLOR)
    ax.tick_params(axis="z", colors=_AXIS_TEXT_COLOR)

    ax.set_xlabel(r"$\alpha$", color=_AXIS_TEXT_COLOR)
    ax.set_ylabel(r"$\beta$", color=_AXIS_TEXT_COLOR)
    if title:
        ax.set_title(title)


def where_argbest(Z, argbest_func):
    # (row, col) in matrix notation.
    return np.unravel_index(argbest_func(Z), Z.shape)


def _plot_2d(
    Z,
    scale_range,
    title,
    ax,
    golden_z,
    loss_name,
    is_golden_at_center,
    where_best_Z,
    best_label,
    golden_accuracy,
    best_net_accuracy,
):
    width = Z.shape[0]

    ticks = np.linspace(0, width, 3)
    tick_labels = list(map(str, [-scale_range, 0, scale_range]))

    sns.heatmap(
        data=Z,
        cmap=_CMAP,
        ax=ax,
    )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, color=_AXIS_TEXT_COLOR)
    ax.set_yticklabels(tick_labels, color=_AXIS_TEXT_COLOR)
    ax.set_xlabel(r"$\alpha$", color=_AXIS_TEXT_COLOR)
    ax.set_ylabel(r"$\beta$", color=_AXIS_TEXT_COLOR)
    ax.tick_params(color=_AXIS_TEXT_COLOR)

    if title:
        ax.set_title(title)

    pixel_center_offset = 0.45
    marker_size = 120
    marker_alpha = 1

    if is_golden_at_center:
        golden_x = golden_y = width // 2
        golden_alpha = 1.0
    else:
        golden_x = golden_y = width + marker_size
        golden_alpha = 0.0

    golden_label = "Golden ("
    best_label += " ("
    if "acc" not in loss_name:
        golden_label += f"{golden_z:.2e}, acc. "
        best_label += f"{Z[where_best_Z]:.2e}, acc. "

    best_label += f"{best_net_accuracy * 100:.1f}%)"
    golden_label += f"{golden_accuracy * 100:.1f}%)"

    ax.scatter(
        golden_x + pixel_center_offset,
        golden_y + pixel_center_offset,
        c=_GOLDEN_COLOR,
        marker="x",
        alpha=golden_alpha,
        s=marker_size,
        label=golden_label,
    )

    where_best_xy = where_best_Z[::-1]  # Switch to (x,y) scatter coordinates.
    if Z[where_best_Z] == golden_z:
        where_best_xy = (golden_x, golden_y)

    ax.scatter(
        where_best_xy[0] + pixel_center_offset,
        where_best_xy[1] + pixel_center_offset,
        c=_BEST_Z_COLOR,
        alpha=marker_alpha,
        marker="+",
        s=marker_size * 1.3,
        label=best_label,
    )

    ax.legend(fontsize=7, loc="lower left")


def _normalize_direction_component(
    one_layer_direction, one_layer_weights, normalization="filter"
):
    # Filter normalization, based on Li et al.
    if normalization == "filter":
        # Normalizes direction component weights feeding one neuron, based on the norm of the corresponding weights in the original network.
        for d, w in zip(one_layer_direction, one_layer_weights):
            # Iterate rows.
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif normalization == "weight":
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        one_layer_direction.mul_(one_layer_weights)


def _normalize_directions_based_on_weights(directions, weights, ignore_biases=False):
    # Based on Li et al. 2018, `https://github.com/tomgoldstein/loss-landscape`.
    for d, w in zip(directions, weights):
        if d.dim() <= 1:
            if ignore_biases:
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            _normalize_direction_component(d, w)


def _get_random_directions_from_weights(weights) -> tuple[torch.Tensor, ...]:
    return tuple(torch.randn(w.size()) for w in weights)


def _create_normalized_direction(net_weights: tuple[torch.Tensor, ...]):
    directions = _get_random_directions_from_weights(net_weights)
    _normalize_directions_based_on_weights(directions, net_weights, ignore_biases=False)
    return directions


def _get_cache_path(net, **kwargs):
    net_hash = networks.net_hash(net)[:8]
    kwargs_hash = utils.get_dict_hash(kwargs)

    return pathlib.Path(f"./cache/net_{net_hash}__args_{kwargs_hash}.pickle")


def _explore_around_net(net, num_points: int, scale_range: int):
    # Return: alphas (mesh), betas (mesh), net thetas (mesh). THETAS[0,0] is origin net.
    net_theta = networks.get_all_params_vector(net)
    net_weights = networks.get_all_weights(net)
    dir1_components = _create_normalized_direction(net_weights)
    dir2_components = _create_normalized_direction(net_weights)

    dir1 = torch.hstack([x.flatten() for x in dir1_components])
    dir2 = torch.hstack([x.flatten() for x in dir2_components])

    alpha = np.linspace(-1 * scale_range, scale_range, num_points).reshape((-1, 1))
    beta = np.linspace(-1 * scale_range, scale_range, num_points).reshape((-1, 1))

    ALPHAS, BETAS = np.meshgrid(alpha, beta)
    THETAS = torch.zeros((num_points, num_points, net_theta.shape[0]))

    for i in range(num_points):
        for j in range(num_points):
            curr_alpha = ALPHAS[i, j]
            curr_beta = BETAS[i, j]

            curr_dir = curr_alpha * dir1 + curr_beta * dir2
            curr_theta = net_theta + curr_dir

            THETAS[i, j] = curr_theta

    return ALPHAS, BETAS, THETAS


def _get_z(theta, z_func, reference_net):
    curr_net = networks.net_from_param_vector(
        param_vector=theta,
        input_size=reference_net.input_size,
        hidden_size=reference_net.hidden_size,
        output_size=reference_net.output_size,
    )

    with torch.no_grad():  # In case of loss calculation.
        return z_func(curr_net)


def _get_Z_around_net(net, z_func, thetas) -> np.ndarray:
    _POOL = multiprocessing.Pool(processes=os.cpu_count() - 1)

    thetas_idxs = np.ndindex(thetas.shape[:2])
    Z_flat = _POOL.map(
        functools.partial(_get_z, z_func=z_func, reference_net=net),
        [thetas[idx] for idx in thetas_idxs],
    )
    return np.reshape(Z_flat, thetas.shape[:2])


def plot_net_probabs(net, n, save_to, title):
    p = 0.3
    corpus = corpora.make_an_bn_from_n_values(n_values=(n,), p=p)

    with torch.no_grad():
        probabs = networks.feed_and_unpack(net, corpus)

    input_labels = list(corpus.input_sequence[0].argmax(dim=-1).numpy())

    _plot_probabs(
        probabs[0].numpy(),
        input_classes=input_labels,
        class_to_label=dict(enumerate("#ab")),
        save_to=save_to,
        title=title,
    )


def plot_memory(net, n, save_to, title):
    p = 0.3
    corpus = corpora.make_an_bn_from_n_values(n_values=(n,), p=p)
    with torch.no_grad():
        c_ts, h_ts, outputs, softmax_ts = networks.get_activations(
            net, inputs=corpus.input_sequence
        )
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    memory_size = net.hidden_size

    colors = sns.color_palette("hls", memory_size)

    values_vector = c_ts
    values_vector = values_vector.squeeze()

    timesteps = list(range(values_vector.shape[0]))
    min_y = 0
    max_y = 0
    for i in range(memory_size):
        vals = values_vector[:, i]
        max_y = max(max_y, vals.max())
        min_y = min(min_y, vals.min())
        sns.lineplot(
            x=timesteps,
            y=vals,
            color=colors[i],
            label=f"$c_{i}$",
            linewidth=3,
            dashes=(5, i + 1),
        )

    class_to_label = dict(enumerate("#ab"))

    step_y = max(1, int(max_y - min_y) // 10)

    input_classes = list(corpus.input_sequence[0].argmax(dim=-1).numpy())

    n = (len(input_classes) - 1) // 2
    n_middle = (n // 2) + 1
    visible_tick_idxs = [
        # SOS, middle of a, middle of b, last b
        0,
        n_middle,
        n + 1,
        n_middle + n,
        len(input_classes)
        # len(input_classes) - 1,
    ]
    x_tick_labels = ["#", "a", "", "b", ""]

    ax.set_xticks(visible_tick_idxs)
    ax.set_xticklabels(x_tick_labels, fontsize=13)

    y_ticks = np.arange(
        (math.ceil(int(min_y) / step_y) - 1) * step_y,
        (math.ceil(int(max_y) / step_y) + 1) * step_y,
        step=step_y,
    )
    ax.set_yticks(y_ticks)

    ax.set_xlabel("Input phase", fontsize=15)
    ax.set_ylabel("$c$", fontsize=15)

    # ax.grid(b=True, color="#bcbcbc")
    # plt.title("Next symbol prediction probabilities", fontsize=17)
    plt.legend(loc="upper right", fontsize=15)
    fig.tight_layout()

    if title:
        ax.set_title(title)

    if save_to:
        fig.savefig(f"./figures/{save_to}", dpi=200)

    plt.show()


def get_Z_ALPHA_BETA_THETA_around_net(
    net, seed: int, num_points: int, scale_range: int, z_func: Callable
):
    utils.seed(seed)

    ALPHAS, BETAS, THETAS = _explore_around_net(
        net=net, num_points=num_points, scale_range=scale_range
    )

    Z = _get_Z_around_net(
        net,
        z_func=z_func,
        thetas=THETAS,
    )

    return Z, ALPHAS, BETAS, THETAS


def _get_l1(x):
    return networks.get_l1(x, l1_lambda=1.0).item()


def _get_l2(x):
    return networks.get_l2(x, l2_lambda=1.0).item()


def get_z_func(
    loss_name, train_corpus, test_corpus, regularization, regularization_lambda
) -> Callable:
    if loss_name == "mdl_train":
        z_func = functools.partial(_get_mdl, corpus=train_corpus)
    elif loss_name == "mdl_test":
        z_func = functools.partial(_get_mdl, corpus=test_corpus)
    elif loss_name == "g":
        z_func = functools.partial(encoding.get_g_encoding_length, zip_bitstring=False)
    elif loss_name == "l1":
        z_func = _get_l1
    elif loss_name == "l2":
        z_func = _get_l2
    elif loss_name.startswith("ce_train"):
        if loss_name.endswith("avg"):
            func = _get_ce_avg
        else:
            func = _get_ce_unreduced
        z_func = functools.partial(
            func,
            corpus=train_corpus,
            regularization=regularization,
            regularization_lambda=regularization_lambda,
        )
    elif loss_name.startswith("ce_test"):
        if loss_name.endswith("avg"):
            func = _get_ce_avg
        else:
            func = _get_ce_unreduced
        z_func = functools.partial(
            func,
            corpus=test_corpus,
            regularization=regularization,
            regularization_lambda=regularization_lambda,
        )
    elif loss_name == "acc_train":
        z_func = functools.partial(networks.get_accuracy, corpus=train_corpus)
    elif loss_name == "acc_test":
        z_func = functools.partial(networks.get_accuracy, corpus=test_corpus)
    else:
        raise ValueError(loss_name)
    return z_func


def viz_2d_3d(
    net: networks.LSTM,
    loss_name: str,
    golden_net: networks.LSTM,
    num_points: int,
    scale_range: int,
    seed: int,
    best_marker_offsets=(0, 0, 0),
    computed_zorder=True,
    show_3d_axis=True,
    show_min=True,
    show_golden=True,
    title=None,
    train_corpus=None,
    test_corpus=None,
    regularization=None,
    regularization_lambda=None,
    save_to=None,
    plot_2d=True,
    plot_3d=True,
):
    assert num_points % 2 == 1, "num_points must be odd so 0 is in range"

    z_func = get_z_func(
        loss_name=loss_name,
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
    )

    Z, ALPHAS, BETAS, THETAS = get_Z_ALPHA_BETA_THETA_around_net(
        net=net,
        seed=seed,
        num_points=num_points,
        scale_range=scale_range,
        z_func=z_func,
    )

    golden_z = z_func(golden_net)

    if plot_2d and plot_3d:
        figwidth = 9
        numcols = 2
    else:
        figwidth = 3.8
        numcols = 1

    sns.reset_defaults()
    fig = plt.figure(figsize=(figwidth, 3), dpi=200)

    if plot_2d:
        ax_2d = fig.add_subplot(1, numcols, 1)
    if plot_3d:
        ax_3d = fig.add_subplot(
            1,
            numcols,
            numcols,
            projection="3d",
            computed_zorder=computed_zorder,
        )

    if "acc" in loss_name:
        argbest_func = np.argmax
        best_label = "Maximum"
    else:
        argbest_func = np.argmin
        best_label = "Minimum"

    where_argbest_coords = where_argbest(Z, argbest_func=argbest_func)

    golden_accuracy = networks.get_accuracy(golden_net, test_corpus)
    best_net_theta = THETAS[where_argbest_coords]
    best_net = networks.net_from_param_vector(
        best_net_theta,
        input_size=net.input_size,
        output_size=net.output_size,
        hidden_size=net.hidden_size,
    )
    center_accuracy = networks.get_accuracy(net, test_corpus)
    best_net_accuracy = networks.get_accuracy(best_net, test_corpus)

    width = Z.shape[0]
    center = (width // 2, width // 2)
    is_golden_at_center = Z[center] == golden_z

    if plot_2d:
        _plot_2d(
            Z=Z,
            scale_range=scale_range,
            ax=ax_2d,
            title=title,
            golden_z=golden_z,
            loss_name=loss_name,
            is_golden_at_center=is_golden_at_center,
            where_best_Z=where_argbest_coords,
            best_label=best_label,
            golden_accuracy=golden_accuracy,
            best_net_accuracy=best_net_accuracy,
        )

    if plot_3d:
        _plot_3d(
            X=ALPHAS,
            Y=BETAS,
            Z=Z,
            best_marker_offsets=best_marker_offsets,
            ax=ax_3d,
            show_3d_axis=show_3d_axis,
            title=title,
            golden_z=golden_z,
            show_min=show_min,
            show_golden=show_golden,
            center_accuracy=center_accuracy,
            is_golden_at_center=is_golden_at_center,
            where_best_xy=where_argbest_coords,
            best_point_label=best_label,
            golden_accuracy=golden_accuracy,
            best_point_accuracy=best_net_accuracy,
        )

    if save_to:
        plt.savefig(
            f"./figures/{save_to}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.tight_layout()
    plt.show()


def _plot_probabs(
    probabs: np.ndarray, input_classes, class_to_label=None, save_to=None, title=None
):
    if probabs.shape[-1] == 1:
        # Binary outputs, output is P(1).
        probabs_ = np.zeros((probabs.shape[0], 2))
        probabs_[:, 0] = (1 - probabs).squeeze()
        probabs_[:, 1] = probabs.squeeze()
        probabs = probabs_

    masked_timesteps = np.where(np.isnan(probabs))[0]
    if len(masked_timesteps):
        first_mask_step = masked_timesteps[0]
        probabs = probabs[:first_mask_step]

    if class_to_label is None:
        class_to_label = {i: str(i) for i in range(len(input_classes))}
    sns.reset_defaults()
    sns.despine()
    sns.set_theme(style="dark")
    # plt.rcParams["xtick.bottom"] = True

    sns.set_style(rc={"axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(probabs.shape[0])

    num_classes = probabs.shape[1]
    width = 1

    colors = sns.color_palette("hls", num_classes)
    for c in range(num_classes):
        sns.barplot(
            x=x,
            y=probabs[:, c],
            label=(f"p({class_to_label[c]})") if num_classes > 1 else "p(1)",
            color=colors[c],
            width=width,
            alpha=0.9 if c > 0 else 1,
            bottom=np.sum(probabs[:, :c], axis=-1),
        )

    n = (len(input_classes) - 1) // 2
    n_middle = (n // 2) + 1
    visible_tick_idxs = [
        # SOS, middle of a, middle of b, last b
        0,
        n_middle,
        n + 1,
        n_middle + n,
        len(input_classes) - 1,
    ]
    ax.set_xticks(visible_tick_idxs)
    x_tick_labels = ["#", "a", "", "b", ""]
    ax.set_xticklabels(x_tick_labels, fontsize=13)

    ax.set_xlabel("Input phase", fontsize=15)
    ax.set_ylabel("Next symbol probability", fontsize=15)

    plt.legend(loc="lower right", fontsize=15)
    fig.tight_layout()

    if title:
        ax.set_title(title)

    if save_to:
        fig.savefig(f"./figures/{save_to}", dpi=200)
    plt.show()
