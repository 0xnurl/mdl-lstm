import numpy as np
import torch

import corpora
import encoding
import experiments
import golden_an_bn_net
import networks
import utils
import visualizations


def test_get_num_params():
    net = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)
    assert networks.get_num_params(net) == 108


def test_get_weights_vector():
    net = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)
    weights = networks.get_all_params_vector(net)
    assert weights.shape == (networks.get_num_params(net),)


def test_net_from_param_vector():
    net_golden = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)
    param_vector = networks.get_all_params_vector(net_golden)
    new_net = networks.net_from_param_vector(
        param_vector=param_vector,
        input_size=net_golden.input_size,
        hidden_size=net_golden.hidden_size,
        output_size=net_golden.output_size,
    )
    new_net_param_vector = networks.get_all_params_vector(new_net)
    assert torch.all(torch.eq(new_net_param_vector, param_vector))


def test_manual_an_bn_lstm_optimal_d_given_g():
    lstm = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)
    an_bn = corpora.make_an_bn_from_n_values(n_values=(3,), p=0.3)
    optimal = -np.log2(0.7) * 3 - np.log2(0.3)
    assert an_bn.optimal_d_given_g == optimal
    d_given_g = encoding.get_data_given_g(lstm, an_bn)
    print(optimal)
    print(d_given_g)
    assert np.round(optimal, decimals=2) == np.round(d_given_g, decimals=2)


def test_golden_an_bn_lstm_outputs():
    lstm = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)
    an_bn = corpora.make_an_bn_from_n_values(n_values=(3, 2, 1), p=0.3)
    probabs_unpacked = networks.feed_and_unpack(lstm, an_bn)

    round_probabs = torch.round(probabs_unpacked, decimals=2)

    assert torch.all(
        torch.eq(
            round_probabs[0],
            torch.Tensor(
                [
                    [0.3, 0.7, 0.0],
                    [0.0, 0.7, 0.3],
                    [0.0, 0.7, 0.3],
                    [0.0, 0.7, 0.3],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
        )
    )


def test_golden_net_encoding_length():
    utils.seed(100)
    an_bn_net = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)
    g_encoding_length = encoding.get_g_encoding_length(an_bn_net, zip_bitstring=False)
    print(f"|G|: {g_encoding_length}")

    assert g_encoding_length == 1057


def test_finetuning_makes_an_bn_net_worse():
    utils.seed(100)
    an_bn = corpora.make_an_bn_sampled(batch_size=500, p=0.3)
    validation = corpora.make_an_bn_from_n_values(n_values=tuple(range(10, 20)), p=0.3)
    lstm = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=6)
    g_before = encoding.get_g_encoding_length(lstm, zip_bitstring=False)
    d_g_before = encoding.get_data_given_g(lstm, an_bn)
    networks.train(
        lstm,
        an_bn,
        seed=100,
        learning_rate=0.001,
        num_epochs=1000,
        regularization=None,
        validation_corpus=validation,
        early_stop_patience=2,
        regularization_lambda=1.0,
    )
    g_after = encoding.get_g_encoding_length(lstm, zip_bitstring=False)
    d_g_after = encoding.get_data_given_g(lstm, an_bn)

    print(g_before, d_g_before)
    print(g_after, d_g_after)

    assert g_after > g_before
    assert d_g_after < d_g_before


def _assert_weight_mean(net, target_mean):
    total_weights = 0
    total = 0
    for p in net.parameters():
        total += torch.sum(p.data).item()
        total_weights += utils.num_items_in_tensor(p.data)
    print(total / total_weights)
    assert round(total / total_weights, 1) == target_mean


def test_init_weights():
    net = golden_an_bn_net.make_an_bn_lstm(p=0.3, hidden_size=3)

    networks.initialize_weights(net, initialization="zeros")
    for p in net.parameters():
        assert torch.all(torch.eq(p, 0))

    utils.seed(1000)
    networks.initialize_weights(net, initialization="uniform")
    _assert_weight_mean(net, 0.5)

    networks.initialize_weights(net, initialization="normal")
    _assert_weight_mean(net, 0.0)


def test_debug_net():
    net = utils.load_trained_net("1c67e83e")

    corpus = corpora.make_an_bn_from_n_values(n_values=(25,), p=0.3)
    with torch.no_grad():
        probabs = networks.feed_and_unpack(net, corpus)

    loss = networks.get_loss_for_corpus(
        net,
        corpus,
        regularization=None,
        regularization_lambda=None,
    )

    input_labels = list(corpus.input_sequence[0].argmax(dim=-1).numpy())

    visualizations._plot_probabs(
        probabs[0].numpy(),
        input_classes=input_labels,
        class_to_label=dict(enumerate("#ab")),
    )
