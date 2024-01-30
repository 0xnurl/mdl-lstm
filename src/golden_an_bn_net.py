import math

import torch

import corpora
import encoding
import networks
import utils

# Great |D:G|, no compromise for |G|
TANH1 = math.tanh(1)
EPSILON = 1 / (2**14 - 1)
LARGE = 2**7 - 1


def make_an_bn_lstm(p, hidden_size=3):
    assert hidden_size >= 3, "Hidden size must be at least 3"
    input_size = 3
    output_size = 3

    # g_t = tanh(W_{ig}*x_t)
    # c_t = g_t
    # 3-place counter: tanh(|#|=1, |#|=1, |a|-|b|)
    W_ig = LARGE * torch.Tensor(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, -1],
        ]
    )

    # Input->hidden gate.
    # o_t = sig(W_{io}*x_t + b_{io}) => mask for c_t based on current input.
    # => h_t = tanh(tanh([1 if #, 1 if a, |a|-|b| if b]))
    W_io = LARGE * torch.Tensor(
        [
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ]
    )

    # weight_ih_l0 = (W_ii|W_if|W_ig|W_io)
    weight_ih_l0 = torch.zeros((4 * hidden_size, input_size))
    utils.add_at(weight_ih_l0, hidden_size * 2, W_ig)
    utils.add_at(weight_ih_l0, hidden_size * 3, W_io)

    bias_io = LARGE * torch.Tensor([-1, -1, -1])
    bias_ii = LARGE * torch.Tensor([1, 1, 1])  # Saturate input gate.
    bias_if = LARGE * torch.Tensor([1, 1, 1])  # Saturate forget gate (=remember).

    # bias_ih_l0 = (b_ii|b_if|b_ig|b_io)
    bias_ih_l0 = torch.zeros((4 * hidden_size,))
    bias_ih_l0[: len(bias_ii)] = bias_ii
    utils.add_at(bias_ih_l0, hidden_size, bias_if)
    utils.add_at(bias_ih_l0, hidden_size * 3, bias_io)

    bias_hh_l0 = torch.zeros((4 * hidden_size,))

    # Build the output layer backwards based on target probabs:
    target_probabs = torch.Tensor(
        [
            [p, 1 - p, 0],  # First step.
            [0, 1 - p, p],  # 'a' phase.
            [0, 0, 1],  # b^{m<n} phase.
            [1, 0, 0],  # b^n step.
        ]
    )
    # Apply log so that softmax is fed correct logits.
    output_layer_weights = torch.log(target_probabs[:3, :] + EPSILON)
    # Use bias to cram fourth state into 3-dim tensor. Fourth state only appear when `h_t` is 0 everywhere.
    output_layer_bias = torch.log(target_probabs[3, :] + EPSILON)
    output_layer_weights = output_layer_weights - output_layer_bias

    # Transpose so that the one-hot `h_t` copies the target probability row.
    output_layer_weights = output_layer_weights.T
    # Divide by tanh(1) because `h_t=tanh(c_t)` and `c_t` is one-valued.
    output_layer_weights /= TANH1

    output_layer = torch.zeros((output_size, hidden_size))
    utils.add_at(output_layer, 0, output_layer_weights)

    lstm = networks.LSTM(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )

    weight_hh_l0 = torch.zeros((4 * hidden_size, hidden_size))
    state_dict = {
        "lstm.weight_ih_l0": weight_ih_l0,
        "lstm.weight_hh_l0": weight_hh_l0,
        "lstm.bias_ih_l0": bias_ih_l0,
        "lstm.bias_hh_l0": bias_hh_l0,
        "output_layer.weight": output_layer,
        "output_layer.bias": output_layer_bias,
    }
    lstm.load_state_dict(state_dict)
    return lstm


def lstm_test(lstm):
    n = 6
    test_corpus = corpora.make_an_bn_from_n_values(n_values=(n,), p=0.3)

    data_given_g = encoding.get_data_given_g(lstm, test_corpus)
    print(
        f"|D:G| for anbn with n={n}:\n"
        f"- Here: {data_given_g:.3f}\n"
        f"- Golden: {test_corpus.optimal_d_given_g:.3f}"
    )

    networks.show_all_activations(
        lstm, inputs=test_corpus.input_sequence[0].unsqueeze(0)
    )


if __name__ == "__main__":
    lstm = make_an_bn_lstm(p=0.3, hidden_size=3)
    lstm_test(lstm)
