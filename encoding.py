import dataclasses
import fractions
import pickle
import zlib

import numpy as np
import torch
from torch import nn

import corpora
import networks

_WEIGHT_DENOMINATOR_PRECISION = 1000


@dataclasses.dataclass(frozen=True)
class _Weight:
    sign: int
    numerator: int
    denominator: int


def _int_to_binary_string(n, sequence_length=None) -> str:
    binary = f"{n:b}"
    if sequence_length is not None:
        binary = ("0" * (sequence_length - len(binary))) + binary
    return binary


def _integer_encoding(n) -> str:
    # Based on Vitanyi & Li's self-delimiting encoding: `unary_encoding(log2_N) + '0' + binary_encoding(N)`.
    assert n >= 0
    if n == 0:
        return "0"
    binary = _int_to_binary_string(n)
    return ("1" * len(binary)) + "0" + binary


def _encode_weight(weight: _Weight) -> str:
    encoding = ""
    encoding += "0" if weight.sign == -1 else "1"
    encoding += _integer_encoding(weight.numerator)
    encoding += _integer_encoding(weight.denominator)
    return encoding


def _float_to_fraction(f) -> fractions.Fraction:
    return fractions.Fraction(f).limit_denominator(_WEIGHT_DENOMINATOR_PRECISION)


def _float_to_rational_weight(f: torch.Tensor) -> _Weight:
    f = f.item()
    fraction = _float_to_fraction(f)
    sign = int(np.sign(f).item())
    return _Weight(
        sign=sign,
        numerator=abs(fraction.numerator),
        denominator=abs(fraction.denominator),
    )


def get_net_encoding(net: networks.LSTM) -> str:
    # Encoding: E(output_size) + E(LSTM) + E(output_weights)
    lstm_encoding = _get_lstm_encoding(net.lstm)
    output_layer_encoding = _get_weights_encoding(net.output_layer.weight)
    return lstm_encoding + output_layer_encoding


def get_data_given_g(net: networks.LSTM, corpus: corpora.Corpus) -> float:
    # Cross-entropy sum in base 2.
    with torch.no_grad():
        (_, weighted_cross_entropy_sum, _) = networks.get_loss_for_corpus(
            net, corpus, regularization=None, regularization_lambda=None
        )

    return float(weighted_cross_entropy_sum.item() / np.log(2))


def _get_weights_encoding(weights: nn.Parameter) -> str:
    rational_weights = [_float_to_rational_weight(w) for w in weights.flatten()]
    return "".join(_encode_weight(w) for w in rational_weights)


def _limit_weight_denominators(weights: torch.Tensor) -> torch.Tensor:
    limited_weights = torch.empty_like(weights)

    for idx, val in np.ndenumerate(weights.numpy()):
        limited_w = float(_float_to_fraction(val.item()))
        limited_weights[idx] = limited_w

    return limited_weights


def get_quantized_net(net: nn.Module) -> nn.Module:
    state_dict = net.state_dict()
    for weights_name, weights in state_dict.items():
        limited_weights = _limit_weight_denominators(weights)
        state_dict[weights_name] = limited_weights
    net_copy = pickle.loads(pickle.dumps(net))
    net_copy.load_state_dict(state_dict, strict=True)
    return net_copy


def _get_lstm_encoding(lstm: nn.LSTM) -> str:
    # Encoding: E(input_size) | E(hidden_size) | E(all weights and biases).
    # Assuming: single-layer network, not bidirectional.

    input_size = lstm.input_size
    hidden_size = lstm.hidden_size

    """
    the learnable input-hidden weights of the kthkth layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size)
    bias_hh_l[k] – the learnable hidden-hidden bias of the kthkth layer (b_hi|b_hf|b_hg|b_ho), of shape (4*hidden_size)
    """
    weight_hh_l0 = lstm.weight_hh_l0
    bias_hh_l0 = lstm.bias_hh_l0

    """
    weight_ih_l[k] – the learnable input-hidden weights of the kthkth layer (W_ii|W_if|W_ig|W_io), of shape (4*hidden_size, input_size)
    bias_ih_l[k] – the learnable input-hidden bias of the kthkth layer (b_ii|b_if|b_ig|b_io), of shape (4*hidden_size)
    """
    weight_ih_l0 = lstm.weight_ih_l0
    bias_ih_l0 = lstm.bias_ih_l0

    input_size_enc = _integer_encoding(input_size)
    hidden_size_enc = _integer_encoding(hidden_size)

    all_weights = [
        weight_hh_l0,
        bias_hh_l0,
        weight_ih_l0,
        bias_ih_l0,
    ]
    all_weights_encodings = "".join(tuple(map(_get_weights_encoding, all_weights)))

    network_encoding = input_size_enc + hidden_size_enc + all_weights_encodings
    return network_encoding


def _get_compressed_bitstring_length(s) -> int:
    return len(zlib.compress(s.encode("utf-8"), level=zlib.Z_BEST_COMPRESSION)) * 8


def get_g_encoding_length(net: networks.LSTM, zip_bitstring: bool):
    g_encoding = get_net_encoding(net)
    if zip_bitstring:
        return _get_compressed_bitstring_length(g_encoding)
    else:
        return len(g_encoding)


def get_mdl_score(
    net: networks.LSTM, corpus: corpora.Corpus, compress_g: bool = True
) -> tuple[float, float, float]:  # Returns (|G|, |D:G|, MDL)
    g_encoding_length = get_g_encoding_length(net, zip_bitstring=compress_g)
    data_given_g = get_data_given_g(net, corpus)
    return g_encoding_length, data_given_g, g_encoding_length + data_given_g
