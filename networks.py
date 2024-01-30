import hashlib
import io
import pickle
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch import nn, optim
from torch.nn.utils import rnn

import corpora
import utils


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.output_layer = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )
        if dropout_rate > 0:
            self.dropout_layer = nn.Dropout(p=dropout_rate)
        else:
            self.dropout_layer = None

    def forward(self, x, h, c):
        timestep_outputs, (h_n, c_n) = self.lstm(x, (h, c))
        if self.dropout_layer is not None:
            timestep_outputs = self.dropout_layer(timestep_outputs.data)
        output = self.output_layer(timestep_outputs.data)
        return output, (h_n, c_n)


def get_weight_per_char_packed(corpus: corpora.Corpus):
    sample_weights = corpus.sample_weights
    max_sequence_length = corpus.input_sequence.shape[1]
    w = torch.Tensor(sample_weights).reshape((-1, 1))
    weights_repeated = np.matmul(w, torch.ones((1, max_sequence_length)))
    weights_repeated[~corpus.input_mask] = corpora.MASK_VALUE

    return pack_masked(weights_repeated, sequence_lengths=corpus.sequence_lengths).data


def _ensure_torch(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(torch.float32)
    return x


def pack_masked(sequences: torch.Tensor, sequence_lengths: tuple[int, ...]):
    return rnn.pack_padded_sequence(
        _ensure_torch(sequences),
        lengths=sequence_lengths,
        batch_first=True,
        enforce_sorted=True,
    )


def _unpack(
    packed_tensor,
    packed_sequence_batch_sizes,
    batch_size,
    max_sequence_length,
    num_classes,
):
    # Assumes sorted indices.
    unpacked = torch.full(
        (batch_size, max_sequence_length, num_classes), corpora.MASK_VALUE
    )
    i = 0
    for t, batch_size in enumerate(packed_sequence_batch_sizes):
        unpacked[:batch_size, t] = packed_tensor[i : i + batch_size]
        i += batch_size
    return unpacked


def get_accuracy(net, corpus) -> float:
    deterministic_mask = corpus.deterministic_steps_mask
    probabs = feed_and_unpack(net, corpus)
    predicted_classes = probabs.argmax(dim=-1)[deterministic_mask]
    targets = corpus.target_sequence.argmax(dim=-1)[deterministic_mask]
    accuracy = (predicted_classes == targets).sum().item() / targets.size()[0]
    return accuracy


def calculate_weighted_cross_entropy(
    targets_packed, outputs_packed, sample_weight_per_char
):
    target_classes = targets_packed.argmax(axis=-1)
    non_reduced_ce = nn.CrossEntropyLoss(reduction="none")(
        outputs_packed, target_classes
    )
    weighted_ce = torch.mul(non_reduced_ce, sample_weight_per_char)
    return weighted_ce.sum()


def get_l1(net, l1_lambda):
    l1 = torch.zeros(1).to(utils.get_net_device(net))
    for weights in net.parameters():
        l1 += l1_lambda * torch.abs(weights).sum()
    return l1


def get_l2(net, l2_lambda):
    l2 = torch.zeros(1).to(utils.get_net_device(net))
    for weights in net.parameters():
        l2 += l2_lambda * torch.square(weights).sum()
    return l2


def get_loss_for_corpus(
    net: LSTM,
    corpus: corpora.Corpus,
    regularization: Optional[str],
    regularization_lambda: Optional[float],
):
    # Return (Average + regularization term, CE sum without regularization, regularization term)

    device = utils.get_net_device(net)

    inputs_packed = pack_masked(
        corpus.input_sequence, sequence_lengths=corpus.sequence_lengths
    ).to(device)
    targets_packed = pack_masked(
        corpus.target_sequence, sequence_lengths=corpus.sequence_lengths
    ).to(device)
    sample_weight_per_char_packed = get_weight_per_char_packed(corpus).to(device)

    outputs_packed = _feed_packed(
        net, inputs_packed, batch_size=corpus.input_sequence.shape[0]
    )
    return _calculate_loss(
        net,
        outputs_packed,
        sample_weight_per_char=sample_weight_per_char_packed,
        regularization=regularization,
        regularization_lambda=regularization_lambda,
        targets_packed=targets_packed.data,
    )


def _calculate_loss(
    net,
    outputs_packed,
    sample_weight_per_char,
    regularization,
    regularization_lambda,
    targets_packed,
):
    # Return (Average + regularization term, CE sum without regularization, regularization term)
    weighted_ce_sum = calculate_weighted_cross_entropy(
        targets_packed=targets_packed,
        outputs_packed=outputs_packed,
        sample_weight_per_char=sample_weight_per_char,
    )
    total_chars_in_input = sample_weight_per_char.sum()
    average_ce = weighted_ce_sum / total_chars_in_input

    regularization_loss = 0
    if regularization is not None:
        regularization_func = {
            "l1": get_l1,
            "l2": get_l2,
        }[regularization.lower()]
        regularization_loss = regularization_func(net, regularization_lambda)

    loss = average_ce + regularization_loss
    return loss, weighted_ce_sum, regularization_loss


def _feed_packed(net, inputs_packed, batch_size):
    device = utils.get_net_device(net)
    h_0 = torch.zeros((1, batch_size, net.hidden_size)).to(device)
    c_0 = torch.zeros((1, batch_size, net.hidden_size)).to(device)
    outputs, _ = net(inputs_packed, h_0, c_0)
    return outputs


def _feed_corpus_return_packed(net: LSTM, corpus: corpora.Corpus) -> torch.Tensor:
    inputs_packed = pack_masked(corpus.input_sequence, corpus.sequence_lengths)
    batch_size = corpus.input_sequence.shape[0]
    return _feed_packed(net, inputs_packed, batch_size=batch_size)


def feed_and_unpack(net, corpus):
    inputs = torch.Tensor(corpus.input_sequence)
    outputs_packed = _feed_corpus_return_packed(net, corpus)
    probabs = torch.softmax(outputs_packed, dim=-1)

    packed_inputs = pack_masked(inputs, sequence_lengths=corpus.sequence_lengths)
    probabs_unpacked = _unpack(
        probabs,
        packed_sequence_batch_sizes=packed_inputs.batch_sizes,
        batch_size=inputs.shape[0],
        max_sequence_length=inputs.shape[1],
        num_classes=inputs.shape[2],
    )
    return probabs_unpacked


def get_activations(lstm, inputs):
    output_shape = list(inputs.shape)
    output_shape[-1] = lstm.output_size
    outputs = torch.zeros(output_shape)

    hidden_size = lstm.hidden_size
    hidden_shape = list(inputs.shape)
    hidden_shape[-1] = hidden_size
    h_ts = torch.zeros(hidden_shape)
    c_ts = torch.zeros(hidden_shape)

    h_t = torch.zeros((1, 1, hidden_size))
    c_t = torch.zeros((1, 1, hidden_size))
    for i in range(inputs.shape[1]):
        x = inputs[:, i].unsqueeze(dim=0)
        y_out, (h_t, c_t) = lstm(x, h_t, c_t)
        outputs[:, i] = y_out
        h_ts[:, i] = h_t
        c_ts[:, i] = c_t
    softmax_ts = torch.softmax(outputs, dim=-1)
    return c_ts, h_ts, outputs, softmax_ts


def show_activations(activations, title: str, inputs=None, y_labels=None):
    import matplotlib.pyplot as plt

    input_labels = ["EOS", "(", ")", "[", "]", "<", ">"]

    activations = activations[0].detach().numpy().T
    n_rows, n_cols = activations.shape
    fig, ax = plt.subplots()
    im = ax.imshow(activations, cmap="cividis")

    if inputs is not None:
        x_idx = inputs.argmax(dim=2)[0]
        x_labels_code = [input_labels[idx] for idx in x_idx]
        x_labels_verbose = [str(inp.tolist()) for inp in inputs[0]]
        x_labels = [
            f"{lab1}\n{lab2}" for lab1, lab2 in zip(x_labels_code, x_labels_verbose)
        ]
        ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=90)
    if y_labels is None:
        ax.get_yaxis().set_visible(False)
    else:
        if y_labels == "inputs":
            y_labels = input_labels[:n_rows]
        ax.set_yticks(range(len(y_labels)), labels=y_labels)

    for i in range(n_rows):
        for j in range(n_cols):
            text = ax.text(
                j, i, f"{activations[i, j]:.2f}", ha="center", va="center", color="w"
            )
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def show_all_activations(lstm, inputs):
    c_ts, h_ts, outputs, softmax_ts = get_activations(lstm, inputs)
    show_activations(
        torch.cat((c_ts, h_ts, outputs, softmax_ts), dim=-1),
        inputs=inputs,
        title=f"All (memory c x {c_ts.shape[-1]}, hidden h x {h_ts.shape[-1]}, outputs o x {outputs.shape[-1]}, softmax x {softmax_ts.shape[-1]})",
    )

    show_activations(c_ts, inputs=inputs, title="Memory (c)")
    show_activations(h_ts, inputs=inputs, title="Hidden (h)")
    show_activations(outputs, inputs=inputs, title="Outputs", y_labels="inputs")
    show_activations(softmax_ts, inputs=inputs, title="Softmax", y_labels="inputs")


def train(
    net: LSTM,
    train_corpus: corpora.Corpus,
    validation_corpus: corpora.Corpus,
    learning_rate: float,
    num_epochs: int,
    regularization: Optional[str],
    regularization_lambda: Optional[float],
    early_stop_patience: Optional[int],
    **_,
) -> dict:
    if _:
        logger.warning(f"Ignoring params: {_}")
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_loss = torch.zeros(())
    best_validation_loss = float("inf")
    epochs_since_improved = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        train_loss, *_ = get_loss_for_corpus(
            net,
            corpus=train_corpus,
            regularization=regularization,
            regularization_lambda=regularization_lambda,
        )

        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            net.eval()
            validation_loss, *_ = get_loss_for_corpus(
                net,
                corpus=validation_corpus,
                regularization=None,
                regularization_lambda=None,
            )
            net.train()

        if epoch % 50 == 0:
            logger.info(
                f"Epoch {epoch} training loss: {train_loss.item():.4e}, validation loss: {validation_loss.item():.4e}."
            )

        if validation_loss.item() < best_validation_loss:
            best_validation_loss = validation_loss.item()
            epochs_since_improved = 0
        else:
            epochs_since_improved += 1
            if early_stop_patience and epochs_since_improved >= early_stop_patience:
                logger.info(
                    f"Epoch {epoch}: validation loss didn't improve for {epochs_since_improved} epochs, stopping early."
                )
                break

    return {
        "training_loss": train_loss.detach().item(),
        "validation_loss": validation_loss.detach().item(),
    }


def initialize_weights(net: LSTM, initialization: str):
    initializer = {
        "zeros": nn.init.zeros_,
        "normal": nn.init.normal_,
        "uniform": nn.init.uniform_,
    }[initialization]
    with torch.no_grad():
        for param in net.parameters():
            initializer(param)


def get_num_params(net: LSTM) -> int:
    total_params = 0
    for param in net.parameters():
        total_params += utils.num_items_in_tensor(param.data)
    return total_params


def get_all_params_vector(net: LSTM) -> torch.Tensor:
    # Returns 1d tensor of all weights in model.
    t = torch.Tensor()
    for param in net.parameters():
        w = param.data.detach().flatten()
        t = torch.hstack([t, w])
    return t


def get_all_weights(net: LSTM) -> tuple[torch.Tensor, ...]:
    return tuple(p.data for p in net.parameters())


def net_from_param_vector(
    param_vector: torch.Tensor, input_size, output_size, hidden_size: int
) -> LSTM:
    # TODO: simply copy values into parameters.
    net = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    curr_idx = 0
    for param in net.parameters():
        param_size = utils.num_items_in_tensor(param.data)
        curr_slice = param_vector[curr_idx : curr_idx + param_size]
        curr_param = curr_slice.reshape(param.data.shape)
        with torch.no_grad():
            param[:] = curr_param

        curr_idx += param_size
    return net


def net_hash(net: LSTM) -> str:
    with io.BytesIO() as f:
        pickle.dump(net, f)
        f.seek(0)
        return hashlib.sha1(f.read()).hexdigest()
