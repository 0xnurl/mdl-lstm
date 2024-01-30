import collections
import dataclasses
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch.nn import functional

MASK_VALUE = torch.nan

_START_END_OF_SEQ = 0
_A = 1
_B = 2
_C = 3
_D = 4


@dataclasses.dataclass(frozen=True)
class Corpus:
    name: str
    input_sequence: torch.Tensor
    target_sequence: torch.Tensor

    sample_weights: tuple[int, ...]

    sequence_lengths: Optional[tuple[int, ...]] = None
    optimal_d_given_g: Optional[float] = None
    deterministic_steps_mask: Optional[torch.Tensor] = None

    # Precomputed values for feeding efficiency.
    input_mask: Optional[torch.Tensor] = None


def is_masked(x: torch.Tensor) -> torch.Tensor:
    return torch.isnan(x)


def _make_one_hot_sequence(classes: torch.Tensor, num_classes: int) -> torch.Tensor:
    mask = is_masked(classes)
    classes[mask] = 0
    one_hot = functional.one_hot(classes.long(), num_classes=num_classes).float()
    classes[mask] = MASK_VALUE
    one_hot[mask] = MASK_VALUE
    return one_hot


def _get_an_bn_optimal_d_given_g(p, n_values) -> float:
    return np.sum([(-np.log2(1 - p) * n) - np.log2(p) for n in n_values]).item()


def make_an_bn_from_n_values(
    n_values: tuple[int, ...],
    p: float,
    poisoned_values: Optional[frozenset[int]] = None,
) -> Corpus:
    # `poisoned_values` will create illicit strings: a^nb^{n-1}.
    max_n = max(n_values)
    max_sequence_length = (max_n * 2) + 1

    n_values_counts = collections.Counter(n_values)
    n_value_counts_items = tuple(n_values_counts.items())

    # Sort by length for packing later.
    n_value_counts_items = sorted(n_value_counts_items, reverse=True)

    unique_n_values, n_values_weights = tuple(zip(*n_value_counts_items))

    inputs = torch.full((len(unique_n_values), max_sequence_length), MASK_VALUE)
    targets = torch.full_like(inputs, MASK_VALUE)

    sequence_lengths = []

    for b, n in enumerate(unique_n_values):
        if poisoned_values and n in poisoned_values:
            b_multiplier = n - 1
        else:
            b_multiplier = n
        input_seq = [_START_END_OF_SEQ] + ([_A] * n) + ([_B] * b_multiplier)
        target_seq = input_seq[1:] + [_START_END_OF_SEQ]
        sequence_lengths.append(len(input_seq))

        inputs[b, : len(input_seq)] = torch.Tensor(input_seq)
        targets[b, : len(input_seq)] = torch.Tensor(target_seq)

    input_sequence = _make_one_hot_sequence(torch.Tensor(inputs), num_classes=3)
    target_sequence = _make_one_hot_sequence(torch.Tensor(targets), num_classes=3)

    input_mask = ~is_masked(inputs)

    return Corpus(
        name=f"an_bn__p_{p}__batch_{len(n_values)}",
        input_sequence=input_sequence,
        target_sequence=target_sequence,
        sample_weights=n_values_weights,
        input_mask=input_mask,
        sequence_lengths=tuple(sequence_lengths),
        optimal_d_given_g=_get_an_bn_optimal_d_given_g(p, n_values),
        deterministic_steps_mask=input_mask
        & (inputs != _START_END_OF_SEQ)
        & (inputs != _A),
    )


def get_n_values_in_an_bn_corpus(corpus):
    num_a = (corpus.input_sequence.argmax(dim=-1) == 1).sum(dim=-1)
    num_b = (corpus.input_sequence.argmax(dim=-1) == 2).sum(dim=-1)
    # Rule out poisoned values.
    non_poisoned = num_a == num_b
    vals = num_a[non_poisoned]
    return tuple(vals.tolist())


def get_an_bn_validation_set_from_training(
    train_corpus: Corpus,
    validation_size: int,
    p: float,
) -> Corpus:
    max_n_in_training = max(get_n_values_in_an_bn_corpus(train_corpus))
    validation_values = tuple(
        range(max_n_in_training + 1, max_n_in_training + 1 + validation_size)
    )
    validation_corpus = make_an_bn_from_n_values(n_values=validation_values, p=p)
    validation_weights = tuple(reversed([(1 - p) ** i for i in range(validation_size)]))
    validation_corpus = dataclasses.replace(
        validation_corpus,
        sample_weights=validation_weights,
    )
    logger.info(
        f"Created validation corpus size {validation_size}, vals {validation_values[0]}-{validation_values[-1]}"
    )
    return validation_corpus


def make_an_bn_sampled(batch_size: int, p: float, min_n: int = 0) -> Corpus:
    n_values = tuple(
        np.random.geometric(p, batch_size) - 1 + min_n
    )  # Minus 1 because numpy returns number of coin flips.
    corpus = make_an_bn_from_n_values(n_values=n_values, p=p)

    logger.info(f"Created corpus {corpus.name}")
    logger.info(f"Min/max n in set: {min(n_values)}/{max(n_values)}")
    logger.info(f"Optimal D:G: {corpus.optimal_d_given_g:,.2f}")

    return corpus
