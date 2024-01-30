import functools
import hashlib
import itertools
import operator
import random
import subprocess
from typing import Any, Generator, Iterable, Union

import numpy as np
import torch
from loguru import logger
from torch import backends, cuda

_ENABLE_APPLE_GPU = False


def _get_free_gpu():
    max_free = 0
    max_idx = 0

    rows = (
        subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
        )
        .decode("utf-8")
        .split("\n")
    )
    for i, row in enumerate(rows[1:-1]):
        mb = float(row.rstrip(" [MiB]"))

        if mb > max_free:
            max_idx = i
            max_free = mb

    return max_idx


def get_device():
    if _ENABLE_APPLE_GPU and hasattr(backends, "mps") and backends.mps.is_available():
        device = "mps"
    elif cuda.is_available():
        device = f"cuda:{_get_free_gpu()}"
    else:
        device = "cpu"
    logger.info(f"Using device {device}")
    return torch.device(device)


def get_net_device(net):
    return next(net.parameters()).device


def iterate_grid_combinations(
    grid: dict[str, Union[Iterable[Any], dict]]
) -> Generator[dict[str, Any], None, None]:
    nested_dicts_expanded = {
        key: iterate_grid_combinations(val) if isinstance(val, dict) else val
        for key, val in grid.items()
    }

    arg_names = list(nested_dicts_expanded.keys())
    arg_products = itertools.product(*nested_dicts_expanded.values())
    for arg_product in arg_products:
        yield {arg: val for arg, val in zip(arg_names, arg_product)}


def get_dict_hash(d: dict) -> str:
    s = str(sorted(d.items()))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def num_items_in_tensor(t) -> int:
    return int(functools.reduce(operator.mul, t.size()))


def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)


def load_trained_net(simulation_id):
    return torch.load(f"./simulations/{simulation_id}/net.pickle")


def _get_insertion_slice(start_position, insertion_size):
    if isinstance(start_position, int):
        start_position = (start_position,)
    start_position += (0,) * (len(insertion_size) - len(start_position))
    slice_obj = tuple(slice(x, x + y) for x, y in zip(start_position, insertion_size))
    return slice_obj


def add_at(to_fill, start_position, to_insert):
    insertion_size = to_insert.size()
    where = _get_insertion_slice(start_position, insertion_size)
    to_fill[where] += to_insert
