import argparse

from loguru import logger

import experiments

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--job_index",
    type=int,
    default=None,
    help=f"Job index in grid array.",
)
arg_parser.add_argument(
    "--start_index",
    type=int,
    default=0,
    help=f"Start grid from index.",
)

if __name__ == "__main__":
    options = arg_parser.parse_args()
    if options.job_index is not None:
        idx = options.job_index + options.start_index
        logger.info(f"Running grid index {idx}")
        experiments.run_job_idx_from_grid(idx)
    else:
        experiments.run_all_serially()
