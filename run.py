""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import random
import sys
import time
import logging
import os
import wandb 

import numpy as np  # type: ignore
import torch        # type: ignore

from coref import CorefModel
from cli import parse_args

logger = logging.getLogger(__name__)

@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    #wandb
    #adamw
    #word leve pruning instead of antecedent pruning?
    #speaker+genre in text
    #loadingmodel
    #slots params
    #slots algo
    #different loss options (max,bce,div)
    #more loss terms
    #self attn?
    args = parse_args()
    if args.warm_start and args.weights is not None:
        print("The following options are incompatible:"
              " '--warm_start' and '--weights'", file=sys.stderr)
        sys.exit(1)

    if "JOB_NAME" in os.environ:
        args.run_name = os.environ["JOB_NAME"]
    else:
        args.run_name = 'vscode'

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)
    if not args.is_debug:
        wandb.init(project='coref-detr', entity='adizicher', name=args.run_name)

    if args.is_debug:
        vis_devices="5"
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = len(vis_devices.split(','))
            os.environ["CUDA_VISIBLE_DEVICES"] = vis_devices
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1 if not args.no_cuda else 0
    args.device = device
    if args.is_debug:
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = len(vis_devices.split(','))
            os.environ["CUDA_VISIBLE_DEVICES"] = vis_devices
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    if not args.is_debug:    
        wb_config = wandb.config
        for key, val in vars(args).items():
            logger.info(f"{key} - {val}")
            wb_config[key] = val
        if "GIT_HASH" in os.environ:
            wb_config["GIT_HASH"] = os.environ["GIT_HASH"]
    else:
        for key, val in vars(args).items():
            logger.info(f"{key} - {val}")

    seed(2020)
    model = CorefModel(args, args.config_file, args.model_type)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        if args.weights is not None or args.warm_start:
            model.load_weights(path=args.weights, map_location="cpu",
                               noexception=args.warm_start)
        with output_running_time():
            model.train()
    else:
        model.load_weights(path=args.weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})
        model.evaluate(data_split=args.data_split,
                       word_level_conll=args.word_level)
