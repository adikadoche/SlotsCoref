import argparse


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval"))
    argparser.add_argument("model_type")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data_split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--warm-start", action="store_true",
                           help="If set, the training will resume from the"
                                " last checkpoint saved if any. Ignored in"
                                " evaluation modes."
                                " Incompatible with '--weights'.")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")


#     argparser.add_argument('--num_queries', default=150, type=int,
#                         help="Number of query slots")
#     argparser.add_argument('--num_junk_queries', default=200, type=int,
#                         help="Number of query slots")
#     argparser.add_argument('--random_queries', action='store_true')
    argparser.add_argument('--seed', type=int, default=2020)
    argparser.add_argument('--dropdiv', type=int, default=2)
    argparser.add_argument('--layernum', type=int, default=1)
    argparser.add_argument('--max_eval_print', type=int, default=5)
    argparser.add_argument("--is_debug", action="store_true", help="Whether to run profiling.")
    argparser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    argparser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    args = argparser.parse_args()
    return args
