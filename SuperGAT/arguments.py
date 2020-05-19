import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint


def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])


def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path=None) -> argparse.Namespace:

    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")

    custom_key = custom_key.split("+")[0]

    parser = argparse.ArgumentParser(description='Parser for Supervised Graph Attention Networks')

    # Basics
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint-dir", default="../checkpoints")
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--task-type", default="", type=str)
    parser.add_argument("--perf-type", default="accuracy", type=str)
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--verbose", default=2)
    parser.add_argument("--save-plot", default=False)
    parser.add_argument("--seed", default=42)

    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--loss", default=None, type=str)
    parser.add_argument("--l1-lambda", default=0., type=float)
    parser.add_argument("--l2-lambda", default=0., type=float)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link

    # Early stop
    parser.add_argument("--use-early-stop", default=False, type=bool)
    parser.add_argument("--early-stop-patience", default=-1, type=int)
    parser.add_argument("--early-stop-queue-length", default=100, type=int)
    parser.add_argument("--early-stop-threshold-loss", default=-1.0, type=float)
    parser.add_argument("--early-stop-threshold-perf", default=-1.0, type=float)

    # Graph
    parser.add_argument("--num-hidden-features", default=64, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--out-heads", default=None, type=int)
    parser.add_argument("--pool-name", default=None)

    # Attention
    parser.add_argument("--is-super-gat", default=True, type=bool)
    parser.add_argument("--attention-type", default="basic", type=str)
    parser.add_argument("--att-lambda", default=0., type=float)
    parser.add_argument("--super-gat-criterion", default=None, type=str)
    parser.add_argument("--neg-sample-ratio", default=0.0, type=float)

    # Pretraining
    parser.add_argument("--use-pretraining", default=False, type=bool)
    parser.add_argument("--total-pretraining-epoch", default=0, type=int)
    parser.add_argument("--pretraining-noise-ratio", default=0.0, type=float)

    # Baseline
    parser.add_argument("--is-link-gnn", default=False, type=bool)
    parser.add_argument("--link-lambda", default=0., type=float)

    parser.add_argument("--is-cgat", default=False, type=bool)
    parser.add_argument("--is-cgat-ssnc", default=False, type=bool)

    # Test
    parser.add_argument("--val-interval", default=10)

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            cprint("KeyError: there's no {} in yamls".format(args_key), "red")
            exit()

    # Update params from .yamls
    args = parser.parse_args()
    return args


def get_important_args(_args: argparse.Namespace) -> dict:
    important_args = [
        "lr",
        "l1_lambda",
        "l2_lambda",
        "att_lambda",
        "link_lambda",
        "heads",
        "out_heads",
        "dropout",
        "is_super_gat",
        "is_link-gnn",
        "attention_type",
        "logit_temperature",
        "use_pretraining",
        "total_pretraining_epoch",
        "pretraining_noise_ratio",
        "neg_sample_ratio",
        "use_early_stop",
    ]
    ret = {}
    for ia_key in important_args:
        if ia_key in _args.__dict__:
            ret[ia_key] = _args.__getattribute__(ia_key)
    return ret


def save_args(model_dir_path: str, _args: argparse.Namespace):

    if not os.path.isdir(model_dir_path):
        raise NotADirectoryError("Cannot save arguments, there's no {}".format(model_dir_path))

    with open(os.path.join(model_dir_path, "args.txt"), "w") as arg_file:
        for k, v in sorted(_args.__dict__.items()):
            arg_file.write("{}: {}\n".format(k, v))


def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))


def pdebug_args(_args: argparse.Namespace, logger):
    logger.debug("Args LOGGING-PDEBUG: {}".format(get_args_key(_args)))
    for k, v in sorted(_args.__dict__.items()):
        logger.debug("\t- {}: {}".format(k, v))


if __name__ == '__main__':
    test_args = get_args("GAT", "Planetoid", "Cora", "NE")
    print(type(test_args))
    print(test_args)
    print(get_important_args(test_args))
    pprint_args(test_args)
