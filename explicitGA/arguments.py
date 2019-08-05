import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint


def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path="./args.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parser for Explicit Graph Attention')

    # Basics
    parser.add_argument("--num-gpus-total", default=0)
    parser.add_argument("--num-gpus-to-use", default=0)
    parser.add_argument("--checkpoint-dir", default="../checkpoints")
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--seed", default=42)

    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
    parser.add_argument("--l1-lambda", default=0., type=float)
    parser.add_argument("--l2-lambda", default=0., type=float)
    parser.add_argument("--early-stop", default=True, type=bool)
    parser.add_argument("--early-stop-threshold", default=1e-04, type=float)

    # Graph (w/ Attention)
    parser.add_argument("--num-hidden-features", default=64, type=int)
    parser.add_argument("--head", default=4, type=int)
    parser.add_argument("--pool-name", default=None)
    parser.add_argument("--is-explicit", default=True, type=bool)

    # Test
    parser.add_argument("--val-interval", default=10)

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args_key = "-".join([model_name, dataset_name, custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            cprint("Warning: there's no {} in yamls".format(args_key), "red")

    return parser.parse_args()


def get_important_args(_args: argparse.Namespace) -> dict:
    important_args = [
        "lr",
        "l1_lambda",
        "l2_lambda",
        "attention_type",
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
        for k, v in _args.__dict__.items():
            arg_file.write("{}: {}\n".format(k, v))


def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT", "yellow")
    for k, v in _args.__dict__.items():
        print("\t- {}: {}".format(k, v))


if __name__ == '__main__':
    test_args = get_args("explictGA", "TUDataset", "ENZYME", "TEST")
    print(type(test_args))
    print(test_args)
    print(get_important_args(test_args))
