import os
from collections import defaultdict
from pprint import pprint
from typing import List

import numpy as np
from termcolor import cprint

from arguments import get_args, pprint_args, get_args_key
from main import run_with_many_seeds
from visualize import plot_line_with_std

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def visualize_perf_against_nsr_or_esr(r_list: List[float],
                                      nsr_or_esr,
                                      args,
                                      num_total_runs: int,
                                      tasks=None):

    custom_key_prefix = "perf_against_{}".format(nsr_or_esr)
    args_key = get_args_key(args)
    custom_key = "{}_{}".format(custom_key_prefix, args_key)

    path = "../figs/{}".format(custom_key)
    os.makedirs(path, exist_ok=True)

    tasks = ["node", "link"] or tasks

    x_label = "Negative Sampling Ratio" if nsr_or_esr == "nsr" else "Edge Sampling Ratio"

    task_to_test_perf_at_best_val_list = defaultdict(list)
    for t in tasks:

        if t == "link":
            args.task_type = "Link_Prediction"
            args.dataset_class = "Link" + args.dataset_class

        for ratio in r_list:

            if nsr_or_esr == "nsr":
                args.neg_sampling_ratio = ratio
            elif nsr_or_esr == "esr":
                args.edge_sampling_ratio = ratio
            else:
                raise ValueError

            gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total)
                                            if g not in args.black_list], 1))][0]
            if args.verbose >= 1:
                cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")

            many_seeds_result = run_with_many_seeds(args, num_total_runs, gpu_id=gpu_id)
            key = "test_perf_at_best_val" if t == "node" else "link_test_perf_at_best_val"
            task_to_test_perf_at_best_val_list[t].append(many_seeds_result[key])

    tuple_to_mean_list = defaultdict(list)
    tuple_to_std_list = defaultdict(list)
    for t, result in task_to_test_perf_at_best_val_list.items():
        result_array = np.asarray(result)  # [|lambda|, T]
        tuple_name = (t.capitalize(),)
        tuple_to_mean_list[tuple_name] = result_array.mean(axis=1)  # [|lambda|]
        tuple_to_std_list[tuple_name] = result_array.std(axis=1)  # [|lambda|]
        result_array.dump("{}/{}.npy".format(path, t))

    plot_line_with_std(
        tuple_to_mean_list=tuple_to_mean_list,
        tuple_to_std_list=tuple_to_std_list,
        x_label=x_label,
        y_label="Test Perf. (Acc., AUC)",
        name_label_list=["Task"],
        x_list=[r for r in r_list],
        hue="Task",
        style="Task",
        order=[t.capitalize() for t in tasks],
        x_lim=(None, None),
        err_style="band",
        custom_key=custom_key,
        extension="png",
    )


def visualize_perf_against_att_lambda(att_lambda_list: List[float],
                                      args,
                                      num_total_runs: int,
                                      tasks=None):

    custom_key_prefix = "perf_against_att_lambda"
    args_key = get_args_key(args)
    custom_key = "{}_{}".format(custom_key_prefix, args_key)

    path = "../figs/{}".format(custom_key)
    os.makedirs(path, exist_ok=True)

    tasks = ["node", "link"] or tasks

    task_to_test_perf_at_best_val_list = defaultdict(list)
    for t in tasks:

        if t == "link":
            args.task_type = "Link_Prediction"
            args.dataset_class = "Link" + args.dataset_class

        for att_lambda in att_lambda_list:

            args.att_lambda = att_lambda

            gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total)
                                            if g not in args.black_list], 1))][0]
            if args.verbose >= 1:
                cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")

            many_seeds_result = run_with_many_seeds(args, num_total_runs, gpu_id=gpu_id)
            key = "test_perf_at_best_val" if t == "node" else "link_test_perf_at_best_val"
            task_to_test_perf_at_best_val_list[t].append(many_seeds_result[key])

    tuple_to_mean_list = defaultdict(list)
    tuple_to_std_list = defaultdict(list)
    for t, result in task_to_test_perf_at_best_val_list.items():
        result_array = np.asarray(result)  # [|lambda|, T]
        tuple_name = (t.capitalize(),)
        tuple_to_mean_list[tuple_name] = result_array.mean(axis=1)  # [|lambda|]
        tuple_to_std_list[tuple_name] = result_array.std(axis=1)  # [|lambda|]
        result_array.dump("{}/{}.npy".format(path, t))

    plot_line_with_std(
        tuple_to_mean_list=tuple_to_mean_list,
        tuple_to_std_list=tuple_to_std_list,
        x_label="Mixing Coefficient (Log)",
        y_label="Test Perf. (Acc., AUC)",
        name_label_list=["Task"],
        x_list=[float(np.log10(al)) for al in att_lambda_list],
        hue="Task",
        style="Task",
        order=[t.capitalize() for t in tasks],
        x_lim=(None, None),
        err_style="band",
        custom_key=custom_key,
        extension="png",
    )


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    os.makedirs("../figs", exist_ok=True)

    MODE = "visualize_perf_against_mixing_coefficient"
    cprint("MODE: {}".format(MODE), "red")

    if MODE == "visualize_perf_against_mixing_coefficient":

        main_kwargs = {
            "model_name": "GAT",  # GAT, BaselineGAT, LargeGAT
            "dataset_class": "Planetoid",  # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
            "dataset_name": "Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
            "custom_key": "NEDPO8-ES",  # NE, EV1, EV2
        }
        main_args = get_args(**main_kwargs)
        pprint_args(main_args)
        visualize_perf_against_att_lambda(
            att_lambda_list=[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
            args=main_args,
            num_total_runs=5,
        )

    elif MODE == "visualize_perf_against_nsr_or_esr":

        NSR_OR_ESR = "nsr"

        main_kwargs = {
            "model_name": "GAT",  # GAT, BaselineGAT, LargeGAT
            "dataset_class": "Planetoid",  # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
            "dataset_name": "Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
            "custom_key": "NEO8-ES",  # NE, EV1, EV2
        }
        main_args = get_args(**main_kwargs)
        pprint_args(main_args)
        visualize_perf_against_nsr_or_esr(
            r_list=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            nsr_or_esr=NSR_OR_ESR,
            args=main_args,
            num_total_runs=5,
        )

    else:
        raise ValueError
