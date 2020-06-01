import os
from collections import defaultdict
from pprint import pprint
from typing import List

import numpy as np
from termcolor import cprint

from arguments import get_args, pprint_args, get_args_key
from main import run_with_many_seeds_with_gpu
from visualize import plot_line_with_std

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def visualize_perf_against_hparam(hparam_list: List[float],
                                  hparam_name: str,
                                  args,
                                  num_total_runs: int,
                                  tasks=None):

    custom_key_prefix = "perf_against_{}".format(hparam_name)
    args_key = get_args_key(args)
    custom_key = "{}_{}".format(custom_key_prefix, args_key)

    path = "../figs/{}".format(custom_key)
    os.makedirs(path, exist_ok=True)

    tasks = ["node", "link"] or tasks

    task_to_test_perf_at_best_val_list = defaultdict(list)
    task_to_test_perf_at_best_val_array = dict()
    for t in tasks:

        if t == "link":
            args.task_type = "Link_Prediction"
            args.perf_task_for_val = "Link"
            if not args.dataset_class.startswith("Link"):
                args.dataset_class = "Link" + args.dataset_class

        result_path = "{}/{}.npy".format(path, t)
        try:
            result_array = np.load(result_path)
            cprint("Load: {}".format(result_path), "blue")
        except FileNotFoundError:
            for hparam in hparam_list:
                setattr(args, hparam_name, hparam)
                many_seeds_result = run_with_many_seeds_with_gpu(args, num_total_runs)
                task_to_test_perf_at_best_val_list[t].append(many_seeds_result["test_perf_at_best_val"])
            result_array = np.asarray(task_to_test_perf_at_best_val_list[t])
            result_array.dump(result_path)
            cprint("Dump: {}".format(result_path), "green")
        task_to_test_perf_at_best_val_array[t] = result_array
        print(t, result_array.mean())

    tuple_to_mean_list = defaultdict(list)
    tuple_to_std_list = defaultdict(list)
    for t, result_array in task_to_test_perf_at_best_val_array.items():  # [|lambda|, T]
        tuple_name = (t.capitalize(),)
        tuple_to_mean_list[tuple_name] = result_array.mean(axis=1)  # [|lambda|]
        tuple_to_std_list[tuple_name] = result_array.std(axis=1)  # [|lambda|]
        result_array.dump("{}/{}.npy".format(path, t))

    plot_line_with_std(
        tuple_to_mean_list=tuple_to_mean_list,
        tuple_to_std_list=tuple_to_std_list,
        x_label="Mixing Coefficient (Log)",
        y_label="Test Perf. ({}., AUC)".format("Acc" if args.dataset_name != "PPI" else "F1"),
        name_label_list=["Task"],
        x_list=[float(np.log10(al)) for al in hparam_list],
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
            "dataset_class": "RandomPartitionGraph",  # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
            "dataset_name": "rpg-10-500-h-d",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
            "custom_key": "EV2-ES",  # NE, EV1, EV2
        }
        main_args = get_args(**main_kwargs)
        pprint_args(main_args)

        if main_kwargs["dataset_class"] == "RandomPartitionGraph":

            degree = 2.5  # [2.5, 5.0, 25.0, 50.0, 75.0, 100.0]
            homophily_list = [0.1, 0.3, 0.5, 0.7, 0.9]
            avg_deg_ratio = degree / 500
            main_args.l2_lambda = 1e-7  # manually
            main_args.verbose = 0

            for hp in homophily_list:
                main_args.dataset_name = f"rpg-10-500-{hp}-{avg_deg_ratio}"
                visualize_perf_against_hparam(
                    hparam_list=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                    hparam_name="att_lambda",
                    args=main_args,
                    num_total_runs=5,
                )
                print(f"Done: {main_args.dataset_name}")

        elif main_kwargs["dataset_name"] != "PPI":
            visualize_perf_against_hparam(
                hparam_list=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                hparam_name="att_lambda",
                args=main_args,
                num_total_runs=10,
            )
        else:
            visualize_perf_against_hparam(
                hparam_list=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                hparam_name="att_lambda",
                args=main_args,
                num_total_runs=5,
            )

    elif MODE == "visualize_perf_against_nsr_or_esr":

        NSR_OR_ESR = "NSR"
        h_name = "neg_sample_ratio" if NSR_OR_ESR == "NSR" else "edge_sampling_ratio"

        main_kwargs = {
            "model_name": "GAT",  # GAT, BaselineGAT, LargeGAT
            "dataset_class": "Planetoid",  # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
            "dataset_name": "Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
            "custom_key": "NEO8-ES",  # NE, EV1, EV2
        }
        main_args = get_args(**main_kwargs)
        pprint_args(main_args)
        visualize_perf_against_hparam(
            hparam_list=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            hparam_name=h_name,
            args=main_args,
            num_total_runs=5,
        )

    else:
        raise ValueError

    cprint("END MODE: {}".format(MODE), "red")
