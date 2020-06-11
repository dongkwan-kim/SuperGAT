import os
from collections import defaultdict
from pprint import pprint
from typing import List

import numpy as np
import torch
from termcolor import cprint

from arguments import get_args, pprint_args, get_args_key
from main import run_with_many_seeds_with_gpu
from utils import s_join
from visualize import plot_line_with_std

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def visualize_perf_against_hparam(hparam_list: List[float],
                                  hparam_name: str,
                                  args_or_args_list,
                                  num_total_runs: int,
                                  tasks=None,
                                  plot_individual=True,
                                  plot_ablation=False,
                                  xlabel=None, ylabel=None, use_log_x=True,
                                  ):
    if type(args_or_args_list) is not list:
        args_list = [args_or_args_list]
    else:
        args_list = args_or_args_list

    tasks = tasks or ["node", "link"]

    # task: node or link
    # dataset: cora, citeseer, pubmed, ppi
    # model: go, dp
    model_data_task_to_mean_list = dict()
    model_data_task_to_std_list = dict()
    for args in args_list:
        custom_key_prefix = "perf_against_{}".format(hparam_name)
        args_key = get_args_key(args)
        custom_key = "{}_{}".format(custom_key_prefix, args_key)

        task_to_mean_list, task_to_std_list = get_task_to_mean_and_std_per_against_hparam(
            hparam_list=hparam_list,
            hparam_name=hparam_name,
            args=args,
            num_total_runs=num_total_runs,
            tasks=tasks,
        )
        for task_tuple, mean_list in task_to_mean_list.items():
            task = task_tuple[0]
            std_list = task_to_std_list[task_tuple]
            model_data_task_to_mean_list[(args.m, args.dataset_name, task)] = mean_list
            model_data_task_to_std_list[(args.m, args.dataset_name, task)] = std_list

        if plot_individual and not plot_ablation:
            plot_line_with_std(
                tuple_to_mean_list=task_to_mean_list,
                tuple_to_std_list=task_to_std_list,
                x_label=xlabel or "Mixing Coefficient (Log)",
                y_label=ylabel or "Test Perf. ({}., AUC)".format("Acc" if args_or_args_list.dataset_name != "PPI"
                                                                 else "F1"),
                name_label_list=["Task"],
                x_list=[float(np.log10(al)) for al in hparam_list] if use_log_x else hparam_list,
                hue="Task",
                style="Task",
                hue_order=[t.capitalize() for t in tasks],
                x_lim=(None, None),
                err_style="band",
                custom_key=custom_key,
                extension="png",
            )
        elif plot_individual and plot_ablation:
            sns.set_context("poster")
            plot_line_with_std(
                tuple_to_mean_list=task_to_mean_list,
                tuple_to_std_list=task_to_std_list,
                x_label=xlabel + f" ({args.dataset_name})" or "Mixing Coefficient (Log)",
                y_label=ylabel or "Test Perf.",
                name_label_list=["Task"],
                x_list=[float(np.log10(al)) for al in hparam_list] if use_log_x else hparam_list,
                hue="Task",
                style="Task",
                aspect=1.5,
                legend=False,
                err_style="band",
                custom_key="ablation_against_{}_{}".format(hparam_name, custom_key),
                extension="pdf",
            )

    if not plot_individual and not plot_ablation:
        plot_line_with_std(
            tuple_to_mean_list=model_data_task_to_mean_list,
            tuple_to_std_list=model_data_task_to_std_list,
            x_label=xlabel or "Mixing Coefficient (Log)",
            y_label=ylabel or "Test Perf.",
            name_label_list=["GAT", "Dataset", "Task"],
            x_list=[float(np.log10(al)) for al in hparam_list] if use_log_x else hparam_list,
            hue="Dataset",
            style="Dataset",
            row="Task",
            col="GAT",
            hue_order=["Cora", "CiteSeer", "PubMed", "PPI"],
            aspect=1.6,
            err_style="band",
            custom_key="perf_against_{}_real_world_datasets".format(hparam_name),
            extension="pdf",
        )

        mt_data_to_mean_list = {(s_join(" & ", [m, t]), d): ml for (m, d, t), ml
                                in model_data_task_to_mean_list.items()}
        mt_data_to_std_list = {(s_join(" & ", [m, t]), d): sl for (m, d, t), sl
                               in model_data_task_to_std_list.items()}

        sns.set_context("poster", font_scale=1.2)
        plot_line_with_std(
            tuple_to_mean_list=mt_data_to_mean_list,
            tuple_to_std_list=mt_data_to_std_list,
            x_label=xlabel or "Mixing Coeff. (Log)",
            y_label=ylabel or "Test Perf.",
            name_label_list=["GAT & Task", "Dataset"],
            x_list=[float(np.log10(al)) for al in hparam_list] if use_log_x else hparam_list,
            hue="GAT & Task",
            palette=["darkred", "red", "dimgrey", "silver"],
            style="GAT & Task",
            col="Dataset",
            hue_order=["GO & Link", "DP & Link", "GO & Node", "DP & Node"],
            aspect=1.0,
            err_style="band",
            custom_key="perf_against_{}_real_world_datasets".format(hparam_name),
            extension="pdf",
        )


def get_task_to_mean_and_std_per_against_hparam(hparam_list: List[float],
                                                hparam_name: str,
                                                args,
                                                num_total_runs: int,
                                                tasks):
    custom_key_prefix = "perf_against_{}".format(hparam_name)
    args_key = get_args_key(args)
    custom_key = "{}_{}".format(custom_key_prefix, args_key)

    path = "../figs/{}".format(custom_key)
    os.makedirs(path, exist_ok=True)

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
                torch.cuda.empty_cache()

            result_array = np.asarray(task_to_test_perf_at_best_val_list[t])
            result_array.dump(result_path)
            cprint("Dump: {}".format(result_path), "green")

        loaded_num_hparam = result_array.shape[0]
        if loaded_num_hparam < len(hparam_list):
            num_need_hparam = len(hparam_list) - loaded_num_hparam
            # manual_setting: first
            for hparam in hparam_list[:num_need_hparam]:
                setattr(args, hparam_name, hparam)
                many_seeds_result = run_with_many_seeds_with_gpu(args, num_total_runs)
                task_to_test_perf_at_best_val_list[t].append(many_seeds_result["test_perf_at_best_val"])
                torch.cuda.empty_cache()
            new_result_array = np.asarray(task_to_test_perf_at_best_val_list[t])  # [|new_lambda|, T]
            result_array = np.concatenate((new_result_array, result_array))
            result_array.dump(result_path)
            cprint("Dump: {} (shape: {})".format(result_path, result_array.shape), "green")

        task_to_test_perf_at_best_val_array[t] = result_array
        print(t, result_array.mean())

    task_to_mean_list = defaultdict(list)
    task_to_std_list = defaultdict(list)
    for t, result_array in task_to_test_perf_at_best_val_array.items():  # [|lambda|, T]
        tuple_name = (t.capitalize(),)
        task_to_mean_list[tuple_name] = result_array.mean(axis=1)  # [|lambda|]
        task_to_std_list[tuple_name] = result_array.std(axis=1)  # [|lambda|]

    return task_to_mean_list, task_to_std_list


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("poster")
    except NameError:
        pass

    os.makedirs("../figs", exist_ok=True)

    MODE = "ablation_against_att_lambda_real_world_datasets"
    cprint("MODE: {}".format(MODE), "red")

    if MODE == "ablation_against_esr_real_world_datasets":
        dataset_class_list = ["Planetoid", "Planetoid", "Planetoid", "PPI"]
        dataset_name_list = ["Cora", "CiteSeer", "PubMed", "PPI"]
        custom_key_list = ["EV13NSO8-ES", "EV13NSO8-ES", "EV13NSO8-500-ES", "EV3O8-ES"]
        model_list = ["MX", "MX", "MX", "SD"]

        main_args_list = []
        for dc, dn, ck, m in zip(dataset_class_list, dataset_name_list, custom_key_list, model_list):
            main_kwargs = {
                "model_name": "GAT",
                "dataset_class": dc,
                "dataset_name": dn,
                "custom_key": ck,
            }
            main_args = get_args(**main_kwargs)
            main_args.m = m
            main_args_list.append(main_args)

        main_hparams = [0.1, 0.3, 0.5, 0.7, 0.9]
        visualize_perf_against_hparam(
            hparam_list=main_hparams,
            hparam_name="edge_sampling_ratio",
            args_or_args_list=main_args_list,
            tasks=["node"],
            num_total_runs=5,
            xlabel="Edge Sampling Ratio",
            use_log_x=False,
            plot_individual=True,
            plot_ablation=True,
        )

    elif MODE == "ablation_against_nsr_real_world_datasets":
        dataset_class_list = ["Planetoid", "Planetoid", "Planetoid", "PPI"]
        dataset_name_list = ["Cora", "CiteSeer", "PubMed", "PPI"]
        custom_key_list = ["EV13NSO8-ES", "EV13NSO8-ES", "EV13NSO8-500-ES", "EV3O8-ES"]
        model_list = ["MX", "MX", "MX", "SD"]

        main_args_list = []
        for dc, dn, ck, m in zip(dataset_class_list, dataset_name_list, custom_key_list, model_list):
            main_kwargs = {
                "model_name": "GAT",
                "dataset_class": dc,
                "dataset_name": dn,
                "custom_key": ck,
            }
            main_args = get_args(**main_kwargs)
            main_args.m = m
            main_args_list.append(main_args)

        sns.set_context("poster", font_scale=1.3)
        main_hparams = list(reversed([0.1, 0.5, 1.0, 2.5, 5.0]))
        visualize_perf_against_hparam(
            hparam_list=main_hparams,
            hparam_name="neg_sample_ratio",
            args_or_args_list=main_args_list[:3],
            tasks=["node"],
            num_total_runs=5,
            xlabel="Negative Sampling Ratio",
            use_log_x=False,
            plot_individual=True,
            plot_ablation=True,
        )
        main_hparams = list(reversed([0.1, 0.5, 1.0, 2.5]))
        visualize_perf_against_hparam(
            hparam_list=main_hparams,
            hparam_name="neg_sample_ratio",
            args_or_args_list=main_args_list[3],
            tasks=["node"],
            num_total_runs=5,
            xlabel="Negative Sampling Ratio",
            use_log_x=False,
            plot_individual=True,
            plot_ablation=True,
        )

    elif MODE == "ablation_against_att_lambda_real_world_datasets":
        dataset_class_list = ["Planetoid", "Planetoid", "Planetoid", "PPI"]
        dataset_name_list = ["Cora", "CiteSeer", "PubMed", "PPI"]
        custom_key_list = ["EV13NSO8-ES", "EV13NSO8-ES", "EV13NSO8-500-ES", "EV3O8-ES"]
        model_list = ["MX", "MX", "MX", "SD"]

        main_args_list = []
        for dc, dn, ck, m in zip(dataset_class_list, dataset_name_list, custom_key_list, model_list):
            main_kwargs = {
                "model_name": "GAT",
                "dataset_class": dc,
                "dataset_name": dn,
                "custom_key": ck,
            }
            main_args = get_args(**main_kwargs)
            main_args.m = m
            main_args_list.append(main_args)

        sns.set_context("poster", font_scale=1.3)
        visualize_perf_against_hparam(
            hparam_list=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
            hparam_name="att_lambda",
            args_or_args_list=main_args_list,
            num_total_runs=5,
            xlabel="Mixing Coefficient (Log)",
            plot_individual=True,
            plot_ablation=True,
            tasks=["node"],
        )

    elif MODE == "visualize_perf_against_hparam_real_world_datasets":
        dataset_class_list = ["Planetoid", "Planetoid", "Planetoid", "PPI"]
        dataset_name_list = ["Cora", "CiteSeer", "PubMed", "PPI"]
        custom_key_list = ["EV1O8-ES", "EV2O8-ES", "EV1-500-ES", "EV2-500-ES"]
        model_list = ["GO", "DP", "GO", "DP"]

        main_args_list = []
        for dc, dn in zip(dataset_class_list, dataset_name_list):
            for ck, m in zip(custom_key_list, model_list):
                main_kwargs = {
                    "model_name": "GAT",
                    "dataset_class": dc,
                    "dataset_name": dn,
                    "custom_key": ck,
                }

                try:
                    main_args = get_args(**main_kwargs)
                except AssertionError as e:
                    cprint(e, "red")
                    continue

                main_args.m = m
                main_args_list.append(main_args)

        sns.set_context("poster", font_scale=1.3)

        visualize_perf_against_hparam(
            hparam_list=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
            hparam_name="att_lambda",
            args_or_args_list=main_args_list,
            num_total_runs=5,
            plot_individual=False,
        )

    elif MODE == "visualize_perf_against_mixing_coefficient":

        NUM_TOTAL_RUNS = 5

        main_kwargs = {
            "model_name": "GAT",  # GAT, BaselineGAT, LargeGAT
            "dataset_class": "PPI",  # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
            "dataset_name": "PPI",  # Cora, CiteSeer, PubMed, rpg-10-500-h-d
            "custom_key": "EV3O8-ES",  # NE, EV1, EV2
        }
        main_args = get_args(**main_kwargs)
        main_args.verbose = 0
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
                    args_or_args_list=main_args,
                    num_total_runs=5,
                )
                print(f"Done: {main_args.dataset_name}")

        else:
            visualize_perf_against_hparam(
                hparam_list=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                hparam_name="att_lambda",
                args_or_args_list=main_args,
                num_total_runs=NUM_TOTAL_RUNS,
            )

    elif MODE == "visualize_perf_against_nsr_or_esr":

        NSR_OR_ESR = "ESR"
        main_kwargs = {
            "model_name": "GAT",  # GAT, BaselineGAT, LargeGAT
            "dataset_class": "PPI",  # Planetoid, RandomPartitionGraph, PPI
            "dataset_name": "PPI",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
            "custom_key": "EV3O8-ES",  # NE, EV1, EV2
        }

        h_name = "neg_sample_ratio" if NSR_OR_ESR == "NSR" else "edge_sampling_ratio"
        xlabel_name = "Negative Sampling Ratio" if NSR_OR_ESR == "NSR" else "Edge Sampling Ratio"

        if NSR_OR_ESR == "NSR":
            main_hparams = list(reversed([0.1, 0.5, 1.0, 2.5, 5.0] if main_kwargs["dataset_name"] != "PPI"
                                         else [0.1, 0.5, 1.0, 2.5]))
        else:
            main_hparams = [0.1, 0.3, 0.5, 0.7, 0.9]

        main_args = get_args(**main_kwargs)
        main_args.verbose = 0
        pprint_args(main_args)
        cprint("--------", "yellow")
        print("h_name", h_name)
        print("xlabel_name", xlabel_name)
        print("NSR_OR_ESR", NSR_OR_ESR)
        cprint("--------", "yellow")
        visualize_perf_against_hparam(
            hparam_list=main_hparams,
            hparam_name=h_name,
            args_or_args_list=main_args,
            tasks=["node"],
            num_total_runs=5,
            xlabel=xlabel_name,
            use_log_x=False,
        )

    else:
        raise ValueError

    cprint("END MODE: {}".format(MODE), "red")
