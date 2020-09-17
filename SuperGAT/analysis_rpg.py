import logging
from collections import defaultdict, OrderedDict
from pprint import pprint
from typing import List, Dict, Tuple
from datetime import datetime
from itertools import chain
import os
import re
import pickle

from torch_geometric.data import Data
from tqdm import tqdm, trange

from arguments import get_args, pprint_args, pdebug_args
from data import get_dataset_or_loader, get_agreement_dist
from main import run, run_with_many_seeds, summary_results, run_with_many_seeds_with_gpu
from utils import s_join, garbage_collection_cuda
from visualize import plot_graph_layout, _get_key, plot_multiple_dist, _get_key_and_makedirs, plot_line_with_std, \
    plot_scatter

import numpy as np
import pandas as pd
from termcolor import cprint
import coloredlogs

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def print_rpg_analysis(deg, hp, legend, custom_key, model="GAT",
                       num_nodes_per_class=500, num_classes=10, print_all=False, print_tsv=True):
    regex = re.compile(r"ms_result_(\d+\.\d+|1e\-\d+)-(\d+\.\d+|1e\-\d+).pkl")

    base_key = "analysis_rpg"
    base_path = os.path.join("../figs", base_key)
    avg_deg_ratio = deg / num_nodes_per_class

    base_kwargs = {
        "model_name": model,
        "dataset_class": "RandomPartitionGraph",
        "dataset_name": f"rpg-{num_classes}-{num_nodes_per_class}-h-d",
        "custom_key": custom_key,
    }
    args = get_args(**base_kwargs)

    dataset_name = f"rpg-{num_classes}-{num_nodes_per_class}-{hp}-{avg_deg_ratio}"
    args.dataset_name = dataset_name
    model_key, model_path = _get_key_and_makedirs(args=args, base_path=base_path, args_prefix=legend)

    bmt = dict()  # best_meta_dict
    max_mean_perf = -1

    for ms_file in os.listdir(model_path):
        result_path = os.path.join(model_path, ms_file)
        many_seeds_result = pickle.load(open(result_path, "rb"))

        match = regex.search(ms_file)
        att_lambda, l2_lambda = float(match.group(1)), float(match.group(2))

        cur_mean_perf = float(np.mean(many_seeds_result["test_perf_at_best_val"]))
        cur_std_perf = float(np.std(many_seeds_result["test_perf_at_best_val"]))

        if print_all:
            print(f"att_lambda: {att_lambda}\tl2_lambda: {l2_lambda}\tperf: {cur_mean_perf} +- {cur_std_perf}")
        if cur_mean_perf > max_mean_perf:
            max_mean_perf = cur_mean_perf
            bmt["mean_perf"] = cur_mean_perf
            bmt["std_perf"] = cur_std_perf
            bmt["att_lambda"] = att_lambda
            bmt["l2_lambda"] = l2_lambda
            bmt["many_seeds_result"] = many_seeds_result

    if print_tsv:
        cprint(s_join("\t", [deg, hp, legend, custom_key,
                             bmt["att_lambda"], bmt["l2_lambda"], bmt["mean_perf"], bmt["std_perf"], ]), "green")
    else:
        cprint(f'att: {bmt["att_lambda"]}\tl2: {bmt["l2_lambda"]}\tperf: {bmt["mean_perf"]} +- {bmt["std_perf"]}',
               "green")

    return bmt


def analyze_rpg_by_degree_and_homophily(degree_list: List[float],
                                        homophily_list: List[float],
                                        legend_list: List[str],
                                        model_list: List[str],
                                        custom_key_list: List[str],
                                        att_lambda_list: List[float],
                                        l2_lambda_list: List[float],
                                        num_total_runs: int,
                                        num_nodes_per_class: int = 500,
                                        num_classes: int = 10,
                                        verbose=2,
                                        is_test=False,
                                        plot_part_by_part=False,
                                        draw_plot=True,
                                        extension="pdf"):
    base_key = "analysis_rpg" + ("" if not is_test else "_test")
    base_path = os.path.join("../figs", base_key)

    best_meta_dict = defaultdict(dict)

    deg_and_legend_to_mean_over_hp_list, deg_and_legend_to_std_over_hp_list = {}, {}

    for deg in degree_list:

        avg_deg_ratio = deg / num_nodes_per_class

        for legend, model, key in zip(legend_list, model_list, custom_key_list):

            base_kwargs = {
                "model_name": model,
                "dataset_class": "RandomPartitionGraph",
                "dataset_name": f"rpg-{num_classes}-{num_nodes_per_class}-h-d",
                "custom_key": key,
            }
            args = get_args(**base_kwargs)
            args.verbose = verbose
            deg_and_legend = (deg, legend)

            if is_test:
                args.epochs = 2

            mean_over_hp_list, std_over_hp_list = [], []
            for hp in homophily_list:

                args.dataset_name = f"rpg-{num_classes}-{num_nodes_per_class}-{hp}-{avg_deg_ratio}"
                model_key, model_path = _get_key_and_makedirs(args=args, base_path=base_path, args_prefix=legend)

                max_mean_perf = -1

                for att_lambda in att_lambda_list:
                    for l2_lambda in l2_lambda_list:
                        args.att_lambda = att_lambda
                        args.l2_lambda = l2_lambda
                        pprint_args(args)

                        result_key = (att_lambda, l2_lambda)
                        result_path = os.path.join(model_path, "ms_result_{}.pkl".format(s_join("-", result_key)))

                        try:
                            many_seeds_result = pickle.load(open(result_path, "rb"))
                            cprint("Load: {}".format(result_path), "blue")

                        except FileNotFoundError:
                            many_seeds_result = run_with_many_seeds_with_gpu(args, num_total_runs)
                            with open(result_path, "wb") as f:
                                pickle.dump(many_seeds_result, f)
                                cprint("Dump: {}".format(result_path), "green")
                                garbage_collection_cuda()
                                cprint("Garbage collected", "green")

                        cur_mean_perf = float(np.mean(many_seeds_result["test_perf_at_best_val"]))
                        cur_std_perf = float(np.std(many_seeds_result["test_perf_at_best_val"]))
                        if cur_mean_perf > max_mean_perf:
                            max_mean_perf = cur_mean_perf
                            best_meta_dict[model_key]["mean_perf"] = cur_mean_perf
                            best_meta_dict[model_key]["std_perf"] = cur_std_perf
                            best_meta_dict[model_key]["att_lambda"] = att_lambda
                            best_meta_dict[model_key]["l2_lambda"] = l2_lambda
                            best_meta_dict[model_key]["many_seeds_result"] = many_seeds_result

                    if not args.is_super_gat:
                        break

                mean_over_hp_list.append(best_meta_dict[model_key]["mean_perf"])
                std_over_hp_list.append(best_meta_dict[model_key]["std_perf"])

            deg_and_legend_to_mean_over_hp_list[deg_and_legend] = mean_over_hp_list
            deg_and_legend_to_std_over_hp_list[deg_and_legend] = std_over_hp_list

    pprint(deg_and_legend_to_mean_over_hp_list)

    if not draw_plot:
        return

    plot_line_with_std(
        tuple_to_mean_list=deg_and_legend_to_mean_over_hp_list,  # (deg, legend) -> List[perf] by homophily
        tuple_to_std_list=deg_and_legend_to_std_over_hp_list,
        x_label="Homophily",
        y_label="Test Accuracy",
        name_label_list=["Avg. Degree", "Model"],
        x_list=homophily_list,
        hue="Model",
        style="Model",
        col="Avg. Degree",
        hue_order=legend_list,
        x_lim=(0, None),
        custom_key=base_key,
        extension=extension,
    )

    hp_and_legend_to_mean_over_deg_list, hp_and_legend_to_std_over_deg_list = defaultdict(list), defaultdict(list)
    legend_to_mean_std_num_agreed_neighbors_list = defaultdict(list)

    for deg, legend in deg_and_legend_to_mean_over_hp_list.keys():
        mean_over_hp_list = deg_and_legend_to_mean_over_hp_list[(deg, legend)]
        std_over_hp_list = deg_and_legend_to_std_over_hp_list[(deg, legend)]
        for hp, mean_of_hp, std_of_hp in zip(homophily_list, mean_over_hp_list, std_over_hp_list):
            hp_and_legend = (hp, legend)
            hp_and_legend_to_mean_over_deg_list[hp_and_legend].append(mean_of_hp)
            hp_and_legend_to_std_over_deg_list[hp_and_legend].append(std_of_hp)

            legend_to_mean_std_num_agreed_neighbors_list[legend].append((mean_of_hp, std_of_hp, hp * deg))

    mean_perf_list = []
    num_agreed_neighbors_list = []
    model_legend_list = []
    for legend, mean_std_num_agr_neighbors_list in legend_to_mean_std_num_agreed_neighbors_list.items():
        for mean_perf, std_perf, num_agr_neighbors in sorted(mean_std_num_agr_neighbors_list, key=lambda t: t[2]):
            mean_perf_list.append(mean_perf)
            model_legend_list.append(legend)
            num_agreed_neighbors_list.append(num_agr_neighbors)

    plot_scatter(
        xs=num_agreed_neighbors_list,
        ys=mean_perf_list,
        hues=model_legend_list,
        xlabel="Avg. Number of Agreed Neighbors",
        ylabel="Test Performance (Acc.)",
        hue_name="Model",
        custom_key=base_key,
    )

    plot_line_with_std(
        tuple_to_mean_list=hp_and_legend_to_mean_over_deg_list,
        tuple_to_std_list=hp_and_legend_to_std_over_deg_list,
        x_label="Avg. Degree",
        y_label="Test Accuracy",
        name_label_list=["Homophily", "Model"],
        x_list=degree_list,
        hue="Model",
        style="Model",
        col="Homophily",
        aspect=0.75,
        hue_order=legend_list,
        x_lim=(0, None),
        custom_key=base_key,
        extension=extension,
    )

    if plot_part_by_part:  # manual.

        # deg: [2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0]
        def filtered_by_hp(hp_list, num_deg=None):
            return (
                {(hp, legend): (mean_list if not num_deg else mean_list[:num_deg])
                 for (hp, legend), mean_list in hp_and_legend_to_mean_over_deg_list.items() if hp in hp_list},
                {(hp, legend): (std_list if not num_deg else std_list[:num_deg])
                 for (hp, legend), std_list in hp_and_legend_to_std_over_deg_list.items() if hp in hp_list}
            )

        hp135_and_legend_to_mean_over_deg_list, hp135_and_legend_to_std_over_deg_list = filtered_by_hp([0.1, 0.3, 0.5])
        hp7_and_legend_to_mean_over_deg_list, hp7_and_legend_to_std_over_deg_list = filtered_by_hp([0.7], num_deg=5)
        hp9_and_legend_to_mean_over_deg_list, hp9_and_legend_to_std_over_deg_list = filtered_by_hp([0.9], num_deg=4)

        plot_line_with_std(
            tuple_to_mean_list=hp135_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp135_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree",
            y_label="Test Accuracy",
            name_label_list=["Homophily", "Model"],
            x_list=degree_list,
            hue="Model",
            style="Model",
            col="Homophily",
            aspect=0.75,
            hue_order=legend_list,
            legend=False,
            x_lim=(0, None),
            custom_key=base_key + "_part135",
            extension=extension,
        )
        plot_line_with_std(
            tuple_to_mean_list=hp7_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp7_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree",
            y_label="Test Accuracy",
            name_label_list=["Homophily", "Model"],
            x_list=degree_list,
            hue="Model",
            style="Model",
            col="Homophily",
            aspect=1.0,
            hue_order=legend_list,
            legend=False,
            x_lim=(0, None),
            use_ylabel=False,
            custom_key=base_key + "_part7",
            extension=extension,
        )
        plot_line_with_std(
            tuple_to_mean_list=hp9_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp9_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree",
            y_label="Test Accuracy",
            name_label_list=["Homophily", "Model"],
            x_list=degree_list,
            hue="Model",
            style="Model",
            col="Homophily",
            aspect=1.0,
            hue_order=legend_list,
            legend="full",
            x_lim=(0, None),
            use_ylabel=False,
            custom_key=base_key + "_part9",
            extension=extension,
        )


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    os.makedirs("../figs", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    MODE = "get_and_print_rpg_analysis"
    cprint("MODE: {}".format(MODE), "red")

    if MODE == "get_and_print_rpg_analysis":

        degree_list = [1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0, 25.0, 40.0, 50.0, 75.0, 100.0]
        homophily_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        model_list = ["GCN", "GAT", "GAT", "GAT"]
        legend_list = ["GCN", "GAT-GO", "SuperGAT-SD", "SuperGAT-MX"]
        custom_key_list = ["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"]

        # model_list = ["GAT", "GAT", "GAT", "GAT"]
        # legend_list = ["GAT-GO", "SuperGAT-SD", "SuperGAT-MX", "SuperGAT-MT"]
        # custom_key_list = ["NE-ES", "EV3-ES", "EV13-ES", "EV20-ES"]

        print(s_join("\t", ["degree", "homophily", "model", "att_lambda", "l2_lambda", "mean_perf", "std_perf"]))
        for _degree in degree_list:
            for _hp in homophily_list:
                for _legend, _custom_key, _model in zip(legend_list, custom_key_list, model_list):
                    print_rpg_analysis(_degree, _hp, _legend, _custom_key, model=_model)

    elif MODE == "analyze_rpg_by_degree_and_homophily_part_by_part":
        analyze_rpg_by_degree_and_homophily(
            degree_list=[2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0],
            homophily_list=[0.1, 0.3, 0.5, 0.7, 0.9],
            legend_list=["GCN", "GAT-GO", "SuperGAT-SD", "SuperGAT-MX"],
            model_list=["GCN", "GAT", "GAT", "GAT"],
            custom_key_list=["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"],
            att_lambda_list=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e-3, 1e-4, 1e-5],
            l2_lambda_list=[1e-7, 1e-5, 1e-3],
            num_total_runs=5,
            plot_part_by_part=True,
            verbose=0,
        )

    elif MODE == "analyze_rpg_by_degree_and_homophily":
        analyze_rpg_by_degree_and_homophily(
            degree_list=[1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0, 25.0, 40.0, 50.0, 75.0, 100.0],
            homophily_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            legend_list=["GCN", "GAT-GO", "SuperGAT-SD", "SuperGAT-MX"],
            model_list=["GCN", "GAT", "GAT", "GAT"],
            custom_key_list=["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"],
            att_lambda_list=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e-3, 1e-4, 1e-5],
            l2_lambda_list=[1e-7, 1e-5, 1e-3],
            num_total_runs=5,
            verbose=0,
        )

    elif MODE == "sandbox_analyze_rpg_by_degree_and_homophily":
        def rev(lst):
            return list(reversed(lst))

        analyze_rpg_by_degree_and_homophily(
            degree_list=[1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0, 25.0, 40.0, 50.0, 75.0, 100.0],
            homophily_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            legend_list=rev(["GCN", "GAT-GO", "SuperGAT-SD", "SuperGAT-MX"]),
            model_list=rev(["GCN", "GAT", "GAT", "GAT"]),
            custom_key_list=rev(["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"]),
            att_lambda_list=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e-3, 1e-4, 1e-5],
            l2_lambda_list=[1e-7, 1e-5, 1e-3],
            num_total_runs=5,
            verbose=0,
            draw_plot=False,
        )

    else:
        raise ValueError

    print("End: {}".format(MODE))
