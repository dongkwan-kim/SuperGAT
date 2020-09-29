import logging
from collections import defaultdict, OrderedDict
from pprint import pprint
from typing import List, Dict, Tuple, Any
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
    plot_scatter, plot_scatter_with_varying_options

import numpy as np
from scipy import stats
import pandas as pd
from termcolor import cprint
import coloredlogs

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def get_rpg_best(deg, hp, legend, custom_key,
                 model="GAT", num_nodes_per_class=500, num_classes=10, verbose=False) -> Dict[str, Any]:
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

        if verbose:
            print(f"att_lambda: {att_lambda}\tl2_lambda: {l2_lambda}\tperf: {cur_mean_perf} +- {cur_std_perf}")

        if cur_mean_perf > max_mean_perf:
            max_mean_perf = cur_mean_perf
            bmt["mean_perf"] = cur_mean_perf
            bmt["std_perf"] = cur_std_perf
            bmt["att_lambda"] = att_lambda
            bmt["l2_lambda"] = l2_lambda
            bmt["many_seeds_result"] = many_seeds_result
    return bmt


def load_or_get_best_rpg_all(degree_list, homophily_list, legend_list, custom_key_list, model_list,
                             path="../figs/analysis_rpg/cache"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "best_rpg_all_d{}_h{}_l{}.pkl".format(
        len(degree_list), len(homophily_list), len(legend_list),
    ))

    try:
        with open(file_path, "rb") as f:
            best_rpg_all_dict = pickle.load(f)
            cprint("Load: {}".format(file_path), "green")
    except FileNotFoundError:
        best_rpg_all_dict = {}
        for degree in tqdm(degree_list):
            for hp in homophily_list:
                for legend, custom_key, model in zip(legend_list, custom_key_list, model_list):
                    rpg_best = get_rpg_best(degree, hp, legend, custom_key, model=model)
                    best_rpg_all_dict[(degree, hp, legend)] = rpg_best
        with open(file_path, "wb") as f:
            pickle.dump(best_rpg_all_dict, f)
            cprint("Dump: {}".format(file_path), "blue")

    return best_rpg_all_dict


def load_or_get_best_rpg_meta(degree_list, homophily_list, legend_list, custom_key_list, model_list,
                              path="../figs/analysis_rpg/cache"):
    file_path = os.path.join(path, "best_rpg_meta_d{}_h{}_l{}.pkl".format(
        len(degree_list), len(homophily_list), len(legend_list),
    ))

    try:
        with open(file_path, "rb") as f:
            best_rpg_meta = pickle.load(f)
            cprint("Load: {}".format(file_path), "green")
    except FileNotFoundError:
        best_rpg_all_dict = load_or_get_best_rpg_all(
            degree_list, homophily_list, legend_list, custom_key_list, model_list, path,
        )
        # Add first_supergat, gat, abs_gain, rel_gain, p-value
        best_rpg_meta = defaultdict(dict)
        for degree in tqdm(degree_list):
            for hp in homophily_list:
                legend_and_mean_perf_and_tpbv_list = [
                    (
                        legend,
                        best_rpg_all_dict[(degree, hp, legend)]["mean_perf"],
                        best_rpg_all_dict[(degree, hp, legend)]["many_seeds_result"]["test_perf_at_best_val"]
                    ) for legend in legend_list
                ]
                legend_and_mean_perf_and_tpbv_list = sorted(
                    legend_and_mean_perf_and_tpbv_list,
                    key=lambda t: -t[1],
                )
                first_supergat, second_supergat, gat = None, None, None
                for lmt in legend_and_mean_perf_and_tpbv_list:
                    if first_supergat is None and "SuperGAT" in lmt[0]:
                        first_supergat = lmt
                    elif second_supergat is None and "SuperGAT" in lmt[0]:
                        second_supergat = lmt
                    if gat is None and lmt[0] == "GAT-GO":
                        gat = lmt

                abs_diff = first_supergat[1] - gat[1]
                abs_gain = 100 * abs_diff
                rel_gain = 100 * (abs_diff / gat[1])

                p_value_btw_sg = stats.ttest_ind(first_supergat[2], second_supergat[2])[1]
                p_value_vs_gat = stats.ttest_ind(first_supergat[2], gat[2])[1]

                best_rpg_meta[(degree, hp)]["first_supergat"] = first_supergat[0]
                best_rpg_meta[(degree, hp)]["second_supergat"] = second_supergat[0]
                best_rpg_meta[(degree, hp)]["abs_gain"] = abs_gain
                best_rpg_meta[(degree, hp)]["rel_gain"] = rel_gain
                best_rpg_meta[(degree, hp)]["p-value (MX/SD)"] = p_value_btw_sg
                best_rpg_meta[(degree, hp)]["p-value (SuperGAT/GAT)"] = p_value_vs_gat

        with open(file_path, "wb") as f:
            pickle.dump(best_rpg_meta, f)
            cprint("Dump: {}".format(file_path), "blue")

    return best_rpg_meta


def visualize_best_rpg_meta(degree_list, homophily_list, legend_list, custom_key_list, model_list,
                            p_value_thres=0.05):
    def nan_to_v(_iters, fill_value=1.0):
        _new_iters = []
        for _it in _iters:
            if isinstance(_it, np.float64) and np.isnan(_it):
                _new_iters.append(fill_value)
            elif isinstance(_it, np.float64) and not np.isnan(_it):
                _new_iters.append(float(_it))
            else:
                _new_iters.append(_it)
        return _new_iters

    best_rpg_meta = load_or_get_best_rpg_meta(
        degree_list, homophily_list, legend_list, custom_key_list, model_list,
    )
    table = [[d, h, *nan_to_v(meta.values())] for (d, h), meta in best_rpg_meta.items()]
    columns = ["Avg. Degree", "Homophily", "Model", "Second", "Abs. Gain (%p)", "Rel. Gain (%)",
               "p-value (MX/SD)", "p-value (SuperGAT/GAT)"]
    df = pd.DataFrame(table, columns=columns)

    # Change Model with non-significance to Any.
    df.loc[df["p-value (MX/SD)"] >= p_value_thres, "Model"] = "SuperGAT-Any"
    df.loc[df["p-value (SuperGAT/GAT)"] >= p_value_thres, "Model"] = "GAT-Any"

    # Add prefix
    df["Model"] = "Syn-" + df["Model"]

    real_world_df = pd.DataFrame([
        [15.85, 0.21, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [2.78, 0.72, 'Real-GAT-Any', '', 6, 10, 0, 0],
        [35.76, 0.81, 'Real-GAT-Any', '', 6, 10, 0, 0],
        [3.9, 0.83, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [6.41, 0.59, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [5.45, 0.81, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [15.48, 0.26, 'Real-GAT-Any', '', 6, 10, 0, 0],
        [8.93, 0.83, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [5.97, 0.81, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [10.08, 0.32, 'Real-GAT-Any', '', 6, 10, 0, 0],
        [1.83, 0.16, 'Real-SuperGAT-SD', '', 6, 10, 0, 0],
        [7.68, 0.63, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [31.13, 0.85, 'Real-GAT-Any', '', 6, 10, 0, 0],
        [14.38, 0.91, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [28, 0.17, 'Real-SuperGAT-SD', '', 6, 10, 0, 0],
        [4.5, 0.79, 'Real-SuperGAT-MX', '', 6, 10, 0, 0],
        [26.4, 0.68, 'Real-SuperGAT-Any', '', 6, 10, 0, 0],
    ], columns=columns)

    df = df.append(real_world_df)

    df["Avg. Degree (Log10)"] = np.log10(df["Avg. Degree"])

    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    kwargs = dict(
        hue_and_style_order=["Syn-SuperGAT-MX", "Syn-SuperGAT-SD", "Syn-SuperGAT-Any", "Syn-GAT-Any",
                             "Real-SuperGAT-MX", "Real-SuperGAT-SD", "Real-SuperGAT-Any", "Real-GAT-Any"],
        markers=["s", "s", "s", "s",
                 "^", "^", "^", "^"],
        palette=["#EF9A9A", "#BBDEFB", "#E1BEE7", "gray",
                 "#D32F2F", "#1976D2", "#7B1FA2", "#2F2F2F"],  # red, blue, purple, black
        custom_key="best_rpg_meta",
        extension="pdf",
    )

    plot_scatter_with_varying_options(
        df, x="Avg. Degree (Log10)", y="Homophily", hue_and_style="Model", size="Abs. Gain (%p)",
        **kwargs
    )
    plot_scatter_with_varying_options(
        df, x="Avg. Degree (Log10)", y="Homophily", hue_and_style="Model", size="Rel. Gain (%)",
        **kwargs
    )
    plot_scatter_with_varying_options(
        df, x="Avg. Degree", y="Homophily", hue_and_style="Model", size="Abs. Gain (%p)",
        **kwargs
    )
    plot_scatter_with_varying_options(
        df, x="Avg. Degree", y="Homophily", hue_and_style="Model", size="Rel. Gain (%)",
        **kwargs
    )


def print_rpg_analysis(deg, hp, legend, custom_key, model="GAT",
                       num_nodes_per_class=500, num_classes=10, print_all=False, print_tsv=True):
    bmt = get_rpg_best(
        deg, hp, legend, custom_key,
        model=model, num_nodes_per_class=num_nodes_per_class, num_classes=num_classes,
        verbose=print_all,
    )
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
                                        draw_diff_between_first=False,
                                        extension="pdf"):
    def to_log10(v, eps=1e-5):
        return float(np.log10(v + eps))

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
        x_label="Avg. Degree (Log10)",  # Log
        y_label="Test Accuracy",
        name_label_list=["Homophily", "Model"],
        x_list=[to_log10(d) for d in degree_list],  # Log
        hue="Model",
        style="Model",
        col="Homophily",
        aspect=0.75,
        hue_order=legend_list,
        x_lim=(None, None),
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

        def get_mean_diff(h_and_l_to_m_over_d_list, first_legend, x100=True):
            h_and_l_to_mean_diff_over_d_list = dict()
            for (hp, legend), mean_list in h_and_l_to_m_over_d_list.items():
                if legend == first_legend:
                    continue
                mean_list_of_first = h_and_l_to_m_over_d_list[(hp, first_legend)]
                mean_diff_list = (np.asarray(mean_list) - np.asarray(mean_list_of_first))
                if x100:
                    mean_diff_list = 100 * mean_diff_list
                mean_diff_list = mean_diff_list.tolist()
                h_and_l_to_mean_diff_over_d_list[(hp, legend)] = mean_diff_list
            return h_and_l_to_mean_diff_over_d_list

        if 0.1 in degree_list:
            b1, b2, b3, b4 = [0.1, 0.3, 0.5], [0.7], [0.9], [0.7, 0.9]
        else:
            b1, b2, b3, b4 = [0.2, 0.4], [0.6], [0.8], [0.6, 0.8]

        hp135_and_legend_to_mean_over_deg_list, hp135_and_legend_to_std_over_deg_list = filtered_by_hp(b1)
        hp7_and_legend_to_mean_over_deg_list, hp7_and_legend_to_std_over_deg_list = filtered_by_hp(b2)
        hp9_and_legend_to_mean_over_deg_list, hp9_and_legend_to_std_over_deg_list = filtered_by_hp(b3)
        hp79_and_legend_to_mean_over_deg_list, hp79_and_legend_to_std_over_deg_list = filtered_by_hp(b4)

        if draw_diff_between_first:
            lf = legend_list[0]
            hp135_and_legend_to_mean_over_deg_list = get_mean_diff(hp135_and_legend_to_mean_over_deg_list, lf)
            hp7_and_legend_to_mean_over_deg_list = get_mean_diff(hp7_and_legend_to_mean_over_deg_list, lf)
            hp79_and_legend_to_mean_over_deg_list = get_mean_diff(hp79_and_legend_to_mean_over_deg_list, lf)
            hp9_and_legend_to_mean_over_deg_list = get_mean_diff(hp9_and_legend_to_mean_over_deg_list, lf)
            hp135_and_legend_to_std_over_deg_list = None
            hp7_and_legend_to_std_over_deg_list = None
            hp79_and_legend_to_std_over_deg_list = None
            hp9_and_legend_to_std_over_deg_list = None
            legend_list = legend_list[1:]
            y_lim = None
            y_label = "Diff. of Test Acc. vs. GO (%p)"
        else:
            y_lim = None
            y_label = "Test Accuracy",

        degree_list = np.log10(degree_list).tolist()

        palette = ["grey", "#1976D2", "#D32F2F"]

        plot_line_with_std(
            tuple_to_mean_list=hp135_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp135_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree (Log10)",
            y_label=y_label,
            name_label_list=["Homophily", "Model"],
            x_list=degree_list,
            hue="Model",
            style="Model",
            col="Homophily",
            aspect=0.9,
            hue_order=legend_list,
            legend=False,
            x_lim=(0, None),
            y_lim=y_lim,
            palette=palette,
            custom_key=base_key + "_part135",
            extension=extension,
        )
        plot_line_with_std(
            tuple_to_mean_list=hp79_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp79_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree (Log10)",
            y_label="Test Accuracy",
            name_label_list=["Homophily", "Model"],
            x_list=degree_list,
            hue="Model",
            style="Model",
            col="Homophily",
            aspect=0.9,
            hue_order=legend_list,
            legend="full",
            x_lim=(0, None),
            y_lim=y_lim,
            use_ylabel=False,
            palette=palette,
            custom_key=base_key + "_part79",
            extension=extension,
        )
        plot_line_with_std(
            tuple_to_mean_list=hp7_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp7_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree (Log10)",
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
            y_lim=y_lim,
            use_ylabel=False,
            palette=palette,
            custom_key=base_key + "_part7",
            extension=extension,
        )
        plot_line_with_std(
            tuple_to_mean_list=hp9_and_legend_to_mean_over_deg_list,
            tuple_to_std_list=hp9_and_legend_to_std_over_deg_list,
            x_label="Avg. Degree (Log10)",
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
            y_lim=y_lim,
            use_ylabel=False,
            palette=palette,
            custom_key=base_key + "_part9",
            extension=extension,
        )


def print_rpg_pivot_table_by_model(degree_list, homophily_list, legend_list, custom_key_list, model_list):
    # (d, h, m) --> {mean_perf: -, std_perf: -, ...}
    rpg_dict = load_or_get_best_rpg_all(degree_list, homophily_list, legend_list, custom_key_list, model_list)
    columns = ["Avg. degree", "Homophily", "Model", "Mean Perf.", "Std Perf.", "Value"]
    df_dict = {k: [] for k in columns}
    for (d, h, m), dct in rpg_dict.items():
        df_dict["Avg. degree"].append(d)
        df_dict["Homophily"].append(h)
        df_dict["Model"].append(m)
        df_dict["Mean Perf."].append(dct["mean_perf"])
        df_dict["Std Perf."].append(dct["std_perf"])
        df_dict["Value"].append('{:.1f} $\pm$ {:.1f}'.format(
            100 * dct["mean_perf"],
            100 * dct["std_perf"],
        ))
    df = pd.DataFrame(df_dict, columns=columns)

    for m in legend_list:
        m_df = df[df["Model"] == m]
        tab = m_df.pivot(index="Homophily", columns="Avg. degree", values="Value")
        print(m)
        print(tab.to_csv(sep="\t"))


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    os.makedirs("../figs", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    degree_list = [1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 32.5, 40.0, 50.0, 75.0, 100.0]
    homophily_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    model_list = ["GCN", "GAT", "GAT", "GAT"]
    legend_list = ["GCN", "GAT-GO", "SuperGAT-SD", "SuperGAT-MX"]
    custom_key_list = ["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"]

    MODE = "print_rpg_pivot_table_by_model"
    cprint("MODE: {}".format(MODE), "red")

    if MODE == "visualize_best_rpg_meta":
        visualize_best_rpg_meta(degree_list, homophily_list, legend_list, custom_key_list, model_list)

    elif MODE == "get_and_print_rpg_analysis":
        print(s_join("\t", ["degree", "homophily", "model", "key", "att_lambda", "l2_lambda", "mean_perf", "std_perf"]))
        for _degree in degree_list:
            for _hp in homophily_list:
                for _legend, _custom_key, _model in zip(legend_list, custom_key_list, model_list):
                    print_rpg_analysis(_degree, _hp, _legend, _custom_key, model=_model)

    elif MODE == "print_rpg_pivot_table_by_model":
        print_rpg_pivot_table_by_model(degree_list, homophily_list, legend_list, custom_key_list, model_list)

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

    elif MODE == "analyze_rpg_by_degree_and_homophily_part_by_part_first_diff":
        analyze_rpg_by_degree_and_homophily(
            degree_list=degree_list,
            homophily_list=[0.2, 0.4, 0.6, 0.8],
            legend_list=["GAT-GO", "GCN", "SuperGAT-SD", "SuperGAT-MX"],
            model_list=["GAT", "GCN", "GAT", "GAT"],
            custom_key_list=["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"],
            att_lambda_list=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e-3, 1e-4, 1e-5],
            l2_lambda_list=[1e-7, 1e-5, 1e-3],
            num_total_runs=5,
            plot_part_by_part=True,
            draw_diff_between_first=True,  # IMPORTANT
            verbose=0,
        )

    elif MODE == "analyze_rpg_by_degree_and_homophily":
        analyze_rpg_by_degree_and_homophily(
            degree_list=degree_list,
            homophily_list=homophily_list,
            legend_list=legend_list,
            model_list=model_list,
            custom_key_list=custom_key_list,
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
