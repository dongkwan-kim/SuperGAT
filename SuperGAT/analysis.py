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
from utils import blind_other_gpus, sigmoid, get_entropy_tensor_by_iter, get_kld_tensor_by_iter, s_join
from visualize import plot_graph_layout, _get_key, plot_multiple_dist, _get_key_and_makedirs, plot_line_with_std, \
    plot_scatter
from layer import negative_sampling, SuperGAT

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, softmax, remove_self_loops, add_self_loops, degree, to_dense_adj
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


def get_homophily(edge_index, y):
    num_nodes = y.size(0)
    try:
        num_labels = y.size(1)  # multi-labels
    except IndexError:
        num_labels = 1
    e_j, e_i = edge_index
    h_list = []
    for node_id in trange(num_nodes):
        neighbors = edge_index[1, e_j == node_id]
        num_neighbors = neighbors.size(0)
        if num_neighbors > 0:
            if num_labels == 1:
                y_i = y[node_id]
                y_of_neighbors = y[neighbors]
                num_neighbors_same_label = (y_of_neighbors == y_i).nonzero().size(0)
                _h = num_neighbors_same_label / num_neighbors
            else:  # multi-label
                y_i = y[node_id]
                y_of_neighbors = y[neighbors]
                num_shared_label_ratio = (((y_i + y_of_neighbors) == 2).sum(dim=1).float() / num_labels).sum()
                _h = num_shared_label_ratio / num_neighbors
        else:
            _h = np.nan
        h_list.append(_h)
    return torch.as_tensor(h_list)


def get_degree_and_homophily(dataset_class, dataset_name, data_root, **kwargs) -> np.ndarray:
    """
    :param dataset_class: str
    :param dataset_name: str
    :param data_root: str
    :return: np.ndarray the shape of which is [N, 2] (degree, homophily) for Ns
    """
    print(f"{dataset_class} / {dataset_name} / {data_root}")
    train_d, val_d, test_d = get_dataset_or_loader(dataset_class, dataset_name, data_root, seed=42, **kwargs)
    if dataset_name in ["PPI", "WebKB4Univ"]:
        cum_sum = 0
        x_list, y_list, edge_index_list = [], [], []
        for _data in chain(train_d, val_d, test_d):
            x_list.append(_data.x)
            y_list.append(_data.y)
            edge_index_list.append(_data.edge_index + cum_sum)
            cum_sum += _data.x.size(0)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        data = train_d[0]
        x, y, edge_index = data.x, data.y, data.edge_index

    deg = degree(edge_index[0], num_nodes=x.size(0))
    homophily = get_homophily(edge_index, y)
    degree_and_homophily = []
    for _deg, _hom in zip(deg, homophily):
        _deg, _hom = int(_deg), float(_hom)
        if _deg != 0:
            degree_and_homophily.append([_deg, _hom])
    return np.asarray(degree_and_homophily)


def analyze_degree_and_homophily(targets=None, extension="png", **data_kwargs):
    dn_to_dg_and_h = OrderedDict()
    targets = targets or ["WebKB4Univ", "WikiCS", "OGB", "PPI", "Planetoid", "RPG"]

    if "WebKB4Univ" in targets:
        degree_and_homophily = get_degree_and_homophily("WebKB4Univ", "WebKB4Univ", data_root="~/graph-data")
        dn_to_dg_and_h["WebKB4Univ"] = degree_and_homophily

    if "WikiCS" in targets:
        degree_and_homophily = get_degree_and_homophily("WikiCS", "WikICS", data_root="~/graph-data", split=0)
        dn_to_dg_and_h["WikiCS"] = degree_and_homophily

    if "OGB" in targets:
        degree_and_homophily = get_degree_and_homophily("PygNodePropPredDataset", "ogbn-arxiv",
                                                        data_root="~/graph-data")
        dn_to_dg_and_h["ogbn-arxiv"] = degree_and_homophily

    if "PPI" in targets:
        degree_and_homophily = get_degree_and_homophily("PPI", "PPI", data_root="~/graph-data")
        dn_to_dg_and_h["PPI"] = degree_and_homophily

    if "Planetoid" in targets:
        for dataset_name in tqdm(["Cora", "CiteSeer", "PubMed"]):
            degree_and_homophily = get_degree_and_homophily("Planetoid", dataset_name, data_root="~/graph-data")
            dn_to_dg_and_h[dataset_name] = degree_and_homophily

    if "RPG" in targets:
        for adr in [0.025, 0.04, 0.01]:
            dataset_name: object
            for dataset_name in tqdm(["rpg-10-500-{}-{}".format(r, adr) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]):
                degree_and_homophily = get_degree_and_homophily("RandomPartitionGraph", dataset_name,
                                                                data_root="~/graph-data")
                dn_to_dg_and_h[dataset_name] = degree_and_homophily

    for dataset_name, degree_and_homophily in dn_to_dg_and_h.items():
        df = pd.DataFrame({
            "degree": degree_and_homophily[:, 0],
            "homophily": degree_and_homophily[:, 1],
        })
        plot = sns.scatterplot(x="homophily", y="degree", data=df,
                               legend=False, palette="Set1",
                               s=15)
        sns.despine(left=False, right=False, bottom=False, top=False)

        _key, _path = _get_key_and_makedirs(args=None, no_args_key="degree_homophily", base_path="../figs")
        plot.get_figure().savefig("{}/fig_{}_{}.{}".format(_path, _key, dataset_name, extension),
                                  bbox_inches='tight')
        plt.clf()
        print("-- {} --".format(dataset_name))
        print("Degree: {} +- {}".format(degree_and_homophily[:, 0].mean(), degree_and_homophily[:, 0].std()))
        print("Homophily: {} +- {}".format(degree_and_homophily[:, 1].mean(), degree_and_homophily[:, 1].std()))


def analyze_link_pred_perfs_for_multiple_models(name_and_kwargs: List[Tuple[str, Dict]], num_total_runs=10):
    logger = logging.getLogger("LPP")
    logging.basicConfig(filename='../logs/{}-{}.log'.format("link_pred_perfs", str(datetime.now())),
                        level=logging.DEBUG)
    coloredlogs.install(level='DEBUG')

    result_list = []
    for _, kwargs in name_and_kwargs:
        args = get_args(**kwargs)
        gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total) if g not in args.black_list], 1))][0]
        if args.verbose >= 1:
            pdebug_args(args, logger)
            cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")

        many_seeds_result = run_with_many_seeds(args, num_total_runs, gpu_id=gpu_id)
        result_list.append(many_seeds_result)

    for results, (name, _) in zip(result_list, name_and_kwargs):
        logger.debug("\n--- {} ---".format(name))
        for line in summary_results(results):
            logger.debug(line)


def plot_kld_jsd_ent(kld_agree_att_by_layer, kld_att_agree_by_layer, jsd_by_layer, entropy_by_layer,
                     kld_agree_unifatt, kld_unifatt_agree, jsd_uniform, entropy_agreement, entropy_uniform,
                     num_layers, model_args, epoch, name_prefix_list, unit_width_per_name=3,
                     ylim_dict=None, width=0.6, extension="png", **kwargs):
    ylim_dict = ylim_dict or dict()

    def _ylim(plot_type):
        try:
            return ylim_dict[plot_type]
        except KeyError:
            return None

    name_list = ["{}-layer-{}".format(name_prefix, i + 1)
                 for name_prefix in name_prefix_list for i in range(num_layers)]

    plot_multiple_dist(kld_agree_att_by_layer + [kld_agree_unifatt],
                       name_list=name_list + ["Uniform"],
                       x="Attention Type ({})".format(model_args.dataset_name), y="KLD(AGR, ATT)",
                       args=model_args, custom_key="KLD_AGR_ATT_{:03d}".format(epoch),
                       ylim=_ylim("KLD_AGR_ATT"), unit_width_per_name=unit_width_per_name,
                       width=width, extension=extension, **kwargs)
    plot_multiple_dist(kld_att_agree_by_layer + [kld_unifatt_agree],
                       name_list=name_list + ["Uniform"],
                       x="Attention Type ({})".format(model_args.dataset_name), y="KLD(ATT, AGR)",
                       args=model_args, custom_key="KLD_ATT_AGR_{:03d}".format(epoch),
                       ylim=_ylim("KLD_ATT_AGR"), unit_width_per_name=unit_width_per_name,
                       width=width, extension=extension, **kwargs)
    plot_multiple_dist(jsd_by_layer + [jsd_uniform],
                       name_list=name_list + ["Uniform"],
                       x="Attention Type ({})".format(model_args.dataset_name), y="JSD",
                       args=model_args, custom_key="JSD_{:03d}".format(epoch),
                       ylim=_ylim("JSD"), unit_width_per_name=unit_width_per_name,
                       width=width, extension=extension, **kwargs)
    plot_multiple_dist(entropy_by_layer + [entropy_agreement, entropy_uniform],
                       name_list=name_list + ["Agreement", "Uniform"],
                       x="Attention Type ({})".format(model_args.dataset_name), y="Entropy",
                       args=model_args, custom_key="ENT_{:03d}".format(epoch),
                       ylim=_ylim("ENT"), unit_width_per_name=unit_width_per_name,
                       width=width, extension=extension, **kwargs)


def get_attention_metric_for_single_model(model, data, device):
    # List[List[torch.Tensor]]: [L, N, [heads, #neighbors]]
    att_dist_by_layer = model.get_attention_dist_by_layer(data.edge_index, data.x.size(0))
    heads = att_dist_by_layer[0][0].size(0)

    agreement_dist = data.agreement_dist  # List[torch.Tensor]: [N, #neighbors]
    agreement_dist_hxn = [ad.expand(heads, -1).to(device) for ad in agreement_dist]  # [N, [heads, #neighbors]]

    uniform_att_dist = [uad.to(device) for uad in data.uniform_att_dist]  # [N, #neighbors]
    uniform_att_dist_hxn = [uad.expand(heads, -1).to(device) for uad in data.uniform_att_dist]

    # Entropy and KLD: [L, N]
    entropy_by_layer = []
    jsd_by_layer, kld_att_agree_by_layer, kld_agree_att_by_layer = [], [], []
    for i, att_dist in enumerate(att_dist_by_layer):  # att_dist: [N, [heads, #neighbors]]

        # Entropy
        entropy = get_entropy_tensor_by_iter(att_dist, is_prob_dist=True)  # [N]
        entropy_by_layer.append(entropy)

        # KLD
        kld_agree_att = get_kld_tensor_by_iter(agreement_dist_hxn, att_dist)  # [N]
        kld_agree_att_by_layer.append(kld_agree_att)

        kld_att_agree = get_kld_tensor_by_iter(att_dist, agreement_dist_hxn)  # [N]
        kld_att_agree_by_layer.append(kld_att_agree)

        # JSD
        jsd = 0.5 * (kld_agree_att + kld_att_agree)
        jsd_by_layer.append(jsd)

    entropy_agreement = get_entropy_tensor_by_iter(agreement_dist_hxn, is_prob_dist=True)  # [N]
    entropy_uniform = get_entropy_tensor_by_iter(uniform_att_dist_hxn, is_prob_dist=True)  # [N]
    kld_agree_unifatt = get_kld_tensor_by_iter(agreement_dist_hxn, uniform_att_dist_hxn)
    kld_unifatt_agree = get_kld_tensor_by_iter(uniform_att_dist_hxn, agreement_dist_hxn)
    jsd_uniform = 0.5 * (kld_agree_unifatt + kld_unifatt_agree)

    return kld_agree_att_by_layer, kld_att_agree_by_layer, jsd_by_layer, entropy_by_layer, \
           kld_agree_unifatt, kld_unifatt_agree, jsd_uniform, entropy_agreement, entropy_uniform


@torch.no_grad()
def get_attention_metric_for_single_model_and_multiple_data(model, data_list, device):
    list_list_of_result = []
    model.eval()
    cprint("Iteration: get_attention_metric_for_single_model_and_multiple_data", "green")
    for data_no, data in enumerate(tqdm(data_list)):
        model(data.x.to(device), data.edge_index.to(device))
        results_in_list_or_tensor = get_attention_metric_for_single_model(model, data, device)
        if len(list_list_of_result) == 0:
            list_list_of_result = [[] for _ in range(len(results_in_list_or_tensor))]
        for idx_ret, ret in enumerate(results_in_list_or_tensor):
            list_list_of_result[idx_ret].append(ret)
            # if ret is list: List[Tensor=[node_size]], length=num_layers
            # if ret is Tensor: Tensor=[node_size]

    list_of_aggr_result = []
    for list_of_result in list_list_of_result:  # list_of_result: List[ret], length=num_data
        if type(list_of_result[0]) == list:
            tensor_of_layer_of_data = list_of_result
            num_layers = len(tensor_of_layer_of_data[0])
            aggr_tensor_list_of_layer = [[] for _ in range(num_layers)]
            for tensor_of_layer in tensor_of_layer_of_data:  # List[Tensor=[node_size]], length=num_layers
                for layer_no, tensor_in_layer in enumerate(tensor_of_layer):
                    aggr_tensor_list_of_layer[layer_no].append(tensor_in_layer)
            aggr_tensor_of_layer = [torch.cat(agg_tensor_list) for agg_tensor_list in aggr_tensor_list_of_layer]
            list_of_aggr_result.append(aggr_tensor_of_layer)

        elif type(list_of_result[0]) == torch.Tensor:
            tensor_of_data = list_of_result
            aggr_tensor = torch.cat(tensor_of_data)
            list_of_aggr_result.append(aggr_tensor)

    return tuple(list_of_aggr_result)


def visualize_attention_metric_for_multiple_models(name_prefix_and_kwargs: List[Tuple[str, Dict]],
                                                   unit_width_per_name=3,
                                                   extension="png"):
    res = None
    total_args, num_layers, custom_key_list, name_prefix_list = None, None, [], []
    kld1_list, kld2_list, jsd_list, ent_list = [], [], [], []  # [L * M, N]
    for name_prefix, kwargs in name_prefix_and_kwargs:
        args = get_args(**kwargs)
        custom_key_list.append(args.custom_key)
        num_layers = args.num_layers

        train_d, val_d, test_d = get_dataset_or_loader(
            args.dataset_class, args.dataset_name, args.data_root,
            batch_size=args.batch_size, seed=args.seed,
        )
        if val_d is None and test_d is None:
            data_list = [train_d[0]]
        else:
            data_list = []
            for _data in chain(train_d, val_d, test_d):
                if _data.x.size(0) != len(_data.agreement_dist):
                    _data.agreement_dist = [_ad for _ad in _data.agreement_dist[0]]
                    _data.uniform_att_dist = [_uad for _uad in _data.uniform_att_dist[0]]
                data_list.append(_data)

        gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total) if g not in args.black_list], 1))][0]

        if args.verbose >= 1:
            pprint_args(args)
            cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")

        device = "cpu" if gpu_id is None \
            else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

        model, ret = run(args, gpu_id=gpu_id, return_model=True)

        kld1_layer, kld2_layer, jsd_layer, ent_layer, *res = \
            get_attention_metric_for_single_model_and_multiple_data(model, data_list, device)
        kld1_list += kld1_layer
        kld2_list += kld2_layer
        jsd_list += jsd_layer
        ent_list += ent_layer
        name_prefix_list.append(name_prefix)
        total_args = args

        torch.cuda.empty_cache()

    total_args.custom_key = "-".join(sorted(custom_key_list))
    plot_kld_jsd_ent(kld1_list, kld2_list, jsd_list, ent_list, *res,
                     num_layers=num_layers, model_args=total_args, epoch=-1,
                     name_prefix_list=name_prefix_list, unit_width_per_name=unit_width_per_name, extension=extension,
                     flierprops={"marker": "x", "markersize": 12})


def get_attention_heatmap_for_single_model(model, data, device):
    cache_list = [m.cache for m in model.modules() if m.__class__.__name__ == SuperGAT.__name__]
    att_list = [cache["att"] for cache in cache_list]  # List of [E, heads]
    for att in att_list:
        print(att.mean(), att.std(), att.max(), att.min())

    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))  # [2, E]

    dense_att_list = []
    for i, att in enumerate(att_list):
        dense_att = to_dense_adj(edge_index.to(device), batch=None, edge_attr=att).squeeze()  # [N, N, heads]
        dense_att_list.append(dense_att)
    return dense_att_list


def visualize_attention_heatmap_for_multiple_models(name_prefix_and_kwargs: List[Tuple[str, Dict]],
                                                    unit_width_per_name=3,
                                                    extension="png"):
    heatmaps_list = []
    total_args, num_layers, custom_key_list, name_prefix_list = None, None, [], []
    for name_prefix, kwargs in name_prefix_and_kwargs:
        args = get_args(**kwargs)
        custom_key_list.append(args.custom_key)
        num_layers = args.num_layers

        train_d, _, _ = get_dataset_or_loader(
            args.dataset_class, args.dataset_name, args.data_root,
            batch_size=args.batch_size, seed=args.seed,
        )
        data = train_d[0]

        args.task_type = "Attention_Dist"  # set cache_attention True after load dataset

        gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total) if g not in args.black_list], 1))][0]

        if args.verbose >= 1:
            pprint_args(args)
            cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")

        device = "cpu" if gpu_id is None \
            else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

        model, ret = run(args, gpu_id=gpu_id, return_model=True)

        heatmaps_nxn = [ah.cpu().numpy() for ah  # [N, N, heads]
                        in get_attention_heatmap_for_single_model(model, data, device)]
        heatmaps_list.append(heatmaps_nxn)
        name_prefix_list.append(name_prefix)
        total_args = args

    total_args.custom_key = "-".join(sorted(custom_key_list))
    for name_prefix, heatmaps in zip(name_prefix_list, heatmaps_list):
        for i, hmp in enumerate(heatmaps):  # [N, N, heads]
            for head in range(hmp.shape[-1]):
                name = "../figs/att_heatmap_{}_layer{}_head{}.{}".format(name_prefix, i + 1, head, extension)
                print(name)
                ax = sns.heatmap(hmp[:, :, head],
                                 vmin=0, vmax=1,
                                 cmap="YlGnBu", xticklabels=False, yticklabels=False)
                ax.set_title("WOW")
                ax.get_figure().savefig(name, bbox_inches='tight')
                plt.clf()


def edge_to_sorted_tuple(e):
    return tuple(sorted([int(e[0]), int(e[1])]))


def get_layer_repr_and_e2att(model, data, _args) -> List[Tuple[torch.Tensor, Dict]]:
    model.set_layer_attrs("cache_attention", True)
    model = model.to("cpu")
    model.eval()
    layer_repr_and_e2att = []

    # edge_index for [2, E + N]
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.x.size(0))

    with torch.no_grad():
        for i, x in enumerate(model.forward_for_all_layers(data.x, data.edge_index)):
            edge_to_attention = defaultdict(list)
            conv = getattr(model, "conv{}".format(i + 1))
            att = conv.cache["att"]  # [E + N, heads]
            mean_att = att.mean(dim=-1)  # [E + N]
            for ma, e in zip(mean_att, edge_index.t()):
                if e[0] != e[1]:
                    edge_to_attention[edge_to_sorted_tuple(e)].append(float(ma))
            layer_repr_and_e2att.append((x, edge_to_attention))
    return layer_repr_and_e2att


def get_first_layer_and_e2att(model, data, _args, with_normalized=False, with_negatives=False, with_fnn=False):
    model = model.to("cpu")
    model.eval()
    with torch.no_grad():

        xs_after_conv1 = model.conv1(data.x, data.edge_index)

        x = torch.matmul(data.x, model.conv1.weight)
        size_i = x.size(0)

        if not with_negatives:
            edge_index_j, edge_index_i = data.edge_index
            x_i = torch.index_select(x, 0, edge_index_i)
            x_j = torch.index_select(x, 0, edge_index_j)

            x_j = x_j.view(-1, model.conv1.heads, model.conv1.out_channels)
            x_i = x_i.view(-1, model.conv1.heads, model.conv1.out_channels)
            alpha = model.conv1._get_attention(edge_index_i, x_i, x_j, size_i, normalize=True, with_negatives=False)

            if with_fnn:
                fnn_alpha = model.conv1.att_scaling * F.elu(alpha) + model.conv1.att_bias
                fnn_alpha = model.conv1.att_scaling_2 * F.elu(fnn_alpha) + model.conv1.att_bias_2
                mean_fnn_alpha = fnn_alpha.mean(dim=-1)

            if with_normalized:
                alpha = softmax(alpha, data.edge_index[1], size_i)

            mean_alpha = alpha.mean(dim=-1)  # [E]

            edge_to_attention = defaultdict(list)
            edge_to_fnn_attention = defaultdict(list)
            for i, e in enumerate(data.edge_index.t()):
                edge_to_attention[edge_to_sorted_tuple(e)].append(float(mean_alpha[i]))

                if with_fnn:
                    edge_to_fnn_attention[edge_to_sorted_tuple(e)].append(float(mean_fnn_alpha[i]))

            if not with_fnn:
                return xs_after_conv1, edge_to_attention
            else:
                return xs_after_conv1, edge_to_attention, edge_to_fnn_attention

        else:
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=x.size(0))
            total_edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
            alpha = model.conv1._get_attention_with_negatives(
                edge_index=data.edge_index,
                neg_edge_index=neg_edge_index,
                x=x,
            )

            if with_fnn:
                fnn_alpha = model.conv1.att_scaling * alpha + model.conv1.att_bias
                fnn_alpha = F.elu(fnn_alpha)
                fnn_alpha = model.conv1.att_scaling_2 * fnn_alpha + model.conv1.att_bias_2
                mean_fnn_alpha = fnn_alpha.mean(dim=-1)

            if with_normalized:
                alpha = softmax(alpha, total_edge_index[1], size_i)

            mean_alpha = alpha.mean(dim=-1)  # [E + neg_E]

            edge_to_attention = defaultdict(list)
            edge_to_fnn_attention = defaultdict(list)
            edge_to_is_negative = dict()
            for i, e in enumerate(total_edge_index.t()):
                edge_to_attention[edge_to_sorted_tuple(e)].append(float(mean_alpha[i]))
                edge_to_is_negative[edge_to_sorted_tuple(e)] = i > total_edge_index.size(1) // 2

                if with_fnn:
                    edge_to_fnn_attention[edge_to_sorted_tuple(e)].append(float(mean_fnn_alpha[i]))

            if not with_fnn:
                return xs_after_conv1, edge_to_attention, edge_to_is_negative
            else:
                return xs_after_conv1, edge_to_attention, edge_to_is_negative, edge_to_fnn_attention


def visualize_glayout_without_training(layout="tsne", **kwargs):
    _args = get_args(**kwargs)
    pprint_args(_args)
    train_d, val_d, test_d = get_dataset_or_loader(
        _args.dataset_class, _args.dataset_name, _args.data_root,
        batch_size=_args.batch_size, seed=_args.seed,
    )
    data = train_d[0]
    plot_graph_layout(data.x.numpy(), data.y.numpy(), data.edge_index.numpy(),
                      args=_args, edge_to_attention=None, key="raw", layout=layout)


def visualize_glayout_with_training_and_attention(**kwargs):
    _args = get_args(**kwargs)
    _args.verbose = 2
    _args.save_model = False
    if not _args.use_early_stop:
        _args.epochs = 300
    pprint_args(_args)

    alloc_gpu = blind_other_gpus(num_gpus_total=_args.num_gpus_total,
                                 num_gpus_to_use=_args.num_gpus_to_use,
                                 black_list=_args.black_list)
    if not alloc_gpu:
        alloc_gpu = [int(np.random.choice([g for g in range(_args.num_gpus_total)
                                           if g not in _args.black_list], 1))]
    cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    model, ret = run(_args, gpu_id=alloc_gpu[0], return_model=True)
    train_d, val_d, test_d = get_dataset_or_loader(
        _args.dataset_class, _args.dataset_name, _args.data_root,
        batch_size=_args.batch_size, seed=_args.seed,
    )
    data = train_d[0]

    for i, (xs_after_conv, edge_to_attention) in enumerate(tqdm(get_layer_repr_and_e2att(model, data, _args))):
        plot_graph_layout(xs_after_conv.numpy(), data.y.numpy(), data.edge_index.numpy(),
                          edge_to_attention=edge_to_attention, args=_args, key="layer-{}".format(i + 1))
        print("plot_graph_layout: layer-{}".format(i + 1))


def get_model_and_preds(data, **kwargs):
    _args = get_args(**kwargs)
    _args.verbose = 1
    _args.save_model = False
    _args.epochs = 300
    pprint_args(_args)

    _alloc_gpu = blind_other_gpus(num_gpus_total=_args.num_gpus_total,
                                  num_gpus_to_use=_args.num_gpus_to_use,
                                  black_list=_args.black_list)
    if _alloc_gpu:
        cprint("Use GPU the ID of which is {}".format(_alloc_gpu), "yellow")
    _alloc_gpu_id = _alloc_gpu[0] if _alloc_gpu else 1

    model, ret = run(_args, gpu_id=_alloc_gpu_id, return_model=True)

    model = model.to("cpu")
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)[data.test_mask].cpu().numpy()
        pred_labels = np.argmax(output, axis=1)
    return model, pred_labels


# noinspection PyTypeChecker
def visualize_attention_dist_by_sample_type(with_normalized=True, with_negatives=True, extension="png", **kwargs):
    _args = get_args(**kwargs)
    _args.verbose = 2
    _args.save_model = False
    if _args.dataset_name == "CiteSeer":
        _args.epochs = 100
    elif _args.dataset_name == "Cora":
        _args.epochs = 100
    elif _args.dataset_name == "PubMed":
        _args.epochs = 200
    else:
        raise ValueError
    pprint_args(_args)

    _alloc_gpu = blind_other_gpus(num_gpus_total=_args.num_gpus_total,
                                  num_gpus_to_use=_args.num_gpus_to_use,
                                  black_list=_args.black_list)
    if _alloc_gpu:
        cprint("Use GPU the ID of which is {}".format(_alloc_gpu), "yellow")
    _alloc_gpu_id = _alloc_gpu[0] if _alloc_gpu else 1

    model, ret = run(_args, gpu_id=_alloc_gpu_id, return_model=True)

    train_d, val_d, test_d = get_dataset_or_loader(
        "Planetoid", _args.dataset_name, _args.data_root,
        batch_size=_args.batch_size, seed=_args.seed,
    )
    data = train_d[0]

    if with_negatives:
        xs_after_conv1, edge_to_attention, edge_to_is_negative = get_first_layer_and_e2att(
            model, data, _args, with_negatives=with_negatives, with_normalized=with_normalized)
    else:
        edge_to_is_negative = None
        xs_after_conv1, edge_to_attention = get_first_layer_and_e2att(
            model, data, _args, with_negatives=with_negatives, with_normalized=with_normalized)

    _data_list = []
    for edge, att in edge_to_attention.items():
        if edge_to_is_negative is not None and edge_to_is_negative[edge]:
            _data_list.append([edge[0], edge[1], float(np.mean(att)), "negative"])
        else:
            _data_list.append([edge[0], edge[1], float(np.mean(att)), "positive"])

    key = _get_key(_args)
    df = pd.DataFrame(_data_list, columns=["i", "j", "un-normalized attention", "sample type"])

    sns.set_context("poster")

    plt.figure(figsize=(6, 8))
    plot = sns.boxplot(x="sample type", y="un-normalized attention", data=df,
                       order=["positive", "negative"], width=0.35,
                       flierprops=dict(marker='x', alpha=.5))

    if _args.dataset_name == "Cora":
        plot.set_ylim(-0.2, 1.5)
    elif _args.dataset_name == "CiteSeer":
        plot.set_ylim(-0.01, 0.09)
    elif _args.dataset_name == "PubMed":
        plot.set_ylim(-0.01, 0.5)
    else:
        raise ValueError

    plot.set_title("{}/{}".format("GAT" if _args.custom_key == "NE" else "Super-GAT",
                                  _args.dataset_name))
    plot.get_figure().savefig("../figs/fig_attention_dist_by_sample_type_{}_{}.{}".format(
        key, "norm" if with_normalized else "unnorm", extension), bbox_inches='tight')
    plt.clf()
    return df


def visualize_edge_fnn(extension="pdf", **kwargs):
    _args = get_args(**kwargs)
    _args.verbose = 2
    _args.save_model = False
    if _args.dataset_name == "CiteSeer":
        _args.epochs = 100
    elif _args.dataset_name == "Cora":
        _args.epochs = 100
    elif _args.dataset_name == "PubMed":
        _args.epochs = 200
    else:
        raise ValueError
    pprint_args(_args)

    _alloc_gpu = blind_other_gpus(num_gpus_total=_args.num_gpus_total,
                                  num_gpus_to_use=_args.num_gpus_to_use,
                                  black_list=_args.black_list)
    if _alloc_gpu:
        cprint("Use GPU the ID of which is {}".format(_alloc_gpu), "yellow")
    _alloc_gpu_id = _alloc_gpu[0] if _alloc_gpu else 1

    model, ret = run(_args, gpu_id=_alloc_gpu_id, return_model=True)

    train_d, val_d, test_d = get_dataset_or_loader(
        "Planetoid", _args.dataset_name, _args.data_root,
        batch_size=_args.batch_size, seed=_args.seed,
    )
    data = train_d[0]

    xs_after_conv1, edge_to_attention, edge_to_is_negative, edge_to_fnn_attention = get_first_layer_and_e2att(
        model, data, _args, with_negatives=True, with_normalized=False, with_fnn=True)

    _data_list = []
    for edge, att in edge_to_attention.items():
        fnn_att = edge_to_fnn_attention[edge]
        if edge_to_is_negative[edge]:
            _data_list.append([edge[0], edge[1], float(np.mean(att)),
                               float(np.mean(fnn_att)), sigmoid(np.mean(fnn_att)), "negative"])
        else:
            _data_list.append([edge[0], edge[1], float(np.mean(att)),
                               float(np.mean(fnn_att)), sigmoid(np.mean(fnn_att)), "positive"])

    # noinspection PyTypeChecker
    key = _get_key(_args)
    pd.set_option('display.expand_frame_repr', False)
    df = pd.DataFrame(_data_list,
                      columns=["i", "j", "un-normalized attention",
                               "f(un-normalized attention)", "phi", "sample type"])

    # sns.set_context("poster")
    sns.set(rc={'text.usetex': True}, style="whitegrid")

    # scatter plot
    plot = sns.scatterplot(x="un-normalized attention", y="f(un-normalized attention)", data=df,
                           hue="sample type", hue_order=["positive", "negative"], s=10, alpha=0.5)
    plot.set_xlabel("$e_{ij}$")
    plot.set_ylabel("$f(e_{ij})$")
    plot.get_figure().savefig("../figs/fig_fnn_scatter_{}.{}".format(key, extension), bbox_inches='tight')
    plt.clf()

    # prob plot
    plot = sns.scatterplot(x="un-normalized attention", y="phi", data=df,
                           hue="sample type", hue_order=["positive", "negative"], s=10, alpha=0.5)
    plot.set_xlabel("$e_{ij}$")
    plot.set_ylabel("$\phi_{ij}$")
    plot.get_figure().savefig("../figs/fig_fnn_prob_{}.{}".format(key, extension), bbox_inches='tight')
    plt.clf()

    # line plot
    min_att_val, max_att_val = np.min(df["un-normalized attention"]), np.max(df["f(un-normalized attention)"])
    att_domain = np.linspace(min_att_val, max_att_val, num=1000)
    att_domain = np.vstack([att_domain] * 8).transpose()  # [1000, heads]

    att_domain_tensor = torch.as_tensor(att_domain).float()
    att_range = model.conv1.att_scaling * att_domain_tensor + model.conv1.att_bias
    att_range = F.elu(att_range)
    att_range = model.conv1.att_scaling_2 * att_range + model.conv1.att_bias_2
    att_range = att_range.detach().numpy()  # [1000, heads]

    _data_list = []
    for d, r_heads in zip(att_domain[:, 0], att_range):
        for i, r in enumerate(r_heads):
            _data_list.append([d, r, i + 1])
    df = pd.DataFrame(_data_list, columns=(["un-normalized attention", "f(un-normalized attention)", "head"]))
    plot = sns.lineplot(x="un-normalized attention", y="f(un-normalized attention)", data=df,
                        hue="head", palette="Set1", legend="full")
    plot.get_figure().savefig("../figs/fig_fnn_line_{}.{}".format(key, extension), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    main_kwargs = {
        "model_name": "GAT",  # GAT, BaselineGAT, LargeGAT
        "dataset_class": "Planetoid",  # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
        "dataset_name": "Cora",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
        "custom_key": "NEO8",  # NE, EV1, EV2
    }

    os.makedirs("../figs", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    MODE = "degree_and_homophily"
    cprint("MODE: {}".format(MODE), "red")

    if MODE == "link_pred_perfs_for_multiple_models":

        def get_main_custom_key_list(dataset_name, prefix_1, prefix_2):
            if "NS" in prefix_1:
                ckl = ["{}O8".format(prefix_1) + ("-ES-Link" if dataset_name != "PubMed" else "-500-ES-Link")]
            elif dataset_name != "PubMed":
                ckl = ["{}O8-ES-Link".format(prefix_1), "{}O8-ES-Link".format(prefix_2)]
            else:
                ckl = ["{}-500-ES-Link".format(prefix_1), "{}-500-ES-Link".format(prefix_2)]
            return ckl


        mode_type = "S2"  # N, S1, S2
        main_kwargs["dataset_class"] = "LinkPlanetoid"
        dataset_name_list = ["Cora", "CiteSeer", "PubMed"]
        if mode_type == "N":
            p1, p2 = "NE", "NEDP"
        elif mode_type == "S1":
            p1, p2 = "EV1", "EV2"
        elif mode_type == "S2":
            p1, p2 = "EV12NS", None
        else:
            raise ValueError("Wrong mode: {}".format(mode_type))

        main_name_and_kwargs = [("{}-{}".format(d, ck), {**main_kwargs, "dataset_name": d, "custom_key": ck})
                                for d in dataset_name_list for ck in get_main_custom_key_list(d, p1, p2)]
        pprint(main_name_and_kwargs)

        analyze_link_pred_perfs_for_multiple_models(main_name_and_kwargs, num_total_runs=10)

    elif MODE == "attention_metric_for_multiple_models":

        sns.set_context("poster", font_scale=1.25)

        is_super_gat = False  # False

        main_kwargs["model_name"] = "LargeGAT"  # GAT, LargeGAT
        main_kwargs["dataset_name"] = "PPI"  # Cora, CiteSeer, PubMed
        main_num_layers = 4  # Only for LargeGAT 3, 4

        if main_kwargs["dataset_name"] != "PPI":
            main_kwargs["dataset_class"] = "ADPlanetoid"  # Fix.
        else:
            main_kwargs["dataset_class"] = "ADPPI"  # Fix

        if not is_super_gat:
            main_name_prefix_list = ["GO", "DP"]
            unit_width = 3
        else:
            main_name_prefix_list = ["SGO", "SDP"]
            unit_width = 3

        if is_super_gat:
            if main_kwargs["dataset_name"] != "PubMed":
                main_custom_key_list = ["EV1O8-ES-ATT", "EV2O8-ES-ATT"]
            else:
                main_custom_key_list = ["EV1-500-ES-ATT", "EV2-500-ES-ATT"]

        elif main_kwargs["model_name"] == "GAT":
            if main_kwargs["dataset_name"] != "PubMed":
                main_custom_key_list = ["NEO8-ES-ATT", "NEDPO8-ES-ATT"]
            else:
                main_custom_key_list = ["NE-500-ES-ATT", "NEDP-500-ES-ATT"]

        elif main_kwargs["model_name"] == "LargeGAT":
            if main_kwargs["dataset_name"] != "PubMed":
                main_custom_key_list = ["NEO8-L{}-ES-ATT".format(main_num_layers),
                                        "NEDPO8-L{}-ES-ATT".format(main_num_layers)]
            else:
                main_custom_key_list = ["NE-600-L{}-ES-ATT".format(main_num_layers),
                                        "NEDP-600-L{}-ES-ATT".format(main_num_layers)]
        else:
            raise ValueError("Wrong model name: {}".format(main_kwargs["model_name"]))
        main_npx_and_kwargs = [(npx, {**main_kwargs, "custom_key": ck}) for npx, ck in zip(main_name_prefix_list,
                                                                                           main_custom_key_list)]
        pprint(main_npx_and_kwargs)
        visualize_attention_metric_for_multiple_models(main_npx_and_kwargs,
                                                       unit_width_per_name=unit_width, extension="pdf")

    elif MODE == "visualize_attention_heatmap_for_multiple_models":

        # sns.set_context("poster", font_scale=1.25)

        is_super_gat = False  # False
        main_kwargs["model_name"] = "GAT"  # GAT, LargeGAT
        main_kwargs["dataset_name"] = "Cora"  # Cora, CiteSeer, PubMed

        unit_width = 3
        main_name_prefix_list = ["SDP", "GO", "DP"]
        main_custom_key_list = ["NESDPO8", "NEO8", "NEDPO8"]
        main_npx_and_kwargs = [(npx, {**main_kwargs, "custom_key": ck}) for npx, ck in zip(main_name_prefix_list,
                                                                                           main_custom_key_list)]

        visualize_attention_heatmap_for_multiple_models(main_npx_and_kwargs,
                                                        unit_width_per_name=unit_width, extension="png")

    elif MODE == "link_pred_perfs_for_multiple_models_synthetic":
        k = "EV12NSO8"
        ck = "{}-ES-Link".format(k)
        main_kwargs["model_name"] = "GAT"
        main_kwargs["dataset_class"] = "LinkRandomPartitionGraph"
        main_name_and_kwargs = []
        for _d in [0.01, 0.025, 0.04]:
            for h in [0.1, 0.3, 0.5, 0.7, 0.9]:
                d = "rpg-10-500-{}-{}".format(h, _d)
                main_name_and_kwargs.append(("{}-{}".format(d, ck),
                                             {**main_kwargs, "dataset_name": d, "custom_key": ck}))
        pprint(main_name_and_kwargs)
        analyze_link_pred_perfs_for_multiple_models(main_name_and_kwargs, num_total_runs=10)

    elif MODE == "attention_metric_for_multiple_models_synthetic":
        main_kwargs["model_name"] = "GAT"
        main_kwargs["dataset_class"] = "ADRandomPartitionGraph"
        sns.set_context("poster", font_scale=1.25)
        for _d in [0.01, 0.025, 0.04]:
            for h in [0.1, 0.3, 0.5, 0.7, 0.9]:
                main_kwargs["dataset_name"] = "rpg-10-500-{}-{}".format(h, _d)

                main_npx_and_kwargs = []
                for npx, ck in zip(["SG", "GO", "DP"], ["EV12NSO8", "NEO8", "NEDPO8"]):
                    main_npx_and_kwargs.append((npx, {**main_kwargs, "custom_key": "{}-ES-ATT".format(ck)}))
                pprint(main_npx_and_kwargs)
                visualize_attention_metric_for_multiple_models(main_npx_and_kwargs,
                                                               unit_width_per_name=3, extension="pdf")

    elif MODE == "glayout_without_training":
        layout_shape = "tsne"  # tsne, spring, kamada_kawai
        visualize_glayout_without_training(layout=layout_shape, **main_kwargs)

    elif MODE == "small_synthetic_examples":
        layout_shape = "tsne"
        c, n = 5, 100
        main_kwargs["dataset_class"] = "RandomPartitionGraph"
        for d in [2.5, 20.0]:
            ad = d / n
            for r in [0.1, 0.5, 0.9]:
                main_kwargs["dataset_name"] = "rpg-{}-{}-{}-{}".format(c, n, r, ad)
                visualize_glayout_without_training(layout=layout_shape, **main_kwargs)
                print("Done: {}".format(main_kwargs["dataset_name"]))

    elif MODE == "glayout_with_training_and_attention":
        visualize_glayout_with_training_and_attention(**main_kwargs)

    elif MODE == "degree_and_homophily":
        analyze_degree_and_homophily()

    elif MODE == "get_and_print_rpg_analysis":

        degree_list = [2.5, 5.0, 25.0, 50.0]
        homophily_list = [0.1, 0.3, 0.5, 0.7, 0.9]

        legend_list = ["GAT-GO", "SuperGAT-SD", "SuperGAT-MX", "SuperGAT-MT"]
        custom_key_list = ["NE-ES", "EV3-ES", "EV13-ES", "EV20-ES"]

        print(s_join("\t", ["degree", "homophily", "model", "att_lambda", "l2_lambda", "mean_perf", "std_perf"]))
        for _degree in degree_list:
            for _hp in homophily_list:
                for _legend, _custom_key in zip(legend_list, custom_key_list):
                    print_rpg_analysis(_degree, _hp, _legend, _custom_key, model="GAT")

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
            degree_list=[2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0],
            homophily_list=[0.1, 0.3, 0.5, 0.7, 0.9],
            legend_list=["GCN", "GAT-GO", "SuperGAT-SD", "SuperGAT-MX"],
            model_list=["GCN", "GAT", "GAT", "GAT"],
            custom_key_list=["NE-ES", "NE-ES", "EV3-ES", "EV13-ES"],
            att_lambda_list=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e-3, 1e-4, 1e-5],
            l2_lambda_list=[1e-7, 1e-5, 1e-3],
            num_total_runs=5,
            verbose=0,
        )

    elif MODE == "attention_dist_by_sample_type":  # deprecated
        data_frame_list = []
        for dn in ["Cora", "CiteSeer", "PubMed"]:
            for ck in ["NE", "EV1"]:
                for with_norm in [True, False]:
                    main_kwargs = dict(
                        model_name="GAT",  # GAT, BaselineGAT
                        dataset_class="Planetoid",
                        dataset_name=dn,  # Cora, CiteSeer, PubMed
                        custom_key=ck,  # NE, EV1, EV2, NR, RV1
                    )
                    with_neg = not with_norm
                    data_frame_list.append(visualize_attention_dist_by_sample_type(
                        **main_kwargs,
                        with_negatives=with_neg,
                        with_normalized=with_norm,
                    ))
        idx = 0
        for dn in ["Cora", "CiteSeer"]:
            for ck in ["NE", "EV1"]:
                for nn in [False]:
                    df_idx = data_frame_list[idx]
                    cprint("{}/{}/{}".format(ck, dn, nn), "yellow")
                    print("MEAN")
                    print(df_idx[df_idx["sample type"] == "positive"].mean())
                    print("STD")
                    print(df_idx[df_idx["sample type"] == "positive"].std())
                    cprint("-------", "yellow")
                    idx += 1

    elif MODE == "edge_fnn":  # deprecated
        for dn in ["Cora", "CiteSeer", "PubMed"]:
            main_kwargs = dict(
                model_name="GAT",  # GAT, BaselineGAT
                dataset_class="Planetoid",
                dataset_name=dn,  # Cora, CiteSeer, PubMed
                custom_key="EV1",  # NE, EV1, EV2, NR, RV1
            )
            visualize_edge_fnn(extension="pdf", **main_kwargs)

    else:
        raise ValueError

    print("End: {}".format(MODE))
