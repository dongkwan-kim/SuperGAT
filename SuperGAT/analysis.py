import logging
from collections import defaultdict, OrderedDict
from pprint import pprint
from typing import List, Dict, Tuple
from datetime import datetime
from itertools import chain
import os
import re
import multiprocessing as mp

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


def _get_h_of_one_node_torch(node_id, edge_index, e_j, y, num_labels):
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
    return _h


def _get_h_of_one_node_numpy(node_id, edge_index, e_j, y, num_labels):
    neighbors = edge_index[1, e_j == node_id]
    num_neighbors = neighbors.shape[0]
    if num_neighbors > 0:
        if num_labels == 1:
            y_i = y[node_id]
            y_of_neighbors = y[neighbors]
            num_neighbors_same_label = np.sum(y_of_neighbors == y_i)
            _h = num_neighbors_same_label / num_neighbors
        else:  # multi-label
            raise NotImplementedError
    else:
        _h = np.nan
    return _h


def _get_h_of_one_node_numpy_global(node_id, num_labels):
    global EDGE_INDEX, Y
    e_j, _ = EDGE_INDEX
    return _get_h_of_one_node_numpy(node_id, EDGE_INDEX, e_j, Y, num_labels)


def get_homophily(edge_index, y, use_multiprocessing=False):
    y = y.squeeze()
    try:
        num_labels = y.size(1)  # multi-labels
        use_numpy = False
    except IndexError:
        num_labels = 1
        use_numpy = True

    cprint(f"use_numpy: {use_numpy} / use_mp: {use_multiprocessing}", "green")

    if use_numpy and not use_multiprocessing:
        edge_index, y = edge_index.numpy(), y.numpy()
        e_j, e_i = edge_index
        h_list = []
        for node_id in trange(y.shape[0]):
            h_list.append(_get_h_of_one_node_numpy(node_id, edge_index, e_j, y, num_labels))
    elif not use_numpy and not use_multiprocessing:
        e_j, e_i = edge_index
        h_list = []
        for node_id in trange(y.size(0)):
            h_list.append(_get_h_of_one_node_torch(node_id, edge_index, e_j, y, num_labels))
    else:
        edge_index, y = edge_index.numpy(), y.numpy()

        global EDGE_INDEX, Y
        EDGE_INDEX, Y = edge_index, y

        pool = mp.Pool(mp.cpu_count())
        h_list = pool.starmap(
            _get_h_of_one_node_numpy_global,
            [(node_id, num_labels) for node_id in range(y.shape[0])],
        )
    return torch.as_tensor(h_list)


def get_degree_and_homophily(dataset_class, dataset_name, data_root,
                             use_multiprocessing=False, **kwargs) -> np.ndarray:
    """
    :param dataset_class: str
    :param dataset_name: str
    :param data_root: str
    :param use_multiprocessing:
    :return: np.ndarray the shape of which is [N, 2] (degree, homophily) for Ns
    """
    print(f"{dataset_class} / {dataset_name} / {data_root}")
    train_d, val_d, test_d = get_dataset_or_loader(dataset_class, dataset_name, data_root, seed=42, **kwargs)
    if dataset_name in ["PPI", "WebKB4Univ", "CLUSTER"]:
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
        try:
            num_labels = y.size(1)  # multi-labels
        except IndexError:
            num_labels = 1
    else:
        data = train_d[0]
        x, y, edge_index = data.x, data.y, data.edge_index
        num_labels = 1

    deg = degree(edge_index[0], num_nodes=x.size(0))
    homophily = get_homophily(edge_index, y, use_multiprocessing=use_multiprocessing)
    degree_and_homophily = []
    for _deg, _hom in zip(deg, homophily):
        _deg, _hom = int(_deg), float(_hom)
        if _deg != 0:
            degree_and_homophily.append([_deg, _hom])
    return np.asarray(degree_and_homophily)


def analyze_degree_and_homophily(targets=None, extension="png", **data_kwargs):
    dn_to_dg_and_h = OrderedDict()
    targets = targets or [
        "Reddit",
        "Crocodile", "Squirrel", "Chameleon",
        "MyCitationFull", "MyCoauthor", "MyAmazon",
        "Flickr", "CLUSTER", "WikiCS", "ogbn-arxiv", "ogbn-products",
        "PPI", "Planetoid", "RPG",
    ]

    if "Reddit" in targets:
        degree_and_homophily = get_degree_and_homophily(
            "Reddit", "Reddit",
            data_root="~/graph-data", use_multiprocessing=True, size=[10, 5], num_hops=2,
        )
        dn_to_dg_and_h["Reddit"] = degree_and_homophily

    if "Squirrel" in targets:
        degree_and_homophily = get_degree_and_homophily("Squirrel", "Squirrel", data_root="~/graph-data")
        dn_to_dg_and_h["Squirrel"] = degree_and_homophily

    if "Chameleon" in targets:
        degree_and_homophily = get_degree_and_homophily("Chameleon", "Chameleon", data_root="~/graph-data")
        dn_to_dg_and_h["Chameleon"] = degree_and_homophily

    if "Crocodile" in targets:
        degree_and_homophily = get_degree_and_homophily("Crocodile", "Crocodile", data_root="~/graph-data")
        dn_to_dg_and_h["Crocodile"] = degree_and_homophily

    if "MyCitationFull" in targets:
        degree_and_homophily = get_degree_and_homophily("MyCitationFull", "Cora", data_root="~/graph-data")
        dn_to_dg_and_h["CoraFull"] = degree_and_homophily
        degree_and_homophily = get_degree_and_homophily("MyCitationFull", "Cora_ML", data_root="~/graph-data")
        dn_to_dg_and_h["Cora_ML"] = degree_and_homophily
        degree_and_homophily = get_degree_and_homophily("MyCitationFull", "DBLP", data_root="~/graph-data")
        dn_to_dg_and_h["DBLP"] = degree_and_homophily

    if "MyCoauthor" in targets:
        degree_and_homophily = get_degree_and_homophily("MyCoauthor", "CS", data_root="~/graph-data")
        dn_to_dg_and_h["CS"] = degree_and_homophily
        degree_and_homophily = get_degree_and_homophily("MyCoauthor", "Physics", data_root="~/graph-data")
        dn_to_dg_and_h["Physics"] = degree_and_homophily

    if "MyAmazon" in targets:
        degree_and_homophily = get_degree_and_homophily("MyAmazon", "Photo", data_root="~/graph-data")
        dn_to_dg_and_h["Photo"] = degree_and_homophily
        degree_and_homophily = get_degree_and_homophily("MyAmazon", "Computers", data_root="~/graph-data")
        dn_to_dg_and_h["Computers"] = degree_and_homophily

    if "Flickr" in targets:
        degree_and_homophily = get_degree_and_homophily("Flickr", "Flickr", data_root="~/graph-data")
        dn_to_dg_and_h["Flickr"] = degree_and_homophily

    if "CLUSTER" in targets:
        degree_and_homophily = get_degree_and_homophily("GNNBenchmarkDataset", "CLUSTER", data_root="~/graph-data")
        dn_to_dg_and_h["CLUSTER"] = degree_and_homophily

    if "WebKB4Univ" in targets:
        degree_and_homophily = get_degree_and_homophily("WebKB4Univ", "WebKB4Univ", data_root="~/graph-data")
        dn_to_dg_and_h["WebKB4Univ"] = degree_and_homophily

    if "WikiCS" in targets:
        degree_and_homophily = get_degree_and_homophily("WikiCS", "WikICS", data_root="~/graph-data", split=0)
        dn_to_dg_and_h["WikiCS"] = degree_and_homophily

    if "ogbn-products" in targets:
        degree_and_homophily = get_degree_and_homophily(
            "PygNodePropPredDataset", "ogbn-products",
            data_root="~/graph-data", use_multiprocessing=True, size=[10, 5], num_hops=2,
        )
        dn_to_dg_and_h["ogbn-products"] = degree_and_homophily

    if "ogbn-arxiv" in targets:
        degree_and_homophily = get_degree_and_homophily(
            "PygNodePropPredDataset", "ogbn-arxiv",
            data_root="~/graph-data", use_multiprocessing=True,
        )
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

    elif MODE == "degree_and_homophily":
        analyze_degree_and_homophily(["MyCoauthor"])
        # analyze_degree_and_homophily(["ogbn-arxiv"])

    else:
        raise ValueError

    print("End: {}".format(MODE))
