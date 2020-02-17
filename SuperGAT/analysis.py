import logging
from collections import defaultdict, OrderedDict
from pprint import pprint
from typing import List, Dict, Tuple
from datetime import datetime
import os
from tqdm import tqdm

from arguments import get_args, pprint_args, pdebug_args
from data import get_dataset_or_loader, get_agreement_dist
from main import run, run_with_many_seeds, summary_results
from utils import blind_other_gpus, sigmoid, get_entropy_tensor_by_iter, get_kld_tensor_by_iter
from visualize import plot_graph_layout, _get_key, plot_multiple_dist, _get_key_and_makedirs, plot_line_with_std
from layer import negative_sampling

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph, softmax, remove_self_loops, add_self_loops, degree
import numpy as np
import pandas as pd
from termcolor import cprint
import coloredlogs
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def get_degree_and_homophily(dataset_class, dataset_name, data_root) -> np.ndarray:
    """
    :param dataset_class: str
    :param dataset_name: str
    :param data_root: str
    :return: np.ndarray the shape of which is [N, 2] (degree, homophily) for Ns
    """

    def get_h(agr_dist):
        agr_dist = agr_dist.cpu().numpy()
        agr_counts = (agr_dist == np.max(agr_dist)).sum()
        return agr_counts / len(agr_dist)

    train_d, val_d, test_d = get_dataset_or_loader(dataset_class, dataset_name, data_root, seed=42)
    data = train_d[0]

    x, y, edge_index = data.x, data.y, data.edge_index

    deg = degree(edge_index[0])
    agr = get_agreement_dist(edge_index, y, with_self_loops=False, epsilon=0)

    degree_and_homophily = []
    for i, (d, a) in enumerate(zip(deg, agr)):
        if len(a) == 0:
            assert d == 0
            degree_and_homophily.append([d, 1])
        else:
            h = get_h(a)
            degree_and_homophily.append([d, h])
    return np.asarray(degree_and_homophily)


def analyze_degree_and_homophily(extension="png", **data_kwargs):
    dn_to_dg_and_h = OrderedDict()

    for dataset_name in tqdm(["Cora", "CiteSeer", "PubMed"]):
        degree_and_homophily = get_degree_and_homophily("Planetoid", dataset_name, data_root="~/graph-data")
        dn_to_dg_and_h[dataset_name] = degree_and_homophily

    for adr in [0.025, 0.04, 0.01]:
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
                               legend=False, palette="Set1")
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


def get_attention_metric_for_single_model(model, batch, device):
    # List[List[torch.Tensor]]: [L, N, [heads, #neighbors]]
    att_dist_by_layer = model.get_attention_dist_by_layer(batch.edge_index, batch.x.size(0))
    heads = att_dist_by_layer[0][0].size(0)

    agreement_dist = batch.agreement_dist  # List[torch.Tensor]: [N, #neighbors]
    agreement_dist_hxn = [ad.expand(heads, -1).to(device) for ad in agreement_dist]  # [N, [heads, #neighbors]]

    uniform_att_dist = [uad.to(device) for uad in batch.uniform_att_dist]  # [N, #neighbors]
    uniform_att_dist_hxn = [uad.expand(heads, -1).to(device) for uad in batch.uniform_att_dist]

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

        train_d, _, _ = get_dataset_or_loader(
            args.dataset_class, args.dataset_name, args.data_root,
            batch_size=args.batch_size, seed=args.seed,
        )
        batch = train_d[0]

        gpu_id = [int(np.random.choice([g for g in range(args.num_gpus_total) if g not in args.black_list], 1))][0]

        if args.verbose >= 1:
            pprint_args(args)
            cprint("Use GPU the ID of which is {}".format(gpu_id), "yellow")

        device = "cpu" if gpu_id is None \
            else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

        model, ret = run(args, gpu_id=gpu_id, return_model=True)

        kld1_layer, kld2_layer, jsd_layer, ent_layer, *res = get_attention_metric_for_single_model(model, batch, device)
        kld1_list += kld1_layer
        kld2_list += kld2_layer
        jsd_list += jsd_layer
        ent_list += ent_layer
        name_prefix_list.append(name_prefix)
        total_args = args

    total_args.custom_key = "-".join(sorted(custom_key_list))
    plot_kld_jsd_ent(kld1_list, kld2_list, jsd_list, ent_list, *res,
                     num_layers=num_layers, model_args=total_args, epoch=-1,
                     name_prefix_list=name_prefix_list, unit_width_per_name=unit_width_per_name, extension=extension,
                     flierprops={"marker": "x", "markersize": 12})


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
        "dataset_class": "RandomPartitionGraph", # ADPlanetoid, LinkPlanetoid, Planetoid, RandomPartitionGraph
        "dataset_name": "rpg-10-500-0.1-0.025",  # Cora, CiteSeer, PubMed, rpg-10-500-0.9-0.025
        "custom_key": "NEO8",  # NE, EV1, EV2
    }

    os.makedirs("../figs", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)

    MODE = "attention_metric_for_multiple_models_synthetic"
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

        is_super_gat = True  # False

        main_kwargs["model_name"] = "GAT"  # GAT, LargeGAT
        main_kwargs["dataset_name"] = "PubMed"  # Cora, CiteSeer, PubMed
        main_kwargs["dataset_class"] = "ADPlanetoid"  # Fix.
        main_num_layers = 4  # Only for LargeGAT 3, 4

        if not is_super_gat:
            main_name_prefix_list = ["GO", "DP"]
            unit_width = 3
        else:
            main_name_prefix_list = ["SG"]
            unit_width = 3

        if is_super_gat:
            if main_kwargs["dataset_name"] != "PubMed":
                main_custom_key_list = ["EV12NSO8-ES-ATT"]
            else:
                main_custom_key_list = ["EV12NSO8-500-ES-ATT"]

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
        for d in [0.01, 0.04]:
            for r in [0.1, 0.5, 0.9]:
                main_kwargs["dataset_name"] = "rpg-{}-{}-{}-{}".format(c, n, r, d)
                visualize_glayout_without_training(layout=layout_shape, **main_kwargs)
                print("Done: {}".format(main_kwargs["dataset_name"]))

    elif MODE == "glayout_with_training_and_attention":
        visualize_glayout_with_training_and_attention(**main_kwargs)

    elif MODE == "degree_and_homophily":
        analyze_degree_and_homophily()

    elif MODE == "performance_synthetic":
        plot_line_with_std(
            tuple_to_mean_list={
                (20, "GCN"): [0.24008, 0.46788, 0.85649, 0.98591, 1],
                (20, "GAT-GO8"): [0.35761, 0.55295, 0.84871, 0.98708, 0.99771],
                (20, "GAT-DP8"): [0.40997, 0.53756, 0.76837, 0.91099, 0.9713],
                (20, "SuperGAT"): [0.38751, 0.61508, 0.87356, 0.98674, 0.99955],
                (12.5, "GCN"): [0.25025, 0.3936, 0.72674, 0.94985, 0.99694],
                (12.5, "GAT-GO8"): [0.31979, 0.46839, 0.7487, 0.95127, 0.9948],
                (12.5, "GAT-DP8"): [0.39322, 0.49541, 0.67294, 0.86431, 0.95439],
                (12.5, "SuperGAT"): [0.35552, 0.5193, 0.78104, 0.94951, 0.99646],
                (5, "GCN"): [0.27805, 0.30722, 0.54352, 0.69841, 0.88001],
                (5, "GAT-GO8"): [0.2978, 0.36233, 0.54607, 0.70381, 0.88528],
                (5, "GAT-DP8"): [0.39047, 0.44268, 0.56799, 0.65115, 0.80486],
                (5, "SuperGAT"): [0.32552, 0.40137, 0.59095, 0.74638, 0.90562],
            },
            tuple_to_std_list={
                (20, "GCN"): [0.005764859062, 0.006408244689, 0.003850960919, 0.001312211873, 0],
                (20, "GAT-GO8"): [0.01277567611, 0.01050464183, 0.0116724419, 0.003164427278, 0.001321325092],
                (20, "GAT-DP8"): [0.009649305675, 0.0103472895, 0.01333540776, 0.01174946382, 0.02417043649],
                (20, "SuperGAT"): [0.01565534733, 0.01074865573, 0.00828893238, 0.003288221404, 0.000668954408],
                (12.5, "GCN"): [0.006174746958, 0.007363423117, 0.0006329730072, 0.003491060011, 0.0005624944444],
                (12.5, "GAT-GO8"): [0.01237682916, 0.01146463693, 0.0111350797, 0.006684093057, 0.001542724862],
                (12.5, "GAT-DP8"): [0.009930337356, 0.01046335988, 0.01658060313, 0.01661005418, 0.02025778616],
                (12.5, "SuperGAT"): [0.014369746, 0.01068877916, 0.008749765711, 0.006091789556, 0.001283900308],
                (5, "GCN"): [0.006107986575, 0.006671701432, 0.00847995283, 0.004002736564, 0.004652945304],
                (5, "GAT-GO8"): [0.01141227409, 0.0117711979, 0.0113518765, 0.009844485766, 0.00579496333],
                (5, "GAT-DP8"): [0.01070649803, 0.009168293189, 0.0083791348, 0.01342488361, 0.01143767459],
                (5, "SuperGAT"): [0.01224620758, 0.00980576871, 0.009284799405, 0.007544242838, 0.006159188258],
            },
            x_label="Homophily",
            y_label="Test Accuracy",
            name_label_list=["Avg. Degree", "Model"],
            x_list=[0.1, 0.3, 0.5, 0.7, 0.9],
            hue="Model",
            style="Model",
            col="Avg. Degree",
            order=["GCN", "GAT-GO8", "GAT-DP8", "SuperGAT"],
            x_lim=(0, None),
            custom_key="performance-synthetic",
            extension="pdf",
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
