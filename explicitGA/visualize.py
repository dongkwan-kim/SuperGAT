from layer import negative_sampling
from main import *

from sklearn.manifold import TSNE
from torch_geometric.utils import subgraph, softmax
import networkx as nx
import matplotlib as mpl
import torch.nn.functional as F


def _get_key(args):
    return "{}-{}-{}".format(args.model_name, args.dataset_name, args.custom_key) if args is not None else "raw"


def plot_nodes_by_tsne(xs, ys, args=None, extension="png"):
    x_embed = TSNE(n_components=2).fit_transform(xs)

    df = pd.DataFrame({
        "x_coord": x_embed[:, 0],
        "y_coord": x_embed[:, 1],
        "class": ys,
    })
    plot = sns.scatterplot(x="x_coord", y="y_coord", hue="class", data=df,
                           legend=False, palette="Set1")
    plot.set_xlabel("")
    plot.set_ylabel("")
    plot.get_xaxis().set_visible(False)
    plot.get_yaxis().set_visible(False)
    sns.despine(left=False, right=False, bottom=False, top=False)

    key = _get_key(args)
    plot.get_figure().savefig("../figs/fig_tsne_{}.{}".format(key, extension), bbox_inches='tight')
    plt.clf()


def plot_graph_layout(xs, ys, edge_index, edge_to_attention, args=None, extension="png", layout="tsne"):
    G = nx.Graph()
    G.add_edges_from([(i, j) for i, j in np.transpose(edge_index)])

    if layout == "tsne":
        x_embed = TSNE(n_components=2).fit_transform(xs)
        pos = {xid: x_embed[xid] for xid in range(len(xs))}
    else:
        pos = nx.layout.spring_layout(G)

    n_classes = len(np.unique(ys))

    node_sizes = 4
    node_cmap = plt.cm.get_cmap("Set1")
    class_to_node_color = {c: node_cmap(c / n_classes) for c in range(n_classes)}
    node_color_list = [class_to_node_color[y] for y in ys]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_color_list, alpha=0.5)

    if edge_to_attention is not None:
        edge_color = [float(np.mean(edge_to_attention[tuple(sorted(e))])) for e in G.edges]
        edge_kwargs = dict(edge_color=edge_color, edge_cmap=plt.cm.Greys, width=1.25, alpha=0.5,
                           vmin=np.min(edge_color)/2, vmax=np.max(edge_color)*2)
    else:
        edge_kwargs = dict(edge_color="grey", width=0.5, alpha=0.3)

    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, **edge_kwargs)

    ax = plt.gca()
    ax.set_axis_off()

    key = _get_key(args)
    plt.savefig("../figs/fig_glayout_{}.{}".format(key, extension), bbox_inches='tight')
    plt.clf()


def visualize_without_training(**kwargs):
    _args = get_args(**kwargs)
    train_d, val_d, test_d = get_dataset_or_loader(
        "Planetoid", _args.dataset_name, _args.data_root,
        batch_size=_args.batch_size, seed=_args.seed,
    )
    data = train_d[0]

    plot_graph_layout(data.x.numpy(), data.y.numpy(), data.edge_index.numpy(), edge_to_attention=None)


def get_first_layer_and_e2att(model, data, _args, negative_normalized=False, with_negatives=False):

    model = model.to("cpu")
    model.eval()
    with torch.no_grad():

        xs_after_conv1 = model.conv1(data.x, data.edge_index)

        x = torch.matmul(data.x, model.conv1.weight)
        size_i = x.size(0)

        def edge_to_sorted_tuple(e):
            return tuple(sorted([int(e[0]), int(e[1])]))

        if not with_negatives:
            edge_index_j, edge_index_i = data.edge_index
            x_i = torch.index_select(x, 0, edge_index_i)
            x_j = torch.index_select(x, 0, edge_index_j)

            x_j = x_j.view(-1, model.conv1.heads, model.conv1.out_channels)
            x_i = x_i.view(-1, model.conv1.heads, model.conv1.out_channels)
            alpha = model.conv1._get_attention(edge_index_i, x_i, x_j, size_i, normalize=True, with_negatives=False)
            mean_alpha = alpha.mean(dim=-1)  # [E]

            edge_to_attention = defaultdict(list)
            for i, e in enumerate(data.edge_index.t()):
                edge_to_attention[edge_to_sorted_tuple(e)].append(float(mean_alpha[i]))

            return xs_after_conv1, edge_to_attention

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
            if negative_normalized:
                alpha = softmax(alpha, total_edge_index[1], size_i)

            mean_alpha = alpha.mean(dim=-1)  # [E + neg_E]

            edge_to_attention = defaultdict(list)
            edge_to_is_negative = dict()
            for i, e in enumerate(total_edge_index.t()):
                edge_to_attention[edge_to_sorted_tuple(e)].append(float(mean_alpha[i]))
                edge_to_is_negative[edge_to_sorted_tuple(e)] = i > total_edge_index.size(1) // 2

            return xs_after_conv1, edge_to_attention, edge_to_is_negative


def visualize_with_training(**kwargs):
    _args = get_args(**kwargs)
    _args.verbose = 2
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

    train_d, val_d, test_d = get_dataset_or_loader(
        "Planetoid", _args.dataset_name, _args.data_root,
        batch_size=_args.batch_size, seed=_args.seed,
    )
    data = train_d[0]

    xs_after_conv1, edge_to_attention = get_first_layer_and_e2att(model, data, _args)
    plot_graph_layout(xs_after_conv1.numpy(), data.y.numpy(), data.edge_index.numpy(),
                      edge_to_attention=edge_to_attention, args=_args)


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


def attention_analysis(negative_normalized=True, **kwargs):
    _args = get_args(**kwargs)
    _args.verbose = 2
    _args.save_model = False
    if _args.dataset_name == "CiteSeer":
        _args.epochs = 200
    elif _args.dataset_name == "Cora":
        _args.epochs = 100
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

    xs_after_conv1, edge_to_attention, edge_to_is_negative = get_first_layer_and_e2att(
        model, data, _args, with_negatives=True, negative_normalized=negative_normalized)

    _data_list = []
    for edge, att in edge_to_attention.items():
        if edge_to_is_negative[edge]:
            _data_list.append([edge[0], edge[1], float(np.mean(att)), "negative"])
        else:
            _data_list.append([edge[0], edge[1], float(np.mean(att)), "positive"])

    key = _get_key(_args)
    df = pd.DataFrame(_data_list, columns=["i", "j", "un-normalized attention", "sample type"])

    sns.set_context("poster")

    plt.figure(figsize=(6, 8))
    plot = sns.boxplot(x="sample type", y="un-normalized attention", data=df,
                       order=["positive", "negative"], width=0.35)

    if _args.dataset_name == "Cora":
        plot.set_ylim(-0.2, 1.5)
    elif _args.dataset_name == "CiteSeer":
        plot.set_ylim(-0.01, 0.09)

    plot.set_title("{}/{}".format("GAT" if _args.custom_key == "NE" else "Super-GAT",
                                  _args.dataset_name))
    plot.get_figure().savefig("../figs/fig_attention_{}_{}.pdf".format(
        key, "norm" if negative_normalized else "unnorm"), bbox_inches='tight')
    plt.clf()
    return df


if __name__ == '__main__':

    sns.set(style="white")

    vis_kwargs = dict(
        model_name="GAT",  # GAT, BaselineGAT
        dataset_class="Planetoid",
        dataset_name="CiteSeer",  # Cora, CiteSeer, PubMed
        custom_key="NE",  # NE, EV1, EV2, NR, RV1
    )
    
    os.makedirs("../figs", exist_ok=True)

    MODE = "attention_analysis"

    if MODE == "without_training":
        visualize_without_training(**vis_kwargs)
    elif MODE == "with_training":
        visualize_with_training(**vis_kwargs)
    elif MODE == "attention_analysis":
        data_frame_list = []
        for dn in ["Cora", "CiteSeer"]:
            for ck in ["NE", "EV1"]:
                for nn in [False]:
                    vis_kwargs = dict(
                        model_name="GAT",  # GAT, BaselineGAT
                        dataset_class="Planetoid",
                        dataset_name=dn,  # Cora, CiteSeer, PubMed
                        custom_key=ck,  # NE, EV1, EV2, NR, RV1
                    )
                    data_frame_list.append(attention_analysis(**vis_kwargs, negative_normalized=nn))
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
    else:
        raise ValueError
