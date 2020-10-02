from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import SAGEConv

from torch_geometric.nn.conv import GCNConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import negative_sampling, batched_negative_sampling
import torch_geometric.nn.inits as tgi

from layer import is_pretraining
from layer_cgat import CGATConv
from data import getattr_d, get_dataset_or_loader


def _get_gn_cls(cls_name: str):
    if cls_name == "LinkGAT":
        return GATConv
    elif cls_name == "LinkGCN":
        return GCNConv
    elif cls_name == "LinkSAGE":
        return SAGEConv
    else:
        raise ValueError


def _get_gn_kwargs(cls_name: str, args, **kwargs):
    if cls_name == "LinkGAT":
        return {"heads": args.heads, "dropout": args.dropout, **kwargs}
    elif cls_name == "LinkGCN":
        return {}
    elif cls_name == "LinkSAGE":
        return {}
    else:
        raise ValueError


def _get_last_features(cls_name: str, args):
    if cls_name == "LinkGAT":
        return args.num_hidden_features * args.heads
    elif cls_name == "LinkGCN":
        return args.num_hidden_features
    elif cls_name == "LinkSAGE":
        return args.num_hidden_features
    else:
        raise ValueError


class MLPNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(MLPNet, self).__init__()
        self.args = args

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.fc = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.Linear(num_input_features, args.num_hidden_features),
            nn.ELU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.num_hidden_features, num_classes),
        )
        pprint(next(self.modules()))

    def forward(self, x, *args, **kwargs):
        return self.fc(x)


class CGATNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(CGATNet, self).__init__()
        self.args = args

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        kwargs = {"use_topk_softmax": args.use_topk_softmax}
        if args.use_topk_softmax:
            kwargs["aggr_k"] = args.aggr_k
        else:
            kwargs["dropout"] = args.dropout

        self.conv1 = CGATConv(
            num_input_features, args.num_hidden_features,
            heads=args.heads, concat=True,
            margin_graph=args.margin_graph,
            margin_boundary=args.margin_boundary,
            num_neg_samples_per_edge=args.num_neg_samples_per_edge,
            **kwargs,
        )

        self.conv2 = CGATConv(
            args.num_hidden_features * args.heads, num_classes,
            heads=(args.out_heads or args.heads), concat=False,
            margin_graph=args.margin_graph,
            margin_boundary=args.margin_boundary,
            num_neg_samples_per_edge=args.num_neg_samples_per_edge,
            **kwargs
        )

        pprint(next(self.modules()))

    def forward(self, x, edge_index, batch=None, **kwargs):
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkGNN(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(LinkGNN, self).__init__()
        self.args = args

        gn_layer = _get_gn_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")
        self.neg_sample_ratio = args.neg_sample_ratio

        self.conv1 = gn_layer(
            num_input_features, args.num_hidden_features,
            **_get_gn_kwargs(args.model_name, args,
                             concat=True),
        )
        self.conv2 = gn_layer(
            _get_last_features(args.model_name, args), num_classes,
            **_get_gn_kwargs(args.model_name, args,
                             heads=(args.out_heads or args.heads),
                             concat=False),
        )

        if args.is_link_gnn:
            self.r_scaling_11, self.r_bias_11 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
            self.r_scaling_12, self.r_bias_12 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
            self.r_scaling_21, self.r_bias_21 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))
            self.r_scaling_22, self.r_bias_22 = Parameter(torch.Tensor(1)), Parameter(torch.Tensor(1))

        self.cache = {"num_updated": 0, "batch": None, "x_conv1": None, "x_conv2": None, "label": None}

        self.reset_parameters()
        pprint(next(self.modules()))

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("r_scaling"):
                tgi.ones(param)
            elif name.startswith("r_bias"):
                tgi.zeros(param)

    def forward(self, x, edge_index, batch=None, **kwargs):

        # Labels
        if self.training and self.cache["label"] is None and self.args.is_link_gnn:
            device = next(self.parameters()).device
            num_pos, num_neg = edge_index.size(1), int(self.neg_sample_ratio * edge_index.size(1))
            label = torch.zeros(num_pos + num_neg).float().to(device)
            label[:edge_index.size(1)] = 1.
            self._update_cache("label", label)

        self._update_cache("batch", batch)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        self._update_cache("x_conv1", x)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        self._update_cache("x_conv2", x)

        return x

    def forward_to_reconstruct_edges(self, x, edge_index, r_scaling_1, r_bias_1, r_scaling_2, r_bias_2, batch=None):
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param r_scaling_1: [1]
        :param r_scaling_2: [1]
        :param r_bias_1: [1]
        :param r_bias_2: [1]
        :param batch: [N]
        :return: Reconstructed edges [2, E + neg_E] (0 <= v <=  1)
        """

        if batch is None:
            num_neg_samples = int(self.neg_sample_ratio * edge_index.size(1))
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=x.size(0),
                num_neg_samples=num_neg_samples)
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index=edge_index,
                batch=batch)

        total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]
        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, F]

        recon = torch.einsum("ef,ef->e", x_i, x_j)  # [E + neg_E]
        recon = r_scaling_1 * F.elu(recon) + r_bias_1
        recon = r_scaling_2 * F.elu(recon) + r_bias_2
        return recon

    def _update_cache(self, key, val):
        self.cache[key] = val
        self.cache["num_updated"] += 1

    @staticmethod
    def get_reconstruction_loss(model, edge_index, edge_sampling_ratio=1.0, criterion=None):

        device = next(model.parameters()).device
        criterion = nn.BCEWithLogitsLoss() if criterion is None else eval(criterion)

        loss_list = []

        batch = model.cache["batch"]
        label = model.cache["label"]
        num_total_samples = label.size(0)
        num_to_sample = int(num_total_samples * edge_sampling_ratio)

        for layer_id in range(1, 3):
            x = model.cache["x_conv{}".format(layer_id)]

            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            recon = model.forward_to_reconstruct_edges(
                x, edge_index,
                r_scaling_1=getattr(model, "r_scaling_{}1".format(layer_id)),
                r_scaling_2=getattr(model, "r_scaling_{}2".format(layer_id)),
                r_bias_1=getattr(model, "r_bias_{}1".format(layer_id)),
                r_bias_2=getattr(model, "r_bias_{}2".format(layer_id)),
                batch=batch,
            )  # [E + neg_E]

            permuted = torch.randperm(num_total_samples).to(device)
            loss = criterion(recon[permuted][:num_to_sample], label[permuted][:num_to_sample])
            loss_list.append(loss)

        return sum(loss_list)

    @staticmethod
    def mix_reconstruction_loss_with_pretraining(loss, model, edge_index, mixing_weight,
                                                 edge_sampling_ratio=1.0, criterion=None,
                                                 current_epoch=None, pretraining_epoch=None):
        if mixing_weight == 0:
            return loss

        current_pretraining = is_pretraining(current_epoch, pretraining_epoch)

        if (current_pretraining is None) or (not current_pretraining):
            w1, w2 = 1.0, mixing_weight  # Forbid pre-training or normal-training
        else:
            w1, w2 = 0.0, 1.0  # Pre-training

        loss = w1 * loss + w2 * LinkGNN.get_reconstruction_loss(
            model=model,
            edge_index=edge_index,
            edge_sampling_ratio=edge_sampling_ratio,
            criterion=criterion,
        )
        return loss


if __name__ == '__main__':
    from arguments import get_args

    main_args = get_args(
        model_name="GCN",
        dataset_class="PPI",
        dataset_name="PPI",
        custom_key="NE",
    )

    train_d, val_d, test_d = get_dataset_or_loader(
        main_args.dataset_class, main_args.dataset_name, main_args.data_root,
        batch_size=main_args.batch_size, seed=main_args.seed,
    )

    _m = LinkGNN(main_args, train_d)

    for b in train_d:
        ob = _m(b.x, b.edge_index)
        print(b.x.size(), b.edge_index.size(), ob.size())
