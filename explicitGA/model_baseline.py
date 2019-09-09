import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from termcolor import cprint

from typing import Tuple, List

from torch_geometric.utils import remove_self_loops, add_self_loops

from layer import negative_sampling, batched_negative_sampling
from data import getattr_d
from model import to_pool_cls


class LinearWrapper(nn.Linear):

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, x, edge_index=None, *args, **kwargs):
        return super().forward(x)


def _get_gn_cls(cls_name: str):
    if cls_name == "BaselineGAT":
        return GATConv
    elif cls_name == "BaselineGCN":
        return GCNConv
    elif cls_name == "MLP":
        return LinearWrapper
    else:
        raise ValueError


def _get_gn_kwargs(cls_name: str, args, **kwargs):
    if cls_name == "BaselineGAT":
        return {"heads": args.heads, "dropout": args.dropout, **kwargs}
    elif cls_name == "BaselineGCN":
        return {}
    elif cls_name == "MLP":
        return {}
    else:
        raise ValueError


def _get_last_features(cls_name: str, args):
    if cls_name == "BaselineGAT":
        return args.num_hidden_features * args.heads
    elif cls_name == "BaselineGCN":
        return args.num_hidden_features
    elif cls_name == "MLP":
        return args.num_hidden_features
    else:
        raise ValueError


class BaselineGNNet(nn.Module):

    def __init__(self, args, dataset_or_loader):

        super(BaselineGNNet, self).__init__()

        self.args = args

        gn_layer = _get_gn_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = gn_layer(
            num_input_features, args.num_hidden_features,
            **_get_gn_kwargs(args.model_name, args),
        )

        self.conv2 = gn_layer(
            _get_last_features(args.model_name, args), num_classes,
            **_get_gn_kwargs(args.model_name, args, heads=1),
        )

        if args.pool_name is not None:
            self.pool = to_pool_cls(args.pool_name)
            self.fc = nn.Linear(num_classes, num_classes)

        self.residuals = {"num_updated": 0, "x_conv1": None, "x_conv2": None, "batch": None}

    def forward(self, x, edge_index, batch=None):

        self._update_residuals("batch", batch)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        self._update_residuals("x_conv1", x)
        x = F.elu(self.conv1(x, edge_index))

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        self._update_residuals("x_conv2", x)
        x = F.elu(self.conv2(x, edge_index))

        if self.args.pool_name is not None:
            x = self.pool(x, batch)
            x = self.fc(x)

        x = F.log_softmax(x, dim=1)
        return x

    def forward_to_reconstruct_edges(self, x, edge_index, batch=None):
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param batch: [N]
        :return: Reconstructed edges [2, E + neg_E] (0 <= v <=  1)
        """

        if batch is None:
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=x.size(0))
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index=edge_index,
                batch=batch)

        total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]
        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, F]

        recon = torch.einsum("ef,ef->e", x_i, x_j)  # [E + neg_E]
        return recon

    def get_reconstruction_loss(self, edge_index, criterion=None):

        device = next(self.parameters()).device

        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        loss_list = []

        batch = self.residuals["batch"]
        for layer_id in range(1, 3):
            x = self.residuals["x_conv{}".format(layer_id)]
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            recon = self.forward_to_reconstruct_edges(x, edge_index, batch)  # [E + neg_E]

            num_pos_samples = edge_index.size(1)
            num_total_samples = recon.size(0)
            num_to_sample = int(num_total_samples * self.args.edge_sampling_ratio)

            label = torch.zeros(num_total_samples).to(device)
            label[:num_pos_samples] = 1
            label = label.float()

            permuted = torch.randperm(num_total_samples).to(device)

            loss = criterion(recon[permuted][:num_to_sample], label[permuted][:num_to_sample])
            loss_list.append(loss)

        total_loss = self.args.recon_lambda * sum(loss_list)
        return total_loss

    def _update_residuals(self, key, val):
        self.residuals[key] = val
        self.residuals["num_updated"] += 1

