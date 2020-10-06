import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from layer import SuperGAT
from data import get_dataset_or_loader, getattr_d

from pprint import pprint
from typing import Tuple, List


def _get_gat_cls(attention_name: str):
    if attention_name in ["GAT", "GATPPI", "LargeGAT"]:
        return SuperGAT
    else:
        raise ValueError("{} is not proper name".format(attention_name))


def _inspect_attention_tensor(x, edge_index, att_res) -> bool:
    num_pos_samples = edge_index.size(1) + x.size(0)

    if att_res["att_with_negatives"] is not None \
            and (num_pos_samples == 13264 or
                 num_pos_samples == 12431 or
                 num_pos_samples == 0):

        att_with_negatives = att_res["att_with_negatives"].mean(dim=-1)
        att_with_negatives_cloned = att_with_negatives.clone()
        att_with_negatives_cloned = torch.sigmoid(att_with_negatives_cloned)

        pos_samples = att_with_negatives_cloned[:num_pos_samples]
        neg_samples = att_with_negatives_cloned[num_pos_samples:]

        print()
        pos_m, pos_s = float(pos_samples.mean()), float(pos_samples.std())
        cprint("TPos: {} +- {} ({})".format(pos_m, pos_s, pos_samples.size()), "blue")
        neg_m, neg_s = float(neg_samples.mean()), float(neg_samples.std())
        cprint("TNeg: {} +- {} ({})".format(neg_m, neg_s, neg_samples.size()), "blue")
        return True
    else:
        return False


class SuperGATNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super().__init__()
        self.args = args

        gat_cls = _get_gat_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = gat_cls(
            num_input_features, args.num_hidden_features,
            heads=args.heads, dropout=args.dropout, concat=True,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio, use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg, scaling_factor=args.scaling_factor,
        )

        self.conv2 = gat_cls(
            args.num_hidden_features * args.heads, num_classes,
            heads=(args.out_heads or args.heads), dropout=args.dropout, concat=False,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio, use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg, scaling_factor=args.scaling_factor,
        )

        pprint(next(self.modules()))

    def forward_for_all_layers(self, x, edge_index, batch=None, **kwargs):
        x1 = F.dropout(x, p=self.args.dropout, training=self.training)
        x1 = self.conv1(x1, edge_index, batch=batch, **kwargs)
        x2 = F.elu(x1)
        x2 = F.dropout(x2, p=self.args.dropout, training=self.training)
        x2 = self.conv2(x2, edge_index, batch=batch, **kwargs)
        return x1, x2

    def forward(self, x, edge_index, batch=None, **kwargs) -> torch.Tensor:

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index, batch=batch, **kwargs)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index, batch=batch, **kwargs)

        if self.training and self.args.verbose >= 3:
            _inspect_attention_tensor(x, edge_index, self.conv2.cache)

        return x

    def set_layer_attrs(self, name, value):
        setattr(self.conv1, name, value)
        setattr(self.conv2, name, value)

    def get_attention_dist_by_layer(self, edge_index, num_nodes) -> List[List[torch.Tensor]]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return List[List[torch.Tensor]]: [L, N, [#neighbors, heads]]
        """
        return [
            self.conv1.get_attention_dist(edge_index, num_nodes),
            self.conv2.get_attention_dist(edge_index, num_nodes),
        ]


class LargeSuperGATNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super().__init__()
        self.args = args
        self.num_layers = self.args.num_layers

        gat_cls = _get_gat_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        conv_common_kwargs = dict(
            dropout=args.dropout,
            is_super_gat=args.is_super_gat,
            attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio, edge_sample_ratio=args.edge_sampling_ratio,
            pretraining_noise_ratio=args.pretraining_noise_ratio,
            use_pretraining=args.use_pretraining,
            to_undirected_at_neg=args.to_undirected_at_neg,
            scaling_factor=args.scaling_factor,
        )
        self.conv_list = []
        self.bn_list = []
        for conv_id in range(1, self.num_layers + 1):
            if conv_id == 1:  # first layer
                in_channels, out_channels = num_input_features, args.num_hidden_features
                heads, concat = args.heads, True
            elif conv_id == self.num_layers:  # last layer
                in_channels, out_channels = args.num_hidden_features * args.heads, num_classes
                heads, concat = args.out_heads or args.heads, False
            else:
                in_channels, out_channels = args.num_hidden_features * args.heads, args.num_hidden_features
                heads, concat = args.heads, True
            # conv
            conv = gat_cls(in_channels, out_channels, heads=heads, concat=concat, **conv_common_kwargs)
            conv_name = "conv{}".format(conv_id)
            self.conv_list.append(conv)
            setattr(self, conv_name, conv)
            self.add_module(conv_name, conv)
            # bn
            if args.use_bn and conv_id != self.num_layers:  # not last layer
                bn = nn.BatchNorm1d(out_channels * heads)
                bn_name = "bn{}".format(conv_id)
                self.bn_list.append(bn)
                setattr(self, bn_name, bn)
                self.add_module(bn_name, bn)

        pprint(next(self.modules()))

    def forward(self, x, edge_index, batch=None, **kwargs) -> torch.Tensor:
        for conv_idx, conv in enumerate(self.conv_list):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = conv(x, edge_index, **kwargs)
            if conv_idx != self.num_layers - 1:
                if self.args.use_bn:
                    x = self.bn_list[conv_idx](x)
                x = F.elu(x)
        return x

    def set_layer_attrs(self, name, value):
        for conv in self.conv_list:
            setattr(conv, name, value)

    def get_attention_dist_by_layer(self, edge_index, num_nodes) -> List[List[torch.Tensor]]:
        """
        :param edge_index: tensor the shape of which is [2, E]
        :param num_nodes: number of nodes
        :return List[List[torch.Tensor]]: [L, N, [#neighbors, heads]]
        """
        attention_dist_by_layer = []
        for conv in self.conv_list:
            attention_dist_by_layer.append(conv.get_attention_dist(edge_index, num_nodes))
        return attention_dist_by_layer
