from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

import torch_geometric.transforms as T
import torch_geometric.nn as pygnn

from layer import SupervisedGAT
from data import get_dataset_or_loader, getattr_d

from typing import Tuple, List


def _get_gat_cls(attention_name: str):
    if attention_name == "GAT" or attention_name == "GATPPI":
        return SupervisedGAT
    else:
        raise ValueError("{} is not proper name".format(attention_name))


def _inspect_attention_tensor(x, edge_index, att_res) -> bool:
    num_pos_samples = edge_index.size(1) + x.size(0)

    if att_res["att_with_negatives"] is not None \
            and (num_pos_samples == 13264 or
                 num_pos_samples == 12431 or
                 num_pos_samples == 0):

        att_with_negatives = att_res["att_with_negatives"]
        att_with_negatives_cloned = att_with_negatives.clone()
        att_with_negatives_cloned = torch.sigmoid(att_with_negatives_cloned)

        if len(att_with_negatives.size()) == 2:
            pos_samples = att_with_negatives_cloned[:num_pos_samples, 0]
            neg_samples = att_with_negatives_cloned[num_pos_samples:, 0]
        else:
            pos_samples = att_with_negatives_cloned[:num_pos_samples]
            neg_samples = att_with_negatives_cloned[num_pos_samples:]

        print()
        pos_m, pos_s = float(pos_samples.mean()), float(pos_samples.std())
        cprint("TPos: {} +- {}".format(pos_m, pos_s), "blue")
        neg_m, neg_s = float(neg_samples.mean()), float(neg_samples.std())
        cprint("TNeg: {} +- {}".format(neg_m, neg_s), "blue")
        return True
    else:
        return False


def to_pool_cls(pool_name):
    if pool_name in pygnn.glob.__all__ or pool_name in pygnn.pool.__all__:
        return eval("tgnn.{}".format(pool_name))
    else:
        raise ValueError("{} is not in {} or {}".format(pool_name, pygnn.glob.__all__, pygnn.pool.__all__))


class BaseSupervisedGATNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super().__init__()
        self.args = args

    def forward(self, x, edge_index, batch=None):
        raise NotImplementedError

    def get_supervised_attention_loss(self, criterion=None):

        assert self.args.is_super_gat
        if self.args.att_lambda == 0:
            return 0

        device = next(self.parameters()).device

        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = eval(criterion)

        loss_list = []
        att_residuals_list = [(m, m.residuals) for m in self.modules()
                              if m.__class__.__name__ == SupervisedGAT.__name__]

        for module, att_res in att_residuals_list:

            # Attention (X)
            att = att_res["att_with_negatives"]  # [E + neg_E, heads]
            num_total_samples = att.size(0)
            num_to_sample = int(num_total_samples * self.args.edge_sampling_ratio)

            # Labels (Y)
            label = att_res["att_label"]  # [E + neg_E]

            permuted = torch.randperm(num_total_samples).to(device)

            att = att.mean(dim=-1)  # [E + neg_E]
            loss = criterion(att[permuted][:num_to_sample], label[permuted][:num_to_sample])

            loss_list.append(loss)

        total_loss = self.args.att_lambda * sum(loss_list)
        return total_loss


class SupervisedGATNet(BaseSupervisedGATNet):

    def __init__(self, args, dataset_or_loader):
        super(SupervisedGATNet, self).__init__(args, dataset_or_loader)

        gat_cls = _get_gat_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = gat_cls(
            num_input_features, args.num_hidden_features,
            heads=args.heads, dropout=args.dropout, concat=True,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
        )

        self.conv2 = gat_cls(
            args.num_hidden_features * args.heads, num_classes,
            heads=(args.out_heads or args.heads), dropout=args.dropout, concat=False,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
        )

        if args.pool_name is not None:
            self.pool = to_pool_cls(args.pool_name)
            self.fc = nn.Linear(num_classes, num_classes)

        pprint(next(self.modules()))

    def forward(self, x, edge_index, batch=None) -> torch.Tensor:

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        if self.training and self.args.verbose >= 2:
            _inspect_attention_tensor(x, edge_index, self.conv2.residuals)

        if self.args.pool_name is not None:
            x = self.pool(x, batch)
            x = self.fc(x)

        return x


class SupervisedGATNetPPI(BaseSupervisedGATNet):

    def __init__(self, args, dataset_or_loader):
        super(SupervisedGATNetPPI, self).__init__(args, dataset_or_loader)

        gat_cls = _get_gat_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = gat_cls(
            num_input_features, args.num_hidden_features,
            heads=args.heads, dropout=args.dropout, concat=True,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
        )
        self.lin1 = nn.Linear(num_input_features, args.num_hidden_features * args.heads)

        self.conv2 = gat_cls(
            args.num_hidden_features * args.heads, args.num_hidden_features,
            heads=args.heads, dropout=args.dropout, concat=True,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
        )
        if self.args.use_skip_connect_for_2:
            self.lin2 = nn.Linear(args.num_hidden_features * args.heads, args.num_hidden_features * args.heads)

        self.conv3 = gat_cls(
            args.num_hidden_features * args.heads, num_classes,
            heads=(args.out_heads or args.heads), dropout=args.dropout, concat=False,
            is_super_gat=args.is_super_gat, attention_type=args.attention_type,
            super_gat_criterion=args.super_gat_criterion,
            neg_sample_ratio=args.neg_sample_ratio,
        )
        self.lin3 = nn.Linear(args.num_hidden_features * args.heads, num_classes)

        pprint(next(self.modules()))

    def forward(self, x, edge_index, batch=None) -> torch.Tensor:

        x = self.conv1(x, edge_index) + self.lin1(x)
        x = F.elu(x)

        if self.args.use_skip_connect_for_2:
            x = self.conv2(x, edge_index) + self.lin2(x)
        else:
            x = self.conv2(x, edge_index) + x
        x = F.elu(x)

        x = self.conv3(x, edge_index) + self.lin3(x)

        # if self.training and self.args.verbose >= 2:
        #     _inspect_attention_tensor(x, edge_index, self.conv2.residuals)

        return x
