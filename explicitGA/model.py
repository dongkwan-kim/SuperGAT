import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

import torch_geometric.transforms as T
import torch_geometric.nn as pygnn

from attention import ExplicitGAT
from data import get_dataset_or_loader, getattr_d

from typing import Tuple, List


def _get_gat_cls(attention_name: str):
    if attention_name == "GAT":
        return ExplicitGAT
    else:
        raise ValueError


def _inspect_attention_tensor(x, edge_index, att_res) -> bool:
    num_pos_samples = edge_index.size(1) + x.size(0)

    if att_res and (num_pos_samples == 13264 or
                    num_pos_samples == 12431 or
                    num_pos_samples == 0):

        total_att = att_res["att_with_negatives"]
        total_att_cloned = total_att.clone()
        total_att_cloned = torch.sigmoid(total_att_cloned)

        if len(total_att.size()) == 2:
            pos_samples = total_att_cloned[:num_pos_samples, 0]
            neg_samples = total_att_cloned[num_pos_samples:, 0]
        else:
            pos_samples = total_att_cloned[:num_pos_samples]
            neg_samples = total_att_cloned[num_pos_samples:]

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


class ExplicitGATNet(nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(ExplicitGATNet, self).__init__()

        self.args = args

        gat_cls = _get_gat_cls(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = gat_cls(
            num_input_features, args.num_hidden_features,
            heads=args.head, dropout=args.dropout,
            is_explicit=args.is_explicit, explicit_type=args.explicit_type,
        )

        self.conv2 = gat_cls(
            args.num_hidden_features * args.head, num_classes,
            heads=1, dropout=args.dropout,
            is_explicit=args.is_explicit, explicit_type=args.explicit_type,
        )

        if args.pool_name is not None:
            self.pool = to_pool_cls(args.pool_name)
            self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x, edge_index, batch=None) -> Tuple[torch.Tensor, None or List[torch.Tensor]]:

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        if self.training:
            _inspect_attention_tensor(x, edge_index, self.conv2.att_residuals)

        if self.args.pool_name is not None:
            x = self.pool(x, batch)
            x = self.fc(x)

        x = F.log_softmax(x, dim=1)
        return x

    def get_explicit_attention_loss(self, num_pos_samples, criterion=None):

        assert self.args.is_explicit

        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        loss_list = []
        att_residuals_list = [m.att_residuals for m in self.modules()
                              if m.__class__.__name__ == ExplicitGAT.__name__]

        for att_res in att_residuals_list:

            att = att_res["att_with_negatives"]
            num_total_samples = att.size(0)
            num_to_sample = int(num_total_samples * self.args.edge_sampling_ratio)

            att = att.mean(dim=-1)  # [E + neg_E]

            label = torch.zeros(num_total_samples)
            label[:num_pos_samples] = 1
            label = label.float()

            permuted = torch.randperm(num_total_samples)

            loss = criterion(att[permuted][:num_to_sample], label[permuted][:num_to_sample])
            loss_list.append(loss)

        total_loss = self.args.att_lambda * sum(loss_list)
        return total_loss
