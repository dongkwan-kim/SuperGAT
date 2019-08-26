from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch_geometric.nn as tgnn

from attention import ExplicitGAT
from data import get_dataset_or_loader, getattr_d


def _get_attention_layer(attention_name: str):
    if attention_name == "GAT":
        return ExplicitGAT
    else:
        raise ValueError


def _to_pool_cls(pool_name):
    if pool_name in tgnn.glob.__all__ or pool_name in tgnn.pool.__all__:
        return eval("tgnn.{}".format(pool_name))
    else:
        raise ValueError("{} is not in {} or {}".format(pool_name, tgnn.glob.__all__, tgnn.pool.__all__))


def _inspect_attention_tensor(x, edge_index, att_res) -> bool:
    num_pos_samples = edge_index.size(1) + x.size(0)

    if att_res and (num_pos_samples == 13264 or
                    num_pos_samples == 12431 or
                    num_pos_samples == 0):

        total_att = att_res["total_alpha"]
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


class GATNet(torch.nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(GATNet, self).__init__()

        self.args = args

        attention_layer = _get_attention_layer(self.args.model_name)

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = attention_layer(
            num_input_features, args.num_hidden_features,
            heads=args.head, dropout=args.dropout,
            is_explicit=args.is_explicit, explicit_type=args.explicit_type,
        )

        self.conv2 = attention_layer(
            args.num_hidden_features * args.head, num_classes,
            heads=1, dropout=args.dropout,
            is_explicit=args.is_explicit, explicit_type=args.explicit_type,
        )

        if args.pool_name is not None:
            self.pool = _to_pool_cls(args.pool_name)
            self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x, edge_index, batch=None) -> Tuple[torch.Tensor, None or List[torch.Tensor]]:

        x = F.dropout(x, p=0.6, training=self.training)
        x, att_res_1 = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x, att_res_2 = self.conv2(x, edge_index)
        x = F.elu(x)

        if self.training and _inspect_attention_tensor(x, edge_index, att_res_2):
            pass

        if self.args.pool_name is not None:
            x = self.pool(x, batch)
            x = self.fc(x)

        x = F.log_softmax(x, dim=1)

        explicit_attentions = [att_res_1, att_res_2] if att_res_1 is not None else None

        return x, explicit_attentions
