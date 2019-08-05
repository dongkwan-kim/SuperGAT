import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch_geometric.nn as tgnn

from attention import ExplicitGAT
from data import get_dataset_or_loader, getattr_d


def get_model_cls(model_name):
    if model_name == "GATNet":
        return GATNet
    else:
        raise ValueError


def _to_pool_cls(pool_name):
    if pool_name in tgnn.glob.__all__ or pool_name in tgnn.pool.__all__:
        return eval("tgnn.{}".format(pool_name))
    else:
        raise ValueError("{} is not in {} or {}".format(pool_name, tgnn.glob.__all__, tgnn.pool.__all__))


class GATNet(torch.nn.Module):

    def __init__(self, args, dataset_or_loader):
        super(GATNet, self).__init__()

        self.args = args

        num_input_features = getattr_d(dataset_or_loader, "num_node_features")
        num_classes = getattr_d(dataset_or_loader, "num_classes")

        self.conv1 = ExplicitGAT(
            num_input_features, args.num_hidden_features,
            heads=args.head, dropout=args.dropout, is_explicit=args.is_explicit,
        )

        num_input_features *= args.head
        self.conv2 = ExplicitGAT(
            args.num_hidden_features * args.head, args.num_hidden_features,
            heads=args.head, dropout=0., is_explicit=args.is_explicit,
        )
        self.conv3 = ExplicitGAT(
            args.num_hidden_features * args.head, args.num_hidden_features,
            heads=1, dropout=0., is_explicit=args.is_explicit,
        )

        if args.pool_name is not None:
            self.pool = _to_pool_cls(args.pool_name)

        self.fc = nn.Sequential(
            nn.Linear(args.num_hidden_features, args.num_hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(args.num_hidden_features, num_classes),
        )

    def forward(self, x, edge_index, batch=None):

        x, att1 = self.conv1(x, edge_index)
        x = F.relu(x)

        x, att2 = self.conv2(x, edge_index)
        x = F.relu(x)

        x, att3 = self.conv3(x, edge_index)

        if self.args.pool_name is not None:
            x = self.pool(x, batch)

        x = self.fc(x)

        return x
