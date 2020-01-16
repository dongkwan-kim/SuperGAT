import torch
import torch_geometric as pyg
from termcolor import cprint
from torch_geometric.datasets import *
from torch_geometric.data import DataLoader, InMemoryDataset
from torch_geometric.utils import is_undirected, to_undirected
import torch_geometric.transforms as T

import os
from pprint import pprint
from typing import Tuple, Callable

from layer import negative_sampling


def get_one_link_edge_index(edge_index):
    """Remove (j, i) if there are (i, j) and (j, i)
    :param edge_index: undirected edge_index the size of which is [2, 2E]
    :return: one_link_edge_index the size of which is [2, X] (E <= X < 2E)
    """
    one_link_edge_index = torch.sort(edge_index.t())[0].t()
    one_link_edge_index = torch.unique(one_link_edge_index, dim=1)
    return one_link_edge_index


class LinkPlanetoid(Planetoid):

    def __init__(self, root, name, train_val_test_ratio=None):
        super().__init__(root, name)
        self.train_val_test_ratio = train_val_test_ratio or (0.8, 0.1, 0.1)

        tpei, vei, tei = self.train_val_test_split()
        self.train_pos_edge_index = tpei  # undirected [2, E * 0.8]
        self.val_edge_index = vei  # undirected [2, E * 0.1 * 2]
        self.test_edge_index = tei  # undirected [2, E * 0.1 * 2]

        self.val_edge_y = self.get_edge_y(self.val_edge_index.size(1))
        self.test_edge_y = self.get_edge_y(self.test_edge_index.size(1))

        # Remove validation/test pos edges from self.edge_index
        self.data.edge_index = self.train_pos_edge_index

    def _sample_train_neg_edge_index(self, is_undirected_edges=True):
        must_not_tnei = torch.cat([self.train_pos_edge_index, self.val_edge_index, self.test_edge_index], dim=1)
        neg_edge_index = negative_sampling(
            edge_index=must_not_tnei,
            num_nodes=self.data.x.size(0),
            num_neg_samples=self.train_pos_edge_index.size(1) // 2,
        )
        if is_undirected_edges:
            neg_edge_index = to_undirected(neg_edge_index)
        return neg_edge_index

    def train_val_test_split(self):
        x, edge_index = self.data.x, self.data.edge_index
        num_nodes, num_edges = x.size(0), edge_index.size(1)  # [N], [2, uE]

        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=x.size(0),
            num_neg_samples=int(edge_index.size(1) * 1.5),
        )
        neg_edge_index = get_one_link_edge_index(neg_edge_index)[:, :num_edges // 2]  # [2, dE]

        # Remove (j, i) if there are (i, j) and (j, i): is_undirected
        if is_undirected(edge_index):
            edge_index = get_one_link_edge_index(edge_index)  # [2, dE]
            num_edges = num_edges // 2

        num_pos_train, num_pos_val, _ = [round(num_edges * r) for r in self.train_val_test_ratio]

        pos_permuted = torch.randperm(edge_index.size(1))  # [dE]
        neg_permuted = torch.randperm(neg_edge_index.size(1))  # [dE]

        train_pos_edge_index = edge_index[:, pos_permuted][:, :num_pos_train]
        val_pos_edge_index = edge_index[:, pos_permuted][:, num_pos_train:num_pos_train + num_pos_val]
        test_pos_edge_index = edge_index[:, pos_permuted][:, num_pos_train + num_pos_val:]
        num_pos_test = test_pos_edge_index.size(1)

        test_neg_edge_index = neg_edge_index[:, neg_permuted][:, :num_pos_test]
        val_neg_edge_index = neg_edge_index[:, neg_permuted][:, num_pos_test:num_pos_test + num_pos_val]

        val_edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], dim=1)
        test_edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=1)

        if is_undirected(self.data.edge_index):
            train_pos_edge_index = to_undirected(train_pos_edge_index)
            val_edge_index = to_undirected(val_edge_index)
            test_edge_index = to_undirected(test_edge_index)
        return train_pos_edge_index, val_edge_index, test_edge_index

    @staticmethod
    def get_edge_y(num_edges, pos_num_or_ratio=0.5, device=None):
        num_pos = pos_num_or_ratio if isinstance(pos_num_or_ratio, int) else int(pos_num_or_ratio * num_edges)
        y = torch.zeros(num_edges).float()
        y[:num_pos] = 1.
        y = y if device is None else y.to(device)
        return y

    def __getitem__(self, item) -> torch.Tensor:
        """
        :param item:
        :return: Draw negative samples, and return [2, E * 0.8 * 2] tensor
        """
        datum = super().__getitem__(item)

        # Sample negative training samples from the negative sample pool
        train_neg_edge_index = self._sample_train_neg_edge_index(is_undirected(self.test_edge_index))
        train_edge_index = torch.cat([self.train_pos_edge_index, train_neg_edge_index], dim=1)
        train_edge_y = self.get_edge_y(train_edge_index.size(1),
                                       pos_num_or_ratio=self.train_pos_edge_index.size(1),
                                       device=train_edge_index.device)

        # Add attributes for edge prediction
        datum.__setitem__("train_edge_index", train_edge_index)
        datum.__setitem__("val_edge_index", self.val_edge_index)
        datum.__setitem__("test_edge_index", self.test_edge_index)
        datum.__setitem__("train_edge_y", train_edge_y)
        datum.__setitem__("val_edge_y", self.val_edge_y)
        datum.__setitem__("test_edge_y", self.test_edge_y)
        return datum


def get_dataset_class_name(dataset_name: str) -> str:
    dataset_name = dataset_name.lower()
    if dataset_name in ["cora", "citeseer", "pubmed"]:
        return "Planetoid"
    elif dataset_name in ['ptc_mr', 'nci1', 'proteins', 'dd', 'collab', 'imdb-binary', 'imdb-multi']:
        return "TUDataset"
    elif dataset_name in ["ppi"]:
        return "PPI"
    else:
        raise ValueError


def get_dataset_class(dataset_class: str) -> Callable[..., InMemoryDataset]:
    assert dataset_class in (pyg.datasets.__all__ + ["LinkPlanetoid"])
    return eval(dataset_class)


def get_dataset_or_loader(dataset_class: str, dataset_name: str or None, root: str,
                          batch_size: int = 128,
                          train_val_test: Tuple[float, float, float] = (0.9 * 0.9, 0.9 * 0.1, 0.1),
                          seed: int = 42, **kwargs)\
        -> Tuple[InMemoryDataset or DataLoader, DataLoader or None, DataLoader or None]:
    """
    Note that datasets structure in torch_geometric varies.
    :param dataset_class: ['KarateClub', 'TUDataset', 'Planetoid', 'CoraFull', 'Coauthor', 'Amazon', 'PPI', 'Reddit',
                           'QM7b', 'QM9', 'Entities', 'GEDDataset', 'MNISTSuperpixels', 'FAUST', 'DynamicFAUST',
                           'ShapeNet', 'ModelNet', 'CoMA', 'SHREC2016', 'TOSCA', 'PCPNetDataset', 'S3DIS',
                           'GeometricShapes', 'BitcoinOTC', 'ICEWS18', 'GDELT']
    :param dataset_name: Useful names are below.
        - For graph classification [From 1]
            TUDataset: PTC_MR, NCI1, PROTEINS, DD, COLLAB, IMDB-BINARY, IMDB-MULTI [2]
            Entities: MUTAG
        - For node classification (semi-supervised setting) [From 3]
            Planetoid: CiteSeer, Cora, PubMed [4]
            PPI: None [5]
        [1] An End-to-End Deep Learning Architecture for Graph Classification (AAAI 2018)
        [2] https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        [3] SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS (ICLR 2017)
        [4] Revisiting Semi-Supervised Learning with Graph Embeddings
        [5] Predicting Multicellular Function through Multi-layer Tissue Networks
    :param root: data path
    :param batch_size:
    :param train_val_test:
    :param seed: 42
    :param kwargs:
    :return:
    """
    if dataset_class != "PPI":
        kwargs["name"] = dataset_name

    torch.manual_seed(seed)
    root = os.path.join(root, dataset_name or dataset_class)
    dataset_cls = get_dataset_class(dataset_class)

    if dataset_class in ["TUDataset"]:  # Graph
        dataset = dataset_cls(root=root, **kwargs).shuffle()
        n_samples = len(dataset)
        train_samples = int(n_samples * train_val_test[0])
        val_samples = int(n_samples * train_val_test[1])
        train_dataset = dataset[:train_samples]
        val_dataset = dataset[train_samples:train_samples+val_samples]
        test_dataset = dataset[train_samples+val_samples:]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    elif dataset_class in ["Planetoid", "LinkPlanetoid"]:  # Node or Link (One graph with given mask)
        dataset = dataset_cls(root=root, **kwargs)
        return dataset, None, None

    elif dataset_class in ["PPI"]:  # Node (Multiple graphs)
        train_dataset = dataset_cls(root=root, split='train', **kwargs)
        val_dataset = dataset_cls(root=root, split='val', **kwargs)
        test_dataset = dataset_cls(root=root, split='test', **kwargs)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        raise ValueError


def getattr_d(dataset_or_loader, name):
    if isinstance(dataset_or_loader, DataLoader):
        return getattr(dataset_or_loader.dataset, name)
    else:
        return getattr(dataset_or_loader, name)


def _test_data(dataset_class: str, dataset_name: str or None, root: str, *args, **kwargs):

    def print_d(dataset_or_loader, prefix: str):
        if dataset_or_loader is None:
            return
        elif isinstance(dataset_or_loader, DataLoader):
            dataset = dataset_or_loader.dataset
        else:
            dataset = dataset_or_loader
        cprint("{} {} of {} (path={})".format(prefix, dataset_name, dataset_class, root), "yellow")
        print("\t- is_directed: {}".format(dataset[0].is_directed()))
        print("\t- num_classes: {}".format(dataset.num_classes))
        print("\t- num_graph: {}".format(len(dataset)))
        if dataset.data.x is not None:
            print("\t- num_node_features: {}".format(dataset.num_node_features))
            print("\t- num_nodes: {}".format(dataset.data.x.size(0)))
        print("\t- num_batches: {}".format(len(dataset_or_loader)))
        for batch in dataset_or_loader:
            print("\t- 1st batch: {}".format(batch))
            break
        if "train_mask" in dataset[0].__dict__:  # Cora, Citeseer, Pubmed
            print("\t- #train: {} / #val: {} / #test: {}".format(
                int(dataset[0].train_mask.sum()),
                int(dataset[0].val_mask.sum()),
                int(dataset[0].test_mask.sum()),
            ))

    try:
        train_d, val_d, test_d = get_dataset_or_loader(dataset_class, dataset_name, root, *args, **kwargs)
        print_d(train_d, "[Train]")
        print_d(val_d, "[Val]")
        print_d(test_d, "[Test]")
    except NotImplementedError:
        cprint("NotImplementedError for {}, {}, {}".format(dataset_class, dataset_name, root), "red")


if __name__ == '__main__':

    # Link Prediction
    _test_data("LinkPlanetoid", "Cora", "~/graph-data")
    _test_data("LinkPlanetoid", "CiteSeer", "~/graph-data")
    _test_data("LinkPlanetoid", "PubMed", "~/graph-data")

    # Node Classification
    _test_data("Planetoid", "Cora", '~/graph-data')
    _test_data("Planetoid", "CiteSeer", '~/graph-data')
    _test_data("Planetoid", "PubMed", '~/graph-data')
    _test_data("PPI", "PPI", '~/graph-data')

    # Graph Classification
    _test_data("TUDataset", "MUTAG", '~/graph-data')
    _test_data("TUDataset", "PTC_MR", "~/graph-data")
    _test_data("TUDataset", "NCI1", "~/graph-data")
    _test_data("TUDataset", "PROTEINS", "~/graph-data")
    _test_data("TUDataset", "DD", "~/graph-data")
    _test_data("TUDataset", "COLLAB", "~/graph-data")
    _test_data("TUDataset", "IMDB-BINARY", "~/graph-data")
    _test_data("TUDataset", "IMDB-MULTI", "~/graph-data")
