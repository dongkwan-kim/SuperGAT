from typing import Tuple, Callable

import torch
import torch_geometric
from termcolor import cprint
from torch_geometric.datasets import *
from torch_geometric.data import DataLoader, InMemoryDataset
import torch_geometric.transforms as T


import os
from pprint import pprint


def get_dataset_class(dataset_class: str) -> Callable[..., InMemoryDataset]:
    assert dataset_class in torch_geometric.datasets.__all__
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
    elif dataset_class in ["Planetoid"]:  # Node (One graph with given mask)
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

    train_d, val_d, test_d = get_dataset_or_loader(dataset_class, dataset_name, root, *args, **kwargs)
    print_d(train_d, "[Train]")
    print_d(val_d, "[Val]")
    print_d(test_d, "[Test]")


if __name__ == '__main__':

    # Node Classification
    _test_data("Planetoid", "CiteSeer", '~/graph-data')
    _test_data("Planetoid", "Cora", '~/graph-data')
    _test_data("Planetoid", "PubMed", '~/graph-data')
    _test_data("PPI", None, '~/graph-data')

    # Graph Classification
    _test_data("TUDataset", "MUTAG", '~/graph-data')
    _test_data("TUDataset", "PTC_MR", "~/graph-data")
    _test_data("TUDataset", "NCI1", "~/graph-data")
    _test_data("TUDataset", "PROTEINS", "~/graph-data")
    _test_data("TUDataset", "DD", "~/graph-data")
    _test_data("TUDataset", "COLLAB", "~/graph-data")
    _test_data("TUDataset", "IMDB-BINARY", "~/graph-data")
    _test_data("TUDataset", "IMDB-MULTI", "~/graph-data")
