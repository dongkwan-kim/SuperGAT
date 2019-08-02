from typing import Tuple

import torch_geometric
from torch_geometric.datasets import *
from torch_geometric.data import DataLoader

import os
from pprint import pprint


def get_dataset_class(dataset_class: str):
    assert dataset_class in torch_geometric.datasets.__all__
    return eval(dataset_class)


def get_dataloader(dataset_class: str, dataset_name: str, root: str,
                   train_val_test: Tuple[float, float, float] = (0.9 * 0.9, 0.9 * 0.1, 0.1),
                   seed: int = 42, **kwargs):
    """
    Note that datasets structure in torch_geometric varies.
    :param dataset_class: ['KarateClub', 'TUDataset', 'Planetoid', 'CoraFull', 'Coauthor', 'Amazon', 'PPI', 'Reddit',
                           'QM7b', 'QM9', 'Entities', 'GEDDataset', 'MNISTSuperpixels', 'FAUST', 'DynamicFAUST',
                           'ShapeNet', 'ModelNet', 'CoMA', 'SHREC2016', 'TOSCA', 'PCPNetDataset', 'S3DIS',
                           'GeometricShapes', 'BitcoinOTC', 'ICEWS18', 'GDELT']
    :param dataset_name: https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
    :param root:
    :param train_val_test:
    :param seed:
    :param kwargs:
    :return:
    """

    root = os.path.join(root, dataset_name)
    dataset_cls = get_dataset_class(dataset_class)
    dataset = dataset_cls(root=root, name=dataset_name, **kwargs)

    raise NotImplementedError


if __name__ == '__main__':
    dlr = get_dataloader("TUDataset", "ENZYMES", "../data")
