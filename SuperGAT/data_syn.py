# Copyright 2019 Sami Abu-El-Haija. All Rights Reserved.
# Original code & data: https://github.com/samihaija/mixhop/blob/master/data/synthetic

import pickle
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx, subgraph


class HomophilySynthetic(InMemoryDataset):

    def __init__(self, root, name, num_train_per_class=20, num_val_per_class=50, num_test_per_class=100,
                 transform=None, pre_transform=None):
        self.name = name  # hs-{}
        self.homophily = float(name.split("-")[1])
        self.num_train_per_class = num_train_per_class
        self.num_val_per_class = num_val_per_class
        self.num_test_per_class = num_test_per_class
        super(HomophilySynthetic, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["ind.n5000-h{}-c10.graph".format(self.homophily),
                "ind.n5000-h{}-c10.allx".format(self.homophily),
                "ind.n5000-h{}-c10.ally".format(self.homophily)]

    @property
    def processed_file_names(self):
        return ['data-h{}.pt'.format(self.homophily)]

    def download(self):
        print("Please download manually from: https://github.com/samihaija/mixhop/blob/master/data/synthetic")  # todo
        pass

    def process(self):
        graph_path, x_path, y_path = [os.path.join(self.root, rf) for rf in self.raw_file_names]
        graph = _unpickle(graph_path)  # node to neighbors: Dict[int, List[int]]
        x = np.load(x_path)  # ndarray the shape of which is [5000, 2]
        y_one_hot = np.load(y_path)  # ndarray the shape of which is [5000, 10]
        y = np.argmax(y_one_hot, axis=1)

        nx_g = nx.Graph()
        for u, neighbors in graph.items():
            nx_g.add_edges_from([(u, v) for v in neighbors])
        for x_id, (x_features, y_features) in enumerate(zip(x, y)):
            nx_g.add_node(x_id, x=x_features, y=y_features)

        data = from_networkx(nx_g)
        mask_dict = self._get_split_mask(data.y)
        for k, v in mask_dict.items():
            setattr(data, k, v)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def _get_split_mask(self, y):

        def _mask0(sz):
            return torch.zeros(sz, dtype=torch.uint8)

        def _fill_mask(index, mask0):
            index = torch.Tensor(index).long()
            mask0[index] = 1
            return mask0

        classes = torch.unique(y)
        train, val, test = [], [], []
        train_mask, val_mask, test_mask = _mask0(len(y)), _mask0(len(y)), _mask0(len(y))

        num_total = self.num_train_per_class + self.num_val_per_class + self.num_test_per_class
        for c in classes:
            indices_c = [i for _, i in zip(range(num_total), torch.utils.data.SubsetRandomSampler((y == c).nonzero()))]
            train += indices_c[:self.num_train_per_class]
            val += indices_c[self.num_train_per_class:self.num_train_per_class+self.num_val_per_class]
            test += indices_c[-self.num_test_per_class:]

        return {
            "train_mask": _fill_mask(train, train_mask),
            "val_mask": _fill_mask(val, val_mask),
            "test_mask": _fill_mask(test, test_mask),
        }


def _unpickle(_path):
    u = pickle._Unpickler(open(_path, "rb"))
    u.encoding = 'latin1'
    return u.load()


def make_x(path, homophily=0.5, save=False):
    x_path = os.path.join(path, "synthetic", "ind.n5000-h{}-c10.allx".format(homophily))
    y_path = os.path.join(path, "synthetic", "ind.n5000-h{}-c10.ally".format(homophily))

    y = np.load(y_path)

    num_classes = y.shape[1]
    variance_factor = 350
    start_cov = np.array(
        [[70.0 * variance_factor, 0.0],
         [0.0, 20.0 * variance_factor]])

    cov = start_cov
    theta = np.pi * 2 / num_classes
    rotation_mat = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    radius = 300
    allx = np.zeros(shape=[len(y), 2], dtype='float32')
    plt.figure(figsize=(40, 40))
    for cls, theta in enumerate(np.arange(0, np.pi * 2, np.pi * 2 / num_classes)):
        gaussian_y = radius * np.cos(theta)
        gaussian_x = radius * np.sin(theta)
        num_points = np.sum(y.argmax(axis=1) == cls)
        coord_x, coord_y = np.random.multivariate_normal(
            [gaussian_x, gaussian_y], cov, num_points).T
        cov = rotation_mat.T.dot(cov.dot(rotation_mat))

        # Belonging to class cls
        example_indices = np.nonzero(y[:, cls] == 1)[0]
        random.shuffle(example_indices)
        allx[example_indices, 0] = coord_x
        allx[example_indices, 1] = coord_y

    if save:
        np.save(open(x_path, 'w'), allx)

    return allx


if __name__ == '__main__':

    for hi in range(4, 10):
        h = round(float(0.1 * hi), 1)
        hs = HomophilySynthetic(root="~/graph-data/synthetic", name="h-{}".format(h))
