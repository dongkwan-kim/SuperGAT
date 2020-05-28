# Copyright 2019 Sami Abu-El-Haija. All Rights Reserved.
# Original code & data: https://github.com/samihaija/mixhop/blob/master/data/synthetic

import pickle
import random
import os

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx


class RandomPartitionGraph(InMemoryDataset):

    def __init__(self, root, name, num_train_per_class=20, num_val_per_class=50, num_test_per_class=100,
                 transform=None, pre_transform=None):
        # rpg-{n_classes}-{nodes_per_class}-{p_in_ratio}-{avg_degree_ratio}
        #   e.g., rpg-10-500-0.9-0.025
        self.name = name
        parsed = self.name.split("-")[1:]
        self.n_classes, self.nodes_per_class = int(parsed[0]), int(parsed[1])
        self.avg_degree_ratio = float(parsed[3])
        self.p_in = float(parsed[2]) * self.avg_degree_ratio
        self.p_out = self._get_p_out()
        self.num_train_per_class = num_train_per_class
        self.num_val_per_class = num_val_per_class
        self.num_test_per_class = num_test_per_class

        super(RandomPartitionGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data.num_nodes = self.data.x.size(0)

    def get(self, idx):
        self.data.__num_nodes__ = [self.data.x.size(0)]
        got = super().get(idx)
        return got

    def _get_p_out(self):
        assert self.avg_degree_ratio - self.p_in > 0
        assert self.n_classes >= 2
        # N * p_in + (C - 1) * N * p_out = avg_degree_ratio * N
        p_out = (self.avg_degree_ratio - self.p_in) / (self.n_classes - 1)
        return p_out

    @property
    def degree(self):
        return self.avg_degree_ratio * self.nodes_per_class

    @property
    def raw_dir(self):
        return os.path.join(self.root, "RandomPartitionGraph", self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, "RandomPartitionGraph", self.name, 'processed')

    @property
    def raw_file_names(self):
        return ["{}.graph".format(self.name),
                "{}.allx.npy".format(self.name),
                "{}.ally.npy".format(self.name)]

    @property
    def processed_file_names(self):
        return ['rpg-data-{}.pt'.format(self.name)]

    def download(self):
        if os.path.isfile(os.path.join(self.root, self.raw_file_names[2])):
            return

        sizes = [self.nodes_per_class for _ in range(self.n_classes)]
        G = nx.random_partition_graph(sizes, self.p_in, self.p_out, directed=False)
        y = [G._node[n]["block"] for n in range(len(G.nodes))]
        y_dict = {n: G._node[n]["block"] for n in range(len(G.nodes))}
        nx.set_node_attributes(G, y_dict, "y")
        for n in range(len(G.nodes)):
            del G._node[n]["block"]

        adj_dict = {u: list(v_dict.keys()) for u, v_dict in G.adj.items()}
        pickle.dump(adj_dict, open(self.raw_paths[0], "wb"))

        y_one_hot = np.eye(self.n_classes)[y]
        np.save(self.raw_paths[2], y_one_hot)

        make_x(path=self.raw_dir, name=self.name, y_one_hot=y_one_hot, save=True)

    def process(self):
        graph_path, x_path, y_path = self.raw_paths
        graph = _unpickle(graph_path)  # node to neighbors: Dict[int, List[int]]
        x = np.load(x_path)  # ndarray the shape of which is [N, 2]
        x = (x - x.mean()) / x.std()
        y_one_hot = np.load(y_path)  # ndarray the shape of which is [N, C]
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
            return torch.zeros(sz, dtype=torch.bool)

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
            val += indices_c[self.num_train_per_class:self.num_train_per_class + self.num_val_per_class]
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


def make_x(path, name, y_one_hot=None, save=False):
    if y_one_hot is None:  # one-hot ndarray the shape of which is (N, C)
        y_path = os.path.join(path, "{}.ally.npy".format(name))  # e.g., ind.n5000-h{}-c10
        y_one_hot = np.load(y_path)

    num_classes = y_one_hot.shape[1]
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
    allx = np.zeros(shape=[len(y_one_hot), 2], dtype='float32')
    for cls, theta in enumerate(np.arange(0, np.pi * 2, np.pi * 2 / num_classes)):
        gaussian_y = radius * np.cos(theta)
        gaussian_x = radius * np.sin(theta)
        num_points = np.sum(y_one_hot.argmax(axis=1) == cls)
        coord_x, coord_y = np.random.multivariate_normal(
            [gaussian_x, gaussian_y], cov, num_points).T
        cov = rotation_mat.T.dot(cov.dot(rotation_mat))

        # Belonging to class cls
        example_indices = np.nonzero(y_one_hot[:, cls] == 1)[0]
        random.shuffle(example_indices)
        allx[example_indices, 0] = coord_x
        allx[example_indices, 1] = coord_y

    if save:
        x_path = os.path.join(path, "{}.allx.npy".format(name))
        np.save(x_path, allx)

    return allx


if __name__ == '__main__':

    MODE = "RPG"
    for p_in_ratio in [0.9, 0.7, 0.5, 0.3, 0.1]:
        rpg = RandomPartitionGraph(root="~/graph-data",
                                   name="rpg-10-500-{}-0.005".format(p_in_ratio))
        print(rpg.degree)
        for b in rpg:
            print(b)
        print("---")
