import inspect
import random

import torch
import torch_geometric as pyg
from copy import deepcopy

from sklearn.decomposition import PCA
from termcolor import cprint
from torch_geometric.datasets import *
from torch_geometric.data import DataLoader, InMemoryDataset, Data, NeighborSampler
from torch_geometric.transforms import Compose
from torch_geometric.utils import is_undirected, to_undirected, degree, sort_edge_index, remove_self_loops, \
    add_self_loops, negative_sampling, train_test_split_edges
from torch_geometric.io import read_npz
import numpy as np

import os
from pprint import pprint
from typing import Tuple, Callable, List

from tqdm import tqdm

import ogb
from ogb.nodeproppred import PygNodePropPredDataset

from data_saint import MyGraphSAINTRandomWalkSampler, DisjointGraphSAINTRandomWalkSampler
from data_sampler import MyNeighborSampler
from data_syn import RandomPartitionGraph
from data_transform import DigitizeY, ToUndirected
from data_utils import mask_init, mask_getitem, collate_and_pca, get_loader_and_dataset_kwargs
from data_webkb4univ import WebKB4Univ
from data_flickr import Flickr
from data_wikics import WikiCS
from data_snap import SNAPDataset, Crocodile, Squirrel, Chameleon
from data_reddit import MyReddit
from utils import negative_sampling_numpy

from multiprocessing import Process, Queue
import os


class ENSPlanetoid(Planetoid):
    """Efficient Negative Sampling for Planetoid"""

    def __init__(self, root, name, neg_sample_ratio, q_trial=110):
        super().__init__(root, name)

        self.neg_sample_ratio = neg_sample_ratio

        x, edge_index = self.data.x, self.data.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self.edge_index_with_self_loops_numpy = edge_index_with_self_loops.numpy()
        self.num_pos_samples = self.edge_index_with_self_loops_numpy.shape[1]

        self.q = Queue()
        self.q_trial = q_trial
        self.put_train_edge_index_to_queue(5, self.q)

        self.need_neg_samples = True

    def train(self):
        self.need_neg_samples = True

    def eval(self):
        self.need_neg_samples = False

    def get_train_edge_index_numpy(self):
        # Sample negative training samples from the negative sample pool
        # We need numpy version, because it is stuck if child process touch torch.Tensor.
        neg_edge_index = negative_sampling_numpy(
            edge_index_numpy=self.edge_index_with_self_loops_numpy,
            num_nodes=self.data.x.size(0),
            num_neg_samples=int(self.neg_sample_ratio * self.num_pos_samples),
        )
        train_edge_index_numpy = np.concatenate([self.edge_index_with_self_loops_numpy, neg_edge_index], axis=1)
        return train_edge_index_numpy

    def put_train_edge_index_to_queue(self, n, q):
        for _ in range(n):
            q.put(self.get_train_edge_index_numpy())

    def get_train_edge_index_from_queue(self):
        train_edge_index_from_queue = torch.Tensor(self.q.get()).long()
        if self.q.qsize() <= 5:
            p = Process(target=self.put_train_edge_index_to_queue, args=(self.q_trial, self.q))
            p.daemon = True
            p.start()
        return train_edge_index_from_queue

    def __getitem__(self, item) -> torch.Tensor:
        """
        :param item:
        :return: Draw negative samples, and return [2, E] tensor
        """
        datum = super().__getitem__(item)
        # Add attributes for edge prediction
        if self.need_neg_samples:
            train_edge_index = self.get_train_edge_index_from_queue()
            datum.__setitem__("train_edge_index", train_edge_index)
        return datum


def get_agreement_dist(edge_index: torch.Tensor, y: torch.Tensor,
                       with_self_loops=True, return_agree_dist_sum=False,
                       epsilon=1e-11) -> List[torch.Tensor] or (List, List):
    """
    :param edge_index: tensor the shape of which is [2, E]
    :param y: tensor the shape of which is [N]
    :param with_self_loops: add_self_loops if True
    :param return_agree_dist_sum: whether return the sum of agreement dist
    :param epsilon: small float number for stability.
    :return: Tensor list L the length of which is N.
        L[i] = tensor([..., a(y_j, y_i), ...]) for e_{ji} \in {E}
            - a(y_j, y_i) = 1 / L[i].sum() if y_j = y_i,
            - a(y_j, y_i) = 0 otherwise.
    """
    y = y.squeeze()
    num_nodes = y.size(0)

    # Add self-loops and sort by index
    if with_self_loops:
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # [2, E + N]
    edge_index, _ = sort_edge_index(edge_index, num_nodes=num_nodes)

    agree_dist_list = []
    agree_dist_sum_list = []
    for node_idx, label in enumerate(tqdm(y)):
        neighbors, _ = edge_index[:, edge_index[1] == node_idx]
        y_neighbors = y[neighbors]
        if len(label.size()) == 0:
            agree_dist = (y_neighbors == label).float()
        else:  # multi-label case
            agree_dist = (y_neighbors * label).float().sum(dim=1)

        if return_agree_dist_sum:
            agree_dist_sum_list.append(agree_dist.sum().item())

        if int(agree_dist.sum()) != 0:
            agree_dist[agree_dist == 0] = epsilon  # For KLD
            agree_dist = agree_dist / agree_dist.sum()
        else:
            agree_dist[:] = 1.0
            agree_dist = agree_dist / agree_dist.sum()

        agree_dist_list.append(agree_dist)

    if not return_agree_dist_sum:
        return agree_dist_list  # [N, #neighbors]
    else:
        return agree_dist_list, agree_dist_sum_list  # [N, #neighbors], [N]


def get_uniform_dist_like(dist_list: List[torch.Tensor]) -> List[torch.Tensor]:
    uniform_dist_list = []
    for dist in dist_list:
        ones = torch.ones_like(dist)  # [#neighbors]
        ones = ones / ones.size(0)
        uniform_dist_list.append(ones)
    return uniform_dist_list  # [N, #neighbors]


class FullPlanetoid(Planetoid):

    def __init__(self, root, name):
        super().__init__(root, name)
        self.data.train_mask[:] = True
        self.data.train_mask[self.data.test_mask] = False
        self.data.train_mask[self.data.test_mask] = False


class MyCitationFull(CitationFull):

    def __init__(self, root, name, transform=None, pre_transform=None, seed=0):
        self.pca_dim = 500
        self.name = name
        super().__init__(root, name, transform, pre_transform)
        mask_init(self, seed=12345 + seed)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        return mask_getitem(self, datum)

    def download(self):
        return super().download()

    def process(self):
        if self.name.lower() == "cora":
            data = read_npz(self.raw_paths[0])
            data = data if self.pre_transform is None else self.pre_transform(data)
            data, slices = collate_and_pca(self, [data], pca_dim=self.pca_dim)
            torch.save((data, slices), self.processed_paths[0])
        else:
            return super().process()


class MyCoauthor(Coauthor):

    def __init__(self, root, name, transform=None, pre_transform=None, seed=0):
        self.pca_dim = 500
        super().__init__(root, name, transform, pre_transform)
        mask_init(self, seed=12345 + seed)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        return mask_getitem(self, datum)

    def download(self):
        return super().download()

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_{}'.format(self.pca_dim))

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = collate_and_pca(self, [data], pca_dim=self.pca_dim)
        torch.save((data, slices), self.processed_paths[0])


class MyAmazon(Amazon):

    def __init__(self, root, name, transform=None, pre_transform=None, seed=0):
        super().__init__(root, name, transform, pre_transform)
        mask_init(self, seed=12345 + seed)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        return mask_getitem(self, datum)

    def download(self):
        return super().download()

    def process(self):
        return super().process()


class ADPlanetoid(Planetoid):

    def __init__(self, root, name):
        super().__init__(root, name)
        y, edge_index = self.data.y, self.data.edge_index
        self.agreement_dist = get_agreement_dist(edge_index, y)
        self.uniform_att_dist = get_uniform_dist_like(self.agreement_dist)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        datum.__setitem__("agreement_dist", self.agreement_dist)
        datum.__setitem__("uniform_att_dist", self.uniform_att_dist)
        return datum


class ADPPI(PPI):

    def __init__(self, root, split):
        super().__init__(root, split)
        self.agreement_dist_list = []
        self.uniform_att_dist_list = []

        self.initialized = False
        for data in self:
            edge_index, y = data.edge_index, data.y
            agreement_dist = get_agreement_dist(edge_index, y)
            uniform_att_dist = get_uniform_dist_like(agreement_dist)
            self.agreement_dist_list.append(agreement_dist)
            self.uniform_att_dist_list.append(uniform_att_dist)
        self.initialized = True

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        if self.initialized:
            datum.__setitem__("agreement_dist", self.agreement_dist_list[item])
            datum.__setitem__("uniform_att_dist", self.uniform_att_dist_list[item])
        return datum


class LinkPPI(PPI):

    def __init__(self, root, split, train_val_test_ratio=None, seed=42):
        super().__init__(root, split)
        self.split = split
        self.train_val_test_ratio = train_val_test_ratio or (1.0 - 0.15, 0.05, 0.1)
        self.seed = seed

        if split == 'train':
            data_list = []
            for g in tqdm(self):
                train_e, _ = self.train_test_split(g.x, g.edge_index, 1 - self.train_val_test_ratio[0])
                g.edge_index = train_e
                data_list.append(g)
            self.data, self.slices = self.collate(data_list)

        elif split == 'val':
            val_ratio = self.train_val_test_ratio[1]
            data_list = []
            val_edge_index_list = []
            for g in self:
                train_e, val_e = self.train_test_split(g.x, g.edge_index, val_ratio)
                g.edge_index = train_e
                data_list.append(g)
                val_edge_index_list.append(val_e)
            self.val_edge_index_list = val_edge_index_list
            self.val_edge_y_list = [self.get_edge_y(vei.size(1)) for vei in val_edge_index_list]

        elif split == 'test':
            test_ratio = self.train_val_test_ratio[2]
            data_list = []
            test_edge_index_list = []
            for g in self:
                train_e, test_e = self.train_test_split(g.x, g.edge_index, test_ratio)
                g.edge_index = train_e
                data_list.append(g)
                test_edge_index_list.append(test_e)
            self.test_edge_index_list = test_edge_index_list
            self.test_edge_y_list = [self.get_edge_y(vei.size(1)) for vei in test_edge_index_list]

    def train_test_split(self, x, edge_index, ratio):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        data = deepcopy(self.data)
        data.edge_index = edge_index
        data.x = x

        data = train_test_split_edges(data, 0, ratio)
        data.__delattr__("train_neg_adj_mask")

        test_edge_index = torch.cat([to_undirected(data.test_pos_edge_index),
                                     to_undirected(data.test_neg_edge_index)], dim=1)

        return data.train_pos_edge_index, test_edge_index

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
        try:
            if self.split == "val":
                datum.__setitem__("val_edge_index", self.val_edge_index_list[item])
                datum.__setitem__("val_edge_y", self.val_edge_y_list[item])
            elif self.split == "test":
                datum.__setitem__("test_edge_index", self.test_edge_index_list[item])
                datum.__setitem__("test_edge_y", self.test_edge_y_list[item])
        except AttributeError as e:
            print(e)
        return datum


class LinkPlanetoid(Planetoid):

    def __init__(self, root, name, train_val_test_ratio=None, seed=42):
        super().__init__(root, name)
        self.train_val_test_ratio = train_val_test_ratio or (1.0 - 0.15, 0.05, 0.1)
        self.seed = seed

        tpei, vei, tei = self.train_val_test_split()
        self.train_pos_edge_index = tpei  # undirected [2, E * 0.85]
        self.val_edge_index = vei  # undirected [2, E * 0.05 * 2]
        self.test_edge_index = tei  # undirected [2, E * 0.1 * 2]

        self.val_edge_y = self.get_edge_y(self.val_edge_index.size(1))
        self.test_edge_y = self.get_edge_y(self.test_edge_index.size(1))

        # Remove validation/test pos edges from self.edge_index
        self.data.edge_index = self.train_pos_edge_index

    def train_val_test_split(self):
        x, edge_index = self.data.x, self.data.edge_index

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        data = deepcopy(self.data)
        data.edge_index = edge_index

        # train_pos_edge_index=[2, E * 0.85] (undirected)
        # val_neg/pos_edge_index=[2, E/2 * 0.05] (not undirected)
        # test_neg/pos_edge_index: [2, E/2 * 0.1] (not undirected)
        data = train_test_split_edges(data, *self.train_val_test_ratio[1:])
        data.__delattr__("train_neg_adj_mask")

        test_edge_index = torch.cat([to_undirected(data.test_pos_edge_index),
                                     to_undirected(data.test_neg_edge_index)], dim=1)

        if data.val_pos_edge_index.size(1) > 0:
            val_edge_index = torch.cat([to_undirected(data.val_pos_edge_index),
                                        to_undirected(data.val_neg_edge_index)], dim=1)
        else:
            val_edge_index = test_edge_index

        return data.train_pos_edge_index, val_edge_index, test_edge_index

    def _sample_train_neg_edge_index(self, is_undirected_edges=True):
        num_pos_samples = self.train_pos_edge_index.size(1)
        num_neg_samples = num_pos_samples // 2 if is_undirected_edges else num_pos_samples
        neg_edge_index = negative_sampling(
            edge_index=self.train_pos_edge_index,
            num_nodes=self.data.x.size(0),
            num_neg_samples=num_neg_samples,
        )
        return to_undirected(neg_edge_index) if is_undirected_edges else neg_edge_index

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
        datum.__setitem__("train_edge_y", train_edge_y)
        datum.__setitem__("val_edge_index", self.val_edge_index)
        datum.__setitem__("val_edge_y", self.val_edge_y)
        datum.__setitem__("test_edge_index", self.test_edge_index)
        datum.__setitem__("test_edge_y", self.test_edge_y)
        return datum


class ADRandomPartitionGraph(RandomPartitionGraph):

    def __init__(self, root, name):
        super().__init__(root, name)
        y, edge_index = self.data.y, self.data.edge_index
        self.agreement_dist = get_agreement_dist(edge_index, y)
        self.uniform_att_dist = get_uniform_dist_like(self.agreement_dist)

    def __getitem__(self, item) -> torch.Tensor:
        datum = super().__getitem__(item)
        datum.__setitem__("agreement_dist", self.agreement_dist)
        datum.__setitem__("uniform_att_dist", self.uniform_att_dist)
        return datum


class LinkRandomPartitionGraph(RandomPartitionGraph):

    def __init__(self, root, name, train_val_test_ratio=None, seed=42):
        super().__init__(root, name)
        self.train_val_test_ratio = train_val_test_ratio or (1.0 - 0.15, 0.05, 0.1)
        self.seed = seed

        tpei, vei, tei = self.train_val_test_split()
        self.train_pos_edge_index = tpei  # undirected [2, E * 0.85]
        self.val_edge_index = vei  # undirected [2, E * 0.05 * 2]
        self.test_edge_index = tei  # undirected [2, E * 0.1 * 2]

        self.val_edge_y = self.get_edge_y(self.val_edge_index.size(1))
        self.test_edge_y = self.get_edge_y(self.test_edge_index.size(1))

        # Remove validation/test pos edges from self.edge_index
        self.data.edge_index = self.train_pos_edge_index

    def train_val_test_split(self):
        x, edge_index = self.data.x, self.data.edge_index

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        data = deepcopy(self.data)
        data.edge_index = edge_index

        # train_pos_edge_index=[2, E * 0.85] (undirected)
        # val_neg/pos_edge_index=[2, E/2 * 0.05] (not undirected)
        # test_neg/pos_edge_index: [2, E/2 * 0.1] (not undirected)
        data = train_test_split_edges(data, *self.train_val_test_ratio[1:])
        data.__delattr__("train_neg_adj_mask")

        test_edge_index = torch.cat([to_undirected(data.test_pos_edge_index),
                                     to_undirected(data.test_neg_edge_index)], dim=1)

        if data.val_pos_edge_index.size(1) > 0:
            val_edge_index = torch.cat([to_undirected(data.val_pos_edge_index),
                                        to_undirected(data.val_neg_edge_index)], dim=1)
        else:
            val_edge_index = test_edge_index

        return data.train_pos_edge_index, val_edge_index, test_edge_index

    def _sample_train_neg_edge_index(self, is_undirected_edges=True):
        num_pos_samples = self.train_pos_edge_index.size(1)
        num_neg_samples = num_pos_samples // 2 if is_undirected_edges else num_pos_samples
        neg_edge_index = negative_sampling(
            edge_index=self.train_pos_edge_index,
            num_nodes=self.data.x.size(0),
            num_neg_samples=num_neg_samples,
        )
        return to_undirected(neg_edge_index) if is_undirected_edges else neg_edge_index

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
        datum.__setitem__("train_edge_y", train_edge_y)
        datum.__setitem__("val_edge_index", self.val_edge_index)
        datum.__setitem__("val_edge_y", self.val_edge_y)
        datum.__setitem__("test_edge_index", self.test_edge_index)
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
    assert dataset_class in (pyg.datasets.__all__ +
                             [
                                 "ENSPlanetoid", "LinkPlanetoid", "ADPlanetoid", "FullPlanetoid", "HomophilySynthetic",
                                 "LinkPPI", "ADPPI",
                                 "RandomPartitionGraph", "LinkRandomPartitionGraph", "ADRandomPartitionGraph",
                                 "WebKB4Univ", "PygNodePropPredDataset", "WikiCS", "GNNBenchmarkDataset", "Flickr",
                                 "MyAmazon", "MyCoauthor", "MyCitationFull",
                                 "Crocodile", "Squirrel", "Chameleon",
                                 "MyReddit",
                             ]), f"{dataset_class} is not good."
    return eval(dataset_class)


def get_dataset_or_loader(dataset_class: str, dataset_name: str or None, root: str,
                          batch_size: int = 1024,
                          train_val_test: Tuple[float, float, float] = (0.9 * 0.9, 0.9 * 0.1, 0.1),
                          seed: int = 42, num_splits: int = 1, sampler="NeighborSampler",
                          **kwargs):
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
    :param num_splits: 1
    :param sampler: currently only supports MyNeighborSampler and NeighborSampler
    :param kwargs:
    :return:
    """

    cprint("Now loading dataset: {} / {}".format(dataset_class, dataset_name), "green")
    root = os.path.expanduser(root)

    if dataset_class not in ["PPI", "WebKB4Univ", "LinkPPI", "ADPPI", "Reddit", "WikiCS", "MyReddit"]:
        kwargs["name"] = dataset_name

    torch.manual_seed(seed)
    if dataset_class in ["HomophilySynthetic"]:
        root = os.path.join(root, "synthetic")
    dataset_cls = get_dataset_class(dataset_class)

    if dataset_class in ["TUDataset"]:  # Graph
        dataset = dataset_cls(root=root, **kwargs).shuffle()
        n_samples = len(dataset)
        train_samples = int(n_samples * train_val_test[0])
        val_samples = int(n_samples * train_val_test[1])
        train_dataset = dataset[:train_samples]
        val_dataset = dataset[train_samples:train_samples + val_samples]
        test_dataset = dataset[train_samples + val_samples:]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    # Node or Link (One graph with given mask)
    elif dataset_class.endswith("Planetoid"):  # "Planetoid", "LinkPlanetoid", "ADPlanetoid", "ENSPlanetoid", ...
        dataset = dataset_cls(root=root, **kwargs)
        return dataset, None, None

    elif dataset_class in ["HomophilySynthetic",
                           "RandomPartitionGraph", "LinkRandomPartitionGraph", "ADRandomPartitionGraph"]:
        dataset = dataset_cls(root=root, **kwargs)
        return dataset, None, None

    elif dataset_class in ["PPI", "LinkPPI", "ADPPI"]:  # Node (Multiple graphs)
        val_dataset = dataset_cls(root=root, split='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        train_dataset = dataset_cls(root=root, split='train', **kwargs)
        test_dataset = dataset_cls(root=root, split='test', **kwargs)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return train_loader, val_loader, test_loader

    elif dataset_class in ["WebKB4Univ"]:
        dataset = dataset_cls(root=root, **kwargs)
        train_dataset = dataset[2:]
        val_dataset = dataset[0:1]
        test_dataset = dataset[1:2]
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader, test_loader

    elif dataset_class in ["Reddit"]:
        root = os.path.join(root, "reddit")

        if "NeighborSampler" in sampler:  # MyNeighborSampler, NeighborSampler
            use_negative_sampling = (sampler == "MyNeighborSampler")
            loader_kwargs, dataset_kwargs = get_loader_and_dataset_kwargs(cls=MyNeighborSampler, **kwargs)
            dataset = dataset_cls(root=root, **dataset_kwargs)
            train_loader = MyNeighborSampler(
                data=dataset[0], batch_size=batch_size, bipartite=False, shuffle=True,
                use_negative_sampling=use_negative_sampling, **loader_kwargs,
            )
            eval_loader = MyNeighborSampler(
                data=dataset[0], batch_size=batch_size, bipartite=False, shuffle=False,
                use_negative_sampling=False, **loader_kwargs,
            )

        else:
            raise ValueError

        return dataset, train_loader, eval_loader

    elif dataset_class in ["MyReddit"]:
        root = os.path.join(root, "reddit")
        if "NeighborSampler" in sampler:  # MyNeighborSampler, NeighborSampler
            loader_kwargs, dataset_kwargs = get_loader_and_dataset_kwargs(cls=MyNeighborSampler, **kwargs)
            dataset = dataset_cls(root=root, batch_size=batch_size, **kwargs)
            eval_loader = MyNeighborSampler(
                data=dataset.data_xy, batch_size=batch_size, bipartite=False, shuffle=False,
                use_negative_sampling=False, **loader_kwargs,
            )

        elif "SAINT" in sampler:
            dataset = dataset_cls(
                root=root, batch_size=batch_size,
                size=kwargs["size"],
                neg_sample_ratio=kwargs["neg_sample_ratio"],
            )
            eval_loader = NeighborSampler(
                data=dataset.data_xy, batch_size=batch_size, bipartite=False, shuffle=False,
                num_hops=len(kwargs["size"]), size=kwargs["size"],
            )

        else:
            raise NotImplementedError

        return dataset, dataset, eval_loader

    elif dataset_class in ["WikiCS"]:
        root = os.path.join(root, "WikiCS")
        _split = kwargs["split"]
        dataset = dataset_cls(root=root)
        dataset.data.stopping_mask = None
        dataset.data.train_mask = dataset.data.train_mask[:, _split]
        dataset.data.val_mask = dataset.data.val_mask[:, _split]
        return dataset, None, None

    elif dataset_class in ["MyCoauthor"]:
        _name = kwargs["name"]
        root = os.path.join(root, f"{dataset_class}{_name}")
        dataset = dataset_cls(root=root, name=_name.lower(), seed=seed % num_splits)
        return dataset, None, None

    elif dataset_class in ["MyAmazon", "MyCitationFull"]:
        _name = kwargs["name"]
        if _name.lower() == "corafull":
            _name = "Cora"
        root = os.path.join(root, f"{dataset_class}{_name.capitalize()}")
        dataset = dataset_cls(root=root, name=_name.lower(), seed=seed % num_splits)
        return dataset, None, None

    elif dataset_class in ["Crocodile", "Squirrel", "Chameleon"]:
        dataset = dataset_cls(root=root, seed=seed % num_splits)
        return dataset, None, None

    elif dataset_class in ["GNNBenchmarkDataset"]:
        train_dataset = dataset_cls(root=root, split="train", **kwargs)
        val_dataset = dataset_cls(root=root, split="val", **kwargs)
        test_dataset = dataset_cls(root=root, split="test", **kwargs)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader, test_loader

    elif dataset_class in ["Flickr"]:
        root = os.path.join(root, "flickr")
        dataset = dataset_cls(root=root)
        return dataset, None, None

    elif dataset_class == "PygNodePropPredDataset":

        if dataset_name == "ogbn-arxiv":
            dataset_kwargs, loader_kwargs = kwargs, {}
            transform_to_undirected = False if ("to_undirected" not in dataset_kwargs) else kwargs["to_undirected"]
            if "to_undirected" in dataset_kwargs:
                del dataset_kwargs["to_undirected"]
            if transform_to_undirected:
                dataset_kwargs["transform"] = ToUndirected()
        elif dataset_name == "ogbn-proteins":
            dataset_kwargs, loader_kwargs = kwargs, {}
            from data_transform import ToSparseTensor
            dataset_kwargs["pre_transform"] = ToSparseTensor()
        elif dataset_name == "ogbn-products":
            loader_kwargs, dataset_kwargs = get_loader_and_dataset_kwargs(**kwargs)
        else:
            raise ValueError

        dataset = PygNodePropPredDataset(root=root, **dataset_kwargs)
        split_idx = dataset.get_idx_split()
        dataset.train_mask = split_idx["train"]
        dataset.val_mask = split_idx["valid"]
        dataset.test_mask = split_idx["test"]

        if dataset_name == "ogbn-arxiv":
            return dataset, None, None
        elif dataset_name == "ogbn-proteins":
            raise NotImplementedError
        elif dataset_name == "ogbn-products":
            raise NotImplementedError

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
        elif "train_mask" in dataset.__dict__:
            print("\t- #train: {} / #val: {} / #test: {}".format(
                int(dataset.train_mask.size(0)),
                int(dataset.val_mask.size(0)),
                int(dataset.test_mask.size(0)),
            ))

    def print_l(loader, prefix):
        cprint("[{}] {} of {} (path={})".format(prefix, dataset_name, dataset_class, root), "yellow")
        for i, b in enumerate(loader):
            print("batch {}: ".format(i), b)
            if i == 2:
                break

    try:
        _dl = get_dataset_or_loader(dataset_class, dataset_name, root, *args, **kwargs)
        if isinstance(_dl[0], PygNodePropPredDataset) and _dl[1] is not None:
            full_dataset, train_loader, eval_loader = _dl
            print_l(train_loader(full_dataset.train_mask), "[Train Loader]")
            print_l(train_loader(full_dataset.val_mask), "[Eval Loader] ")
            print_d(full_dataset, "[Dataset]")
        elif _dl[1] is not None and isinstance(_dl[1], NeighborSampler):
            full_dataset, train_loader, eval_loader = _dl
            print_l(train_loader(full_dataset[0].train_mask), "[Train Loader]")
            print_l(eval_loader(full_dataset[0].val_mask), "[Eval Loader] ")
            print_d(full_dataset, "[Dataset]")
        elif isinstance(_dl[0], MyReddit) and _dl[0].sampler_type != "walk":
            full_dataset, train_loader, eval_loader = _dl
            print_l(train_loader, "[Train Loader]")
            print_l(eval_loader(full_dataset.val_mask), "[Eval Loader] ")
            print_d(full_dataset, "[Dataset]")
        elif isinstance(_dl[0], MyReddit) and _dl[0].sampler_type == "walk":
            full_dataset, train_loader, eval_loader = _dl
            print_l(train_loader, "[Loader]")
        else:
            train_d, val_d, test_d = _dl
            print_d(train_d, "[Train]")
            print_d(val_d, "[Val]")
            print_d(test_d, "[Test]")

    except NotImplementedError:
        cprint("NotImplementedError for {}, {}, {}".format(dataset_class, dataset_name, root), "red")


if __name__ == '__main__':
    _test_data("PygNodePropPredDataset", "ogbn-arxiv", '~/graph-data')

    _test_data("MyReddit", "MyReddit", '~/graph-data', batch_size=4096,
               sampler="GraphSAINTRandomWalkSampler",
               size=[5, 5], num_hops=2, neg_sample_ratio=0.5, num_version=2)
    _test_data("Reddit", "Reddit", '~/graph-data', batch_size=512,
               sampler="MyNeighborSampler", size=[15, 10], num_hops=2, neg_sample_ratio=0.5)
    _test_data("PygNodePropPredDataset", "ogbn-products", '~/graph-data',
               size=[10, 5], num_hops=2)
    _test_data("PygNodePropPredDataset", "ogbn-arxiv", '~/graph-data')

    _test_data("MyCitationFull", "CoraFull", '~/graph-data')
    _test_data("MyCoauthor", "CS", '~/graph-data')
    _test_data("Chameleon", "Chameleon", '~/graph-data')
    exit()

    _test_data("MyCoauthor", "Physics", '~/graph-data')
    _test_data("Squirrel", "Squirrel", '~/graph-data')
    _test_data("Crocodile", "Crocodile", '~/graph-data')
    _test_data("MyCitationFull", "Cora_ML", '~/graph-data')
    _test_data("MyAmazon", "Computers", '~/graph-data')
    _test_data("MyAmazon", "Photo", '~/graph-data')
    _test_data("Flickr", "Flickr", '~/graph-data')
    _test_data("WebKB4Univ", "WebKB4Univ", '~/graph-data')
    _test_data("WikiCS", "WikiCS", '~/graph-data', split=0)
    _test_data("MyCitationFull", "Cora", '~/graph-data')
    _test_data("MyCitationFull", "DBLP", '~/graph-data')

    _test_data("Reddit", "Reddit", '~/graph-data')
    _test_data("ADPPI", "ADPPI", '~/graph-data')

    _test_data("LinkPlanetoid", "Cora", "~/graph-data")
    _test_data("LinkPPI", "LinkPPI", '~/graph-data')

    _test_data("FullPlanetoid", "Cora", '~/graph-data')
    _test_data("FullPlanetoid", "CiteSeer", '~/graph-data')
    _test_data("FullPlanetoid", "PubMed", '~/graph-data')

    _test_data("CitationFull", "Cora", '~/graph-data')
    _test_data("CitationFull", "CiteSeer", '~/graph-data')

    # WebKB Four University
    _test_data("WebKB4Univ", "WebKB4Univ", '~/graph-data')

    # Efficient Negative Sampling
    _test_data("ENSPlanetoid", "Cora", '~/graph-data', neg_sample_ratio=0.5)
    _test_data("ENSPlanetoid", "CiteSeer", '~/graph-data', neg_sample_ratio=0.5)
    _test_data("ENSPlanetoid", "PubMed", '~/graph-data', neg_sample_ratio=0.5)

    num_nodes_per_class = 500
    for d in [2.5, 5.0, 50.0, 100.0]:
        for h in [0.1, 0.3, 0.5, 0.7, 0.9]:
            avg_degree_ratio = d / num_nodes_per_class
            rpg_name = "rpg-10-{}-{}-{}".format(num_nodes_per_class, h, avg_degree_ratio)
            # RandomPartitionGraph
            _test_data("RandomPartitionGraph", rpg_name, "~/graph-data")
            # ADRandomPartitionGraph
            _test_data("ADRandomPartitionGraph", rpg_name, "~/graph-data")
            # LinkRandomPartitionGraph
            _test_data("LinkRandomPartitionGraph", rpg_name, "~/graph-data")

    # Link Prediction
    _test_data("LinkPlanetoid", "Cora", "~/graph-data")
    _test_data("LinkPlanetoid", "CiteSeer", "~/graph-data")
    _test_data("LinkPlanetoid", "PubMed", "~/graph-data")

    # Attention Dist
    _test_data("ADPlanetoid", "Cora", '~/graph-data')

    # Node Classification
    _test_data("Planetoid", "Cora", '~/graph-data')
    _test_data("Planetoid", "CiteSeer", '~/graph-data')
    _test_data("Planetoid", "PubMed", '~/graph-data')
    _test_data("PPI", "PPI", '~/graph-data')

    # Deprecated
    _test_data("GNNBenchmarkDataset", "CLUSTER", '~/graph-data')

    # Graph Classification
    _test_data("TUDataset", "MUTAG", '~/graph-data')
    _test_data("TUDataset", "PTC_MR", "~/graph-data")
    _test_data("TUDataset", "NCI1", "~/graph-data")
    _test_data("TUDataset", "PROTEINS", "~/graph-data")
    _test_data("TUDataset", "DD", "~/graph-data")
    _test_data("TUDataset", "COLLAB", "~/graph-data")
    _test_data("TUDataset", "IMDB-BINARY", "~/graph-data")
    _test_data("TUDataset", "IMDB-MULTI", "~/graph-data")
