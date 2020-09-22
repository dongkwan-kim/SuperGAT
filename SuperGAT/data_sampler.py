from __future__ import division

import copy
import random
import time

import warnings
from pprint import pprint

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import degree, segregate_self_loops, negative_sampling
from torch_geometric.data.sampler import NeighborSampler
import torch.multiprocessing as mp
import numpy as np

from torch_sparse import SparseTensor

from utils import negative_sampling_numpy, iter_window, grouper

try:
    from torch_cluster import neighbor_sampler
except ImportError:
    neighbor_sampler = None


def fetch_and_generate(iters, func, num_proc):
    with mp.Pool(processes=num_proc) as pool:
        async_result_list = []
        for _iter in iters:
            ret_val = pool.apply_async(func, _iter)
            async_result_list.append(ret_val)
        for async_result in async_result_list:
            synced = async_result.get()
            yield synced


class MyNeighborSampler(NeighborSampler):

    def __init__(self, data, size, num_hops,
                 batch_size=1, shuffle=False, drop_last=False, bipartite=True,
                 add_self_loops=False, flow='source_to_target',
                 use_negative_sampling=False, neg_sample_ratio=None):

        self.N = N = data.num_nodes
        self.E = data.num_edges
        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        if use_negative_sampling:
            assert neg_sample_ratio is not None
            assert neg_sample_ratio > 0.0

        self.use_negative_sampling = use_negative_sampling
        self.num_proc = mp.cpu_count()
        self.neg_sample_ratio = neg_sample_ratio
        super().__init__(data, size, num_hops, batch_size, shuffle, drop_last, bipartite, add_self_loops, flow)

    def __produce_subgraph__(self, b_id):
        r"""Produces a :obj:`Data` object holding the subgraph data for a given
        mini-batch :obj:`b_id`."""

        n_ids = [b_id]
        e_ids = []
        edge_indices = []

        for l in range(self.num_hops):
            e_id = neighbor_sampler(n_ids[-1], self.cumdeg, self.size[l])
            n_id = self.edge_index_j.index_select(0, e_id)
            n_id = n_id.unique(sorted=False)
            n_ids.append(n_id)
            e_ids.append(self.e_assoc.index_select(0, e_id))
            edge_index = self.data.edge_index.index_select(1, e_ids[-1])
            edge_indices.append(edge_index)

        n_id = torch.unique(torch.cat(n_ids, dim=0), sorted=False)
        self.tmp[n_id] = torch.arange(n_id.size(0))
        e_id = torch.cat(e_ids, dim=0)
        edge_index = self.tmp[torch.cat(edge_indices, dim=1)]

        num_nodes = n_id.size(0)
        idx = edge_index[0] * num_nodes + edge_index[1]
        idx, inv = idx.unique(sorted=False, return_inverse=True)
        edge_index = torch.stack([idx / num_nodes, idx % num_nodes], dim=0)
        e_id = e_id.new_zeros(edge_index.size(1)).scatter_(0, inv, e_id)

        # n_id: original ID of nodes in the whole sub-graph.
        # b_id: original ID of nodes in the training graph.
        # sub_b_id: sampled ID of nodes in the training graph.

        # Get full-subgraph for negative sampling.
        # Will be deleted at __call__.
        if self.use_negative_sampling:
            adj, _ = self.adj.saint_subgraph(n_id)
            row, col, edge_idx = adj.coo()
            full_edge_index = torch.stack([row, col], dim=0)
        else:
            full_edge_index = None

        return Data(edge_index=edge_index, e_id=e_id, n_id=n_id, b_id=b_id,
                    sub_b_id=self.tmp[b_id], full_edge_index=full_edge_index, num_nodes=num_nodes)

    def __call__(self, subset=None):
        r"""Returns a generator of :obj:`DataFlow` that iterates over the nodes
        in :obj:`subset` in a mini-batch fashion.

        Args:
            subset (LongTensor or BoolTensor, optional): The initial nodes to
                propagate messages to. If set to :obj:`None`, will iterate over
                all nodes in the graph. (default: :obj:`None`)
        """
        if self.bipartite:
            produce = self.__produce_bipartite_data_flow__
        else:
            produce = self.__produce_subgraph__

        if not self.use_negative_sampling:
            for n_id in self.__get_batches__(subset):
                yield produce(n_id)
        else:
            yield from self.__call_with_negatives__(subset, produce)

    def __call_with_negatives__(self, subset, produce):
        for n_id_group in grouper(self.__get_batches__(subset), self.num_proc):
            # print("produce start ~ ", end=""); t0 = time.time()
            produced_data_list = [produce(n_id) for n_id in n_id_group]
            # print("end: {}".format(time.time() - t0))
            ns_generator = fetch_and_generate(
                iters=[(p_data.full_edge_index.numpy(),
                        p_data.n_id.size(0),
                        int(self.neg_sample_ratio * p_data.edge_index.size(1)))
                       for p_data in produced_data_list],
                func=negative_sampling_numpy,
                num_proc=self.num_proc,
            )
            # print("yield start ~ ", end=""); t0 = time.time()
            for p_data, ns in zip(produced_data_list, ns_generator):
                del p_data.full_edge_index
                p_data.neg_edge_index = torch.as_tensor(ns).long()
                yield p_data
            # print("end: {}".format(time.time() - t0))


class RandomIndexSampler(torch.utils.data.Sampler):
    def __init__(self, num_nodes: int, num_parts: int, shuffle: bool = False):
        self.N = num_nodes
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):
        n_id = torch.randint(self.num_parts, (self.N,), dtype=torch.long)
        n_ids = [(n_id == i).nonzero(as_tuple=False).view(-1)
                 for i in range(self.num_parts)]
        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts


class RandomNodeSampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using :obj:`RandomNodeSampler`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        shuffle (bool, optional): If set to :obj:`True`, the data is reshuffled
            at every epoch (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """

    def __init__(self, data, num_parts: int, shuffle: bool = False, **kwargs):
        assert data.edge_index is not None

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None

        super(RandomNodeSampler, self).__init__(
            self, batch_size=1,
            sampler=RandomIndexSampler(self.N, num_parts, shuffle),
            collate_fn=self.__collate__, **kwargs)

    def __getitem__(self, idx):
        return idx

    def __collate__(self, node_idx):
        node_idx = node_idx[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        adj, _ = self.adj.saint_subgraph(node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if item.size(0) == self.N:
                data[key] = item[node_idx]
            elif item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        return data
