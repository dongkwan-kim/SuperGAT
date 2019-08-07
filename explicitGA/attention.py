from typing import Tuple

import numpy as np
import torch
from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, subgraph

from torch_geometric.nn.inits import glorot, zeros, ones

from utils import get_cartesian, create_hash

import time
from math import sqrt


class ExplicitGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 is_explicit=True, hash_to_neg_possible_edges: dict = None, possible_edges_factor: int = None,
                 att_criterion: str = None, **kwargs):
        super(ExplicitGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_explicit = is_explicit
        self.possible_edges_factor = possible_edges_factor

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.att_criterion = att_criterion
        self.att_scaling = Parameter(torch.Tensor(1))
        self.att_bias = Parameter(torch.Tensor(1))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.cached_alpha = None
        self.hash_to_neg_possible_edges = hash_to_neg_possible_edges if hash_to_neg_possible_edges is not None else dict()

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)
        ones(self.att_scaling)
        zeros(self.att_bias)

    def forward(self, x, edge_index, size=None, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param size:
        :param batch: None or [B]
        :return:
        """
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # [N, F0] * [F0, heads * F] = [N, heads * F]
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        propagated = self.propagate(edge_index, size=size, x=x)

        if self.training and self.is_explicit:
            neg_edge_index = self._sample_negative_edges(
                max_n_neg_edges=edge_index.size(1),
                edge_index=edge_index,
                n_nodes=x.size(0),
                batch=batch,
            )
            neg_alpha = self._get_attention_of_negative_edges(neg_edge_index=neg_edge_index, x=x)  # [neg_E, heads]
            total_alpha = torch.cat([self.cached_alpha, neg_alpha])  # [E + neg_E, heads]
            reduced_alpha = total_alpha.mean(dim=1)  # [E + neg_E]

            m, s = reduced_alpha.mean(), reduced_alpha.std()
            reduced_alpha = torch.sigmoid(self.att_scaling * ((reduced_alpha - m) / s) + self.att_bias)

            if "CrossEntropyLoss" in self.att_criterion:
                target_alpha = torch.stack([reduced_alpha, 1 - reduced_alpha]).t()  # [E + neg_E, 2]
            else:  # MSELoss, L1Loss
                target_alpha = reduced_alpha
        else:
            target_alpha = None

        self.cached_alpha = None

        return propagated, target_alpha

    def _get_attention(self, edge_index_i, x_i, x_j, size_i):
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads]
        """
        # Compute attention coefficients.
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [E, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)
        return alpha

    # noinspection PyMethodOverriding
    def message(self, edge_index_i, x_i, x_j, size_i):
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads * F]
        :param x_j: [E, heads * F]
        :param size_i: N
        :return: [E, heads, F]
        """
        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E, heads, F]

        # Compute attention coefficients. [E, heads]
        alpha = self._get_attention(edge_index_i, x_i, x_j, size_i)

        # Caching
        self.cached_alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # [E, heads, F] * [E, heads, 1] = [E, heads, F]
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        """
        :param aggr_out: [N, heads, F]
        :return: [N, heads * F]
        """
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def _sample_negative_edges(self,
                               max_n_neg_edges: int,
                               edge_index: torch.Tensor,
                               n_nodes: int,
                               batch: torch.Tensor) -> torch.Tensor:
        """
        :param max_n_neg_edges: neg_E
        :param edge_index: [2, E]
        :param n_nodes: N
        :param batch: [B]
        :return: tensor of [2, neg_E]
        """

        n_possible_edges = n_nodes ** 2

        if batch is not None:
            n_batches = batch.max() + 1
            batch_numpy = batch.numpy()
            nodes_list = [np.argwhere(batch_numpy == n).squeeze() for n in range(n_batches)]
            current_edges_list = [subgraph(torch.as_tensor(nodes), edge_index)[0] for nodes in nodes_list]
        else:
            n_batches = 1
            nodes_list = [np.arange(n_nodes)]
            current_edges_list = [edge_index]
        assert n_batches == 1, "T.T"

        edge_hash_list = [create_hash({"current_edges": current_edges, "n_nodes": n_nodes})
                          for current_edges in current_edges_list]
        n_current_edges_list = [current_edges.size(1) for current_edges in current_edges_list]
        n_neg_edges_list = [min(max_n_neg_edges, n_current_edges, n_possible_edges - n_current_edges)
                            for n_current_edges in n_current_edges_list]

        neg_edges_list = []
        for edge_hash, n_neg_edges, nodes in zip(edge_hash_list, n_neg_edges_list, nodes_list):

            if n_nodes > 10000:  # Scalability
                n_sample_nodes = int(sqrt(self.possible_edges_factor * n_neg_edges)) + 1
                nodes_x = np.random.choice(nodes, n_sample_nodes, replace=False)
                nodes_y = np.random.choice(nodes, n_sample_nodes, replace=False)
            else:
                nodes_x, nodes_y = nodes, nodes

            if edge_hash not in self.hash_to_neg_possible_edges:
                node_pairs = get_cartesian(nodes_x, nodes_y)  # ndarray of [N^2, 2]
                np.random.shuffle(node_pairs)
                node_pairs = {tuple(sorted(e)) for e in node_pairs[:self.possible_edges_factor * n_neg_edges]}
                neg_possible_edges = np.asarray(list(node_pairs - {tuple(e) for e in edge_index.t().numpy()}))
                self.hash_to_neg_possible_edges[edge_hash] = neg_possible_edges
            else:
                neg_possible_edges = self.hash_to_neg_possible_edges[edge_hash]  # Use cached one.

            np.random.shuffle(neg_possible_edges)
            neg_edges = torch.as_tensor(neg_possible_edges[:n_neg_edges]).t()  # [neg_E(i), 2]
            neg_edges_list.append(neg_edges)

        neg_edges_index = torch.cat(neg_edges_list, dim=1)
        return neg_edges_index

    def _get_attention_of_negative_edges(self, neg_edge_index, x) -> torch.Tensor:
        """
        :param neg_edge_index: [2, neg_E]
        :param x: [N, heads * F]
        :return: [neg_E, heads]
        """
        neg_edge_index_j, neg_edge_index_i = neg_edge_index  # [neg_E]
        x_i = torch.index_select(x, 0, neg_edge_index_i)  # [neg_E, heads * F]
        x_j = torch.index_select(x, 0, neg_edge_index_j)  # [neg_E, heads * F]
        size_i = x.size(0)  # N

        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E, heads, F]

        alpha = self._get_attention(neg_edge_index_i, x_i, x_j, size_i)
        return alpha


    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
