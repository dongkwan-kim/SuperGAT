from typing import Tuple

import numpy as np
import torch
from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, subgraph

from torch_geometric.nn.inits import glorot, zeros, ones

import time
import random


def negative_sampling(pos_edge_index, num_nodes, max_num_samples=None):

    max_num_samples = max_num_samples or pos_edge_index.size(1)
    num_samples = min(max_num_samples,
                      num_nodes * num_nodes - pos_edge_index.size(1))

    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.as_tensor(random.sample(rng, num_samples))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.as_tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


# TODO: scalability
def batch_negative_sampling(pos_edge_index: torch.Tensor,
                            num_nodes: int,
                            batch: torch.Tensor,
                            max_num_samples: int = None) -> torch.Tensor:
    """
    :param pos_edge_index: [2, E]
    :param num_nodes: N
    :param batch: [B]
    :param max_num_samples: neg_E
    :return: tensor of [2, neg_E]
    """

    if batch is not None:
        n_batches = batch.max() + 1
        batch_numpy = batch.numpy()
        nodes_list = [np.argwhere(batch_numpy == n).squeeze() for n in range(n_batches)]
        current_edges_list = [subgraph(torch.as_tensor(nodes), pos_edge_index, relabel_nodes=True)[0]
                              for nodes in nodes_list]
    else:
        n_batches = 1
        nodes_list = [np.arange(num_nodes)]
        current_edges_list = [pos_edge_index]

    neg_edges_index_list = []
    prev_node_idx = 0
    for curr_nodes, curr_edge_index in zip(nodes_list, current_edges_list):
        curr_nodes -= prev_node_idx
        curr_neg_edges_index = negative_sampling(curr_edge_index,
                                                 num_nodes=len(curr_nodes),
                                                 max_num_samples=max_num_samples)
        curr_neg_edges_index += prev_node_idx
        neg_edges_index_list.append(curr_neg_edges_index)
        prev_node_idx += len(curr_nodes)
    neg_edges_index = torch.cat(neg_edges_index_list, dim=1)
    return neg_edges_index


class ExplicitGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 is_explicit=True, possible_edges_factor: int = None,
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
        self.att = Parameter(torch.Tensor(2, heads, 2 * out_channels))

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
            neg_edge_index = batch_negative_sampling(
                pos_edge_index=edge_index,
                num_nodes=x.size(0),
                batch=batch,
            )
            # noinspection PyTypeChecker
            neg_alpha = self._get_raw_attention_of_negative_edges(
                neg_edge_index=neg_edge_index, x=x)  # [2, neg_E, heads]
            total_alpha = torch.cat([self.cached_alpha, neg_alpha], dim=-2)  # [2, E + neg_E, heads]
            reduced_alpha = total_alpha.mean(dim=-1)  # [2, E + neg_E]
            target_alpha = torch.nn.LogSoftmax(dim=1)(reduced_alpha.t())  # [E + neg_E, 2]
        else:
            target_alpha = None

        self.cached_alpha = None

        return propagated, target_alpha

    def _get_attention(self, edge_index_i, x_i, x_j, size_i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads], [2, E, heads]
        """
        # Compute attention coefficients.
        # [E, heads, 2F] * [2, heads, 2F] -> [2, E, heads]
        raw_alpha = torch.einsum("ehf,phf->peh",
                                 torch.cat([x_i, x_j], dim=-1),
                                 self.att)
        alpha = F.leaky_relu(raw_alpha[0], self.negative_slope)
        return softmax(alpha, edge_index_i, size_i), raw_alpha

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
        alpha, raw_alpha = self._get_attention(edge_index_i, x_i, x_j, size_i)

        # Caching
        self.cached_alpha = raw_alpha

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

    def _get_raw_attention_of_negative_edges(self, neg_edge_index, x) -> torch.Tensor:
        """
        :param neg_edge_index: [2, neg_E]
        :param x: [N, heads * F]
        :return: [2, neg_E, heads]
        """

        if neg_edge_index.size(1) <= 0:
            return torch.zeros((2, 0, self.heads))

        neg_edge_index_j, neg_edge_index_i = neg_edge_index  # [neg_E]
        x_i = torch.index_select(x, 0, neg_edge_index_i)  # [neg_E, heads * F]
        x_j = torch.index_select(x, 0, neg_edge_index_j)  # [neg_E, heads * F]
        size_i = x.size(0)  # N

        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E, heads, F]

        alpha, raw_alpha = self._get_attention(neg_edge_index_i, x_i, x_j, size_i)
        return raw_alpha

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
