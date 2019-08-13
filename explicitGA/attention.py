from typing import Tuple

import numpy as np
import torch
from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, subgraph

import torch_geometric.nn.inits as tgi

import time
import random


def negative_sampling(pos_edge_index, num_nodes, max_num_samples=None):
    max_num_samples = max_num_samples or pos_edge_index.size(1)
    num_samples = min(max_num_samples,
                      num_nodes * num_nodes - pos_edge_index.size(1))

    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1])
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes ** 2)
    perm = torch.tensor(random.sample(rng, num_samples))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


def batch_negative_sampling(pos_edge_index: torch.Tensor,
                            num_nodes: int = None,
                            batch: torch.Tensor = None,
                            max_num_samples: int = None) -> torch.Tensor:
    """
    :param pos_edge_index: [2, E]
    :param num_nodes: N
    :param batch: [B]
    :param max_num_samples: neg_E
    :return: tensor of [2, neg_E]
    """
    assert (num_nodes is not None) or (batch is not None), "Either num_nodes or batch must not be None"

    if batch is not None:
        nodes_list = [torch.nonzero(batch == b).squeeze() for b in range(batch.max() + 1)]
        num_nodes_list = [len(nodes) for nodes in nodes_list]
        pos_edge_index_list = [subgraph(torch.as_tensor(nodes), pos_edge_index, relabel_nodes=True)[0]
                               for nodes in nodes_list]
    else:
        num_nodes_list = [num_nodes]
        pos_edge_index_list = [pos_edge_index]

    neg_edges_index_list = []
    prev_node_idx = 0
    for num_nodes, pos_edge_index_of_one_graph in zip(num_nodes_list, pos_edge_index_list):
        neg_edges_index = negative_sampling(pos_edge_index_of_one_graph,
                                            num_nodes=num_nodes,
                                            max_num_samples=max_num_samples)
        neg_edges_index += prev_node_idx
        neg_edges_index_list.append(neg_edges_index)
        prev_node_idx += num_nodes
    neg_edges_index = torch.cat(neg_edges_index_list, dim=1)
    return neg_edges_index


class ExplicitGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 is_explicit=True, att_criterion: str = None,
                 att_head_type: str = None, att_hidden_features: int = None, **kwargs):
        super(ExplicitGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_explicit = is_explicit

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att_criterion = att_criterion
        self.att_head_type = att_head_type
        self.att_hidden_features = att_hidden_features
        if self.is_explicit:
            if self.att_head_type == "multi":
                self.att_base = Parameter(torch.Tensor(att_hidden_features, heads, 2 * out_channels))
                self.att_base_bias = Parameter(torch.Tensor(att_hidden_features, heads))
                self.att_out_1 = Parameter(torch.Tensor(1, att_hidden_features))
                self.att_out_2 = Parameter(torch.Tensor(2, att_hidden_features))
                self.att_out_bias_1 = Parameter(torch.Tensor(1))
                self.att_out_bias_2 = Parameter(torch.Tensor(2))
            else:
                self.att_base = Parameter(torch.Tensor(att_hidden_features, heads, 2 * out_channels))
                self.att_base_bias = Parameter(torch.Tensor(att_hidden_features, heads))
                self.att_out_2 = Parameter(torch.Tensor(2, att_hidden_features))
                self.att_out_1 = Parameter(torch.Tensor(1, 2))
                self.att_out_bias_1 = Parameter(torch.Tensor(1))
                self.att_out_bias_2 = Parameter(torch.Tensor(2))
        else:
            self.att_base = Parameter(torch.Tensor(att_hidden_features, heads, 2 * out_channels))
            self.att_base_bias = Parameter(torch.Tensor(att_hidden_features, heads))
            self.att_out_1 = Parameter(torch.Tensor(1, att_hidden_features))
            self.att_out_2 = None
            self.att_out_bias_1, self.att_out_bias_2 = None, None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.cached_alpha_2 = None

        self.reset_parameters()

    def reset_parameters(self):
        tgi.glorot(self.weight)
        tgi.zeros(self.bias)
        tgi.glorot(self.att_base)
        tgi.zeros(self.att_base_bias)
        tgi.zeros(self.att_out_bias_1)
        tgi.zeros(self.att_out_bias_2)

        if self.att_out_1 is not None and len(self.att_out_1.size()) > 1:
            tgi.glorot(self.att_out_1)
        else:
            tgi.ones(self.att_out_1)

        if self.att_out_2 is not None and len(self.att_out_2.size()) > 1:
            tgi.glorot(self.att_out_2)
        else:
            tgi.zeros(self.att_out_2)

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
            neg_alpha = self._get_negative_edge_att2(neg_edge_index=neg_edge_index, x=x)  # [2, neg_E, heads]
            total_alpha_2 = torch.cat([self.cached_alpha_2, neg_alpha], dim=-2)  # [2, E + neg_E, heads]
            total_alpha_2 = total_alpha_2.mean(dim=-1)  # [2, E + neg_E]
            total_alpha_2 = torch.nn.LogSoftmax(dim=1)(total_alpha_2.t())  # [E + neg_E, 2]
        else:
            total_alpha_2 = None

        self.cached_alpha_2 = None

        return propagated, total_alpha_2

    def _get_attention(self, edge_index_i, x_i, x_j, size_i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads], [2, E, heads]
        """

        def _add_bias(tensor_peh_or_eh, tensor_p):
            if len(tensor_peh_or_eh.size()) == 3:
                return (tensor_peh_or_eh.permute((1, 2, 0)) + tensor_p).permute((2, 0, 1))
            else:
                return (tensor_peh_or_eh.permute((1, 0)) + tensor_p).permute((1, 0))

        # Compute attention coefficients.

        # [E, heads, 2F] * [X=1~, heads, 2F] -> [X=1~, E, heads]
        alpha_base = torch.einsum("ehf,xhf->xeh",
                                  torch.cat([x_i, x_j], dim=-1),
                                  self.att_base)
        alpha_base = _add_bias(alpha_base, self.att_base_bias.t())

        if not self.is_explicit:
            # Attention, [X, E, heads] * [1, X] -> [E, heads]
            alpha_1 = torch.einsum("xeh,px->eh", alpha_base, self.att_out_1)
            alpha_2 = None

        elif self.att_head_type == "multi":
            # Link prediction, [X, E, heads] * [2, X] -> [2, E, heads]
            alpha_2 = torch.einsum("xeh,px->peh", alpha_base, self.att_out_2)
            alpha_2 = _add_bias(alpha_2, self.att_out_bias_2)

            # Attention, [X, E, heads] * [1, X] -> [E, heads]
            alpha_1 = torch.einsum("xeh,px->eh", alpha_base, self.att_out_1)
            alpha_1 = _add_bias(alpha_1, self.att_out_bias_1)
        else:
            # Link prediction, [X, E, heads] * [2, X] -> [2, E, heads]
            alpha_2 = torch.einsum("xeh,px->peh", alpha_base, self.att_out_2)
            alpha_2 = _add_bias(alpha_2, self.att_out_bias_2)

            # Attention, [2, E, heads] * [1, 2] -> [E, heads]
            alpha_1 = torch.einsum("xeh,px->eh", alpha_2, self.att_out_1)
            alpha_1 = _add_bias(alpha_1, self.att_out_bias_1)

        alpha_1 = F.leaky_relu(alpha_1, self.negative_slope)
        alpha_1 = softmax(alpha_1, edge_index_i, size_i)

        return alpha_1, alpha_2

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
        alpha_1, alpha_2 = self._get_attention(edge_index_i, x_i, x_j, size_i)

        # Caching
        self.cached_alpha_2 = alpha_2

        # Sample attention coefficients stochastically.
        alpha_1 = F.dropout(alpha_1, p=self.dropout, training=self.training)

        # [E, heads, F] * [E, heads, 1] = [E, heads, F]
        return x_j * alpha_1.view(-1, self.heads, 1)

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

    def _get_negative_edge_att2(self, neg_edge_index, x) -> torch.Tensor:
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

        alpha_1, alpha_2 = self._get_attention(neg_edge_index_i, x_i, x_j, size_i)
        return alpha_2

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
