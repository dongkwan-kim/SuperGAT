from typing import Tuple

import numpy as np
import torch
from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, subgraph, degree
import torch_geometric.nn.inits as tgi

import time
import random


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None):
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '2*|edges| > num_nodes^2' case.
    num_neg_samples = min(num_neg_samples,
                          num_nodes * num_nodes - edge_index.size(1))

    idx = (edge_index[0] * num_nodes + edge_index[1]).to('cpu')

    rng = range(num_nodes**2)
    perm = torch.as_tensor(random.sample(rng, num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.as_tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(edge_index.device)


def batched_negative_sampling(edge_index, batch, num_neg_samples=None):
    split = degree(batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)
    num_nodes = degree(batch, dtype=torch.long)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])

    neg_edge_indices = []
    for edge_index, N, C in zip(edge_indices, num_nodes.tolist(),
                                cum_nodes.tolist()):
        neg_edge_index = negative_sampling(edge_index - C, N,
                                           num_neg_samples) + C
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)


class ExplicitGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 is_explicit=True, explicit_type="basic", **kwargs):
        super(ExplicitGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_explicit = is_explicit
        self.explicit_type = explicit_type

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if self.is_explicit:

            if self.explicit_type == "two_layer_scaling":
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))
                self.att_scaling = Parameter(torch.Tensor(heads))
                self.att_bias = Parameter(torch.Tensor(heads))
                self.att_scaling_2 = Parameter(torch.Tensor(heads))
                self.att_bias_2 = Parameter(torch.Tensor(heads))

            elif self.explicit_type == "divided_head":
                self.att_mh_1 = Parameter(torch.Tensor(out_channels, heads, 2 * out_channels))
                self.att_mh_2_not_neg = Parameter(torch.Tensor(1, heads, out_channels))
                self.att_mh_2_neg = Parameter(torch.Tensor(1, heads, out_channels))

        else:
            self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.cached_pos_alpha = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        tgi.glorot(self.weight)
        tgi.zeros(self.bias)
        for name, param in self.named_parameters():
            if name.startswith("att_scaling"):
                tgi.ones(param)
            elif name.startswith("att_bias"):
                tgi.zeros(param)
            elif name.startswith("att_mh"):
                tgi.glorot(param)

    def forward(self, x, edge_index, size=None, batch=None):
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param size:
        :param batch: None or [B]
        :return:
        """
        residuals = {}

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
            if batch is None:
                neg_edge_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=x.size(0))
            else:
                neg_edge_index = batched_negative_sampling(
                    edge_index=edge_index,
                    batch=batch)

            total_alpha = self._get_attention_with_negatives(
                edge_index=edge_index,
                neg_edge_index=neg_edge_index,
                x=x,
            )  # [E + neg_E, heads]

            #total_alpha = self._degree_scaling(total_alpha, edge_index, neg_edge_index, x.size(0))

            if self.explicit_type == "two_layer_scaling":
                total_alpha = self.att_scaling * total_alpha + self.att_bias
                total_alpha = F.elu(total_alpha)
                total_alpha = self.att_scaling_2 * total_alpha + self.att_bias_2

            residuals = {
                "total_alpha": total_alpha,
                "pos_alpha": self.cached_pos_alpha,
                **residuals,
            }

        return propagated, residuals

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

        self.cached_pos_alpha = alpha

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

    def _get_attention(self, edge_index_i, x_i, x_j, size_i, **kwargs) -> torch.Tensor:
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads]
        """

        # Compute attention coefficients.

        if self.explicit_type == "basic" or self.explicit_type == "two_layer_scaling":
            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh",
                                   torch.cat([x_i, x_j], dim=-1),
                                   self.att_mh_1)

        elif self.explicit_type == "divided_head":
            # [E, heads, 2F] * [F(=x), heads, 2F] -> [F(=x), E, heads]
            alpha = torch.einsum("ehf,xhf->xeh",
                                   torch.cat([x_i, x_j], dim=-1),
                                   self.att_mh_1)
            alpha = F.elu(alpha)
            alpha = F.dropout(alpha, training=self.training)

            # [F, E, heads] * [1, heads, F] -> [E, heads]
            with_negatives = kwargs["with_negatives"] if "with_negatives" in kwargs else False
            if not with_negatives:
                alpha = torch.einsum("feh,xhf->eh", alpha, self.att_mh_2_not_neg)
            else:
                alpha = torch.einsum("feh,xhf->eh", alpha, self.att_mh_2_neg)

        else:
            raise ValueError

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        return alpha

    def _get_attention_with_negatives(self, edge_index, neg_edge_index, x):
        """
        :param edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]
        :param x: [N, heads * F]
        :return: [E + neg_E, heads]
        """

        total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        if neg_edge_index.size(1) <= 0:
            return torch.zeros((2, 0, self.heads))

        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, heads * F]
        size_i = x.size(0)  # N

        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]

        alpha = self._get_attention(total_edge_index_i, x_i, x_j, size_i, with_negatives=True)
        return alpha

    def _degree_scaling(self, attention_eh, edge_index, neg_edge_index, num_nodes):
        total_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)  # [2, E + neg_E]
        node_degree = degree(total_edge_index[1], num_nodes)  # [N]
        total_edge_degree = node_degree[total_edge_index[1]]  # [E + neg_E]
        attention_eh = torch.einsum("e,eh->eh", total_edge_degree, attention_eh)  # [E + neg_E, heads]
        attention_eh /= node_degree.mean()
        return attention_eh

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
