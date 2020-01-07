import numpy as np
from termcolor import cprint
import torch
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, subgraph, degree
import torch_geometric.nn.inits as tgi

import time
import random

from utils import get_accuracy, to_one_hot, np_sigmoid


def negative_sampling(edge_index, num_nodes=None, num_neg_samples=None):
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '2*|edges| > num_nodes^2' case.
    num_neg_samples = min(num_neg_samples,
                          num_nodes * num_nodes - edge_index.size(1))

    idx = (edge_index[0] * num_nodes + edge_index[1]).to('cpu')

    rng = range(num_nodes ** 2)
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


class SuperGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0, bias=True,
                 is_super_gat=True, attention_type="basic", super_gat_criterion=None, neg_sample_ratio=0.0, **kwargs):
        super(SuperGAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.is_super_gat = is_super_gat
        self.attention_type = attention_type
        self.super_gat_criterion = super_gat_criterion
        self.neg_sample_ratio = neg_sample_ratio

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if self.is_super_gat:

            if self.attention_type == "gat_originated":
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))
                self.att_scaling = Parameter(torch.Tensor(heads))
                self.att_bias = Parameter(torch.Tensor(heads))
                self.att_scaling_2 = Parameter(torch.Tensor(heads))
                self.att_bias_2 = Parameter(torch.Tensor(heads))

            elif self.attention_type == "dot_product":
                self.att_scaling = Parameter(torch.Tensor(heads))
                self.att_bias = Parameter(torch.Tensor(heads))
                self.att_scaling_2 = Parameter(torch.Tensor(heads))
                self.att_bias_2 = Parameter(torch.Tensor(heads))

            elif self.attention_type.endswith("mask"):
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))
                self.att_scaling = Parameter(torch.Tensor(heads))
                self.att_bias = Parameter(torch.Tensor(heads))
                self.att_scaling_2 = Parameter(torch.Tensor(heads))
                self.att_bias_2 = Parameter(torch.Tensor(heads))

            elif self.attention_type.endswith("mask_only"):
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            else:
                raise ValueError

        else:
            if self.attention_type == "gat_originated" or self.attention_type == "basic":
                self.att_mh_1 = Parameter(torch.Tensor(1, heads, 2 * out_channels))

            elif self.attention_type == "dot_product":
                pass

            else:
                raise ValueError

        self.residuals = {"num_updated": 0, "att_with_negatives": None, "att_label": None}

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

    def forward(self, x, edge_index, size=None, batch=None, attention_edge_index=None):
        """
        :param x: [N, F]
        :param edge_index: [2, E]
        :param size:
        :param batch: None or [B]
        :param attention_edge_index: [2, E'], Use for link prediction
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

        if (self.is_super_gat and self.training) or (attention_edge_index is not None):

            if attention_edge_index is not None:
                neg_edge_index = None

            elif batch is None:
                # For compatibility, use x.size(0) if neg_sample_ratio is not given
                num_neg_samples = x.size(0) if self.neg_sample_ratio <= 0.0 \
                    else int(self.neg_sample_ratio * edge_index.size(1))
                neg_edge_index = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=x.size(0),
                    num_neg_samples=num_neg_samples,
                )
            else:
                neg_edge_index = batched_negative_sampling(
                    edge_index=edge_index,
                    batch=batch)

            att_with_negatives = self._get_attention_with_negatives(
                x=x,
                edge_index=edge_index,
                neg_edge_index=neg_edge_index,
                total_edge_index=attention_edge_index,
            )  # [E + neg_E, heads]

            # Labels
            if self.training:
                device = next(self.parameters()).device
                att_label = torch.zeros(att_with_negatives.size(0)).float().to(device)
                att_label[:edge_index.size(1)] = 1.
            else:
                att_label = None
            self._update_residuals("att_label", att_label)

            if self.attention_type in [
                "gat_originated", "dot_product", "logit_mask", "prob_mask", "tanh_mask",
            ]:
                att_with_negatives = self.att_scaling * F.elu(att_with_negatives) + self.att_bias
                att_with_negatives = self.att_scaling_2 * F.elu(att_with_negatives) + self.att_bias_2

            self._update_residuals("att_with_negatives", att_with_negatives)

        return propagated

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

    def _get_attention(self, edge_index_i, x_i, x_j, size_i, normalize=True, with_negatives=False,
                       **kwargs) -> torch.Tensor:
        """
        :param edge_index_i: [E]
        :param x_i: [E, heads, F]
        :param x_j: [E, heads, F]
        :param size_i: N
        :return: [E, heads]
        """

        # Compute attention coefficients.

        if self.attention_type == "basic" or self.attention_type == "gat_originated":
            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh",
                                 torch.cat([x_i, x_j], dim=-1),
                                 self.att_mh_1)

        elif self.attention_type == "dot_product":
            # [E, heads, F] * [E, heads, F] -> [E, heads]
            alpha = torch.einsum("ehf,ehf->eh", x_i, x_j)

        elif self.attention_type.endswith("mask") or self.attention_type.endswith("mask_only"):

            # [E, heads, F] * [E, heads, F] -> [E, heads]
            logits = torch.einsum("ehf,ehf->eh", x_i, x_j)
            if with_negatives:
                return logits

            # [E, heads, 2F] * [1, heads, 2F] -> [E, heads]
            alpha = torch.einsum("ehf,xhf->eh",
                                 torch.cat([x_i, x_j], dim=-1),
                                 self.att_mh_1)

            # [E, heads] * [E, heads] -> [E, heads]
            if self.attention_type.startswith("logit_mask"):
                alpha = torch.einsum("eh,eh->eh", alpha, logits)
            elif self.attention_type.startswith("prob_mask"):
                alpha = torch.einsum("eh,eh->eh", alpha, torch.sigmoid(logits))
            elif self.attention_type.startswith("tanh_mask"):
                alpha = torch.einsum("eh,eh->eh", alpha, torch.tanh(logits))
            else:
                raise ValueError

        else:
            raise ValueError

        if normalize:
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, edge_index_i, size_i)

        return alpha

    def _get_attention_with_negatives(self, x, edge_index, neg_edge_index, total_edge_index=None):
        """
        :param x: [N, heads * F]
        :param edge_index: [2, E]
        :param neg_edge_index: [2, neg_E]
        :param total_edge_index: [2, E + neg_E], if total_edge_index is given, use it.
        :return: [E + neg_E, heads]
        """

        if neg_edge_index is not None and neg_edge_index.size(1) <= 0:
            neg_edge_index = torch.zeros((2, 0, self.heads))

        if total_edge_index is None:
            total_edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)  # [2, E + neg_E]

        total_edge_index_j, total_edge_index_i = total_edge_index  # [E + neg_E]
        x_i = torch.index_select(x, 0, total_edge_index_i)  # [E + neg_E, heads * F]
        x_j = torch.index_select(x, 0, total_edge_index_j)  # [E + neg_E, heads * F]
        size_i = x.size(0)  # N

        x_j = x_j.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]
        if x_i is not None:
            x_i = x_i.view(-1, self.heads, self.out_channels)  # [E + neg_E, heads, F]

        alpha = self._get_attention(total_edge_index_i, x_i, x_j, size_i,
                                    normalize=False, with_negatives=True)
        return alpha

    def __repr__(self):
        return '{}({}, {}, heads={}, attention_type={}, neg_sample_ratio={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.attention_type, self.neg_sample_ratio,
        )

    def _update_residuals(self, key, val):
        self.residuals[key] = val
        self.residuals["num_updated"] += 1

    @staticmethod
    def get_supervised_attention_loss(model, mixing_weight, edge_sampling_ratio=1.0, criterion=None):

        assert model.args.is_super_gat
        if mixing_weight == 0:
            return 0

        loss_list = []
        att_residuals_list = [(m, m.residuals) for m in model.modules()
                              if m.__class__.__name__ == SuperGAT.__name__]

        device = next(model.parameters()).device
        criterion = nn.BCEWithLogitsLoss() if criterion is None else eval(criterion)
        for module, att_res in att_residuals_list:

            # Attention (X)
            att = att_res["att_with_negatives"]  # [E + neg_E, heads]
            num_total_samples = att.size(0)
            num_to_sample = int(num_total_samples * edge_sampling_ratio)

            # Labels (Y)
            label = att_res["att_label"]  # [E + neg_E]

            att = att.mean(dim=-1)  # [E + neg_E]
            permuted = torch.randperm(num_total_samples).to(device)
            loss = criterion(att[permuted][:num_to_sample], label[permuted][:num_to_sample])
            loss_list.append(loss)

        total_loss = mixing_weight * sum(loss_list)
        return total_loss

    @staticmethod
    def get_link_pred_acc_by_attention(model, edge_y, layer_idx=-1):
        """
        :param model: GNN model (nn.Module)
        :param edge_y: [E_pred] tensor
        :param layer_idx: layer idx of GNN models
        :return:
        """
        att_residuals_list = [m.residuals for m in model.modules() if m.__class__.__name__ == SuperGAT.__name__]
        att_res = att_residuals_list[layer_idx]

        att = att_res["att_with_negatives"]  # [E + neg_E, heads]
        att = att.mean(dim=-1)  # [E + neg_E]

        edge_probs = 1. - np_sigmoid(att.cpu().numpy())
        edge_outputs = np.transpose(np.vstack([1. - edge_probs, edge_probs]))
        pred_acc = get_accuracy(edge_outputs, to_one_hot(edge_y.cpu().int(), 2))
        return pred_acc
