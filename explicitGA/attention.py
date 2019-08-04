from typing import Dict, Tuple

import torch_geometric as tg
import torch
import torch.nn as nn
from termcolor import cprint


class GAttModule(nn.Module):

    def __init__(self, in_features: int, out_features: int, dropout: float, heads: int, concat: bool,
                 forward_with_attention: bool,
                 *args, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.heads = heads
        self.concat = concat
        self.forward_with_attention = forward_with_attention

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    def register_parameter_all(self):
        for name, param in self._parameters.items():
            self.register_parameter(name, param)

    def initialize_parameters(self):
        self.register_parameter_all()
        self.reset_parameters()

    def forward(self, x, edge_index) -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GAT(GAttModule):

    def __init__(self, in_features: int, out_features: int, dropout: float, heads: int, concat: bool,
                 forward_with_attention: bool,
                 alpha: float):
        super().__init__(in_features, out_features, dropout, heads, concat, forward_with_attention)

        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.att = nn.Parameter(torch.Tensor(self.heads, 2 * self.out_features))

        self.initialize_parameters()

    def forward(self, x, edge_index) -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
        pass


if __name__ == '__main__':
    gat = GAT(10, 15, 0.8, 1, True, True, 1.)
