import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_max, scatter_mean


class GaANConv(MessagePassing):
    r"""The gated attentional operator from the `"GaAN: Gated Attention
    Networks for Learning on Large and Spatiotemporal Graphs"
    <https://arxiv.org/abs/1803.07294>`_ paper
    """

    def __init__(self, in_channels, out_channels, key_and_query_channels, value_channels, projected_channels,
                 heads=1, negative_slope=0.1, **kwargs):
        super(GaANConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_and_query_channels = key_and_query_channels
        self.value_channels = value_channels
        self.projected_channels = projected_channels
        self.heads = heads
        self.negative_slope = negative_slope

        self.lin_za = Linear(in_channels, heads * key_and_query_channels)
        self.lin_xa = Linear(in_channels, heads * key_and_query_channels)
        self.lin_v = Linear(in_channels, heads * value_channels)
        self.lin_o = Linear(in_channels + heads * value_channels, out_channels)
        self.lin_g = Linear(in_channels + projected_channels + in_channels, heads)
        self.lin_m = Linear(in_channels, projected_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_za.reset_parameters()
        self.lin_xa.reset_parameters()
        self.lin_v.reset_parameters()
        self.lin_o.reset_parameters()
        self.lin_g.reset_parameters()
        self.lin_m.reset_parameters()

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        x_xa = self.lin_xa(x)
        z_za = self.lin_za(x)
        z_v = F.leaky_relu(self.lin_v(x), self.negative_slope)
        z_m = self.lin_m(x)

        out = self.propagate(edge_index, size=size, x=x, z=x, x_xa=x_xa, z_za=z_za, z_v=z_v, z_m=z_m)
        out = torch.cat([x, out], dim=1)
        out = self.lin_o(out)
        return out

    def message(self, edge_index_i, x, z_j, x_xa_i, z_za_j, z_v_j, z_m_j, size_i):

        x_xa_i = x_xa_i.view(-1, self.heads, self.key_and_query_channels)
        z_za_j = z_za_j.view(-1, self.heads, self.key_and_query_channels)
        z_v_j = z_v_j.view(-1, self.heads, self.value_channels)

        # Compute multi-head attention coefficients.
        phi_w = torch.einsum("ehf,ehf->eh", x_xa_i, z_za_j)
        w = softmax(phi_w, edge_index_i, size_i)

        # Compute gated attention coefficients.
        max_pooled_z_m_j = scatter_max(z_m_j, edge_index_i, dim=0, dim_size=size_i)[0]
        mean_pooled_z_m_j = scatter_mean(z_j, edge_index_i, dim=0, dim_size=size_i)
        g = self.lin_g(torch.cat([x, max_pooled_z_m_j, mean_pooled_z_m_j], dim=1))
        g = torch.sigmoid(g)
        g = g[edge_index_i]
        return z_v_j * g.view(-1, self.heads, 1) * w.view(-1, self.heads, 1)  # [E, heads, d_v]

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.value_channels)  # [N, heads, d_v]
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


if __name__ == '__main__':
    _x = torch.Tensor(7, 13)
    _ei = torch.Tensor([[0, 0, 0, 1, 1, 2, 3],
                        [0, 1, 2, 3, 4, 5, 6]]).long()

    _l = GaANConv(in_channels=13,
                  out_channels=17,
                  key_and_query_channels=9,
                  value_channels=11,
                  projected_channels=5,
                  heads=2)
    print(_l)
    print(_l(_x, _ei).size())
