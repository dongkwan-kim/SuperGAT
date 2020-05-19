import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, structured_negative_sampling

from torch_geometric.nn.inits import glorot, zeros
# from ..inits import glorot, zeros

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_max, scatter_add


def topk_softmax(src, index, k, num_nodes=None):
    r"""Computes a sparsely evaluated softmax using only top-k values.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then select top-k values for each group, compute the softmax individually.
    The output of not selected indices will be zero.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        k (int): The number of indexes to select from each group.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    num_nodes = maybe_num_nodes(index, num_nodes)

    # Create mask the topk values of which are 1., otherwise 0.
    out = src.clone()
    topk_mask = torch.zeros_like(out)
    for _ in range(k):
        v_max = scatter_max(out, index, dim=0, dim_size=num_nodes)[0]
        i_max = (out == v_max[index])
        topk_mask[i_max] = 1.
        out[i_max] = float("-Inf")

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()

    # Mask except topk values
    out = out * topk_mask

    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out


class CGATConv(MessagePassing):
    r"""The constrained graph attentional operator from the
    `"Improving Graph Attention Networks with Large Margin-based Constraints"
    <https://arxiv.org/abs/1910.11945>`_ paper
    .. math::

    .. math::

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        margin_graph (float, optional): slack variable which controls
            the margin between attention values regarding to the graph
            structure (default: :obj:`0.1`)
        margin_boundary (float, optional): slack variable which controls
            the margin between attention values regarding to the class
            boundary (default: :obj:`0.1`)
        use_topk_softmax (bool, optional):
        aggr_k (int, optional): aggregate function only makes use of the
            features from the neighbors with top k attention weights rather
            than all neighbors. (default: :obj:`4`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 margin_graph=0.1, margin_boundary=0.1,
                 use_topk_softmax=True, aggr_k=4,
                 num_neg_samples_per_edge=5, **kwargs):
        super(CGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # CGAT's own hyper-parameters
        self.margin_graph = margin_graph
        self.margin_boundary = margin_boundary
        self.use_topk_softmax = use_topk_softmax
        self.aggr_k = aggr_k
        self.num_neg_samples_per_edge = num_neg_samples_per_edge
        self.cache = {"edge_index": None, "att_pos": None, "att_neg": None}

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))
        self.cache["edge_index"] = edge_index

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        propagated = self.propagate(edge_index, size=size, x=x)

        if self.training:
            att_neg_list = []
            for _ in range(self.num_neg_samples_per_edge):
                edge_j, edge_i, edge_k = structured_negative_sampling(
                    edge_index=edge_index,
                    num_nodes=x.size(0),
                )
                x_j, x_k = x[edge_j], x[edge_k]
                att_neg = self.get_unnormalized_attention(x_j, x_k)
                att_neg_list.append(att_neg)
            self.cache["att_neg"] = torch.stack(att_neg_list, dim=-1)  # [E, heads, num_neg]

        return propagated

    def get_unnormalized_attention(self, x_i, x_j):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        return alpha

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        alpha = self.get_unnormalized_attention(x_i, x_j)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Cache alpha for regularization loss.
        self.cache["att_pos"] = alpha  # [E, heads]

        # Get top k attention coefficients only.
        if self.use_topk_softmax:
            if self.training:
                alpha = topk_softmax(alpha, edge_index_i, self.aggr_k, size_i)
            else:
                alpha = softmax(alpha, edge_index_i, size_i)
        else:
            alpha = softmax(alpha, edge_index_i, size_i)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j.view(-1, self.heads, self.out_channels) * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    @staticmethod
    def _get_edge_mask_by_class(edge_index, masked_y, same_class=True):
        edge_j, edge_i = edge_index
        if same_class:
            class_mask = masked_y[edge_j] == masked_y[edge_i]
        else:
            class_mask = masked_y[edge_j] != masked_y[edge_i]
        edge_mask = ((masked_y[edge_j] >= 0) &
                     (masked_y[edge_i] >= 0) &
                     class_mask)
        return edge_mask

    def get_graph_structure_constraint_loss(self, masked_y):
        r"""Loss from Graph Structure based Constraint.

        Args:
            masked_y (LongTensor): Node targets where val/test nodes are
                masked by `-1`.
        :return:
        """
        att_pos = self.cache["att_pos"]  # [E, heads]
        att_neg = self.cache["att_neg"]  # [E, heads, num_neg]
        edge_index = self.cache["edge_index"]
        num_edges = edge_index.size(1)

        is_edge_same_class = self._get_edge_mask_by_class(edge_index, masked_y, same_class=True)  # bool of [E]

        # \sum_{i, j \in N_i^+} \sum_{k in V \ N_i}
        #   max(0, \phi(v_{i}, v_{k}) + \zeta_{g} - \phi(v_{i}, v_{j}))
        att_pos = att_pos.view(num_edges, self.heads, 1) \
            .expand(-1, -1, self.num_neg_samples_per_edge)  # [E, heads, num_neg]
        loss = F.relu(att_neg[is_edge_same_class] + self.margin_graph
                      - att_pos[is_edge_same_class])  # [E*, heads, num_neg]
        return loss.mean()

    def get_graph_structure_constraint_loss_for_ssnc(self):
        r"""Modified Loss from Graph Structure based Constraint for the
            semi-supervised node classification task, by not using label
            information.
        :return:
        """
        att_pos = self.cache["att_pos"]  # [E, heads]
        att_neg = self.cache["att_neg"]  # [E, heads, num_neg]
        edge_index = self.cache["edge_index"]
        num_edges = edge_index.size(1)

        # \sum_{i, j \in N_i} \sum_{k in V \ N_i}
        #   max(0, \phi(v_{i}, v_{k}) + \zeta_{g} - \phi(v_{i}, v_{j}))
        att_pos = att_pos.view(num_edges, self.heads, 1) \
            .expand(-1, -1, self.num_neg_samples_per_edge)  # [E, heads, num_neg]
        loss = F.relu(att_neg + self.margin_graph - att_pos)  # [E, heads, num_neg]
        return loss.mean()

    def get_class_boundary_constraint_loss(self, masked_y):
        r"""Loss from Class Boundary Constraint.

        Args:
            masked_y (LongTensor): Node targets where val/test nodes are
                masked by `-1`.
        :return:
        """
        att_pos = self.cache["att_pos"]  # [E, heads]
        edge_index = self.cache["edge_index"]

        # bool of [E]
        is_edge_same_class = self._get_edge_mask_by_class(edge_index, masked_y, same_class=True)
        is_edge_diff_class = self._get_edge_mask_by_class(edge_index, masked_y, same_class=False)

        edge_index_same_class = edge_index[:, is_edge_same_class]  # [2, E^+]
        edge_index_diff_class = edge_index[:, is_edge_diff_class]  # [2, E^-]
        att_pos_same_class = att_pos[is_edge_same_class]  # [E^+, heads]
        att_pos_diff_class = att_pos[is_edge_diff_class]  # [E^-, heads]

        # \sum_{i, j \in N_i^+} \sum_{k in N_i^-}
        #   max(0, \phi(v_{i}, v_{k}) + \zeta_{b} - \phi(v_{i}, v_{j}))
        loss_list = []
        for e_i in torch.unique(edge_index_diff_class[0]):

            att_ij = att_pos_same_class[edge_index_same_class[0] == e_i]  # [E^{+, e_i}, heads]
            att_ik = att_pos_diff_class[edge_index_diff_class[0] == e_i]  # [E^{-, e_i}, heads]

            # [E^{+, e_i} * E^{-, e_i}, heads]: get computation of all combinations of ik and ij
            loss = (att_ik.unsqueeze(1) + self.margin_boundary - att_ij.unsqueeze(0)).view(-1, self.heads)
            loss_list.append(F.relu(loss))

        loss = torch.cat(loss_list, dim=0)
        return loss.mean()

    @staticmethod
    def mix_regularization_loss(loss, model, masked_y, graph_lambda, boundary_lambda):
        cgat_layers = [m for m in model.modules() if m.__class__.__name__ == CGATConv.__name__]
        for layer in cgat_layers:
            loss += graph_lambda * layer.get_graph_structure_constraint_loss(masked_y)
            loss += boundary_lambda * layer.get_class_boundary_constraint_loss(masked_y)
        return loss

    @staticmethod
    def mix_regularization_loss_for_ssnc(loss, model, graph_lambda):
        cgat_layers = [m for m in model.modules() if m.__class__.__name__ == CGATConv.__name__]
        for layer in cgat_layers:
            loss += graph_lambda * layer.get_graph_structure_constraint_loss_for_ssnc()
        return loss

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


if __name__ == '__main__':

    MODE = "TOPK_SM"

    if MODE == "CONV":
        from torch_geometric.utils import to_undirected

        cgat = CGATConv(11, 13, heads=3)
        _x = torch.randn(20, 11).float()
        _y = torch.Tensor([0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1,
                           2, 2, 2, 2, 2,
                           3, 3, 3, 3, 3]).long()
        _y_mask = torch.Tensor([1, 1, 0, 0, 0,
                                1, 1, 0, 0, 0,
                                1, 1, 0, 0, 0,
                                1, 1, 0, 0, 0]).bool()
        _edge_index = torch.Tensor([[1, 2, 3, 4, 5, 6, 7],
                                    [0, 0, 0, 0, 0, 0, 0]]).long()
        _edge_index = to_undirected(_edge_index)
        print("edge_index_with_self_loops", add_self_loops(_edge_index, num_nodes=_x.size(0))[0].size())
        cgat(_x, _edge_index)
        _masked_y = _y
        _masked_y[~_y_mask] = -1
        print(cgat.get_graph_structure_constraint_loss(_masked_y))
        print(cgat.get_class_boundary_constraint_loss(_masked_y))

    elif MODE == "TOPK_SM":
        _src = torch.Tensor(
            [[1, 2, 9],
             [4, 5, 6],
             [5, 6, 7],
             [7, 0, 3],
             [3, -2, 1],
             [-1, -1, -1],
             [0, 1, 2]]
        ).float()  # [E, heads]
        _index = torch.Tensor(
            [0, 0, 0, 1, 1, 1, 2]
        ).long()  # [E]

        """
        mask:
        tensor([[False, False,  True],
                [ True,  True, False],
                [ True,  True,  True],
                [ True,  True,  True],
                [ True, False,  True],
                [False,  True, False],
                [ True,  True,  True]])
        value:
        tensor([[0.0000, 0.0000, 0.8808],
                [0.2689, 0.2689, 0.0000],
                [0.7311, 0.7311, 0.1192],
                [0.9820, 0.7311, 0.8808],
                [0.0180, 0.0000, 0.1192],
                [0.0000, 0.2689, 0.0000],
                [1.0000, 1.0000, 1.0000]])
        """
        _ks = topk_softmax(_src, _index, 2)
        print(_ks)
