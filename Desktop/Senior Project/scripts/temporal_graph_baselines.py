"""
Temporal baselines for comparing GCN-style aggregation vs graph attention.

Both use the same outer recurrence as TGCN in run_tgcn_time_multiseed: per time
bucket, update node states with a graph layer, then merge with a GRU cell over
time. Link scores use the same dot-product decoder as the rest of the project.

Graph "transformer" here follows the common *graph attention* stack implemented
by PyG's TransformerConv (multi-head attention over **neighbors**). Full
**global** self-attention is O(n^2) in nodes and is usually impractical for
fleet-scale daily graphs; if your professor requires global attention, subsample
nodes or use a published sparse / linear-attention variant and cite it.

GCN baseline: two GCNConv layers + GRUCell (temporal state).
Graph-transformer baseline: two TransformerConv layers + GRUCell.

``TGCNPure`` is the T-GCN gated cell from Zhao et al., implemented with only
``torch_geometric.nn.GCNConv`` (same structure as ``torch_geometric_temporal``'s
``TGCN`` class). It avoids ``torch_sparse``, which the *package* import graph of
``torch_geometric_temporal`` otherwise pulls in on some installs.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, TransformerConv


class TGCNPure(nn.Module):
    """
    Temporal Graph Convolutional gated cell (T-GCN), GCNConv-only.

    Logic matches ``torch_geometric_temporal.nn.recurrent.TGCN`` (Zhao et al.);
    see upstream ``temporalgcn.py`` in the pytorch_geometric_temporal repository.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_z = GCNConv(
            in_channels, out_channels, improved=improved, cached=cached, add_self_loops=add_self_loops
        )
        self.linear_z = nn.Linear(2 * out_channels, out_channels)
        self.conv_r = GCNConv(
            in_channels, out_channels, improved=improved, cached=cached, add_self_loops=add_self_loops
        )
        self.linear_r = nn.Linear(2 * out_channels, out_channels)
        self.conv_h = GCNConv(
            in_channels, out_channels, improved=improved, cached=cached, add_self_loops=add_self_loops
        )
        self.linear_h = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, hidden):
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        z = torch.cat([self.conv_z(x, edge_index, edge_weight), hidden], dim=1)
        z = torch.sigmoid(self.linear_z(z))
        r = torch.cat([self.conv_r(x, edge_index, edge_weight), hidden], dim=1)
        r = torch.sigmoid(self.linear_r(r))
        h_tilde = torch.cat([self.conv_h(x, edge_index, edge_weight), hidden * r], dim=1)
        h_tilde = torch.tanh(self.linear_h(h_tilde))
        return z * hidden + (1.0 - z) * h_tilde


class TemporalGCNGRU(nn.Module):
    """Two-layer GCN per snapshot; GRU carries state across buckets."""

    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int | None = None):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else max(out_channels, 64)
        self.conv1 = GCNConv(in_channels, h)
        self.conv2 = GCNConv(h, out_channels)
        self.gru = nn.GRUCell(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, hidden):
        del edge_weight  # unused; API matches TGCN
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        if hidden is None:
            return h
        return self.gru(h, hidden)


class TemporalGraphTransformerGRU(nn.Module):
    """Two-layer TransformerConv (neighbor attention) per snapshot + GRU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else max(out_channels, 64)
        if h % heads != 0:
            h = h + (heads - h % heads)
        self.conv1 = TransformerConv(
            in_channels, h // heads, heads=heads, concat=True, dropout=dropout, beta=True
        )
        self.conv2 = TransformerConv(
            h, out_channels, heads=max(1, heads // 2), concat=False, dropout=dropout, beta=True
        )
        self.gru = nn.GRUCell(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, hidden):
        del edge_weight
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        if hidden is None:
            return h
        return self.gru(h, hidden)
