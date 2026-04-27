# -*- coding: utf-8 -*-

"""
gnn_model.py - GNN architectures for ML-P prolongation prediction.

Two architectures are available:

1. EdgeWeightedSAGE (v1):
   - GraphSAGE-style with learned edge weights
   - Simpler, faster inference
   - Good baseline

2. AttentionGNN (v2):
   - Graph Transformer with multi-head attention + dropout
   - Learns WHICH neighbors matter (not just how much)
   - Better generalization across matrix types
   - Recommended for production use

Both predict per-node interpolation strengths s_i in R (pre-sigmoid),
which the GNNProlongationSolver maps to (0,1) and uses to build P.

No external dependencies beyond PyTorch — no torch_geometric needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# V1: EDGE-WEIGHTED GRAPHSAGE LAYER
# ============================================================

class EdgeWeightedSAGEConv(nn.Module):
    """
    GraphSAGE-style convolution with learned edge weights.

    For each node i:
        w_ij  = MLP(edge_attr_ij)                      edge weight
        agg_i = sum_j(w_ij * h_j) / sum_j(w_ij)       weighted mean
        h_i'  = MLP([h_i || agg_i])                    update
    """
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 3) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        N   = x.shape[0]
        src = edge_index[0]
        dst = edge_index[1]

        edge_weights = self.edge_mlp(edge_attr)          # (E, 1)

        agg = torch.zeros(N, x.shape[1], device=x.device)
        cnt = torch.zeros(N, 1,          device=x.device)
        agg.index_add_(0, dst, x[src] * edge_weights)
        cnt.index_add_(0, dst, edge_weights)
        agg = agg / cnt.clamp(min=1e-12)

        return self.node_mlp(torch.cat([x, agg], dim=1))


class SAGEProlongationGNN(nn.Module):
    """
    GraphSAGE-style GNN for per-node interpolation strength prediction.

    Input:
        x          : (N, node_feat_dim)
        edge_index : (2, E)
        edge_attr  : (E, 3)  [|a_ij|, a_ij/a_ii, same_cluster]
    Output:
        (N, 1)  raw logits — apply sigmoid for (0, 1)
    """
    def __init__(
        self,
        node_feat_dim: int = 10,
        hidden: int = 64,
        edge_dim: int = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden)
        self.convs = nn.ModuleList([
            EdgeWeightedSAGEConv(hidden, hidden, edge_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr))
        return self.head(h)


# ============================================================
# V2: ATTENTION-BASED GRAPH TRANSFORMER LAYER
# ============================================================

class AttentionSAGEConv(nn.Module):
    """
    Graph Transformer-style layer with multi-head attention.

    For each node i:
        e_ij  = (Q_i · K_j) * scale + W_e(edge_attr_ij)   attention score
        a_ij  = softmax_j(e_ij)                             attention weight
        agg_i = sum_j(a_ij * V_j)                          attended aggregation
        h_i'  = MLP([h_i || agg_i])                        update

    Unlike fixed edge weights, attention LEARNS which neighbors matter —
    key for generalization across different matrix types.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = out_dim // num_heads
        self.dropout   = dropout
        self.scale     = self.head_dim ** -0.5

        self.W_q = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.W_e = nn.Linear(edge_dim, num_heads, bias=False)
        self.W_o = nn.Linear(num_heads * self.head_dim, out_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        N   = x.shape[0]
        src = edge_index[0]
        dst = edge_index[1]

        Q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim)

        edge_bias = self.W_e(edge_attr)   # (E, num_heads)
        attn = (Q[dst] * K[src]).sum(dim=-1) * self.scale + edge_bias
        attn = F.leaky_relu(attn, negative_slope=0.2)

        # Scatter softmax per dst node
        attn_exp = torch.exp(attn - attn.max())
        attn_sum = torch.zeros(N, self.num_heads, device=x.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_norm = attn_exp / attn_sum[dst].clamp(min=1e-12)

        if self.training:
            attn_norm = F.dropout(attn_norm, p=self.dropout)

        weighted_V = V[src] * attn_norm.unsqueeze(-1)   # (E, H, D)
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        agg.index_add_(0, dst, weighted_V)

        agg = self.W_o(agg.view(N, self.num_heads * self.head_dim))
        return self.node_mlp(torch.cat([x, agg], dim=1))


class AttentionProlongationGNN(nn.Module):
    """
    Graph Transformer GNN with multi-head attention and dropout.

    Recommended over SAGEProlongationGNN for production — better
    generalization across diverse matrix types due to learned attention.

    Input:
        x          : (N, node_feat_dim)
        edge_index : (2, E)
        edge_attr  : (E, 3)
    Output:
        (N, 1)  raw logits — apply sigmoid for (0, 1)
    """
    def __init__(
        self,
        node_feat_dim: int = 10,
        hidden: int = 64,
        edge_dim: int = 3,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList([
            AttentionSAGEConv(hidden, hidden, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            h = norm(h + drop(conv(h, edge_index, edge_attr)))
        return self.head(h)


# ============================================================
# SAVE / LOAD — unified for both architectures
# ============================================================

def save_gnn_model(
    model: nn.Module,
    norm: dict,
    path: str,
    architecture: str = "attention",
    hidden: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.3,
) -> None:
    """
    Save a GNN model checkpoint.

    Parameters
    ----------
    model : nn.Module
        Trained GNN model (SAGEProlongationGNN or AttentionProlongationGNN).
    norm : dict
        Normalization statistics with keys:
        'node_feat_dim', 'X_mean', 'X_std', 'edge_mean', 'edge_std'.
    path : str
        Output .pth file path.
    architecture : str
        'sage' or 'attention'. Stored in checkpoint for load_gnn_model.
    """
    checkpoint = {
        "state_dict":    model.state_dict(),
        "architecture":  architecture,
        "node_feat_dim": norm["node_feat_dim"],
        "hidden":        hidden,
        "num_layers":    num_layers,
        "num_heads":     num_heads,
        "dropout":       dropout,
        "X_mean":        norm["X_mean"],
        "X_std":         norm["X_std"],
        "edge_mean":     norm["edge_mean"],
        "edge_std":      norm["edge_std"],
    }
    torch.save(checkpoint, path)


def load_gnn_model(path: str, device: str = "cpu") -> tuple:
    """
    Load a GNN model from checkpoint.

    Returns
    -------
    model : SAGEProlongationGNN or AttentionProlongationGNN
        Loaded model in eval mode with dropout disabled.
    norm : dict
        Normalization statistics.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", "sage")   # backwards compatible

    if arch == "attention":
        model = AttentionProlongationGNN(
            node_feat_dim=ckpt["node_feat_dim"],
            hidden=ckpt["hidden"],
            num_layers=ckpt["num_layers"],
            num_heads=ckpt.get("num_heads", 4),
            dropout=ckpt.get("dropout", 0.3),
        )
    else:
        model = SAGEProlongationGNN(
            node_feat_dim=ckpt["node_feat_dim"],
            hidden=ckpt["hidden"],
            num_layers=ckpt["num_layers"],
        )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()   # disables dropout at inference time

    norm = {
        "node_feat_dim": ckpt["node_feat_dim"],
        "X_mean":        ckpt["X_mean"],
        "X_std":         ckpt["X_std"],
        "edge_mean":     ckpt["edge_mean"],
        "edge_std":      ckpt["edge_std"],
    }
    return model, norm