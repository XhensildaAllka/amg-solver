"""
training/train_gnn.py
======================
Unified training script for GNN-P prolongation models.

Supports two architectures:
    sage       — EdgeWeightedSAGEConv (simpler, faster)
    attention  — Graph Transformer with multi-head attention (recommended)

Each sample is a full graph (not a flat vector), so training iterates
over graphs one at a time. The best model (lowest loss) is saved
automatically during training.

Usage
-----
    # Train attention model (recommended)
    python training/train_gnn.py \\
        --dataset gnn_dataset.pkl \\
        --output gnn_model_attn.pth \\
        --arch attention

    # Train SAGE model (faster)
    python training/train_gnn.py \\
        --dataset gnn_dataset.pkl \\
        --output gnn_model_sage.pth \\
        --arch sage \\
        --hidden 64 --epochs 100
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from amg.gnn_model import (
    SAGEProlongationGNN,
    AttentionProlongationGNN,
    save_gnn_model,
)


# ============================================================
# NORMALIZATION
# ============================================================

def compute_normalization(graphs: list) -> tuple:
    """Compute global mean/std for node features and edge attributes."""
    all_x  = np.vstack([g["node_features"] for g in graphs])
    all_ea = np.vstack([g["edge_attr"]     for g in graphs])

    X_mean    = all_x.mean(axis=0,  keepdims=True).astype(np.float32)
    X_std     = all_x.std(axis=0,   keepdims=True).astype(np.float32) + 1e-8
    edge_mean = all_ea.mean(axis=0, keepdims=True).astype(np.float32)
    edge_std  = all_ea.std(axis=0,  keepdims=True).astype(np.float32) + 1e-8

    return X_mean, X_std, edge_mean, edge_std


def graph_to_tensors(graph: dict, X_mean, X_std, edge_mean, edge_std) -> tuple:
    """Normalize and convert one graph dict to torch tensors."""
    x  = (graph["node_features"] - X_mean) / X_std
    ea = (graph["edge_attr"]     - edge_mean) / edge_std
    y  = graph["targets"].reshape(-1, 1)
    return (
        torch.tensor(x,                 dtype=torch.float32),
        torch.tensor(graph["edge_index"], dtype=torch.long),
        torch.tensor(ea,                dtype=torch.float32),
        torch.tensor(y,                 dtype=torch.float32),
    )


# ============================================================
# TRAINING LOOP
# ============================================================

def train(
    dataset_path: str,
    save_path: str,
    architecture: str = "attention",
    hidden: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.3,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    log_every: int = 10,
) -> None:
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "rb") as f:
        graphs = pickle.load(f)

    print(f"  Graphs : {len(graphs)}")
    print(f"  Nodes  : {sum(g['N'] for g in graphs)}")
    print(f"  Edges  : {sum(g['edge_index'].shape[1] for g in graphs)}")

    X_mean, X_std, edge_mean, edge_std = compute_normalization(graphs)
    node_feat_dim = graphs[0]["node_features"].shape[1]

    norm = {
        "node_feat_dim": node_feat_dim,
        "X_mean":        X_mean,
        "X_std":         X_std,
        "edge_mean":     edge_mean,
        "edge_std":      edge_std,
    }

    # Build model
    if architecture == "attention":
        model = AttentionProlongationGNN(
            node_feat_dim=node_feat_dim,
            hidden=hidden,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    elif architecture == "sage":
        model = SAGEProlongationGNN(
            node_feat_dim=node_feat_dim,
            hidden=hidden,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Use 'attention' or 'sage'.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nArchitecture : {architecture}")
    print(f"Parameters   : {n_params:,}")
    print(f"Hidden       : {hidden},  Layers: {num_layers},  Epochs: {epochs}")
    if architecture == "attention":
        print(f"Heads        : {num_heads},  Dropout: {dropout}")

    opt       = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20,
    )
    loss_fn   = nn.MSELoss()
    best_loss = float("inf")

    print(f"\nTraining for {epochs} epochs over {len(graphs)} graphs...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss    = 0.0
        total_samples = 0

        perm = np.random.permutation(len(graphs))
        for idx in perm:
            x, ei, ea, y = graph_to_tensors(
                graphs[idx], X_mean, X_std, edge_mean, edge_std
            )
            opt.zero_grad()
            pred = model(x, ei, ea)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss    += loss.item() * len(y)
            total_samples += len(y)

        epoch_loss /= total_samples
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_gnn_model(
                model, norm, save_path,
                architecture=architecture,
                hidden=hidden,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{epochs}  "
                f"loss={epoch_loss:.6f}  "
                f"best={best_loss:.6f}  "
                f"lr={opt.param_groups[0]['lr']:.2e}"
            )

    print(f"\n✓ Training complete. Best loss: {best_loss:.6f}")
    print(f"✓ Best model saved to: {save_path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # *** CONFIGURE HERE ***
    # --------------------------------------------------------
    THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(THIS_DIR, "datasets")
    MODELS_DIR  = os.path.join(THIS_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Default dataset and output model — saved inside the library
    DEFAULT_DATASET = os.path.join(DATASET_DIR, "gnn_dataset_proxy.pkl")
    DEFAULT_MODEL   = os.path.join(MODELS_DIR,  "gnn_model_attn_proxy.pth")
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(description="Train GNN-P prolongation model")
    parser.add_argument("--dataset",      default=DEFAULT_DATASET,
                        help=f"Path to .pkl dataset (default: {DEFAULT_DATASET})")
    parser.add_argument("--output",       default=DEFAULT_MODEL,
                        help=f"Output .pth model path (default: {DEFAULT_MODEL})")
    parser.add_argument("--arch",         default="attention",
                        choices=["attention", "sage"],
                        help="GNN architecture (default: attention)")
    parser.add_argument("--hidden",       type=int, default=64)
    parser.add_argument("--num_layers",   type=int, default=3)
    parser.add_argument("--num_heads",    type=int, default=4,
                        help="Attention heads (attention arch only)")
    parser.add_argument("--dropout",      type=float, default=0.3,
                        help="Dropout rate (attention arch only)")
    parser.add_argument("--epochs",       type=int, default=200)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_every",    type=int, default=10)
    args = parser.parse_args()

    print(f"Dataset : {args.dataset}")
    print(f"Output  : {args.output}")

    train(
        dataset_path=args.dataset,
        save_path=args.output,
        architecture=args.arch,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )