"""
mlp_model.py - MLP architecture for ML-P prolongation prediction.

The model predicts a per-node interpolation strength s_i in R
(pre-sigmoid), which is then mapped to (0,1) via sigmoid in the solver.

The model is trained to predict how much each fine node should
interpolate from neighboring clusters (vs staying in its own cluster).
High strength -> richer interpolation; low strength -> near injection.

Architecture: Linear(in) -> ReLU -> Linear(256) -> ReLU -> Linear(1)
"""

import torch
import torch.nn as nn
import numpy as np


class MLProlongationMLP(nn.Module):
    """
    MLP for predicting per-node prolongation interpolation strength.

    Parameters
    ----------
    in_dim : int
        Number of input features per node (10 by default from
        mlp_features.extract_mlP_features).
    hidden : int, optional
        Hidden layer size. Default is 256.
    """

    def __init__(self, in_dim: int = 10, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_mlP_model(path: str, device: str = "cpu") -> tuple:
    """
    Load a trained MLProlongationMLP from a .pth checkpoint.

    Parameters
    ----------
    path : str
        Path to the .pth checkpoint saved by train_mlP.py.
    device : str, optional
        Device to load onto. Default is 'cpu'.

    Returns
    -------
    model : MLProlongationMLP
        Loaded model in eval mode.
    norm : dict
        Normalization statistics with keys:
        'X_mean', 'X_std', 'y_mean', 'y_std'.
    """
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = MLProlongationMLP(
        in_dim=ckpt["in_dim"],
        hidden=ckpt.get("hidden", 256),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    norm = {
        "X_mean": ckpt["X_mean"],
        "X_std":  ckpt["X_std"],
        "y_mean": ckpt.get("y_mean", np.zeros((1, 1))),
        "y_std":  ckpt.get("y_std",  np.ones((1, 1))),
    }
    return model, norm