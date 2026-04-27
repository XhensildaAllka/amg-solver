# -*- coding: utf-8 -*-

"""
ml_weight_predictor.py - MLP for predicting per-cluster Jacobi weights.

Contains the neural network architecture, training loop, and
save/load utilities. The model predicts one optimal omega value
per METIS cluster from 7 local cluster features.

Architecture:
    Input (7 features) -> Linear(64) -> ReLU -> Linear(32) -> ReLU
    -> Linear(1) -> Sigmoid -> output in (0, 1)

The raw output is then dynamically scaled to [0.7, 0.95] in
MLAMGSolver to match the high-performance range found empirically.

Features (7):
    0: Relative cluster size (normalized by mean cluster size)
    1: log10 of mean absolute diagonal value
    2: Mean row norm / diagonal ratio
    3: Mean diagonal dominance ratio
    4: Local spectral radius estimate (Power Method, 3 iters)
    5: Off-diagonal strength coefficient of variation
    6: Mean normalized node index (spatial position hint)
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

logger = logging.getLogger(__name__)


class WeightPredictorMLP(nn.Module):
    """
    Multi-Layer Perceptron for predicting optimal Jacobi damping weights.

    Parameters
    ----------
    input_features : int, optional
        Number of input features per cluster. Default is 7.
    hidden_layers : list of int, optional
        Sizes of hidden layers. Default is [64, 32].
    """

    def __init__(
        self,
        input_features: int = 7,
        hidden_layers: list = None,
    ) -> None:
        super(WeightPredictorMLP, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers = []
        in_dim = input_features
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # output in (0, 1)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_weight_predictor(
    model: WeightPredictorMLP,
    train_loader,
    val_loader=None,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cpu",
) -> tuple:
    """
    Train the MLP using MSE loss and Adam optimizer.

    Parameters
    ----------
    model : WeightPredictorMLP
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader, optional
        Validation data loader.
    epochs : int, optional
        Number of training epochs. Default is 100.
    lr : float, optional
        Learning rate. Default is 0.001.
    device : str, optional
        'cpu' or 'cuda'. Default is 'cpu'.

    Returns
    -------
    model : WeightPredictorMLP
        Trained model in eval mode.
    history : dict
        Dictionary with 'train_loss' and 'val_loss' lists.
    """
    model = model.to(device)
    # L2 regularization (weight_decay) reduces overfitting to specific grid sizes
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}

    logger.info("Starting training on %s for %d epochs...", device, epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        history["train_loss"].append(avg_train)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    predictions = model(batch_features.to(device))
                    loss = criterion(predictions, batch_targets.to(device))
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            history["val_loss"].append(avg_val)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Epoch %03d | Train MSE: %.6f | Val MSE: %.6f",
                    epoch + 1, avg_train, avg_val,
                )

    model.eval()
    return model, history


def save_model(model: WeightPredictorMLP, filepath: str) -> None:
    """
    Save model weights and metadata to a .pth file.

    Parameters
    ----------
    model : WeightPredictorMLP
        Trained model to save.
    filepath : str
        Output file path (e.g., 'amg_weight_predictor.pth').
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": 7,
        },
        filepath,
    )
    logger.info("Model saved to %s", filepath)


def load_model(filepath: str, device: str = "cpu") -> WeightPredictorMLP:
    """
    Load a saved model from a .pth file.

    Parameters
    ----------
    filepath : str
        Path to the saved .pth file.
    device : str, optional
        Device to load the model onto. Default is 'cpu'.

    Returns
    -------
    WeightPredictorMLP
        Loaded model in eval mode.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model = WeightPredictorMLP(
        input_features=checkpoint.get("input_features", 7)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded from %s", filepath)
    return model