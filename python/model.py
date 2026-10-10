"""Floating-point v1 network: 784 inputs -> 64 ReLU neurons -> 10 raw scores."""

import torch
from torch import nn


class MNISTMLP(nn.Module):
    """Accept binary float32 inputs shaped (batch, 784); return (batch, 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(inputs)))
