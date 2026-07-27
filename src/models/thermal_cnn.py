"""Pipeline 1 — Thermal CNN encoder. Single 48x48 thermal image -> 256-dim expression feature vector."""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int, dropout: float):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ThermalCNNEncoder(nn.Module):
    """(N, 1, 48, 48) -> (N, 256)"""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.block1 = ConvBlock(1, 32, num_convs=2, dropout=0.25)     # -> (32, 24, 24)
        self.block2 = ConvBlock(32, 64, num_convs=2, dropout=0.25)    # -> (64, 12, 12)
        self.block3 = ConvBlock(64, 128, num_convs=1, dropout=0.25)   # -> (128, 6, 6)
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 6, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)
