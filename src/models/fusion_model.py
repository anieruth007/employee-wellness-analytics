"""Fusion layer — concat(expression_vec[256], personality_proxy_vec[2]) -> 3-way engagement classifier."""
import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """(N, 256) expression features + (N, 2) personality proxy -> (N, 3) engagement logits.

    Returns raw logits (no softmax) so training can use nn.CrossEntropyLoss directly —
    softmax is applied separately at inference time via `predict_proba`, to avoid a
    double-softmax when training with CrossEntropyLoss (which applies log-softmax internally).
    """

    def __init__(self, expression_dim: int = 256, proxy_dim: int = 2, hidden_dims=(128, 64), num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        fused_dim = expression_dim + proxy_dim
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(fused_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, expression_vec: torch.Tensor, proxy_vec: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([expression_vec, proxy_vec], dim=1)
        return self.net(fused)

    @torch.no_grad()
    def predict_proba(self, expression_vec: torch.Tensor, proxy_vec: torch.Tensor) -> torch.Tensor:
        logits = self.forward(expression_vec, proxy_vec)
        return torch.softmax(logits, dim=1)


class EngagementModel(nn.Module):
    """End-to-end wrapper: thermal image + proxy vector -> engagement logits."""

    def __init__(self, cnn_encoder: nn.Module, fusion_classifier: FusionClassifier):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.fusion_classifier = fusion_classifier

    def forward(self, image: torch.Tensor, proxy_vec: torch.Tensor) -> torch.Tensor:
        expression_vec = self.cnn_encoder(image)
        return self.fusion_classifier(expression_vec, proxy_vec)
