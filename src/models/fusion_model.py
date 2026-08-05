"""Classifier head — 256-dim CNN expression features -> 3-way engagement classifier.

The N/C personality-proxy vector is deliberately NOT fed into this classifier. The proxy
is computed via a deterministic threshold rule on ROI temperatures, and
synthesize_engagement_label() derives the training label from that exact same proxy —
so if the proxy were also handed to the classifier as input, the model could hit ~100%
accuracy by trivially inverting its own label-generation rule instead of learning
anything from the thermal image (confirmed: an earlier run with proxy_dim=2 concatenated
into this classifier hit val_acc=1.000 by epoch 4). The proxy still drives label
synthesis and the dashboard's personality-aware explanation — just never model input.
"""
import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """(N, 256) expression features -> (N, 3) engagement logits.

    Returns raw logits (no softmax) so training can use nn.CrossEntropyLoss directly —
    softmax is applied separately at inference time via `predict_proba`, to avoid a
    double-softmax when training with CrossEntropyLoss (which applies log-softmax internally).
    """

    def __init__(self, expression_dim: int = 256, hidden_dims=(128, 64), num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(expression_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, expression_vec: torch.Tensor) -> torch.Tensor:
        return self.net(expression_vec)

    @torch.no_grad()
    def predict_proba(self, expression_vec: torch.Tensor) -> torch.Tensor:
        logits = self.forward(expression_vec)
        return torch.softmax(logits, dim=1)


class EngagementModel(nn.Module):
    """End-to-end wrapper: thermal image -> engagement logits. Proxy is not part of the
    model at all — it's computed separately (see src/roi/) for labeling and explanation.
    """

    def __init__(self, cnn_encoder: nn.Module, fusion_classifier: FusionClassifier):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.fusion_classifier = fusion_classifier

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        expression_vec = self.cnn_encoder(image)
        return self.fusion_classifier(expression_vec)
