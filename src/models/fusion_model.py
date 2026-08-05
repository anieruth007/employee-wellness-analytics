"""Classifier head — ResNet18 (frozen, room-temperature-pretrained) expression features
[512-dim] + raw ROI temperature vector [5-dim] -> 3-way engagement classifier.

Reintroduces ROI-derived input into the classifier (previously removed entirely — see git
history around "Fix class-imbalance handling" / target-leakage — when the input was the
binary N/C proxy that directly determines the training label via
synthesize_engagement_label(), causing ~100% accuracy through trivial label inversion).

This time the classifier receives raw_temperature_vector()'s continuous ROI temperatures
(nose/forehead/periorbital/lip + differential index) rather than the thresholded binary
proxy. This is less directly leaky — these are a superset of information the binary proxy
is thresholded FROM, not the literal label-generating value — but they're still
correlated with the label, since the label is itself a threshold function of these same
quantities. Worth scrutinizing results with this in mind, not treating it as leakage-free.
"""
import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    """(N, 512) expression features + (N, 5) raw ROI temperatures -> (N, 3) engagement logits.

    Returns raw logits (no softmax) so training can use nn.CrossEntropyLoss directly —
    softmax is applied separately at inference time via `predict_proba`, to avoid a
    double-softmax when training with CrossEntropyLoss (which applies log-softmax internally).
    """

    def __init__(self, expression_dim: int = 512, roi_dim: int = 5, hidden_dims=(128, 64), num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        fused_dim = expression_dim + roi_dim
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

    def forward(self, expression_vec: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([expression_vec, roi_features], dim=1)
        return self.net(fused)

    @torch.no_grad()
    def predict_proba(self, expression_vec: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(expression_vec, roi_features)
        return torch.softmax(logits, dim=1)


class EngagementModel(nn.Module):
    """End-to-end wrapper: thermal image + raw ROI temperature vector -> engagement logits.

    `cnn_encoder` (the ResNet18 backbone, see src/models/resnet_backbone.py::load_frozen_backbone)
    is expected to be frozen (requires_grad=False on all its params). Its forward pass runs
    under torch.no_grad() here regardless, and the train()/eval() override below keeps it
    pinned in eval() mode even when the overall model is in train() mode for the
    classifier head — without this, BatchNorm layers inside the backbone would keep
    updating their running statistics from new data despite being "frozen" (requires_grad
    only blocks gradient updates to the affine parameters, not BatchNorm's running-stat
    tracking, which happens unconditionally in train mode).
    """

    def __init__(self, cnn_encoder: nn.Module, fusion_classifier: FusionClassifier):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.fusion_classifier = fusion_classifier

    def forward(self, image: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            expression_vec = self.cnn_encoder(image)
        return self.fusion_classifier(expression_vec, roi_features)

    def train(self, mode: bool = True) -> "EngagementModel":
        super().train(mode)
        self.cnn_encoder.eval()  # frozen backbone always stays in eval mode
        return self
