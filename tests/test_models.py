import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fusion_model import EngagementModel, FusionClassifier
from src.models.thermal_cnn import ThermalCNNEncoder


def test_thermal_cnn_encoder_output_shape():
    encoder = ThermalCNNEncoder(feature_dim=256)
    x = torch.randn(4, 1, 48, 48)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_fusion_classifier_output_shape():
    classifier = FusionClassifier(expression_dim=256, proxy_dim=2, hidden_dims=(128, 64), num_classes=3, dropout=0.3)
    expr = torch.randn(4, 256)
    proxy = torch.randn(4, 2)

    logits = classifier(expr, proxy)
    assert logits.shape == (4, 3)

    probs = classifier.predict_proba(expr, proxy)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


def test_engagement_model_end_to_end_shape():
    encoder = ThermalCNNEncoder(feature_dim=256)
    classifier = FusionClassifier(expression_dim=256, proxy_dim=2, hidden_dims=(128, 64), num_classes=3, dropout=0.3)
    model = EngagementModel(encoder, classifier)

    image = torch.randn(2, 1, 48, 48)
    proxy = torch.randn(2, 2)
    logits = model(image, proxy)
    assert logits.shape == (2, 3)
