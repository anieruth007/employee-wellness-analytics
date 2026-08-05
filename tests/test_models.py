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
    classifier = FusionClassifier(expression_dim=512, roi_dim=5, hidden_dims=(128, 64), num_classes=3, dropout=0.3)
    expr = torch.randn(4, 512)
    roi_features = torch.randn(4, 5)

    logits = classifier(expr, roi_features)
    assert logits.shape == (4, 3)

    probs = classifier.predict_proba(expr, roi_features)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


def test_engagement_model_end_to_end_shape():
    # ThermalCNNEncoder stands in for the real (ResNet18) backbone here — EngagementModel
    # is generic over any encoder module, so this keeps the test fast/network-free while
    # still verifying the forward(image, roi_features) plumbing and the frozen-backbone
    # train()/eval() override.
    encoder = ThermalCNNEncoder(feature_dim=512)
    for p in encoder.parameters():
        p.requires_grad = False
    classifier = FusionClassifier(expression_dim=512, roi_dim=5, hidden_dims=(128, 64), num_classes=3, dropout=0.3)
    model = EngagementModel(encoder, classifier)

    image = torch.randn(2, 1, 48, 48)
    roi_features = torch.randn(2, 5)
    logits = model(image, roi_features)
    assert logits.shape == (2, 3)

    model.train()
    assert not model.cnn_encoder.training  # frozen backbone must stay in eval mode
    assert model.fusion_classifier.training
