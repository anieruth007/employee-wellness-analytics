"""ResNet18 backbone adapted for 1-channel thermal input, pretrained on ImageNet.

Used to pretrain a thermal-pattern feature extractor on the room-temperature-condition
task (large, ~10k images, cleanly labeled directly from filenames — see
src/data/thermal_dataset.py's RoomTempDataset) before reuse as the expression/thermal
feature extractor for engagement classification (small, ~6.6k images, noisily labeled
via ROI-threshold proxy). See scripts/train_room_temp_backbone.py for the pretraining loop.
"""
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

DEFAULT_BACKBONE_CHECKPOINT = Path("checkpoints/room_temp_backbone/backbone_only.pt")


def build_resnet18_1ch(pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ResNet18 with its first conv layer adapted to 1-channel input
    (RGB filters averaged across the channel dim, preserving pretrained low-level feature
    detectors) and its final FC layer removed — outputs raw 512-dim pooled features.

    NOTE: ResNet18 was designed/pretrained on 224x224 inputs; our thermal pipeline uses
    48x48 images. The architecture runs fine at 48x48 (AdaptiveAvgPool2d handles any
    spatial size), but the pretrained filters' effective receptive fields won't be as
    well-matched to the input as they were at the original resolution — a known caveat of
    reusing ImageNet backbones on much smaller inputs, not something this function corrects.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv

    model.fc = nn.Identity()  # remove classification head -> raw 512-dim pooled features
    return model


def load_frozen_backbone(
    checkpoint_path: Union[str, Path] = DEFAULT_BACKBONE_CHECKPOINT,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load the room-temperature-pretrained ResNet18 backbone and freeze all its
    parameters — used as a fixed feature extractor for engagement classification.

    Builds the architecture WITHOUT downloading ImageNet weights (pretrained=False) since
    the real weights being loaded here are the room-temperature-pretrained ones, not raw
    ImageNet — downloading ImageNet weights just to immediately overwrite them would be
    wasted work.
    """
    backbone = build_resnet18_1ch(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device or "cpu")
    backbone.load_state_dict(state_dict)
    # map_location only affects where torch.load() initially materializes the checkpoint
    # tensors — load_state_dict() copies values in-place into the module's existing
    # (CPU-resident) parameters regardless, so the module itself must still be moved
    # explicitly. Relying on a later `.to(device)` on some wrapping module is what made
    # this look like it worked in train.py/inference.py — fixed properly here instead.
    if device is not None:
        backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone


class RoomTempClassifier(nn.Module):
    """ResNet18 (1-channel) backbone + FC(512->num_classes) head, for room-temperature-
    condition pretraining. After training, `.backbone` is saved separately (no head) and
    reused as a feature extractor.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.backbone = build_resnet18_1ch(pretrained=pretrained)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
