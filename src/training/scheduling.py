"""Shared LR scheduling — cosine annealing with an optional linear warmup phase.
Used by both train.py (engagement classifier) and scripts/train_room_temp_backbone.py.
"""
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    """Cosine annealing with a linear warmup phase (configs/cnn_config.yaml: scheduler:
    cosine_with_warmup). warmup_epochs<=0 collapses to plain cosine annealing.
    """
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs)

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
