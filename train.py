"""Train the engagement classifier head on top of a FROZEN, room-temperature-pretrained
ResNet18 backbone (see scripts/train_room_temp_backbone.py for the backbone pretraining
stage, and src/models/resnet_backbone.py::load_frozen_backbone).

Only the classifier head's parameters are updated — the backbone never sees a gradient.

Usage:
    python train.py
"""
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data.preprocessing import TRAIN_TRANSFORM
from src.data.thermal_dataset import ThermalDataset, build_weighted_sampler, compute_class_weights
from src.models.fusion_model import EngagementModel, FusionClassifier
from src.models.resnet_backbone import load_frozen_backbone
from src.roi.labeling import CLASS_NAMES
from src.training.scheduling import build_scheduler


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(cnn_cfg: dict, fusion_cfg: dict):
    raw_dir = cnn_cfg["data"]["raw_dir"]
    input_size = cnn_cfg["model"]["input_size"]
    seed = fusion_cfg["data"]["seed"]

    # Two ThermalDataset instances over the SAME underlying files/cache — one with
    # augmentation (TRAIN_TRANSFORM) for the training split, one without (default
    # EVAL_TRANSFORM) for val/test. Since random_split's index partitioning depends only
    # on dataset length + generator seed (not the transform), calling it twice with
    # freshly-seeded generators of the same seed value reproduces the identical split on
    # both instances — we just take the train indices from one and val/test from the other.
    train_dataset = ThermalDataset(raw_dir=raw_dir, input_size=input_size, transform=TRAIN_TRANSFORM)
    eval_dataset = ThermalDataset(raw_dir=raw_dir, input_size=input_size)

    val_split = fusion_cfg["data"]["val_split"]
    test_split = fusion_cfg["data"]["test_split"]
    n = len(train_dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_set, _, _ = random_split(train_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
    _, val_set, test_set = random_split(eval_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

    # Disengaged is a minority class — oversample it in training via WeightedRandomSampler
    # only (not also via a weighted loss, which would double-correct for imbalance).
    # Val/test loaders stay unweighted for an honest evaluation signal.
    num_classes = fusion_cfg["model"]["num_classes"]
    class_weights = compute_class_weights([train_dataset.labels[i] for i in train_set.indices], num_classes=num_classes)
    sampler = build_weighted_sampler(train_dataset.labels, train_set.indices, num_classes=num_classes)

    batch_size = cnn_cfg["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, class_weights


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    for images, roi_features, labels in loader:
        images, roi_features, labels = images.to(device), roi_features.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, roi_features)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    cnn_cfg = load_config("configs/cnn_config.yaml")
    fusion_cfg = load_config("configs/fusion_config.yaml")
    set_seed(fusion_cfg["data"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = fusion_cfg["model"]["num_classes"]
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cnn_cfg, fusion_cfg)
    print("Class weights (inverse frequency, applied to the WeightedRandomSampler only — "
          "not also to the loss, to avoid double-correcting for imbalance):")
    for name, weight in zip(CLASS_NAMES, class_weights.tolist()):
        print(f"  {name}: {weight:.4f}")

    backbone_checkpoint = fusion_cfg["training"]["backbone_checkpoint"]
    encoder = load_frozen_backbone(backbone_checkpoint, device=device)
    print(f"\nLoaded frozen backbone from {backbone_checkpoint} "
          f"({sum(p.numel() for p in encoder.parameters())} params, all frozen)")

    classifier = FusionClassifier(
        expression_dim=fusion_cfg["model"]["expression_dim"],
        roi_dim=fusion_cfg["model"]["roi_dim"],
        hidden_dims=tuple(fusion_cfg["model"]["hidden_dims"]),
        num_classes=fusion_cfg["model"]["num_classes"],
        dropout=fusion_cfg["model"]["dropout"],
    )
    model = EngagementModel(encoder, classifier).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params (classifier head only): {sum(p.numel() for p in trainable_params)}")

    # Training-loop hyperparameters (epochs, lr, scheduler, early stopping, batch_size)
    # live in configs/cnn_config.yaml — the single authoritative source for this training
    # loop. fusion_config.yaml's training section only holds fusion-specific fields
    # (checkpoint_dir, backbone_checkpoint).
    train_cfg = cnn_cfg["training"]
    total_epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    patience = train_cfg["early_stopping_patience"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, total_epochs, warmup_epochs)

    checkpoint_dir = Path(fusion_cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(total_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"epoch {epoch+1}/{total_epochs} "
              f"lr={optimizer.param_groups[0]['lr']:.6f} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

        # Early stopping on val_loss (smoother/less noisy than accuracy on a small,
        # imbalanced validation set). The best-val_loss checkpoint, not whatever epoch
        # patience happens to run out on, is what gets used for final reporting below —
        # by the time patience is exhausted the model has typically already started
        # overfitting for several epochs.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_val_loss_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} — val_loss hasn't improved in "
                      f"{patience} epochs (best val_loss={best_val_loss:.4f})")
                break

    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")

    # Reload the best-val_loss checkpoint for final test-set reporting, rather than
    # whatever epoch the loop happened to stop on.
    model.load_state_dict(torch.load(checkpoint_dir / "best_val_loss_model.pt", map_location=device))

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal test_loss={test_loss:.4f} test_acc={test_acc:.3f} (best-val_loss checkpoint)")

    # Raw accuracy is a misleading headline metric here — WeightedRandomSampler trains
    # the model toward a balanced effective prediction distribution, while the test set
    # follows the natural class distribution. Per-class precision/recall/F1 + confusion
    # matrix show whether the model learned anything about minority classes that accuracy hides.
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, roi_features, labels in test_loader:
            logits = model(images.to(device), roi_features.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    class_indices = list(range(num_classes))
    print(f"\nConfusion matrix (rows=true, cols=predicted), order = {CLASS_NAMES}:")
    print(confusion_matrix(all_labels, all_preds, labels=class_indices))
    print("\nPer-class metrics (best-val_loss checkpoint, test set):")
    print(classification_report(all_labels, all_preds, labels=class_indices, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    main()
