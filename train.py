"""Train the end-to-end thermal engagement model (CNN encoder + fusion classifier).

BiLSTM removal means the model is small enough (48x48 input) to train jointly in a
single stage on a 4GB-VRAM GPU — no separate CNN/fusion training stages needed.

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

from src.data.thermal_dataset import ThermalDataset, build_weighted_sampler, compute_class_weights
from src.models.fusion_model import EngagementModel, FusionClassifier
from src.models.thermal_cnn import ThermalCNNEncoder

CLASS_NAMES = ["Disengaged", "Neutral", "Engaged"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(cnn_cfg: dict, fusion_cfg: dict):
    dataset = ThermalDataset(raw_dir=cnn_cfg["data"]["raw_dir"], input_size=cnn_cfg["model"]["input_size"])

    val_split = fusion_cfg["data"]["val_split"]
    test_split = fusion_cfg["data"]["test_split"]
    n = len(dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(fusion_cfg["data"]["seed"])
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    # Disengaged is a minority class — oversample it in training via WeightedRandomSampler
    # only (not also via a weighted loss, which would double-correct for imbalance).
    # Val/test loaders stay unweighted for an honest evaluation signal.
    class_weights = compute_class_weights([dataset.labels[i] for i in train_set.indices])
    sampler = build_weighted_sampler(dataset.labels, train_set.indices)

    batch_size = cnn_cfg["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, class_weights


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    for images, proxies, labels in loader:
        images, proxies, labels = images.to(device), proxies.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, proxies)
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

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cnn_cfg, fusion_cfg)
    print("Class weights (inverse frequency, applied to the WeightedRandomSampler only — "
          "not also to the loss, to avoid double-correcting for imbalance):")
    for name, weight in zip(CLASS_NAMES, class_weights.tolist()):
        print(f"  {name}: {weight:.4f}")

    encoder = ThermalCNNEncoder(feature_dim=cnn_cfg["model"]["feature_dim"])
    classifier = FusionClassifier(
        expression_dim=fusion_cfg["model"]["expression_dim"],
        proxy_dim=fusion_cfg["model"]["proxy_dim"],
        hidden_dims=tuple(fusion_cfg["model"]["hidden_dims"]),
        num_classes=fusion_cfg["model"]["num_classes"],
        dropout=fusion_cfg["model"]["dropout"],
    )
    model = EngagementModel(encoder, classifier).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=fusion_cfg["training"]["lr"], weight_decay=fusion_cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fusion_cfg["training"]["epochs"])

    checkpoint_dir = Path(fusion_cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(fusion_cfg["training"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"epoch {epoch+1}/{fusion_cfg['training']['epochs']} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal test_loss={test_loss:.4f} test_acc={test_acc:.3f}")


if __name__ == "__main__":
    main()
