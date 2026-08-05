"""Pretrain a ResNet18 backbone on room-temperature-condition classification.

Charlotte-ThermalFace was captured at 4 room-temperature conditions, encoded directly in
each filename (see src/data/thermal_dataset.py::parse_room_condition). This gives a much
larger (~10.1k images, vs ~6.6k for the proxy-labeled engagement task) and cleanly-labeled
task — no face detection needed, since the label comes from the filename, not the image
content — used here to pretrain a better thermal-pattern feature extractor than training a
small CNN from scratch on the noisier, proxy-derived engagement labels.

Usage:
    python scripts/train_room_temp_backbone.py
"""
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import TRAIN_TRANSFORM
from src.data.thermal_dataset import RoomTempDataset, build_weighted_sampler, compute_class_weights
from src.models.resnet_backbone import RoomTempClassifier
from src.training.scheduling import build_scheduler

CLASS_NAMES = ["Condition 1", "Condition 2", "Condition 3", "Condition 4"]
NUM_CLASSES = 4
CHECKPOINT_DIR = Path("checkpoints/room_temp_backbone")


def build_dataloaders(raw_dir: str, input_size: int, batch_size: int, val_split: float, test_split: float, seed: int):
    # Two RoomTempDataset instances over the same files — one augmented (training split),
    # one not (val/test) — with independently-seeded identical random_split calls so both
    # instances partition into the same indices. Same trick as train.py's build_dataloaders.
    train_dataset = RoomTempDataset(raw_dir=raw_dir, input_size=input_size, transform=TRAIN_TRANSFORM)
    eval_dataset = RoomTempDataset(raw_dir=raw_dir, input_size=input_size)

    n = len(train_dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_set, _, _ = random_split(train_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
    _, val_set, test_set = random_split(eval_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

    class_weights = compute_class_weights([train_dataset.labels[i] for i in train_set.indices], num_classes=NUM_CLASSES)
    sampler = build_weighted_sampler(train_dataset.labels, train_set.indices, num_classes=NUM_CLASSES)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, class_weights


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def per_class_accuracy(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    correct = Counter()
    total = Counter()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            for p, l in zip(preds.tolist(), labels.tolist()):
                total[l] += 1
                if p == l:
                    correct[l] += 1
    return {c: (correct[c] / total[c] if total[c] else 0.0) for c in range(num_classes)}


def main():
    with open("configs/cnn_config.yaml") as f:
        cnn_cfg = yaml.safe_load(f)

    raw_dir = cnn_cfg["data"]["raw_dir"]
    input_size = cnn_cfg["model"]["input_size"]
    seed = cnn_cfg["data"]["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Show class distribution BEFORE any training starts.
    probe_dataset = RoomTempDataset(raw_dir=raw_dir, input_size=input_size)
    total = len(probe_dataset)
    counts = Counter(probe_dataset.labels)
    print(f"\n=== Room-temperature condition class distribution ({total} images total) ===")
    for i, name in enumerate(CLASS_NAMES):
        c = counts.get(i, 0)
        print(f"  {name} (label {i}): {c} ({c / total:.1%})")

    batch_size = 64
    total_epochs = 50
    warmup_epochs = 5
    lr = 0.0001

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        raw_dir, input_size, batch_size=batch_size, val_split=0.15, test_split=0.15, seed=seed
    )
    print(f"\nSplit sizes: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print("\nClass weights (WeightedRandomSampler only, not the loss):")
    for name, w in zip(CLASS_NAMES, class_weights.tolist()):
        print(f"  {name}: {w:.4f}")

    model = RoomTempClassifier(num_classes=NUM_CLASSES, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = build_scheduler(optimizer, total_epochs, warmup_epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n=== Training ResNet18 for {total_epochs} epochs (fixed, no early stopping) ===")
    for epoch in range(total_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"epoch {epoch+1}/{total_epochs} lr={optimizer.param_groups[0]['lr']:.6f} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")

    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device))

    per_class_acc = per_class_accuracy(model, val_loader, device)
    print("\n=== Per-class accuracy (val set, best-val_loss checkpoint) ===")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {per_class_acc[i]:.3f}")

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    # Backbone alone (no classification head) — the reusable thermal-pattern feature
    # extractor for engagement classification.
    torch.save(model.backbone.state_dict(), CHECKPOINT_DIR / "backbone_only.pt")
    print(f"\nSaved backbone-only weights to {CHECKPOINT_DIR / 'backbone_only.pt'}")


if __name__ == "__main__":
    main()
