"""Ablation study: how much does each input stream contribute to engagement
classification accuracy?

  Variant A — CNN (ResNet18 backbone) features only, 512-dim, no ROI temperatures
  Variant B — raw ROI temperatures only, 5-dim, no CNN features
  Full model (both, 517-dim) — already trained via train.py, reported here for comparison

Same frozen backbone, same data split, same training hyperparameters as train.py — only
the classifier head's input is ablated, so any accuracy difference is attributable to
what each stream contributes (and, per the leakage discussion, how much of the ROI
stream's contribution is just reconstructing the label-generating threshold rule from
its raw ingredients).

Usage:
    python scripts/run_ablation.py
"""
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.fusion_model import FusionClassifier
from src.models.resnet_backbone import load_frozen_backbone
from src.roi.labeling import CLASS_NAMES
from src.training.scheduling import build_scheduler
from train import build_dataloaders, load_config, set_seed

FULL_MODEL_TEST_ACC = 0.824  # from train.py's run: 512 (CNN) + 5 (ROI) -> 82.4%


def run_epoch_ablation(encoder, classifier, loader, criterion, optimizer, device, train: bool, use_expression: bool, use_roi: bool):
    classifier.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    for images, roi_features, labels in loader:
        images, roi_features, labels = images.to(device), roi_features.to(device), labels.to(device)
        batch_size = labels.size(0)

        if use_expression:
            with torch.no_grad():
                expression_vec = encoder(images)
        else:
            expression_vec = torch.empty(batch_size, 0, device=device)
        roi_in = roi_features if use_roi else torch.empty(batch_size, 0, device=device)

        with torch.set_grad_enabled(train):
            logits = classifier(expression_vec, roi_in)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def run_variant(name, expression_dim, roi_dim, use_expression, use_roi, encoder, cnn_cfg, fusion_cfg,
                 train_loader, val_loader, test_loader, device):
    print(f"\n{'='*70}\n{name}  (expression_dim={expression_dim}, roi_dim={roi_dim})\n{'='*70}")
    set_seed(fusion_cfg["data"]["seed"])

    classifier = FusionClassifier(
        expression_dim=expression_dim,
        roi_dim=roi_dim,
        hidden_dims=tuple(fusion_cfg["model"]["hidden_dims"]),
        num_classes=fusion_cfg["model"]["num_classes"],
        dropout=fusion_cfg["model"]["dropout"],
    ).to(device)

    train_cfg = cnn_cfg["training"]
    total_epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    patience = train_cfg["early_stopping_patience"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, total_epochs, warmup_epochs)

    checkpoint_dir = Path(f"checkpoints/ablation_{name.split(':')[0].strip().lower().replace(' ', '_')}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(total_epochs):
        train_loss, train_acc = run_epoch_ablation(encoder, classifier, train_loader, criterion, optimizer, device, True, use_expression, use_roi)
        val_loss, val_acc = run_epoch_ablation(encoder, classifier, val_loader, criterion, optimizer, device, False, use_expression, use_roi)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{total_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(classifier.state_dict(), checkpoint_dir / "best_val_loss_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    classifier.load_state_dict(torch.load(checkpoint_dir / "best_val_loss_model.pt", map_location=device))

    test_loss, test_acc = run_epoch_ablation(encoder, classifier, test_loader, criterion, optimizer, device, False, use_expression, use_roi)
    print(f"  Final test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    from sklearn.metrics import classification_report, confusion_matrix

    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, roi_features, labels in test_loader:
            images, roi_features = images.to(device), roi_features.to(device)
            batch_size = labels.size(0)
            expr = encoder(images) if use_expression else torch.empty(batch_size, 0, device=device)
            roi_in = roi_features if use_roi else torch.empty(batch_size, 0, device=device)
            logits = classifier(expr, roi_in)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    num_classes = fusion_cfg["model"]["num_classes"]
    class_indices = list(range(num_classes))
    print(f"\n  Confusion matrix ({name}), order = {CLASS_NAMES}:")
    print(confusion_matrix(all_labels, all_preds, labels=class_indices))
    print(f"\n  Per-class metrics ({name}):")
    print(classification_report(all_labels, all_preds, labels=class_indices, target_names=CLASS_NAMES, zero_division=0))

    return test_acc


def main():
    cnn_cfg = load_config("configs/cnn_config.yaml")
    fusion_cfg = load_config("configs/fusion_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _ = build_dataloaders(cnn_cfg, fusion_cfg)
    print(f"Split sizes: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    encoder = load_frozen_backbone(fusion_cfg["training"]["backbone_checkpoint"], device=device)
    encoder.eval()

    results = {"Full model (CNN 512 + ROI 5)": FULL_MODEL_TEST_ACC}

    results["Variant A: CNN only (512)"] = run_variant(
        "Variant A: CNN only", expression_dim=512, roi_dim=0, use_expression=True, use_roi=False,
        encoder=encoder, cnn_cfg=cnn_cfg, fusion_cfg=fusion_cfg,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device,
    )

    results["Variant B: ROI only (5)"] = run_variant(
        "Variant B: ROI only", expression_dim=0, roi_dim=5, use_expression=False, use_roi=True,
        encoder=encoder, cnn_cfg=cnn_cfg, fusion_cfg=fusion_cfg,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device,
    )

    print(f"\n{'='*70}\nAblation summary (test accuracy)\n{'='*70}")
    for name, acc in results.items():
        print(f"  {name}: {acc:.3f}")


if __name__ == "__main__":
    main()
