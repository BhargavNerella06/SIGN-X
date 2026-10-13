"""Train the isolated-sign classifier (Transformer or BiLSTM).

Usage:
    python -m training.train --config training/config.yaml

Produces:
    <checkpoint_dir>/isl_transformer.pt (or isl_bilstm.pt)  — best-val-accuracy checkpoint (used by the running app)
    <checkpoint_dir>/classes.json                            — class list, kept in sync with the checkpoint
    <log_dir>/<run_name>/
        config.yaml               — exact resolved config used (reproducibility, Section 11)
        best_model.pt             — copy of the best checkpoint
        last_model.pt             — final-epoch weights (for overfitting inspection)
        classes.json
        training_history.json    — per-epoch train/val loss & accuracy
        metrics.json              — best epoch's metrics
        experiment_summary.json  — dataset/model/training summary + git commit
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from app.config import BACKEND_DIR, settings
from app.models.classifier import build_model
from app.utils.logging import get_logger
from training.augmentation import AugmentationConfig, build_augmentor
from training.dataset import LandmarkDataset, collate_fn

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=BACKEND_DIR,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _compute_class_weights(dataset: LandmarkDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights: weight_c = N / (num_classes * count_c).
    Documented explicitly (Section 8) rather than silently rebalancing —
    see docs/model.md for when/why this is enabled per experiment."""
    counts = torch.zeros(num_classes)
    for _, label in dataset.samples:
        counts[label] += 1
    counts = counts.clamp(min=1)
    n = counts.sum()
    weights = n / (num_classes * counts)
    return weights


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: str, train: bool) -> tuple[float, float]:
    model.train(mode=train)
    total_loss, total_correct, total_count = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for sequences, masks, labels in loader:
            sequences, masks, labels = sequences.to(device), masks.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(sequences, mask=masks)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SIGN-X isolated-sign classifier.")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BACKEND_DIR / args.config if not config_path.exists() else config_path
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = resolve_device(cfg["training"]["device"])
    logger.info("Training on device: %s", device)

    dataset_dir = BACKEND_DIR / cfg["dataset"]["dataset_path"] / cfg["dataset"]["name"]
    sequence_length = cfg["dataset"]["sequence_length"]

    aug_cfg = AugmentationConfig(**cfg.get("augmentation", {})) if cfg.get("augmentation", {}).get("enabled", True) else AugmentationConfig(enabled=False)
    augmentor = build_augmentor(aug_cfg, seed=cfg["training"]["seed"])

    train_ds = LandmarkDataset(dataset_dir, "train", sequence_length, augment=augmentor)
    val_ds = LandmarkDataset(dataset_dir, "validation", sequence_length, class_names=train_ds.class_names)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    input_dim = train_ds[0][0].shape[-1]
    num_classes = len(train_ds.class_names)
    logger.info("Dataset: %d classes, %d train samples, %d validation samples, input_dim=%d",
                num_classes, len(train_ds), len(val_ds), input_dim)

    model_cfg = cfg["model"]
    model = build_model(
        model_type=model_cfg["type"],
        input_dim=input_dim,
        num_classes=num_classes,
        embedding_dim=model_cfg["embedding_dim"],
        transformer_layers=model_cfg["transformer_layers"],
        transformer_heads=model_cfg["transformer_heads"],
        transformer_ff_dim=model_cfg["transformer_ff_dim"],
        bilstm_hidden_dim=model_cfg["bilstm_hidden_dim"],
        bilstm_layers=model_cfg["bilstm_layers"],
        dropout=model_cfg["dropout"],
        max_seq_len=sequence_length,
    ).to(device)

    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        class_weights = _compute_class_weights(train_ds, num_classes).to(device)
        logger.info("Using inverse-frequency class weights: %s",
                    {train_ds.class_names[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())})
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["training"]["lr_scheduler_factor"],
        patience=cfg["training"]["lr_scheduler_patience"],
    )

    checkpoint_dir = BACKEND_DIR / cfg["training"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "isl_transformer.pt" if model_cfg["type"] == "transformer" else "isl_bilstm.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name

    run_name = f"{model_cfg['type']}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = BACKEND_DIR / cfg["training"]["log_dir"] / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    # Reproducibility record (Section 11): the exact resolved config used.
    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    history = []
    best_val_acc = -1.0
    best_epoch_metrics: dict = {}
    epochs_without_improvement = 0
    patience = cfg["training"]["early_stopping_patience"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.2e",
            epoch, cfg["training"]["epochs"], train_loss, train_acc, val_loss, val_acc, current_lr,
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_epoch_metrics = {"val_accuracy": val_acc, "val_loss": val_loss, "train_accuracy": train_acc,
                                   "train_loss": train_loss, "epoch": epoch}
            _save_checkpoint(checkpoint_path, model, model_cfg, train_ds.class_names, input_dim,
                              sequence_length, best_epoch_metrics, cfg["dataset"]["name"])
            shutil.copy2(checkpoint_path, log_dir / "best_model.pt")
            logger.info("New best model saved (val_acc=%.4f) -> %s", val_acc, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement", patience)
                break

    # last_model.pt: the final epoch's weights, regardless of whether they
    # were the best (useful to check for overfitting past the best epoch).
    final_epoch = history[-1]["epoch"] if history else 0
    _save_checkpoint(
        log_dir / "last_model.pt", model, model_cfg, train_ds.class_names, input_dim, sequence_length,
        {"val_accuracy": history[-1]["val_acc"], "val_loss": history[-1]["val_loss"], "epoch": final_epoch} if history else {},
        cfg["dataset"]["name"],
    )

    with open(log_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(log_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_epoch_metrics, f, indent=2)

    experiment_summary = {
        "run_name": run_name,
        "model_type": model_cfg["type"],
        "dataset": cfg["dataset"]["name"],
        "num_classes": num_classes,
        "class_names": train_ds.class_names,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "input_dim": input_dim,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "seed": cfg["training"]["seed"],
        "git_commit": get_git_commit(),
        "epochs_trained": final_epoch,
        "best_epoch_metrics": best_epoch_metrics,
        "config_path": str(log_dir / "config.yaml"),
        "checkpoint_path": str(checkpoint_path),
    }
    with open(log_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)

    logger.info("Training complete. Best val_acc=%.4f. Experiment artifacts: %s", best_val_acc, log_dir)


def _save_checkpoint(path: Path, model: nn.Module, model_cfg: dict, class_names: list[str],
                      input_dim: int, sequence_length: int, metrics: dict,
                      dataset_name: str = "") -> None:
    torch.save({
        "model_type": model_cfg["type"],
        "input_dim": input_dim,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "config": {**model_cfg, "sequence_length": sequence_length},
        "metrics": metrics,
        "dataset_name": dataset_name,
    }, path)

    with open(path.parent / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names}, f, indent=2)


if __name__ == "__main__":
    main()
