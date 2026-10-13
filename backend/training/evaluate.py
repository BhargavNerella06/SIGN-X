"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python -m training.evaluate --config training/config.yaml
    python -m training.evaluate --config training/config.yaml --checkpoint models/isl_transformer.pt

Writes:
    training/logs/evaluation_results.json
    training/logs/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader

from app.config import BACKEND_DIR
from app.models.classifier import build_model
from app.utils.logging import get_logger
from training.dataset import LandmarkDataset, collate_fn

logger = get_logger(__name__)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader: DataLoader, device: str, num_classes: int) -> dict:
    all_labels, all_preds, all_probs, latencies_ms = [], [], [], []

    model.eval()
    with torch.no_grad():
        for sequences, masks, labels in loader:
            sequences, masks = sequences.to(device), masks.to(device)

            start = time.perf_counter()
            logits = model(sequences, mask=masks)
            latencies_ms.append((time.perf_counter() - start) * 1000.0 / sequences.size(0))

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)

    accuracy = float((all_preds_arr == all_labels_arr).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_arr, all_preds_arr, average="macro", zero_division=0
    )

    k = min(5, num_classes)
    top5 = float(
        top_k_accuracy_score(all_labels_arr, all_probs, k=k, labels=list(range(num_classes)))
    )

    cm = confusion_matrix(all_labels_arr, all_preds_arr, labels=list(range(num_classes)))
    per_class_acc = {}
    for i in range(num_classes):
        total = cm[i].sum()
        per_class_acc[i] = float(cm[i, i] / total) if total > 0 else None

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "top_1_accuracy": accuracy,
        f"top_{k}_accuracy": top5,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "avg_inference_latency_ms": float(np.mean(latencies_ms)),
        "num_test_samples": int(len(all_labels_arr)),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.5), max(5, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SIGN-X checkpoint.")
    parser.add_argument("--config", type=str, default="training/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = BACKEND_DIR / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg["training"]["device"])
    checkpoint_dir = BACKEND_DIR / cfg["training"]["checkpoint_dir"]
    default_ckpt_name = "isl_transformer.pt" if cfg["model"]["type"] == "transformer" else "isl_bilstm.pt"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / default_ckpt_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run training/train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model_cfg = checkpoint["config"]

    model = build_model(
        model_type=checkpoint["model_type"],
        input_dim=checkpoint["input_dim"],
        num_classes=len(class_names),
        embedding_dim=model_cfg["embedding_dim"],
        transformer_layers=model_cfg["transformer_layers"],
        transformer_heads=model_cfg["transformer_heads"],
        transformer_ff_dim=model_cfg["transformer_ff_dim"],
        bilstm_hidden_dim=model_cfg["bilstm_hidden_dim"],
        bilstm_layers=model_cfg["bilstm_layers"],
        dropout=model_cfg["dropout"],
        max_seq_len=model_cfg["sequence_length"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    dataset_dir = BACKEND_DIR / cfg["dataset"]["dataset_path"] / cfg["dataset"]["name"]
    test_ds = LandmarkDataset(dataset_dir, "test", model_cfg["sequence_length"], class_names=class_names)
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    logger.info("Evaluating %s on %d test samples (%d classes)", checkpoint_path, len(test_ds), len(class_names))
    results = evaluate(model, test_loader, device, len(class_names))
    results["checkpoint"] = str(checkpoint_path)
    results["class_names"] = class_names

    log_dir = BACKEND_DIR / cfg["training"]["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    results_path = log_dir / f"evaluation_results_{checkpoint['model_type']}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm_path = log_dir / f"confusion_matrix_{checkpoint['model_type']}.png"
    plot_confusion_matrix(np.array(results["confusion_matrix"]), class_names, cm_path)

    logger.info(
        "Evaluation complete | accuracy=%.4f f1_macro=%.4f | results: %s | confusion matrix: %s",
        results["accuracy"], results["f1_macro"], results_path, cm_path,
    )


if __name__ == "__main__":
    main()
