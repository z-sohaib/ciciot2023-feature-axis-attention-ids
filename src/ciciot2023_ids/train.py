from __future__ import annotations

import json
import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ciciot2023_ids.config import load_config
from ciciot2023_ids.data import load_capped_dataset, make_splits
from ciciot2023_ids.metrics import classification_metrics
from ciciot2023_ids.models.factory import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CICIoT2023 model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
    return 0


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return run_config(config)


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    processed_path = Path(config["data"].get("processed_path", ""))
    if processed_path and processed_path.exists():
        x_train, x_val, x_test, y_train, y_val, y_test, class_names = load_processed_dataset(processed_path)
        print(f"Loaded processed dataset from {processed_path}")
    else:
        frame = load_capped_dataset(config)
        x_train, x_val, x_test, y_train, y_val, y_test, class_names = make_splits(frame, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, input_features=x_train.shape[1], num_classes=len(class_names)).to(device)
    training_config = config["training"]
    criterion = build_loss(y_train, len(class_names), training_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    train_loader = make_loader(x_train, y_train, int(training_config["batch_size"]), shuffle=True)
    val_loader = make_loader(x_val, y_val, int(training_config["batch_size"]), shuffle=False)
    test_loader = make_loader(x_test, y_test, int(training_config["batch_size"]), shuffle=False)

    best_state = None
    best_val_macro_f1 = -1.0
    stale_epochs = 0
    history = []
    start = time.perf_counter()

    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_pred = predict(model, val_loader, device)
        val_metrics = classification_metrics(y_val, val_pred, class_names)
        val_macro_f1 = val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_macro_f1})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_config["early_stopping_patience"]):
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    latency_metrics = measure_inference(model, test_loader, device)
    y_pred = predict(model, test_loader, device)
    metrics = classification_metrics(y_test, y_pred, class_names)
    metrics.update(
        {
            "experiment": config["experiment"]["name"],
            "architecture": config["model"]["architecture"],
            "class_names": class_names,
            "best_val_macro_f1": best_val_macro_f1,
            "training_seconds": time.perf_counter() - start,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(device),
            **latency_metrics,
            "history": history,
        }
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Saved metrics to {run_dir / 'metrics.json'}")
    return metrics


def load_processed_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=False)
    return (
        data["x_train"].astype(np.float32),
        data["x_val"].astype(np.float32),
        data["x_test"].astype(np.float32),
        data["y_train"].astype(np.int64),
        data["y_val"].astype(np.int64),
        data["y_test"].astype(np.int64),
        [str(value) for value in data["class_names"].tolist()],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total_examples += labels.numel()
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []
    for features, _ in loader:
        logits = model(features.to(device))
        predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions)


@torch.no_grad()
def measure_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_examples = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for features, _ in loader:
        logits = model(features.to(device))
        total_examples += logits.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "inference_seconds": float(elapsed),
        "throughput_samples_per_second": float(total_examples / elapsed) if elapsed > 0 else 0.0,
        "latency_ms_per_sample": float((elapsed / total_examples) * 1000.0) if total_examples > 0 else 0.0,
    }


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor_features = torch.from_numpy(features).unsqueeze(1)
    tensor_labels = torch.from_numpy(labels.astype(np.int64))
    return DataLoader(TensorDataset(tensor_features, tensor_labels), batch_size=batch_size, shuffle=shuffle)


def build_loss(y_train: np.ndarray, num_classes: int, config: dict[str, Any], device: torch.device) -> nn.Module:
    class_weight = bool(config.get("class_weight", True))
    label_smoothing = float(config.get("label_smoothing", 0.0))
    loss_name = str(config.get("loss", "cross_entropy")).lower()
    weights = None
    if class_weight:
        weights = torch.tensor(
            compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train),
            dtype=torch.float32,
            device=device,
        )
    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(weight=weights, gamma=float(config.get("focal_gamma", 2.0)))
    raise ValueError(f"Unsupported loss: {loss_name}")


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
