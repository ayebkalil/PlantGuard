from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

import mlflow
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT
from ml.preprocessing.preprocess import build_split_metadata

from ml.model import DEFAULT_BENCHMARK_MODELS, IMG, MEAN, STD, SUPPORTED_MODELS, build_model, build_transforms
from ml.registry import register_checkpoint, promote_checkpoint

SPLITS_DIR = Path("data/processed")
EXPERIMENT_NAME = "plantguard-model-benchmark"
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


class PlantDataset(Dataset):
    def __init__(self, samples, class_to_idx, tf):
        self.samples, self.c2i, self.tf = samples, class_to_idx, tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        with Image.open(s["image_path"]) as image:
            return self.tf(image.convert("RGB")), self.c2i[s["label"]]


@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> dict:
    """Run the model on a loader and return accuracy, macro/weighted F1, loss and raw predictions."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    loss_sum, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.numel()
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(1).tolist())
        total += y.numel()
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "precision": 0.0,
                "recall": 0.0, "loss": 0.0, "y_true": [], "y_pred": []}
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(precision),
        "recall": float(recall),
        "loss": loss_sum / total,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def measure_latency_ms(model: nn.Module, runs: int = 30, warmup: int = 5, size: int = IMG) -> dict:
    """Single-image CPU latency, the same situation as the FastAPI service."""
    cpu_model = model.to("cpu").eval()
    x = torch.randn(1, 3, size, size)
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            cpu_model(x)
        for _ in range(runs):
            start = time.perf_counter()
            cpu_model(x)
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"mean": sum(times) / len(times), "p95": times[int(0.95 * (len(times) - 1))]}


def _configure_mlflow(experiment_name: str = EXPERIMENT_NAME) -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    if tracking_uri.startswith("sqlite:///"):
        Path(tracking_uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def _load_splits(root: Path, rebuild: bool = False) -> dict:
    path = SPLITS_DIR / "splits.json"
    if rebuild or not path.is_file():
        build_split_metadata(root, SPLITS_DIR)
    splits = json.loads(path.read_text(encoding="utf-8"))
    for partition in ("train", "validation", "test"):
        for record in splits[partition]:
            image = (root / record["image_path"]).resolve()
            if not image.is_relative_to(root.resolve()) or not image.is_file():
                raise ValueError(f"Invalid prepared image path: {record['image_path']}")
            record["image_path"] = str(image)
    return splits


def _split_fingerprint(splits: dict) -> str:
    payload = {
        key: sorted((s["label"], s["sha256"]) for s in splits[key])
        for key in ("train", "validation", "test")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _log_confusion_matrix(y_true, y_pred, num_classes: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted class index")
    ax.set_ylabel("True class index")
    ax.set_title("Test confusion matrix (row-normalised)")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    mlflow.log_figure(fig, "test_confusion_matrix.png")
    plt.close(fig)


def train_model(dataset_root=None, epochs=10, batch_size=32, learning_rate=3e-4,
                output_dir=None, max_samples_per_class=None, model_name: str = "mobilenet_v3_small",
                checkpoint_name=None, log_to_mlflow: bool = True, owner: str | None = None,
                pretrained: bool = True, num_workers: int = 2, seed: int = 42,
                experiment_name: str = EXPERIMENT_NAME, rebuild_splits: bool = False,
                image_size: int = IMG, freeze_epochs: int = 0, patience: int = 0, amp: bool = False, prepared: bool = False) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    out_dir = Path(output_dir) if output_dir else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")
    pretrained = pretrained and model_name != "simple_cnn"

    # Each model gets its own files, so training one never overwrites another model's checkpoint.
    # Only run_model_experiments() promotes the winner to plantguard_model.pt.
    checkpoint_name = checkpoint_name or f"plantguard_model_{model_name}.pt"
    model_path = out_dir / checkpoint_name
    metadata_path = out_dir / f"model_metadata_{model_name}.json"

    splits = _load_splits(root, rebuild=rebuild_splits or not prepared)
    class_names = splits["classes"]
    c2i = {c: i for i, c in enumerate(class_names)}

    train_samples = splits["train"]
    if max_samples_per_class:
        by_cls = {}
        for s in train_samples:
            by_cls.setdefault(s["label"], []).append(s)
        train_samples = [s for v in by_cls.values() for s in v[:max_samples_per_class]]

    train_transform, eval_transform = build_transforms(image_size)
    use_amp = amp and device == "cuda"
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=device == "cuda",
                       persistent_workers=num_workers > 0)  # keeps workers alive between epochs (big win on Windows)
    train_dl = DataLoader(PlantDataset(train_samples, c2i, train_transform), shuffle=True,
                          generator=torch.Generator().manual_seed(seed), **loader_args)
    val_dl = DataLoader(PlantDataset(splits["validation"], c2i, eval_transform), **loader_args)
    test_dl = DataLoader(PlantDataset(splits["test"], c2i, eval_transform), **loader_args)

    model = build_model(len(class_names), model_name=model_name, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optional warm-up: train only the new classifier head for the first epochs (backbone frozen).
    # Skips most of the backward pass, so these epochs are ~3-4x faster.
    freeze_epochs = freeze_epochs if model_name != "simple_cnn" else 0
    backbone_frozen = freeze_epochs > 0
    if backbone_frozen:
        head = model.fc if hasattr(model, "fc") else model.classifier
        head_ids = {id(p) for p in head.parameters()}
        for p in model.parameters():
            p.requires_grad = id(p) in head_ids
    scaler = _make_grad_scaler(use_amp)

    run = None
    if log_to_mlflow:
        _configure_mlflow(experiment_name)
        run_name = f"{owner}-{model_name}" if owner else f"{model_name}-epochs{epochs}"
        run = mlflow.start_run(run_name=run_name)
        mlflow.set_tags({
            "owner": owner or "unknown",
            "architecture": model_name,
            "family": "from_scratch" if model_name == "simple_cnn" else "transfer_learning",
            "framework": "pytorch",
            "device": device,
        })
        mlflow.log_params({
            "model_name": model_name,
            "pretrained": pretrained,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "label_smoothing": 0.1,
            "image_size": image_size,
            "freeze_epochs": freeze_epochs,
            "patience": patience,
            "amp": use_amp,
            "seed": seed,
            "max_samples_per_class": max_samples_per_class,
            "dataset_root": root.as_posix(),
            "num_classes": len(class_names),
            "n_train": len(train_samples),
            "n_val": len(splits["validation"]),
            "n_test": len(splits["test"]),
            "split_hash": _split_fingerprint(splits),
            "total_params": total_params,
            "trainable_params": trainable_params,
        })

    best_f1, best_acc, best_epoch, epochs_run = -1.0, 0.0, 0, 0
    train_start = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            epochs_run = epoch
            if backbone_frozen and epoch > freeze_epochs:
                for p in model.parameters():
                    p.requires_grad = True
                backbone_frozen = False
                print(f"[{model_name}] backbone unfrozen from epoch {epoch}")
            model.train()
            running_loss, seen = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * y.size(0)
                seen += y.size(0)
            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step()

            val = evaluate(model, val_dl, device, criterion)
            train_loss = running_loss / max(seen, 1)
            print(f"[{model_name}] epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  val_macro_f1={val['macro_f1']:.4f}")
            if log_to_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "learning_rate_epoch": lr_now,
                    "epoch_time_s": time.perf_counter() - epoch_start,
                }, step=epoch)
            if val["macro_f1"] > best_f1:
                best_f1, best_acc, best_epoch = val["macro_f1"], val["accuracy"], epoch
                torch.save({
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": model_name,
                    "arch": model_name,
                    "image_size": image_size,
                    "mean": MEAN,
                    "std": STD,
                }, model_path)
            elif patience and epoch > freeze_epochs and epoch - best_epoch >= patience:
                print(f"[{model_name}] early stop: no validation improvement for {patience} epochs")
                break
        train_time_s = time.perf_counter() - train_start

        # Final numbers come from the best epoch (chosen on validation), measured once on the test split.
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        test = evaluate(model, test_dl, device, criterion)
        latency = measure_latency_ms(model, size=image_size)
        size_mb = model_path.stat().st_size / 1e6

        metadata = {
            "split_sha256": _split_fingerprint(splits),
            "run_id": run.info.run_id if run else None,
            "image_brightness_mean": json.loads((SPLITS_DIR / "validation.json").read_text(encoding="utf-8"))["train_brightness_mean"],
            "architecture": model_name,
            "model_name": model_name,
            "owner": owner,
            "class_names": class_names,
            "dataset_root": str(root),
            "registry_stage": "candidate",
            "num_classes": len(class_names),
            "trained_on": "PlantVillage color split",
            "val_accuracy": best_acc,
            "val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "test_weighted_f1": test["weighted_f1"],
            "latency_ms_cpu": latency["mean"],
            "latency_p95_ms_cpu": latency["p95"],
            "model_size_mb": size_mb,
            "total_params": total_params,
            "train_time_s": train_time_s,
            "model_path": str(model_path),
        }
        metadata.update(register_checkpoint(model_path, metadata, out_dir))
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        print(f"[{model_name}] best epoch {best_epoch}: val_macro_f1={best_f1:.4f}  "
              f"test_acc={test['accuracy']:.4f}  test_macro_f1={test['macro_f1']:.4f}  "
              f"latency={latency['mean']:.1f}ms  size={size_mb:.1f}MB")

        if log_to_mlflow:
            mlflow.log_metrics({
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "best_val_accuracy": best_acc,
                "best_val_macro_f1": best_f1,
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
                "test_precision_macro": test["precision"],
                "test_recall_macro": test["recall"],
                "test_loss": test["loss"],
                "latency_ms_cpu": latency["mean"],
                "latency_p95_ms_cpu": latency["p95"],
                "model_size_mb": size_mb,
                "train_time_s": train_time_s,
            })
            if test["y_true"]:
                mlflow.log_dict(classification_report(
                    test["y_true"], test["y_pred"], labels=list(range(len(class_names))),
                    target_names=class_names, output_dict=True, zero_division=0),
                    "test_classification_report.json")
                _log_confusion_matrix(test["y_true"], test["y_pred"], len(class_names))
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_artifact(str(metadata_path), artifact_path="models")
        return metadata
    finally:
        if run is not None:
            mlflow.end_run()


def run_model_experiments(dataset_root=None, epochs=5, batch_size=32, learning_rate=3e-4,
                          output_dir=None, max_samples_per_class=None,
                          model_names: Iterable[str] | None = None, owner: str | None = None,
                          pretrained: bool = True, num_workers: int = 2, promote: bool = True,
                          experiment_name: str = EXPERIMENT_NAME, rebuild_splits: bool = False,
                          image_size: int = IMG, freeze_epochs: int = 0, patience: int = 0,
                          amp: bool = False, seed: int = 42, prepared: bool = False,
                          min_macro_f1: float = 0.0, max_latency_ms: float = float("inf")) -> dict:
    model_names = list(model_names or DEFAULT_BENCHMARK_MODELS)
    out_dir = Path(output_dir) if output_dir else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    if rebuild_splits or not prepared:
        build_split_metadata(dataset_root, SPLITS_DIR, random_state=seed)
    results = []
    for model_name in model_names:
        metadata = train_model(
            dataset_root=dataset_root,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=out_dir,
            max_samples_per_class=max_samples_per_class,
            model_name=model_name,
            log_to_mlflow=True,
            owner=owner,
            pretrained=pretrained,
            num_workers=num_workers,
            experiment_name=experiment_name,
            seed=seed,
            prepared=True,
            image_size=image_size,
            freeze_epochs=freeze_epochs,
            patience=patience,
            amp=amp,
        )
        results.append({
            "model_name": model_name,
            "architecture": metadata["architecture"],
            "val_accuracy": metadata["val_accuracy"],
            "val_macro_f1": metadata["val_macro_f1"],
            "test_accuracy": metadata["test_accuracy"],
            "test_macro_f1": metadata["test_macro_f1"],
            "latency_ms_cpu": metadata["latency_ms_cpu"],
            "model_size_mb": metadata["model_size_mb"],
            "model_path": metadata["model_path"],
            "version": metadata["version"],
            "latency_p95_ms_cpu": metadata["latency_p95_ms_cpu"],
        })

    # The winner is chosen on the validation split only (macro F1, faster model wins a tie),
    # so the test split stays an honest final report.
    best_model = max(results, key=lambda item: (item["val_macro_f1"], -item["latency_ms_cpu"]))
    if promote:
        promote_checkpoint(out_dir, best_model["version"], min_macro_f1, max_latency_ms)

    report = {
        "best_model": best_model["model_name"],
        "best_val_accuracy": float(best_model["val_accuracy"]),
        "best_val_macro_f1": float(best_model["val_macro_f1"]),
        "selection_rule": "highest validation macro F1; lower CPU latency breaks ties",
        "experiments": results,
    }
    report_path = out_dir / "experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one or more models and log every run to MLflow for comparison.",
        epilog="Example: python -m ml.training.train --models resnet18 resnet50 --owner Yasmine --epochs 6")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_BENCHMARK_MODELS), choices=SUPPORTED_MODELS,
                        help="architectures to train (default: %(default)s)")
    parser.add_argument("--owner", default=None, help="who owns these runs (logged as an MLflow tag)")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="subsample training data for quick tests")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 2),
                        help="data-loading processes (default: %(default)s)")
    parser.add_argument("--image-size", type=int, default=IMG,
                        help="square input size; 128-160 is 2-4x faster than 224 (default: %(default)s)")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="train only the classifier head for the first N epochs (3-4x faster epochs)")
    parser.add_argument("--patience", type=int, default=0,
                        help="stop a model early after N epochs without validation improvement (0 = off)")
    parser.add_argument("--amp", action="store_true", help="mixed precision, ~2x faster on a CUDA GPU")
    parser.add_argument("--no-pretrained", action="store_true", help="train from random weights")
    parser.add_argument("--no-promote", action="store_true", help="do not copy the winner to models/plantguard_model.pt")
    parser.add_argument("--rebuild-splits", action="store_true", help="recompute data/processed/splits.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_model_experiments(
        dataset_root=args.dataset_root, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, output_dir=args.output_dir,
        max_samples_per_class=args.max_samples_per_class, model_names=args.models,
        owner=args.owner, pretrained=not args.no_pretrained, num_workers=args.num_workers,
        promote=not args.no_promote, experiment_name=args.experiment,
        rebuild_splits=args.rebuild_splits, image_size=args.image_size,
        freeze_epochs=args.freeze_epochs, patience=args.patience, amp=args.amp,
    )
    print(json.dumps(summary, indent=2))
