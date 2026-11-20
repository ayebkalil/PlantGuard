from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
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
from torchvision import models, transforms as T

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT, discover_classes
from ml.preprocessing.preprocess import build_split_metadata

IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SPLITS_DIR = Path("data/processed")
EXPERIMENT_NAME = "plantguard-model-benchmark"
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"

# Every architecture below can be trained and compared in MLflow.
# "simple_cnn" is trained from scratch (baseline), the others use ImageNet transfer learning.
SUPPORTED_MODELS = (
    "simple_cnn",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "convnext_tiny",
)
# Models used when nobody picks any (fast enough for a full benchmark on one laptop).
DEFAULT_BENCHMARK_MODELS = ("simple_cnn", "mobilenet_v3_small", "resnet18", "efficientnet_b0")

def build_transforms(size: int = IMG):
    """(train, eval) transforms for a given square input size. The app reads the size from the checkpoint."""
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(0.4, 0.4, 0.3, 0.05),
        T.RandomApply([T.GaussianBlur(5)], p=0.3),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])
    evaluation = T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size),
                            T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, evaluation


train_tf, eval_tf = build_transforms(IMG)


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
        img = Image.open(s["image_path"]).convert("RGB")
        return self.tf(img), self.c2i[s["label"]]


class SimpleCNN(nn.Module):
    """Small CNN trained from scratch: the baseline every pretrained model must beat."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(num_classes: int, model_name: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes)
    if model_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
        if model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model '{model_name}'. Supported: {SUPPORTED_MODELS}")


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
    """Reuse data/processed/splits.json when it matches this dataset (rebuilding it
    validates every image, which is slow). Same seed + same dataset = same split for everyone."""
    splits_path = SPLITS_DIR / "splits.json"
    if splits_path.exists() and not rebuild:
        with open(splits_path, encoding="utf-8") as fp:
            splits = json.load(fp)
        try:
            same_classes = splits["classes"] == discover_classes(root)
            first = Path(splits["train"][0]["image_path"])
            if same_classes and first.exists() and first.resolve().is_relative_to(root.resolve()):
                return splits
        except (KeyError, IndexError, FileNotFoundError):
            pass
    build_split_metadata(root, SPLITS_DIR, random_state=42)
    with open(splits_path, encoding="utf-8") as fp:
        return json.load(fp)


def _split_fingerprint(splits: dict) -> str:
    """Hash of which image is in which split (ignores machine-specific folder paths), so
    teammates can prove in MLflow that their runs used the exact same train/val/test split."""
    payload = {
        key: sorted((s["label"], Path(s["image_path"].replace("\\", "/")).name) for s in splits[key])
        for key in ("train", "validation", "test")
    }
    return hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:12]


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
                image_size: int = IMG, freeze_epochs: int = 0, patience: int = 0, amp: bool = False) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    splits = _load_splits(root, rebuild=rebuild_splits)
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
            "dataset_root": str(root),
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
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        test = evaluate(model, test_dl, device, criterion)
        latency = measure_latency_ms(model, size=image_size)
        size_mb = model_path.stat().st_size / 1e6

        metadata = {
            "version": "2",
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
                          amp: bool = False) -> dict:
    model_names = list(model_names or DEFAULT_BENCHMARK_MODELS)
    out_dir = Path(output_dir) if output_dir else Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

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
            rebuild_splits=rebuild_splits,
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
        })

    # The winner is chosen on the validation split only (macro F1, faster model wins a tie),
    # so the test split stays an honest final report.
    best_model = max(results, key=lambda item: (item["val_macro_f1"], -item["latency_ms_cpu"]))
    if promote:
        best_checkpoint = Path(best_model["model_path"])
        best_metadata_path = out_dir / f"model_metadata_{best_model['model_name']}.json"
        if best_checkpoint.exists():
            shutil.copy2(best_checkpoint, out_dir / "plantguard_model.pt")
        if best_metadata_path.exists():
            shutil.copy2(best_metadata_path, out_dir / "model_metadata.json")

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