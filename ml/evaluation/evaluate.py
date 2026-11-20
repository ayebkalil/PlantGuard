from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT
from ml.model import build_model, evaluation_transform
from ml.registry import save_json, sha256
from ml.training.train import PlantDataset, _load_splits, _split_fingerprint, evaluate


def evaluate_model(model_path: str | Path = "models/plantguard_model.pt", split: str = "test",
                   dataset_root: str | Path | None = None) -> dict:
    """Evaluate a saved checkpoint (any supported architecture) on the validation or test split."""
    if split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'")

    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    splits = _load_splits(root)
    class_names = splits["classes"]
    class_to_idx = {label: index for index, label in enumerate(class_names)}

    path = Path(model_path)
    metadata_path = path.with_name("model_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if metadata.get("checkpoint") and path.name == "plantguard_model.pt":
        path = path.parent / metadata["checkpoint"]
        if sha256(path) != metadata["checkpoint_sha256"] or metadata["split_sha256"] != _split_fingerprint(splits):
            raise ValueError("The serving checkpoint or the evaluation split has changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != class_names:
        raise RuntimeError("Checkpoint classes do not match the dataset classes.")
    model = build_model(len(class_names), model_name=checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])

    eval_tf = evaluation_transform(checkpoint["image_size"], checkpoint["mean"], checkpoint["std"])
    loader = DataLoader(PlantDataset(splits[split], class_to_idx, eval_tf), batch_size=32)
    result = evaluate(model, loader, "cpu")

    metrics = {
        "model_name": checkpoint["model_name"],
        "split": split,
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["macro_f1"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "samples": len(result["y_true"]),
        "checkpoint_sha256": sha256(path),
        "split_sha256": _split_fingerprint(splits),
    }
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "evaluation.json", {
        "classes": class_names,
        "confusion_matrix": confusion_matrix(result["y_true"], result["y_pred"], labels=labels).tolist(),
        "class_report": classification_report(
            result["y_true"], result["y_pred"], labels=labels, target_names=class_names,
            zero_division=0, output_dict=True,
        ),
    })
    return metrics


if __name__ == "__main__":
    print(evaluate_model())
