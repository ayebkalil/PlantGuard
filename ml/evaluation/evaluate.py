from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT
from ml.training.train import PlantDataset, _load_splits, build_model, build_transforms, evaluate


def evaluate_model(model_path: str | Path = "models/plantguard_model.pt", split: str = "test",
                   dataset_root: str | Path | None = None) -> dict:
    """Evaluate a saved checkpoint (any supported architecture) on the validation or test split."""
    if split not in ("validation", "test"):
        raise ValueError("split must be 'validation' or 'test'")

    root = Path(dataset_root) if dataset_root else DEFAULT_DATASET_ROOT
    splits = _load_splits(root)
    class_names = splits["classes"]
    class_to_idx = {label: index for index, label in enumerate(class_names)}

    checkpoint = torch.load(model_path, map_location="cpu")
    if checkpoint["class_names"] != class_names:
        raise RuntimeError("Checkpoint classes do not match the dataset classes.")
    model = build_model(len(class_names), model_name=checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])

    eval_tf = build_transforms(checkpoint.get("image_size", 224))[1]
    loader = DataLoader(PlantDataset(splits[split], class_to_idx, eval_tf), batch_size=32)
    result = evaluate(model, loader, "cpu")

    metrics = {
        "model_name": checkpoint["model_name"],
        "split": split,
        "accuracy": result["accuracy"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["weighted_f1"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "samples": len(result["y_true"]),
    }
    out_path = Path("mlruns") / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    return metrics


if __name__ == "__main__":
    print(evaluate_model())