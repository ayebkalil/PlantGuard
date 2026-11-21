"""Verify a Kaggle run bundle, import it into MLflow and register its model."""
import argparse
import json
import math
import os
from pathlib import Path
import re

from ml.registry import _copy_checkpoint, save_json, sha256

TASK_KINDS = {"pests": "detection", "weeds": "segmentation", "fruit": "classification", "banana": "classification"}
SELECTION = {"detection": "val_map50_95", "segmentation": "val_mean_iou", "classification": "val_macro_f1"}


def verify_bundle(directory):
    meta = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if TASK_KINDS.get(meta["task"]) != meta["kind"] or meta["kind"] not in SELECTION:
        raise ValueError("Unexpected task or model type")
    if not re.fullmatch(r"[a-f0-9]{32}", meta["source_run_id"]):
        raise ValueError("Unexpected MLflow run ID")
    if not 32 <= meta["image_size"] <= 1024 or not meta["labels"]:
        raise ValueError("Invalid image size or labels")
    if sha256(directory / "model.onnx") != meta["checkpoint_sha256"]:
        raise ValueError("ONNX SHA-256 mismatch")
    if sha256(directory / "weights.pt") != meta["weights_sha256"]:
        raise ValueError("Training weights SHA-256 mismatch")
    from ml.vision.data import split_hash
    manifest = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if split_hash(manifest) != meta["split_sha256"]:
        raise ValueError("Split manifest SHA-256 mismatch")
    return meta


def import_mlflow(directory, meta):
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(f"plantguard-{meta['task']}")
    source_id = meta["source_run_id"]
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=f"tags.source_run_id = '{source_id}' AND attributes.status = 'FINISHED'")
    if not runs.empty:
        return str(runs.iloc[0].run_id)
    with mlflow.start_run(run_name=meta["params"]["architecture"]) as run:
        mlflow.set_tag("source_run_id", source_id)
        mlflow.log_params(meta["params"])
        for record in meta["history"]:
            mlflow.log_metrics(record["metrics"], step=record["step"])
        mlflow.log_metrics(meta["metrics"], step=max((entry["step"] for entry in meta["history"]), default=0))
        for name in ("bundle.json", "model.onnx", "weights.pt", "split.json", "confusion.json"):
            path = directory / name
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id


def install(directory, root, promote=False):
    meta = verify_bundle(directory)
    version = f"{meta['source_run_id'][:12]}-{meta['checkpoint_sha256'][:12]}"
    task_root = root / meta["task"]
    path = task_root / "registry" / version
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "model.onnx"
    if not checkpoint.exists():
        _copy_checkpoint(directory / "model.onnx", checkpoint)
        _copy_checkpoint(directory / "weights.pt", path / "weights.pt")
    if sha256(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Installed checkpoint checksum mismatch")
    run = import_mlflow(directory, meta)
    version_data = {key: value for key, value in meta.items() if key != "history"}
    version_data.update({"version": version, "mlflow_run_id": run,
                         "checkpoint": f"registry/{version}/model.onnx"})
    save_json(path / "metadata.json", version_data)
    if promote:
        metric = SELECTION[meta["kind"]]
        quality, latency = meta["metrics"][metric], meta["metrics"]["cpu_p95_ms"]
        if not (math.isfinite(quality) and 0 < quality <= 1 and math.isfinite(latency) and 0 < latency <= 500):
            raise ValueError("Candidate failed the validation score or CPU latency gate")
        pointer = task_root / "model.json"
        if pointer.exists():
            incumbent = json.loads(pointer.read_text(encoding="utf-8"))
            if (incumbent["split_sha256"] == meta["split_sha256"]
                    and incumbent["metrics"][metric] > quality):
                raise ValueError("Candidate underperforms the serving version on this split")
        save_json(pointer, version_data)
    return version_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--root", type=Path, default=Path("vision-models"))
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(args.directory, args.root, args.promote), indent=2))
