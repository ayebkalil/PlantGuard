"""Train, compare and export the PlantGuard vision models."""
import argparse
import csv
import os
from pathlib import Path
import random
import shutil
import subprocess
from time import perf_counter

import mlflow
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.model import build_model, MEAN, STD
from ml.registry import save_json, sha256
from ml.vision.artifacts import TASK_KINDS
from ml.vision.data import fruit_rows, pest_data, split_classifier, split_hash, weed_data

ARCHS = {"pests": ("yolov8n", "yolov8s", "yolo11s"), "weeds": ("unet16", "unet32"),
         "fruit": ("efficientnet_b0", "mobilenet_v3_small")}


class Samples(Dataset):
    def __init__(self, rows, size, segment, train):
        self.rows, self.size, self.segment, self.train = rows, (size, size), segment, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source, label, _ = self.rows[index]
        if self.segment:
            channels = []
            for path in source:
                with Image.open(path) as channel:
                    samples = np.asarray(channel)
                    if samples.dtype == np.uint16:
                        samples = np.rint(samples.astype(np.float32) / 257).astype(np.uint8)
                    if samples.dtype != np.uint8 or samples.ndim != 2:
                        raise ValueError(f"Expected an 8-bit or 16-bit grayscale band: {path}")
                    channels.append(Image.fromarray(samples))
            image = Image.merge("RGB", channels)
            with Image.open(label) as mask:
                pixels = np.asarray(mask.resize(self.size, Image.Resampling.NEAREST))
            if pixels.ndim != 2 or pixels.max() > 5:
                raise ValueError("Expected indexed WeedsGalore mask values 0 to 5")
            label = np.minimum(pixels, 2).astype(np.int64)
        else:
            with Image.open(source) as photo:
                image = photo.convert("RGB")
        arr = np.array(image.resize(self.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        if self.train and random.random() < .5:
            arr = arr[:, ::-1]
            if self.segment:
                label = label[:, ::-1].copy()
        arr = ((arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)).transpose(2, 0, 1)
        return torch.from_numpy(arr), label


def conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.left = conv_block(3, channels)
        self.middle = conv_block(channels, channels * 2)
        self.bottom = conv_block(channels * 2, channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up_middle = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.up_left = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.decode_middle = conv_block(channels * 4, channels * 2)
        self.decode_left = conv_block(channels * 2, channels)
        self.head = nn.Conv2d(channels, 3, 1)

    def forward(self, x):
        left = self.left(x)
        middle = self.middle(self.pool(left))
        bottom = self.bottom(self.pool(middle))
        middle_out = self.decode_middle(torch.cat((self.up_middle(bottom), middle), dim=1))
        return self.head(self.decode_left(torch.cat((self.up_left(middle_out), left), dim=1)))


def score(model, loader, device, labels, segment):
    count = len(labels)
    confusion = torch.zeros(count * count, dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        for image, truth in loader:
            pred = model(image.to(device)).argmax(1)
            values = truth.to(device).flatten() * count + pred.flatten()
            confusion += torch.bincount(values, minlength=count * count)
    matrix = confusion.cpu().numpy().reshape(count, count)
    tp = matrix.diagonal()
    support = matrix.sum(1)
    union = support + matrix.sum(0) - tp
    iou = tp / np.maximum(union, 1)
    f1 = 2 * tp / np.maximum(support + matrix.sum(0), 1)
    if segment:
        metrics = {"mean_iou": float(iou[union > 0].mean()), "weed_iou": float(iou[labels.index("weed")]),
                   "pixel_accuracy": float(tp.sum() / matrix.sum())}
    else:
        metrics = {"macro_f1": float(f1.mean()), "accuracy": float(tp.sum() / matrix.sum())}
    return metrics, matrix.tolist()


def torch_train(args, model_name, labels, rows, directory, device, history):
    segment = args.task == "weeds"
    metric = "mean_iou" if segment else "macro_f1"
    model = UNet(int(model_name.removeprefix("unet"))) if segment else build_model(len(labels), model_name, not args.scratch)
    model.to(device)
    loaders = {name: DataLoader(Samples(records, args.size, segment, name == "train"),
                                batch_size=args.batch, shuffle=name == "train",
                                generator=torch.Generator().manual_seed(args.seed))
               for name, records in rows.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([.35, 1.5, 3.], device=device) if segment else None)
    best, validation = -1., {}
    checkpoint = directory / "weights.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for image, truth in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = truth.to(device)
            logits = model(image.to(device))
            loss = loss_fn(logits, target)
            if segment:
                probabilities = logits.softmax(1)[:, 1:]
                foreground = nn.functional.one_hot(target, num_classes=3).permute(0, 3, 1, 2)[:, 1:]
                axes = (0, 2, 3)
                dice = (2 * (probabilities * foreground).sum(axes) + 1) / (
                    probabilities.sum(axes) + foreground.sum(axes) + 1
                )
                loss += 1 - dice.mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
        validation_scores, _ = score(model, loaders["val"], device, labels, segment)
        entry = {"train_loss": total / len(loaders["train"]),
                 **{f"val_{key}": value for key, value in validation_scores.items()}}
        mlflow.log_metrics(entry, step=epoch)
        history.append({"step": epoch, "metrics": entry})
        print(epoch, entry, flush=True)
        if validation_scores[metric] > best:
            best, validation = validation_scores[metric], validation_scores
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    test, matrix = score(model, loaders["test"], device, labels, segment)
    save_json(directory / "confusion.json", {"labels": labels, "test_confusion_matrix": matrix})
    x = torch.zeros((1, 3, args.size, args.size))
    model.cpu().eval()
    torch.onnx.export(model, x, str(directory / "model.onnx"), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    return metric, validation, test


def yolo_train(args, name, manifest, directory, device, history):
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    model = YOLO(f"{name}.yaml" if args.scratch else f"{name}.pt")
    schedule = {"optimizer": "AdamW", "lr0": args.lr, "cos_lr": True,
                "close_mosaic": 5, "patience": 10} if args.cosine else {}

    def log_epoch(trainer):
        step = trainer.epoch + 1
        if history and history[-1]["step"] >= step:
            return
        results = Path(trainer.save_dir) / "results.csv"
        if not results.is_file():
            return
        with results.open(newline="") as file:
            row = list(csv.DictReader(file))[-1]
        metrics = {key.strip().replace("(", "_").replace(")", ""): float(value)
                   for key, value in row.items() if key.strip() != "epoch" and value}
        mlflow.log_metrics(metrics, step=step)
        history.append({"step": step, "metrics": metrics})

    model.add_callback("on_fit_epoch_end", log_epoch)
    workers = 0 if os.name == "nt" else 2
    model.train(data=str(manifest), epochs=args.epochs, imgsz=args.size, batch=args.batch,
                seed=args.seed, deterministic=True, device=0 if device == "cuda" else "cpu",
                project=str(directory.resolve()), name="yolo", workers=workers, plots=True, **schedule)
    best = YOLO(str(model.trainer.best))
    def rates(split):
        result = best.val(data=str(manifest), split=split, imgsz=args.size, batch=args.batch,
                          device=0 if device == "cuda" else "cpu", workers=workers, plots=True,
                          project=str(directory.resolve()), name=split)
        return {"map50_95": float(result.box.map), "map50": float(result.box.map50),
                "precision": float(result.box.mp), "recall": float(result.box.mr)}
    validation, test = rates("val"), rates("test")
    path = best.export(format="onnx", imgsz=args.size, opset=17, nms=True, simplify=False,
                       max_det=100, conf=.25, device="cpu")
    shutil.copyfile(path, directory / "model.onnx")
    shutil.copyfile(model.trainer.best, directory / "weights.pt")
    return "map50_95", validation, test


def publish_metadata(args, directory, labels, metrics, digest, provenance, run_id, history, params):
    import onnxruntime as ort

    onnx = directory / "model.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = {session.get_inputs()[0].name: np.zeros((1, 3, args.size, args.size), dtype=np.float32)}
    session.run(None, inputs)
    timings = []
    for _ in range(7):
        start = perf_counter()
        session.run(None, inputs)
        timings.append((perf_counter() - start) * 1000)
    metrics["cpu_p95_ms"] = float(np.percentile(timings, 95))
    meta = {"task": args.task, "kind": TASK_KINDS[args.task], "labels": labels, "image_size": args.size,
            "split_sha256": digest, "dataset": provenance, "metrics": metrics, "params": params,
            "source_run_id": run_id, "history": history, "checkpoint_sha256": sha256(onnx),
            "weights_sha256": sha256(directory / "weights.pt")}
    save_json(directory / "bundle.json", meta)
    mlflow.log_metrics(metrics)
    for name in ("model.onnx", "weights.pt", "bundle.json", "split.json", "confusion.json"):
        path = directory / name
        if path.exists():
            mlflow.log_artifact(str(path))


def train(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.require_gpu and device != "cuda":
        raise RuntimeError("A Kaggle GPU must be selected for this run")
    torch.set_num_threads(2)
    if args.task == "pests":
        labels, rows, manifest, provenance = pest_data(
            args.data, work, args.seed, args.limit, expand_training=args.expand_training
        )
    elif args.task == "weeds":
        labels, rows, provenance = weed_data(args.data, work, args.limit)
        manifest = {key: sorted((paths[0].name, digest) for paths, _, digest in values) for key, values in rows.items()}
    else:
        labels, records, provenance = fruit_rows(args.data, args.limit, args.seed)
        rows, audit = split_classifier(records, labels, args.seed, args.limit)
        provenance.update(audit)
        manifest = {key: sorted((label, digest) for _, label, digest in values) for key, values in rows.items()}

    digest = split_hash(manifest)
    comparison_split = provenance.get("original_split_sha256", digest)
    if args.expected_split and comparison_split != args.expected_split:
        raise ValueError(f"Expected split {args.expected_split}, found {comparison_split}")
    tracking = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(out / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(args.experiment or f"plantguard-{args.task}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    candidates = []
    for model in args.models or ARCHS[args.task][:1]:
        if model not in ARCHS[args.task]:
            raise ValueError(f"{model} is unsupported for {args.task}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        params = {"architecture": model, "seed": args.seed, "epochs": args.epochs, "image_size": args.size,
                  "batch": args.batch, "learning_rate": args.lr, "device": device, "scratch": args.scratch,
                  "preprocessing": "rgb16-scale-dice" if args.task == "weeds" else "dataset-adapter-v1",
                  "git_sha": commit, "split_sha256": digest, "sample_limit": args.limit or 0,
                  "expanded_training": args.expand_training,
                  **{f"{key}_count": len(value) for key, value in manifest.items()}}
        if args.cosine:
            params.update(optimizer="AdamW", lr_schedule="cosine", close_mosaic=5, patience=10)
        with mlflow.start_run(run_name=model) as run:
            directory = out / f"{model}-{run.info.run_id[:8]}"
            directory.mkdir()
            history = []
            mlflow.log_params(params)
            save_json(directory / "split.json", manifest)
            if args.task == "pests":
                metric, val, test = yolo_train(args, model, rows, directory, device, history)
            else:
                metric, val, test = torch_train(args, model, labels, rows, directory, device, history)
            metrics = {f"{split}_{key}": value for split, scores in (("val", val), ("test", test))
                       for key, value in scores.items()}
            publish_metadata(args, directory, labels, metrics, digest, provenance, run.info.run_id, history, params)
            candidates.append({"bundle": str(directory.relative_to(out)), "architecture": model, **metrics})
    winner = max(candidates, key=lambda item: (item[f"val_{metric}"], -item["cpu_p95_ms"]))
    save_json(out / "comparison.json", {"task": args.task, "validation_metric": metric,
                                         "winner": winner["bundle"], "split_sha256": digest, "runs": candidates})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASK_KINDS)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("data/vision"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expand-training", action="store_true", help="use additional IP102 training images while preserving evaluation splits")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--expected-split", help="require this split SHA-256 before training")
    parser.add_argument("--experiment", help="MLflow experiment name")
    parser.add_argument("--cosine", action="store_true", help="use cosine learning-rate decay")
    train(parser.parse_args())
