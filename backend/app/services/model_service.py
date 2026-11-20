import io
import json
import logging
from pathlib import Path
from threading import RLock
import time

import torch
from PIL import Image, ImageOps

from backend.app.config import CONFIDENCE_THRESHOLD, MODEL_META_PATH, MODEL_PATH
from backend.app.services.monitoring import PredictionMonitor
from ml.model import build_model, evaluation_transform
from ml.preprocessing.preprocess import brightness, IMAGE_FORMATS, MAX_PIXELS
from ml.registry import sha256

log = logging.getLogger(__name__)


class InvalidImage(ValueError):
    pass


class ModelService:
    def __init__(self) -> None:
        self.model_path = Path(MODEL_PATH)
        self.meta_path = Path(MODEL_META_PATH)
        self.lock = RLock()
        self.model = None
        self.meta = {}
        self.signature = None
        self.monitor = PredictionMonitor()

    def refresh_metadata(self) -> None:
        with self.lock:
            try:
                stamp = self.meta_path.stat().st_mtime_ns
                if self.signature and self.signature[0] == stamp and self.signature[1] == self.meta_path:
                    checkpoint_path = self.signature[2]
                    if checkpoint_path.stat().st_mtime_ns == self.signature[3]:
                        return
                metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
                root = self.meta_path.resolve().parent
                checkpoint_path = self.model_path
                if metadata.get("checkpoint"):
                    checkpoint_path = (root / metadata["checkpoint"]).resolve()
                    if not checkpoint_path.is_relative_to(root):
                        raise ValueError("Checkpoint path leaves the model directory")
                modified = checkpoint_path.stat().st_mtime_ns
                checksum = metadata.get("checkpoint_sha256")
                if checksum and sha256(checkpoint_path) != checksum:
                    raise ValueError("Checkpoint checksum mismatch")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                labels = checkpoint["class_names"]
                if not labels or labels != metadata["class_names"]:
                    raise ValueError("Model metadata and checkpoint class mappings do not match")
                model_name = checkpoint.get("model_name") or checkpoint["arch"]
                model = build_model(len(labels), model_name=model_name, pretrained=False)
                model.load_state_dict(checkpoint["state_dict"])
                self.transform = evaluation_transform(checkpoint["image_size"], checkpoint["mean"], checkpoint["std"])
                self.model = model.eval()
                self.meta = metadata
                self.signature = (stamp, self.meta_path, checkpoint_path, modified)
                self.monitor.set_model(metadata["version"], metadata.get("image_brightness_mean"))
            except FileNotFoundError:
                if self.model is None:
                    self.monitor.set_model(None, None)
            except Exception:
                log.exception("Could not load the configured checkpoint")

    def predict(self, image_bytes: bytes) -> dict:
        start = time.perf_counter()
        try:
            with Image.open(io.BytesIO(image_bytes), formats=IMAGE_FORMATS) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise InvalidImage("Image exceeds the maximum pixel count.")
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
            raise InvalidImage("Upload a valid JPEG, PNG or BMP image.") from exc

        with self.lock:
            self.refresh_metadata()
            if self.model is None:
                raise RuntimeError("PlantGuard has no serving model. Train or restore a checkpoint first.")
            model, transform, meta = self.model, self.transform, self.meta
        labels, version = meta["class_names"], meta["version"]
        with torch.inference_mode():
            probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
        top_scores, top_indices = torch.topk(probabilities, k=min(3, len(labels)))
        confidence = float(top_scores[0])
        latency = (time.perf_counter() - start) * 1000
        uncertain = confidence < CONFIDENCE_THRESHOLD
        self.monitor.record(version, latency, brightness(image), uncertain)
        return {
            "predicted_class": labels[int(top_indices[0])],
            "confidence": confidence,
            "is_uncertain": uncertain,
            "top_predictions": [
                {"class_name": labels[int(i)], "confidence": float(s)}
                for s, i in zip(top_scores, top_indices)
            ],
            "model_version": version,
            "inference_time_ms": latency,
        }

    def get_model_info(self) -> dict:
        self.refresh_metadata()
        return {
            "model_name": "PlantGuardClassifier",
            "version": self.meta.get("version", "unavailable"),
            "architecture": self.meta.get("architecture", "unavailable"),
            "dataset_root": Path(self.meta.get("dataset_root", "")).name,
            "training_status": "ready" if self.model is not None else "training_required",
            "registry_stage": self.meta.get("registry_stage", "development"),
        }

    def get_monitoring_snapshot(self) -> dict:
        self.refresh_metadata()
        return self.monitor.snapshot()
