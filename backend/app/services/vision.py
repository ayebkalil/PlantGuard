"""CPU inference for the versioned ONNX models served by PlantGuard."""
import base64
import io
import json
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np
from PIL import Image

from backend.app.services.model_service import decode_image
from ml.model import MEAN, STD
from ml.registry import sha256
from ml.vision.artifacts import TASK_KINDS

WEED_COLORS = np.array([[0, 0, 0, 0], [100, 218, 135, 119], [255, 181, 68, 166]], dtype=np.uint8)


def tensor_image(image, size, detection):
    left = top = 0
    scaled_w = scaled_h = size
    if detection:
        ratio = size / max(image.size)
        scaled_w, scaled_h = [max(1, round(dim * ratio)) for dim in image.size]
        left, top = (size - scaled_w) // 2, (size - scaled_h) // 2
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR), (left, top))
        image = canvas
    else:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.
    if not detection:
        pixels = (pixels - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]), (left, top, scaled_w, scaled_h)


def boxed_predictions(output, labels, padding):
    x, y, w, h = padding
    detections = []
    for x1, y1, x2, y2, confidence, class_id in output[0]:
        if not np.isfinite([x1, y1, x2, y2, confidence, class_id]).all() or confidence < .25:
            continue
        label_index = int(class_id)
        if not 0 <= label_index < len(labels):
            continue
        box = [float(value) for value in np.clip([(x1-x)/w, (y1-y)/h, (x2-x)/w, (y2-y)/h], 0., 1.)]
        if box[2] > box[0] and box[3] > box[1]:
            detections.append({"label": labels[label_index], "confidence": float(confidence), "box": box})
    return {"detections": detections}


def segmented_prediction(output, labels):
    mask = output[0].argmax(0)
    pixels = Image.fromarray(WEED_COLORS[mask])
    buf = io.BytesIO()
    pixels.save(buf, format="PNG")
    return {"overlay": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "coverage": [{"label": label, "fraction": float(np.mean(mask == i))}
                         for i, label in enumerate(labels)]}


class VisionService:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.lock = RLock()

    def load(self, task):
        import onnxruntime as ort

        folder = (self.root / task).resolve()
        pointer = folder / "model.json"
        with self.lock:
            stamp = pointer.stat().st_mtime_ns
            cached = self.cache.get(task)
            if cached and cached[0] == stamp and cached[1].stat().st_mtime_ns == cached[2]:
                return cached[3:]
            meta = json.loads(pointer.read_text(encoding="utf-8"))
            if meta["task"] != task or meta["kind"] != TASK_KINDS[task]:
                raise ValueError("Model pointer does not match the requested task")
            path = (folder / meta["checkpoint"]).resolve()
            if not path.is_relative_to(folder) or sha256(path) != meta["checkpoint_sha256"]:
                raise ValueError("Unverified vision checkpoint")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            self.cache[task] = (stamp, path, path.stat().st_mtime_ns, session, meta)
            return session, meta

    def predict(self, task, content):
        start = perf_counter()
        image = decode_image(content)
        session, meta = self.load(task)
        kind, labels = meta["kind"], meta["labels"]
        tensor, padding = tensor_image(image, meta["image_size"], kind == "detection")
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if kind == "detection":
            result = boxed_predictions(output, labels, padding)
        elif kind == "segmentation":
            result = segmented_prediction(output, labels)
        else:
            logits = output[0]
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            top = np.argsort(probabilities)[::-1][:3]
            result = {"top_predictions": [{"label": labels[i], "confidence": float(probabilities[i])} for i in top]}
        return {**result, "kind": kind, "model_version": meta["version"],
                "inference_time_ms": (perf_counter() - start) * 1000}
