import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api import routes
from backend.app.main import app
from backend.app.services.monitoring import PredictionMonitor
from ml.preprocessing.preprocess import build_split_metadata
from ml.registry import promote_checkpoint, register_checkpoint


def test_invalid_and_oversized_uploads_are_rejected(monkeypatch):
    with TestClient(app) as client:
        files = {"file": ("fake.png", b"not an image", "image/png")}
        assert client.post("/predict", files=files).status_code == 400
        monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 4)
        assert client.post("/predict", files=files).status_code == 413
        assert client.post("/predict/batch", files={"files": files["file"]}).status_code == 413
        assert 'status="invalid"' in client.get("/metrics/prometheus").text


def test_splits_are_portable_deterministic_and_deduplicated(tmp_path):
    rng = np.random.default_rng(42)
    data = tmp_path / "dataset"
    for name in ("apple", "grape"):
        folder = data / name
        folder.mkdir(parents=True)
        for index in range(9):
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(folder / f"{index}.png")
        (folder / "copy.png").write_bytes((folder / "0.png").read_bytes())
        (folder / "broken.png").write_bytes(b"bad")
    output = tmp_path / "prepared"
    splits = build_split_metadata(data, output, random_state=5)
    assert splits == build_split_metadata(data, output, random_state=5)
    rows = [record for split in ("train", "validation", "test") for record in splits[split]]
    assert len(rows) == len({row["sha256"] for row in rows}) == 18
    assert all((data / row["image_path"]).is_file() for row in rows)
    report = json.loads((output / "validation.json").read_text())
    assert report["duplicates"] == report["invalid_images"] == 2


def test_registry_rejects_failed_gate_and_tampered_checkpoint(tmp_path):
    checkpoint = tmp_path / "example.pt"
    checkpoint.write_bytes(b"example")
    metadata = {"val_macro_f1": .7, "latency_p95_ms_cpu": 5., "split_sha256": "split-A"}
    version = register_checkpoint(checkpoint, metadata, tmp_path)
    with pytest.raises(ValueError, match="latency gate"):
        promote_checkpoint(tmp_path, version["version"], .8, 500)
    serving = promote_checkpoint(tmp_path, version["version"], .5, 500)
    assert serving["registry_stage"] == "production"
    assert json.loads((tmp_path / "model_metadata.json").read_text())["version"] == version["version"]
    (tmp_path / version["checkpoint"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        promote_checkpoint(tmp_path, version["version"], .5, 500)


def test_monitor_reports_drift_then_resets_for_new_model():
    monitor = PredictionMonitor()
    monitor.set_model("one", .5)
    assert monitor.snapshot()["drift_status"] == "INSUFFICIENT DATA"
    for _ in range(30):
        monitor.record("one", 10, .9, False)
    assert monitor.snapshot()["drift_status"] == "DRIFT DETECTED"
    assert monitor.snapshot()["average_inference_latency_ms"] == 10
    monitor.set_model("two", .8)
    assert monitor.snapshot()["deployed_model"] == "two"
    assert monitor.snapshot()["drift_status"] == "INSUFFICIENT DATA"


def test_real_checkpoint_serves_and_records_prediction(tmp_path, monkeypatch):
    import torch
    from backend.app.services.model_service import ModelService
    from ml.model import build_model, MEAN, STD

    model = build_model(2, "simple_cnn", pretrained=False)
    checkpoint = tmp_path / "plantguard_model.pt"
    torch.save({"state_dict": model.state_dict(), "class_names": ["healthy", "scab"],
                "model_name": "simple_cnn", "image_size": 64, "mean": MEAN, "std": STD}, checkpoint)
    data = {"version": "v-test", "architecture": "simple_cnn", "class_names": ["healthy", "scab"]}
    pointer = tmp_path / "model_metadata.json"
    pointer.write_text(json.dumps(data), encoding="utf-8")
    service = ModelService()
    service.model_path, service.meta_path = checkpoint, pointer
    monkeypatch.setattr(routes, "service", service)
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "#609830").save(image, "PNG")
    with TestClient(app) as client:
        assert client.get("/ready").json()["version"] == "v-test"
        assert client.post("/predict", files={"file": ("input.png", image.getvalue(), "image/png")}).json()["model_version"] == "v-test"
        assert client.get("/metrics").json()["prediction_count"] == 1
