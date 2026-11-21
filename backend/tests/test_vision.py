import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api import routes
from backend.app.main import app
from backend.app.services.vision import VisionService, boxed_predictions, segmented_prediction, tensor_image
from ml.vision.data import split_classifier, voc_boxes
from ml.vision.train import Samples


def test_ip102_annotation_trailing_bytes_and_voc_coordinates(tmp_path):
    annotation = tmp_path / "box.xml"
    annotation.write_text("""<annotation><object><name>aphid</name><bndbox>
        <xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>20</ymax>
        </bndbox></object></annotation>trailing data""")
    assert voc_boxes(annotation, {"aphid": 0}, 20, 40) == [
        "0 0.25000000 0.25000000 0.50000000 0.50000000"
    ]


def test_duplicate_copies_across_fruit_splits_and_conflicting_labels(tmp_path):
    rows = []
    for label in (0, 1):
        for number in range(6):
            png = tmp_path / f"{label}-{number}.png"
            Image.new("RGB", (8, 8), (label * 100, number * 20, 0)).save(png)
            rows.append((png, label, "pool"))
    rows += [(rows[0][0], 1, "pool"), (rows[3][0], 0, "pool")]
    splits, audit = split_classifier(rows, ["a", "b"], 42, None)
    hashes = [digest for values in splits.values() for _, _, digest in values]
    assert audit == {"conflicting_image_hashes_removed": 1, "duplicate_copies_removed": 2}
    assert len(hashes) == len(set(hashes)) == 11


def test_detector_reverses_letterbox_and_ignores_pad():
    tensor, padding = tensor_image(Image.new("RGB", (80, 40)), 64, True)
    assert tensor.shape == (1, 3, 64, 64)
    boxes = np.array([[[0, 16, 64, 48, .9, 0], [1, 2, 7, 8, .9, 0], [0, 0, 0, 0, 0, 0]]])
    assert boxed_predictions(boxes, ["aphid"], padding)["detections"] == [
        {"label": "aphid", "confidence": .9, "box": [0., 0., 1., 1.]}
    ]


def test_segmentation_overlay_and_coverage():
    logits = np.array([[[[1, 1]], [[2, 0]], [[0, 2]]]])
    result = segmented_prediction(logits, ["background", "maize", "weed"])
    assert result["coverage"][-1] == {"label": "weed", "fraction": .5}
    assert result["overlay"].startswith("data:image/png;base64,")


def test_weed_training_preserves_high_bit_depth_band_contrast(tmp_path):
    bands = []
    for name in "RGB":
        path = tmp_path / f"{name}.png"
        Image.fromarray(np.array([[0, 16384], [32768, 65535]], dtype=np.uint16)).save(path)
        bands.append(path)
    mask = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1], [2, 5]], dtype=np.uint8)).save(mask)
    image, target = Samples([(bands, mask, "")], 2, segment=True, train=False)[0]
    assert image[0].unique().numel() == 4
    assert target.tolist() == [[0, 1], [2, 2]]


@pytest.mark.parametrize("task", ("fruit", "pests", "weeds"))
def test_missing_model_and_invalid_upload(task, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "vision", VisionService(tmp_path))
    with TestClient(app) as client:
        assert client.get(f"/vision/{task}").json() == {"status": "training_required"}
        endpoint = f"/vision/{task}/predict"
        assert client.post(endpoint, files={"file": ("x.png", b"broken", "image/png")}).status_code == 400
        image = io.BytesIO()
        Image.new("RGB", (16, 16), "green").save(image, "PNG")
        assert client.post(endpoint, files={"file": ("x.png", image.getvalue(), "image/png")}).status_code == 503


def test_retired_vision_task_is_rejected():
    with TestClient(app) as client:
        assert client.get("/vision/banana").status_code == 422


def test_invalid_model_pointer_cannot_load_external_file(tmp_path):
    folder = tmp_path / "fruit"
    folder.mkdir()
    (folder / "model.json").write_text(json.dumps({"task": "fruit", "kind": "classification",
                                                   "checkpoint": "../private.onnx"}))
    with pytest.raises(ValueError, match="Unverified"):
        VisionService(tmp_path).load("fruit")
