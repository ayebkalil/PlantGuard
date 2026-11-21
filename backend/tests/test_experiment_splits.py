import argparse
from pathlib import Path

import pytest

from ml.training import train as disease_training
from ml.vision import train as vision_training


def test_disease_experiment_rejects_split_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(disease_training, "build_split_metadata", lambda *args, **kwargs: None)
    fingerprint = "a" * 64
    monkeypatch.setattr(disease_training, "_load_splits", lambda *args, **kwargs: {})
    monkeypatch.setattr(disease_training, "_split_fingerprint", lambda _: fingerprint)
    with pytest.raises(ValueError, match=f"found {fingerprint}"):
        disease_training.run_model_experiments(expected_split="b" * 64, promote=False)
    assert not list(Path("models").glob("*.pt"))


def test_vision_experiment_rejects_split_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vision_training, "weed_data", lambda *args: (["background", "maize", "weed"], {"train": []}, {})
    )
    monkeypatch.setattr(
        vision_training.mlflow, "set_tracking_uri", lambda _: pytest.fail("Training reached MLflow after split drift")
    )
    args = argparse.Namespace(task="weeds", data=None, output=tmp_path / "output", work=tmp_path / "work",
                              limit=None, cpu=True, require_gpu=False, expected_split="a" * 64)
    with pytest.raises(ValueError, match="Expected split"):
        vision_training.train(args)

def test_pest_training_expansion_preserves_eval_membership(tmp_path):
    import random

    from PIL import Image

    from ml.vision.data import pest_data, split_hash

    source = tmp_path / "voc"
    source.mkdir()
    trainval = list("abcdefghij")
    (source / "trainval.txt").write_text("\n".join(trainval))
    (source / "test.txt").write_text("z\n")
    random.Random(42).shuffle(trainval)
    repeated_image = trainval[-1]
    xml = """<annotation><object><name>aphid</name><bndbox><xmin>1</xmin>
    <ymin>1</ymin><xmax>12</xmax><ymax>12</ymax></bndbox></object></annotation>"""
    for key in [*trainval, "z"]:
        color = ord("z" if key == repeated_image else key)
        Image.new("RGB", (16, 16), (color, color, color)).save(source / f"{key}.png")
        (source / f"{key}.xml").write_text(xml)

    _, _, baseline, _ = pest_data(source, tmp_path / "base", 42, 1)
    _, yaml_path, expanded, meta = pest_data(
        source, tmp_path / "expanded", 42, 1, expand_training=True
    )

    assert meta["original_split_sha256"] == split_hash(baseline)
    assert expanded["val"] == baseline["val"]
    assert expanded["test"] == baseline["test"]
    assert len(expanded["train"]) > len(baseline["train"])
    assert repeated_image not in {row[0] for row in expanded["train"]}
    assert len((yaml_path.parent / "train.txt").read_text().splitlines()) == len(expanded["train"])
