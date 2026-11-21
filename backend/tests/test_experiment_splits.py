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
