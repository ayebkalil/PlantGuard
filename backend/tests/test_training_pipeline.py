from pathlib import Path

import mlflow
import numpy as np
from PIL import Image

from ml.training.train import run_model_experiments


def _make_fake_dataset(root: Path, classes=("Apple___scab", "Apple___healthy", "Tomato___blight"), per_class=10):
    rng = np.random.default_rng(0)
    for name in classes:
        (root / name).mkdir(parents=True)
        for i in range(per_class):
            pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / name / f"img_{i}.png")


def test_training_logs_comparable_runs_to_mlflow(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    _make_fake_dataset(dataset)
    monkeypatch.chdir(tmp_path)  # splits and models are written relative to the working directory
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    report = run_model_experiments(
        dataset_root=dataset, epochs=1, batch_size=8, output_dir=tmp_path / "models",
        model_names=["simple_cnn"], owner="tester", num_workers=0, experiment_name="pytest-benchmark",
    )

    assert report["best_model"] == "simple_cnn"
    assert (tmp_path / "models" / "plantguard_model.pt").exists()  # winner promoted
    assert (tmp_path / "models" / "plantguard_model_simple_cnn.pt").exists()

    runs = mlflow.search_runs(experiment_names=["pytest-benchmark"])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["tags.owner"] == "tester"
    assert run["params.model_name"] == "simple_cnn"
    for metric in ("best_val_macro_f1", "test_accuracy", "test_macro_f1", "latency_ms_cpu", "model_size_mb"):
        assert run[f"metrics.{metric}"] == run[f"metrics.{metric}"]  # present and not NaN