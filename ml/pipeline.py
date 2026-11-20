"""DVC entry points. The same parameters are used for training and evaluation."""
import argparse
import os
from pathlib import Path

import yaml

from ml.preprocessing.preprocess import build_split_metadata


def configure() -> dict:
    filename = Path(os.getenv("PLANTGUARD_PARAMS", "params.yaml"))
    return yaml.safe_load(filename.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    stage = parser.parse_args().stage
    config = configure()
    data = config["dataset"]
    training = config["training"]
    if stage == "prepare":
        build_split_metadata(data["root"], random_state=data["split_seed"],
                             test_size=data["test_fraction"], val_size=data["val_fraction"])
    elif stage == "train":
        from ml.training.train import run_model_experiments

        run_model_experiments(
            dataset_root=data["root"], image_size=data["image_size"], prepared=True,
            epochs=training["epochs"], batch_size=training["batch_size"],
            learning_rate=training["learning_rate"], seed=training["seed"],
            model_names=training["models"], pretrained=training["pretrained"],
            num_workers=training["num_workers"], experiment_name=training["experiment_name"],
            min_macro_f1=config["registry"]["min_validation_macro_f1"],
            max_latency_ms=config["registry"]["max_cpu_p95_ms"],
        )
    else:
        from ml.evaluation.evaluate import evaluate_model

        print(evaluate_model(dataset_root=data["root"]))


if __name__ == "__main__":
    main()
