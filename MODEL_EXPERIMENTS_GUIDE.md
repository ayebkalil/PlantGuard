# PlantGuard model experiments

The [benchmark procedure](README.md#data-training-and-evaluation) uses DVC and `params.yaml`. The pipeline prepares the pinned PlantVillage sample, trains the baseline CNN and a pretrained MobileNet V3 Small, and evaluates the promoted checkpoint on the authors’ held-out split.

## Run an exploratory experiment

Change the dataset and model arguments as needed. Use a separate output directory and an MLflow experiment name for exploratory runs.

```sh
uv run --locked python -m ml.training.train --dataset-root data/raw/plantvillage --models resnet18 efficientnet_b0 --epochs 6 --seed 42 --num-workers 0 --output-dir models/exploratory --experiment exploratory --no-promote
```

The DVC preparation stage owns `data/processed`. Direct training commands rebuild the split before training. Use a separate working copy for experiments with other datasets so their generated splits stay separate.

See `ml/model.py` for the architecture list and shared evaluation/inference transforms. Each model records epoch metrics and a versioned checkpoint in MLflow. Compare runs with the same split SHA-256. Read the [measured benchmark](reports/benchmark.json), [held-out metrics](reports/metrics.json) and [per-class evaluation](reports/evaluation.json) for the submitted sample.
