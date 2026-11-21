# Fruit quality experiment

Fine-tune EfficientNet-B0 for 12 epochs with cosine decay and mild lighting augmentation that preserves hue. Choose the checkpoint by validation macro F1 and report held-out macro F1 and accuracy.

From this branch, set `MLFLOW_TRACKING_URI` to your team’s MLflow server to compare runs in one place. Otherwise the runner creates a tracking database in the output directory.

```sh
python -m ml.experiments --require-gpu --data /path/to/fruit-dataset --output data/experiment-fruit
```

To reproduce on Kaggle, push your commit and run this from the branch. The script pins the Git commit and uses the parameters in `experiments/candidate.json`.

```sh
python kaggle/prepare_kernel.py fruit --experiment --username YOUR_KAGGLE_USERNAME --output data/kaggle-experiment
kaggle kernels push -p data/kaggle-experiment/fruit --accelerator NvidiaTeslaT4
```

The runner verifies the baseline split fingerprint before training and saves `baseline-comparison.json` with the two run IDs, checkpoint hashes and metric deltas. CPU timings can vary across hosts and runtime sessions. Checkpoint export stays in the experiment output.

