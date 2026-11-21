# Pest detection experiment

Train YOLOv8s for up to 20 epochs at 512 pixels. Use AdamW and cosine decay, and disable mosaic augmentation in the final five epochs. Compare validation and held-out mAP50–95 against the published YOLOv8n run.

From this branch, set `MLFLOW_TRACKING_URI` to your team’s MLflow server to compare runs in one place. Otherwise the runner creates a tracking database in the output directory.

```sh
python -m ml.experiments --require-gpu --output data/experiment-pests
```

To reproduce on Kaggle, push your commit and run this from the branch. The script pins the Git commit and uses the parameters in `experiments/candidate.json`.

```sh
python kaggle/prepare_kernel.py pests --experiment --username YOUR_KAGGLE_USERNAME --output data/kaggle-experiment
kaggle kernels push -p data/kaggle-experiment/pests --accelerator NvidiaTeslaT4
```

The runner verifies the baseline split fingerprint before training and saves `baseline-comparison.json` with the two run IDs, checkpoint hashes and metric deltas. CPU timings can vary across hosts and runtime sessions. Checkpoint export stays in the experiment output.
