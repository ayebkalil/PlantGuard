# Plant disease experiment

Fine-tune a pretrained EfficientNet-B0 on 160-pixel PlantVillage images. Freeze the backbone for the first epoch and choose the checkpoint by validation macro F1.

From this branch, set `MLFLOW_TRACKING_URI` to your team’s MLflow server to compare runs in one place. Otherwise the runner creates a tracking database in the output directory.

```sh
python -m ml.experiments --require-gpu --data data/raw/plantvillage --output data/experiment-disease
```

To reproduce on Kaggle, push your commit and run this from the branch. The script pins the Git commit and uses the parameters in `experiments/candidate.json`.

```sh
python kaggle/prepare_kernel.py disease --experiment --username YOUR_KAGGLE_USERNAME --output data/kaggle-experiment
kaggle kernels push -p data/kaggle-experiment/disease --accelerator NvidiaTeslaT4
```

The runner verifies the baseline split fingerprint before training and saves `baseline-comparison.json` with the two run IDs, checkpoint hashes and metric deltas. CPU timings can vary across hosts and runtime sessions. Checkpoint export stays in the experiment output.
