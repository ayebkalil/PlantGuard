# Weed segmentation experiment

Train U-Net32 with paired image and mask rotations and small brightness shifts. Decay the learning rate over 42 epochs and compare both mean IoU and weed-class IoU at the original 224-pixel resolution.

From this branch, set `MLFLOW_TRACKING_URI` to your team’s MLflow server to compare runs in one place. Otherwise the runner creates a tracking database in the output directory.

```sh
python -m ml.experiments --require-gpu --output data/experiment-weeds
```

To reproduce on Kaggle, push your commit and run this from the branch. The script pins the Git commit and uses the parameters in `experiments/candidate.json`.

```sh
python kaggle/prepare_kernel.py weeds --experiment --username YOUR_KAGGLE_USERNAME --output data/kaggle-experiment
kaggle kernels push -p data/kaggle-experiment/weeds --accelerator NvidiaTeslaT4
```

The runner verifies the baseline split fingerprint before training and saves `baseline-comparison.json` with the two run IDs, checkpoint hashes and metric deltas. CPU timings can vary across hosts and runtime sessions. Checkpoint export stays in the experiment output.

