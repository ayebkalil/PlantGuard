# Expanded IP102 training experiment

Train YOLO11s at 640 pixels on the full clean IP102 training partition. Preserve the validation and test samples from the 1,000-per-split YOLOv8 experiments. The preparer verifies the original split fingerprint and reserves validation and test image hashes before admitting additional training samples. The full training manifest receives its own fingerprint.

```sh
python -m ml.experiments --require-gpu --output data/experiment-pests-expanded
python kaggle/prepare_kernel.py pests --experiment --username YOUR_KAGGLE_USERNAME --output data/kaggle-experiment
```

Compare with YOLOv8s on validation mAP50–95 and check test mAP50–95, precision, recall and CPU latency. The reference candidate scored 9.43% test mAP50–95 and 20.03% recall. The changes in data volume, architecture and input resolution are evaluated together in this recipe.
