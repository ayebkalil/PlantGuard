# Vision models and Kaggle training

| PlantGuard tool | Training data | Candidates | Selection metric |
| --- | --- | --- | --- |
| Plant disease detection | [Pinned PlantVillage benchmark](../README.md#data-training-and-evaluation) | Simple CNN, MobileNet V3 Small | Validation macro F1 |
| Pest identification | [IP102 v1.1 detection subset](https://github.com/xpwu95/IP102) | YOLOv8n, YOLOv8s | Validation box mAP50–95 |
| Weed identification | [Original WeedsGalore RGB bands and semantic masks](https://github.com/GFZ/weedsgalore) | U-Net with 16 or 32 starting channels | Validation mean IoU |
| Fruit quality | [Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) | EfficientNet-B0, MobileNet V3 Small | Validation macro F1 |

The pest loader downloads the authors’ VOC2007 images, annotations and test/trainval lists. The Kaggle classification mirror is optional because boxes are required. It retains the official test set and uses a seeded validation split from trainval. The weed loader combines the original red, green and blue bands for each tile and preserves the authors’ 104/26/26 split. It maps mask IDs to background, maize and weed. The [AgML release](https://huggingface.co/datasets/Project-AgML/WeedsGalore_maize_segmentation) represents the same 156 tiles as individual bands. Fruit quality preserves the provided test split. Identical decoded images are deduplicated and conflicting image labels are excluded before classification splits.

## Train on Kaggle

Generate a private Kaggle script and configure it to fetch a specific pushed Git SHA.

```sh
python kaggle/prepare_kernel.py pests --username YOUR_KAGGLE_USERNAME --output data/kaggle --epochs 50
uvx --with six kaggle kernels push -p data/kaggle/pests --accelerator NvidiaTeslaT4
```

Repeat with `weeds`, `fruit`, or `disease`. The fruit script attaches its Kaggle input automatically. With internet enabled, the weed and pest scripts fetch their source files, and the disease script fetches the pinned PlantVillage benchmark. Use `--dataset OWNER/SLUG` to attach a private source dataset. For disease training, put class folders at the root of that input. Use `--limit` and fewer epochs for a bounded baseline. For parity, compare runs using the same split SHA-256, learning budget and task metric.

## Compare and activate models

Pest, weed, and fruit outputs include a comparison report, the exact split manifest, per-model MLflow history, an ONNX export, and the original trained weights. Disease training uses the existing PlantVillage pipeline and writes its versioned PyTorch checkpoints and MLflow database under `disease/` in the Kaggle output. The [PlantVillage benchmark report](../reports/benchmark.json) records the locally verified disease results. Compare validation results to select a candidate and inspect held-out metrics separately.

```sh
uvx --with six kaggle kernels output OWNER/plantguard-fruit-training -p data/kaggle-output/fruit --file-pattern 'comparison\.json|bundle\.json|split\.json|model\.onnx|weights\.pt|confusion\.json'
uv run --locked python -m ml.vision.artifacts data/kaggle-output/fruit/fruit/RUN_DIRECTORY --promote
uv run --locked mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

Import each run directory to compare candidates side by side. Pass `--promote` for the validation winner to activate it in the frontend. Model registration verifies both weight hashes and the split manifest, imports the original metrics into the task-specific local MLflow experiment, and assigns an immutable ID. Activated ONNX checkpoints are checksum-verified at inference.

The task directories under `vision-models/` are the DVC versioning units. Use `dvc add vision-models/fruit` and commit the resulting `vision-models/fruit.dvc` pointer. Configure an access-controlled DVC remote and use `dvc push vision-models/fruit.dvc` and `dvc pull vision-models/fruit.dvc` to transfer the model files. The Compose backend mounts `vision-models/` read-only.

IP102 is available for academic use. The fruit Kaggle dataset’s metadata lists its license as unknown. WeedsGalore is distributed under CC BY 4.0. Keep restricted artifacts in your private Kaggle outputs or private DVC storage. See the source links for attribution details.

## Measured results

The [versioned benchmark reports](../reports/vision) record the selected run, held-out metrics, exact split digest and MLflow run IDs. The current pest checkpoint is an early YOLOv8n baseline. Its low held-out mAP limits field use. Scores in each report apply to that task and its listed sample size.
