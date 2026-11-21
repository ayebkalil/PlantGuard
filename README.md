# PlantGuard

PlantGuard serves a PyTorch classifier for plant leaf images through FastAPI and a React frontend. The reproducible benchmark uses apple scab, black rot and healthy leaves from PlantVillage.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 22 and Docker with Compose. The lock file targets Python 3.12 on Windows and Linux x86-64. Run the commands from the project root.

```sh
uv sync --locked
npm ci --prefix frontend
uv run --locked python -m pytest --cov=backend.app --cov=ml --cov-report=term-missing
npm run build --prefix frontend
```

## Data, training and evaluation

```sh
uv run --locked python -m ml.data.fetch_plantvillage
uv run --locked dvc status
uv run --locked dvc repro
uv run --locked dvc metrics show
```

The fetcher downloads 396 RGB images using pinned dataset and split revisions from the authors. The DVC pointer records the dataset snapshot. Preparation decodes images, checks format and dimensions, rejects conflicting labels and path escapes, removes exact decoded duplicates and writes relative, content-hashed split manifests. The authors’ test membership is preserved. Validation is drawn from the remaining data, keeping known leaf groups together. Images whose identity is missing from the upstream map are grouped by file identity.

`params.yaml` configures the dataset, seeds, architectures, training and promotion thresholds. Python, NumPy, PyTorch and loader seeds are set. PyTorch requests deterministic algorithms. Library and hardware changes can still affect numeric results. Color augmentation is applied only in training. Evaluation uses the same resize, crop and normalization as inference. Experiments are selected on validation macro F1, with CPU latency as the tiebreaker. The held-out report includes accuracy, macro precision/recall/F1, weighted F1 and per-class confusion counts.

For an expanded dataset, place class folders under a separate local root and update `dataset.root`. Supply a `partitions.json` manifest with entries of the form `{"class/file.jpg": {"partition": "train", "group": "leaf-id"}}` when groups or a fixed test partition are known. Register the new snapshot with DVC and rerun the pipeline. Treat the three-class checkpoint as a benchmark artifact. PlantVillage images are collected under controlled conditions, and performance on field photos requires separate evaluation.

### Track experiments and restore versions

```sh
uv run --locked mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

Open [MLflow](http://127.0.0.1:5000) and select `plantguard-plantvillage-benchmark`. Runs record parameters, the split SHA-256, epoch metrics, latency, classification reports, confusion matrices and checkpoints. Use `MLFLOW_TRACKING_URI` for a team tracking server. Keep the serving deployment and the tracking server on separate trust boundaries.

Each checkpoint has a permanent ID under `models/registry/`. Promotion checks SHA-256, validation macro F1 and CPU p95 latency. `models/model_metadata.json` is the atomic serving pointer. To restore a previously registered and verified version run:

```sh
uv run --locked python -m ml.registry VERSION_ID --min-f1 0.5 --max-latency-ms 500 --rollback
```

Configure a team-approved storage location with `uv run --locked dvc remote add --local -d private-store PATH_OR_URL`. After committing the DVC pointers, run `uv run --locked dvc push data/raw/plantvillage.dvc` to share the source data and `uv run --locked dvc push` to share the pipeline outputs. On another machine, configure the same authorized storage and restore with `uv run --locked dvc pull data/raw/plantvillage.dvc` followed by `uv run --locked dvc pull`. Keep credentials out of Git and mount artifacts read-only for inference. The public repository contains the pointers and evidence reports.

## Run the application

```sh
docker compose up --build -d
```

Open [PlantGuard](http://127.0.0.1:3000). The frontend proxies requests to FastAPI on the Compose network. `/api/health` confirms the process is running, and `/api/ready` confirms a checkpoint loaded. The API accepts bounded JPEG, PNG and BMP uploads at `/api/predict` and batches at `/api/predict/batch`.

For frontend development, run `uv run --locked uvicorn backend.app.main:app --reload` and `npm run dev --prefix frontend` in separate terminals. Copy `.env.example` to `.env` for Compose overrides. The serving container runs as a non-root user with a read-only root and model mount. The upload code decodes images in memory, applies byte and pixel limits and logs aggregate telemetry without filenames or image data.

## Monitoring and delivery

Open [Prometheus](http://127.0.0.1:9090) to query `plantguard_inference_requests_total`, `plantguard_inference_seconds_bucket`, `plantguard_model_info`, and `plantguard_brightness_drift`. `/api/metrics` reports live prediction counts, mean latency, current model and drift status. Brightness drift compares a recent window with the training mean and reports insufficient data until it reaches the minimum sample count. The included alert rules cover downtime, inference errors, p95 latency and drift. The Compose stack uses one inference process. Prometheus persists scraped metrics in its named volume.

GitHub Actions CI tests the locked environment, reports coverage, builds both images and smoke-tests the frontend and API together. On a version tag or manual release, CD reruns verification and publishes both images to GHCR with commit-addressed tags. A deployment can set `PLANTGUARD_BACKEND_IMAGE` and `PLANTGUARD_FRONTEND_IMAGE` to the published digests and update using `docker compose up -d --no-build --pull always` with its verified model volume.

See [the 18-criterion grading guide](reports/RUBRIC.md) and [benchmark metrics](reports/metrics.json).

## References

- [PlantVillage author dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- [Pinned author split lists and leaf map](https://huggingface.co/datasets/mohanty/PlantVillage/tree/afb5ce38128c7bac085b31d499d28b631ef8fc80)
- [DVC pipelines](https://doc.dvc.org/user-guide/pipelines/defining-pipelines)
- [GitHub Actions package publishing](https://docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions)

## Additional vision tools

See the [Kaggle training and model registry guide](kaggle/README.md) for plant disease detection, pest detection, maize-field weed segmentation, and fruit freshness.
