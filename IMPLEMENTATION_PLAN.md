# PlantGuard implementation plan

The project implements plant disease classification, from validating and versioning PlantVillage images through model selection, serving and production telemetry.

The [README](README.md) is the operational guide. The [rubric evidence map](reports/RUBRIC.md) links the implementation to each scoring criterion. The [benchmark record](reports/benchmark.json) contains the observed results and training environment.

## Intended workflow

1. Fetch a pinned dataset and register its snapshot in DVC.
2. Validate images and create deterministic train, validation and test manifests.
3. Compare a baseline CNN and a transfer-learning model in MLflow.
4. Gate promotion on validation macro F1 and CPU p95 latency. Evaluate the winner on held-out images.
5. Serve the checkpoint through FastAPI and the React application in Docker Compose.
6. Measure service health, requests, inference latency, model version and input drift in Prometheus.
7. Verify changes through GitHub Actions and publish release images by Git commit SHA.
