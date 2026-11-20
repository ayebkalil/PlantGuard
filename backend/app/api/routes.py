from fastapi import APIRouter, File, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from backend.app.config import MAX_BATCH_FILES, MAX_UPLOAD_BYTES
from backend.app.schemas.prediction import ModelInfoResponse, PredictionResponse
from backend.app.services.model_service import InvalidImage, ModelService

router = APIRouter()
service = ModelService()


@router.get("/health")
def health():
    return {"status": "ok", "service": "plantguard-backend"}


@router.get("/ready")
def ready():
    info = service.get_model_info()
    if info["training_status"] != "ready":
        raise HTTPException(status_code=503, detail="No serving model is loaded")
    return {"status": "ready", "version": info["version"]}


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    return service.get_model_info()


async def classify(file: UploadFile) -> dict:
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the upload limit")
        result = await run_in_threadpool(service.predict, contents)
    except InvalidImage as exc:
        service.monitor.requests.labels("invalid").inc()
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        service.monitor.requests.labels("invalid").inc()
        raise
    except RuntimeError as exc:
        service.monitor.requests.labels("unavailable").inc()
        raise HTTPException(503, "PlantGuard inference is unavailable") from exc
    except Exception:
        service.monitor.requests.labels("error").inc()
        raise
    service.monitor.requests.labels("success").inc()
    return result


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    return await classify(file)


@router.post("/predict/batch", response_model=dict[str, list[PredictionResponse]])
async def predict_batch(files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(413, "Batch exceeds the image count limit")
    return {"results": [await classify(file) for file in files]}


@router.get("/metrics")
def metrics():
    return service.get_monitoring_snapshot()


@router.get("/metrics/prometheus", include_in_schema=False)
def prometheus_metrics():
    service.refresh_metadata()
    return Response(generate_latest(service.monitor.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})
