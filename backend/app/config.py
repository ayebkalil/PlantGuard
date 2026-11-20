from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "plantguard_model.pt"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", ROOT_DIR / "models" / "model_metadata.json"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "8"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
DRIFT_MIN_SAMPLES = int(os.getenv("DRIFT_MIN_SAMPLES", "20"))
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "200"))
