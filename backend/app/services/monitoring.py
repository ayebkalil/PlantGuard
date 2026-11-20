from collections import deque
import logging
from statistics import mean
from threading import Lock

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from backend.app.config import DRIFT_MIN_SAMPLES, DRIFT_THRESHOLD, DRIFT_WINDOW

log = logging.getLogger(__name__)


class PredictionMonitor:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("plantguard_inference_requests", "Inference requests", ["status"], registry=self.registry)
        self.inference_latency = Histogram(
            "plantguard_inference_seconds", "Successful image inference latency",
            buckets=(.01, .025, .05, .1, .25, .5, 1, 2, 5), registry=self.registry,
        )
        self.drift_score = Gauge("plantguard_brightness_drift", "Relative brightness drift", registry=self.registry)
        self.drift_ready = Gauge("plantguard_brightness_drift_ready", "Whether the drift window is ready", registry=self.registry)
        self.drift_alert = Gauge("plantguard_brightness_drift_alert", "Whether brightness crossed the drift threshold", registry=self.registry)
        self.uncertain = Counter("plantguard_uncertain_predictions", "Predictions below confidence threshold", registry=self.registry)
        self.model_ready = Gauge("plantguard_model_ready", "Whether a model is loaded", registry=self.registry)
        self.model_info = Gauge("plantguard_model_info", "Current serving model version", ["version"], registry=self.registry)
        self.lock = Lock()
        self.version = None
        self.reference = None
        self.recent = deque(maxlen=DRIFT_WINDOW)
        self.count = 0
        self.duration = 0.0
        self.last_status = None

    def set_model(self, version, reference):
        with self.lock:
            if version != self.version:
                if self.version:
                    self.model_info.remove(self.version)
                self.version, self.reference = version, reference
                self.recent.clear()
                self.drift_score.set(0)
                self.drift_ready.set(0)
                self.drift_alert.set(0)
                self.last_status = None
                if version:
                    self.model_info.labels(version).set(1)
            self.model_ready.set(bool(version))

    def record(self, version, latency_ms, brightness, uncertain):
        with self.lock:
            self.count += 1
            self.duration += latency_ms
            self.inference_latency.observe(latency_ms / 1000)
            if uncertain:
                self.uncertain.inc()
            if self.version == version:
                self.recent.append(brightness)
            status, score = self._drift()
            self.drift_score.set(score if score is not None else 0)
            self.drift_ready.set(score is not None)
            self.drift_alert.set(status == "DRIFT DETECTED")
            if status == "DRIFT DETECTED" and status != self.last_status:
                log.warning("Brightness drift detected for model %s (%.3f)", version, score)
            self.last_status = status

    def _drift(self):
        if self.reference is None or len(self.recent) < DRIFT_MIN_SAMPLES:
            return "INSUFFICIENT DATA", None
        score = abs(mean(self.recent) - self.reference) / max(self.reference, 1e-6)
        return ("DRIFT DETECTED" if score > DRIFT_THRESHOLD else "OK"), score

    def snapshot(self) -> dict:
        with self.lock:
            status, score = self._drift()
            return {
                "prediction_count": self.count,
                "average_inference_latency_ms": self.duration / max(self.count, 1),
                "drift_status": status,
                "drift_score": score,
                "window_samples": len(self.recent),
                "deployed_model": self.version or "unavailable",
            }
