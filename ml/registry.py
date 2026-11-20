"""Immutable, verified checkpoints and an atomic serving pointer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import NamedTemporaryFile


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_json(path: Path, value: dict) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_checkpoint(source: Path, destination: Path) -> None:
    with NamedTemporaryFile(dir=destination.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def register_checkpoint(path: Path, metadata: dict, root: Path) -> dict:
    checksum = sha256(path)
    identity = f"{checksum}:{metadata['split_sha256']}:{metadata.get('run_id')}"
    version = hashlib.sha256(identity.encode()).hexdigest()[:20]
    directory = root / "registry" / version
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {
        "version": version, "checkpoint_sha256": checksum,
        "checkpoint": f"registry/{version}/model.pt",
    }
    checkpoint = directory / "model.pt"
    if not checkpoint.exists():
        _copy_checkpoint(path, checkpoint)
    if sha256(checkpoint) != checksum:
        raise ValueError("Registry checkpoint checksum mismatch")
    candidate = directory / "metadata.json"
    if not candidate.exists():
        save_json(candidate, {**metadata, **receipt})
    return receipt


def promote_checkpoint(root: Path, version: str, min_f1: float, max_latency: float,
                       rollback: bool = False) -> dict:
    if not re.fullmatch(r"[0-9a-f]{20}", version):
        raise ValueError("Invalid registered model version")
    if not (0 <= min_f1 <= 1 and max_latency > 0):
        raise ValueError("Invalid promotion thresholds")
    directory = root / "registry" / version
    candidate = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = directory / "model.pt"
    if sha256(checkpoint) != candidate["checkpoint_sha256"]:
        raise ValueError("Registry checkpoint checksum mismatch")
    score, latency = candidate["val_macro_f1"], candidate["latency_p95_ms_cpu"]
    if not (math.isfinite(score) and math.isfinite(latency) and score >= min_f1 and latency <= max_latency):
        raise ValueError("Candidate failed the validation macro F1 or CPU p95 latency gate")
    pointer = root / "model_metadata.json"
    if pointer.exists() and not rollback:
        incumbent = json.loads(pointer.read_text(encoding="utf-8"))
        if (incumbent.get("split_sha256") == candidate["split_sha256"]
                and score < incumbent.get("val_macro_f1", 0)):
            raise ValueError("Candidate underperforms the deployed model on the same split")
    _copy_checkpoint(checkpoint, root / "plantguard_model.pt")
    serving = {**candidate, "registry_stage": "production"}
    save_json(pointer, serving)
    return serving


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a verified checkpoint or restore a historical version.")
    parser.add_argument("version")
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--min-f1", type=float, required=True)
    parser.add_argument("--max-latency-ms", type=float, required=True)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    print(json.dumps(promote_checkpoint(args.root, args.version, args.min_f1, args.max_latency_ms, args.rollback), indent=2))
