from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps

from ml.data.dataset_registry import DEFAULT_DATASET_ROOT, discover_classes

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_FORMATS = ("JPEG", "PNG", "BMP")
MAX_PIXELS = 20_000_000


def brightness(image: Image.Image) -> float:
    return float(np.asarray(image.convert("L").resize((32, 32), Image.Resampling.BOX)).mean() / 255)


def validate_image(path: str | Path) -> bool:
    try:
        with Image.open(path, formats=IMAGE_FORMATS) as image:
            if image.width * image.height > MAX_PIXELS:
                return False
            image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def load_image_array(path: str | Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    with Image.open(path) as image:
        rgb = ImageOps.fit(image.convert("RGB"), target_size, Image.Resampling.BICUBIC)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _partitions(groups, rng, validation_fraction, test_fraction, fixed_test):
    rng.shuffle(groups)
    if len(groups) < (2 if fixed_test else 3):
        raise ValueError("Each class needs distinct leaf/image groups in train, validation and test")
    if not fixed_test:
        count = min(len(groups) - 2, max(1, round(len(groups) * test_fraction)))
        yield "test", groups[:count]
        groups = groups[count:]
    remaining_fraction = 1 - test_fraction if not fixed_test else 1
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction / remaining_fraction)))
    yield "validation", groups[:count]
    yield "train", groups[count:]


def build_split_metadata(dataset_root: str | Path | None = None, output_dir: str | Path | None = None,
                         test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> dict:
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1")
    root = Path(dataset_root or DEFAULT_DATASET_ROOT).resolve()
    output = Path(output_dir or "data/processed")
    classes = discover_classes(root)
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two classes")
    official = root / "partitions.json"
    partitions = json.loads(official.read_text(encoding="utf-8")) if official.is_file() else {}
    samples = {"classes": classes, "train": [], "validation": [], "test": []}
    rng = random.Random(random_state)
    fingerprints = {}
    groups_seen = {}
    skipped = Counter()
    for label in classes:
        groups = defaultdict(list)
        class_dir = root / label
        if class_dir.is_symlink() or class_dir.is_junction():
            raise ValueError(f"Linked class directory is forbidden: {label}")
        paths = sorted((p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS),
                       key=lambda p: (partitions.get(p.relative_to(root).as_posix(), {}).get("partition") != "test", p.name))
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Linked image is forbidden: {path.name}")
            rel = path.relative_to(root).as_posix()
            if partitions and rel not in partitions:
                raise ValueError(f"Image missing from upstream partition manifest: {rel}")
            if not validate_image(path):
                skipped["invalid_images"] += 1
                continue
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in fingerprints:
                if fingerprints[digest] != label:
                    raise ValueError(f"Duplicate image with conflicting labels: {rel}")
                skipped["duplicates"] += 1
                continue
            fingerprints[digest] = label
            source = partitions.get(rel, {})
            partition = source.get("partition", "train")
            if partition not in ("train", "test"):
                raise ValueError(f"Invalid upstream partition for {rel}")
            group = source.get("group", rel)
            if groups_seen.setdefault(group, partition) != partition:
                raise ValueError(f"Leaf group appears in both upstream partitions: {group}")
            record = {"image_path": rel, "label": label, "sha256": digest}
            if partition == "test":
                samples["test"].append(record)
            else:
                groups[group].append(record)
        if partitions and not any(record["label"] == label for record in samples["test"]):
            raise ValueError(f"No valid upstream test images for {label}")
        for partition, subgroup in _partitions(list(groups.values()), rng, val_size, test_size, bool(partitions)):
            samples[partition].extend(record for group in subgroup for record in group)

    reference = []
    for record in samples["train"]:
        with Image.open(root / record["image_path"]) as image:
            reference.append(brightness(image))
    output.mkdir(parents=True, exist_ok=True)
    distribution = Counter(record["label"] for part in ("train", "validation", "test") for record in samples[part])
    reports = {
        "splits.json": samples,
        "class_distribution.json": distribution,
        "validation.json": {
            "seed": random_state, "invalid_images": skipped["invalid_images"], "duplicates": skipped["duplicates"],
            "counts": {part: dict(Counter(row["label"] for row in samples[part]))
                       for part in ("train", "validation", "test")},
            "upstream_test_partition": bool(partitions), "train_brightness_mean": float(np.mean(reference)),
        },
    }
    for name, report in reports.items():
        (output / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return samples


def augment_image(image_array: np.ndarray) -> np.ndarray:
    return np.flip(image_array, axis=1).copy() if np.random.rand() > 0.5 else image_array.copy()
