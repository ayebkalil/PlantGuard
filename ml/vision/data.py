"""Dataset adapters for pest detection, weed segmentation, and fruit freshness."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import tarfile
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from ml.preprocessing.preprocess import VALID_EXTENSIONS
from ml.registry import sha256 as file_hash

WEEDS = "https://doidata.gfz.de/weedsgalore_e_celikkan_2024/weedsgalore-dataset.zip"
WEEDS_REV = "0ddf3a77bce282daf9d795de6f6c3d98e7fc9dcc"
WEEDS_SPLITS = f"https://raw.githubusercontent.com/GFZ/weedsgalore/{WEEDS_REV}/src/datasets/weedsgalore/splits"
IP_CLASSES = "https://raw.githubusercontent.com/xpwu95/IP102/924dda98fbb8d012f5f86276c22e6d5a4f2b6196/classes.txt"
IP_FILES = {
    "Annotations.tar": "1TV1jaAp-XwhDupy_xfy4XPcmSFUb83c8",
    "JPEGImages.tar": "1w7Lr4MQf03uREysDMh5Tr1T2xjfSxhRz",
    "test.txt": "1E7qMewUPvcoKT_T9n7OnOZxX9D9yL38Q",
    "trainval.txt": "1MR85vLr5NKntY1oP8dMD5XCS8me0c4Ux",
}
FRUIT_NAMES = [f"{state} {fruit}" for state in ("fresh", "rotten") for fruit in ("apple", "banana", "orange")]


def download(url, path):
    if not path.exists():
        temporary = path.with_suffix(".part")
        urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def unpack(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".extracted"
    if marker.exists():
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if not (target / name).resolve().is_relative_to(target.resolve()):
                    raise ValueError(f"Unsafe archive entry: {name}")
            source.extractall(target)
    else:
        with tarfile.open(archive) as source:
            source.extractall(target, filter="data")
    marker.touch()


def image_files(directory):
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


def fruit_rows(source, limit=None, seed=42):
    input_root = source or Path("/kaggle/input")
    root = next((p.parent for p in sorted(input_root.rglob("train")) if (p / "freshapples").is_dir()), None)
    if root is None or not (root / "test").is_dir():
        raise ValueError("Attach the fresh/rotten Kaggle dataset containing train and test folders.")
    rows = []
    rng = random.Random(seed)
    for split in ("test", "train"):
        directories = {path.name.casefold().rstrip("s"): path for path in (root / split).iterdir() if path.is_dir()}
        for label, name in enumerate(FRUIT_NAMES):
            directory = directories.get(name.replace(" ", ""))
            if directory is None:
                raise ValueError(f"Fruit archive is missing {name} in {split}. Found: {sorted(directories)}")
            images = image_files(directory)
            if not images:
                raise ValueError(f"No fruit images found in {directory}")
            count = max(3, limit // 5) if limit and split == "test" else limit
            if count and len(images) > count:
                images = rng.sample(images, count)
            rows.extend((path, label, split) for path in images)
    return FRUIT_NAMES, rows, {
        "source": "https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification",
    }


def split_classifier(rows, labels, seed, limit):
    # Prefer the official test split when exact duplicates cross its boundary.
    groups = defaultdict(list)
    seen = {}
    conflicts = set()
    for path, label, split in rows:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            digest = sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if digest in seen and seen[digest][1] != label:
            conflicts.add(digest)
        seen.setdefault(digest, (path, label, split))
    for digest, (path, label, split) in seen.items():
        if digest not in conflicts:
            groups[label, split].append((path, label, digest))
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "val", "test")}
    for i, label in enumerate(labels):
        official = groups[i, "test"]
        pool = groups[i, "train"] or groups[i, "pool"]
        rng.shuffle(pool)
        if limit:
            pool = pool[:limit]
            official = official[:max(3, limit // 5)]
        if len(pool) < 4:
            raise ValueError(f"Too few valid images for {label}")
        val_count = max(1, round(.15 * len(pool)))
        if not official:
            test_count = max(1, round(.15 * len(pool)))
            official, pool = pool[:test_count], pool[test_count:]
        splits["test"].extend(official)
        splits["val"].extend(pool[:val_count])
        splits["train"].extend(pool[val_count:])
    return splits, {
        "conflicting_image_hashes_removed": len(conflicts),
        "duplicate_copies_removed": len(rows) - len(seen),
    }


def weed_data(source, work, limit):
    if source is None:
        archive = download(WEEDS, work / "weedsgalore.zip")
        source = work / "weedsgalore"
        unpack(archive, source)
        provenance = {"source": WEEDS, "archive_sha256": file_hash(archive)}
    else:
        provenance = {"source": WEEDS, "input": "attached author archive"}
    base = next(source.rglob("semantics")).parent.parent
    result = {}
    for split in ("train", "val", "test"):
        ids = download(f"{WEEDS_SPLITS}/{split}.txt", work / f"weeds-{split}.txt").read_text().splitlines()
        result[split] = []
        for tile in ids[:limit]:
            date = base / tile[:10]
            bands = tuple(date / "images" / f"{tile}_{band}.png" for band in "RGB")
            mask = date / "semantics" / f"{tile}.png"
            if not all(p.is_file() for p in (*bands, mask)):
                raise FileNotFoundError(f"Missing WeedsGalore tile or RGB band: {tile}")
            pixel_digest = sha256("".join(file_hash(p) for p in (*bands, mask)).encode()).hexdigest()
            result[split].append((bands, mask, pixel_digest))
    return ["background", "maize", "weed"], result, {**provenance, "split_revision": WEEDS_REV}


def download_ip102(work):
    import gdown

    root = work / "ip102"
    root.mkdir(exist_ok=True)
    for name, drive_id in IP_FILES.items():
        destination = root / name
        if not destination.exists():
            temporary = destination.with_suffix(".part")
            if gdown.download(id=drive_id, output=str(temporary), quiet=False) is None:
                raise RuntimeError(f"Official IP102 download failed for {name}. Attach the VOC2007 files.")
            temporary.replace(destination)
        if name.endswith(".tar"):
            unpack(destination, root / destination.stem)
    return root


def voc_boxes(path, classes, width, height):
    root = read_voc(path)
    rows = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "").strip()
        if label not in classes:
            raise ValueError(f"Unknown IP102 annotation class {label!r}")
        box = obj.find("bndbox")
        x1 = max(0, float(box.findtext("xmin")) - 1)
        y1 = max(0, float(box.findtext("ymin")) - 1)
        x2 = min(width, float(box.findtext("xmax")))
        y2 = min(height, float(box.findtext("ymax")))
        if x2 <= x1 or y2 <= y1:
            continue
        values = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
        rows.append(f"{classes[label]} " + " ".join(f"{value:.8f}" for value in values))
    return rows


def read_voc(path):
    # Some official IP102 files have extra bytes after the first VOC document.
    data = path.read_bytes()
    marker = b"</annotation>"
    end = data.find(marker)
    if end >= 0:
        data = data[:end + len(marker)]
    return ET.fromstring(data)


def pest_data(source, work, seed, limit, *, expand_training=False):
    import yaml

    if expand_training and not limit:
        raise ValueError("Training expansion requires the original sample limit")
    root = source or download_ip102(work)
    annotations = {p.stem: p for p in root.rglob("*.xml")}
    images = {p.stem: p for p in image_files(root)}
    def official(name):
        path = next(root.rglob(f"{name}.txt"))
        return sorted({line.split()[0] for line in path.read_text().splitlines() if line.strip()})

    fit, test = official("trainval"), official("test")
    if set(fit) & set(test):
        raise ValueError("IP102 trainval and test IDs overlap")
    random.Random(seed).shuffle(fit)
    count = max(1, round(.15 * len(fit)))
    splits = {"train": fit[count:], "val": fit[:count], "test": test}
    missing = set(fit + test) - (images.keys() & annotations.keys())
    if missing:
        raise ValueError(f"Official IP102 detection subset incomplete ({len(missing)} missing image/XML pairs)")
    # Annotation names are the detection taxonomy, independent of the classification labels file.
    invalid_xml = set()
    observed = set()
    for key in fit + test:
        try:
            names = read_voc(annotations[key]).findall("object/name")
            observed.update(name.text.strip() for name in names if name.text and name.text.strip())
        except ET.ParseError:
            invalid_xml.add(key)
    raw_labels = sorted(observed)
    if not raw_labels:
        raise ValueError("The IP102 input contains no bounding box labels")
    indices = dict(zip(raw_labels, range(len(raw_labels))))
    labels = raw_labels
    if all(label.isdigit() and 0 <= int(label) < 102 for label in raw_labels):
        taxonomy = download(IP_CLASSES, work / "ip102-classes.txt").read_text(encoding="utf-8")
        names = {int(number) - 1: name.strip() for number, name in (line.strip().split(maxsplit=1)
                 for line in taxonomy.splitlines() if line.strip())}
        labels = [f"{names[int(label)]} (IP102 {label})" for label in raw_labels]
    destination = work / ("yolo-expanded" if expand_training else "yolo")
    digests = {}
    image_splits = {}
    def add_images(split, names):
        picture_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        picture_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for key in names:
            img = images[key]
            image_sha = file_hash(img)
            if image_sha in image_splits and image_splits[image_sha] != split:
                continue
            with Image.open(img) as image:
                boxes = voc_boxes(annotations[key], indices, *image.size)
            if not boxes:
                continue
            image_splits[image_sha] = split
            target = picture_dir / img.name
            if not target.exists() or file_hash(target) != image_sha:
                shutil.copyfile(img, target)
            (label_dir / f"{img.stem}.txt").write_text("\n".join(boxes), newline="\n")
            digests[split].append((key, image_sha, file_hash(annotations[key])))

    for split, names in splits.items():
        digests[split] = []
        add_images(split, [key for key in names if key not in invalid_xml][:limit])
        if not digests[split]:
            raise ValueError(f"IP102 split {split} has no boxed images")

    original_split_sha = split_hash(digests)
    if expand_training:
        # Reserve the exact original eval set before admitting more training images.
        add_images("train", [key for key in splits["train"] if key not in invalid_xml][limit:])
    manifest = destination / "data.yaml"
    split_paths = {}
    for split, records in digests.items():
        split_file = destination / f"{split}.txt"
        split_file.write_text("".join(
            f"./images/{split}/{images[key].name}\n" for key, _, _ in records
        ), encoding="utf-8")
        split_paths[split] = split_file.name
    manifest.write_text(yaml.safe_dump({"path": str(destination.resolve()), **split_paths, "names": labels}))
    return labels, manifest, digests, {"source": "https://github.com/xpwu95/IP102", "release": "v1.1 VOC2007",
                                      "official_test": True, "annotation_classes": len(labels),
                                      "original_split_sha256": original_split_sha,
                                      "invalid_annotations_omitted": len(invalid_xml)}


def split_hash(splits):
    return sha256(json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
