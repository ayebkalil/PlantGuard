"""Fetch the pinned three-class PlantVillage benchmark used by the demo."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import random
from time import sleep
from urllib.parse import quote
from urllib.request import urlopen

SOURCE = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
METADATA = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/afb5ce38128c7bac085b31d499d28b631ef8fc80"
CLASSES = ("Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy")
ROOT = Path("data/raw/plantvillage")


def fetch(url: str, limit: int = 4_000_000) -> bytes:
    for attempt in range(4):
        try:
            with urlopen(url, timeout=35) as response:
                content = response.read(limit + 1)
            if len(content) > limit:
                raise ValueError("Upstream file exceeded its size limit")
            return content
        except OSError:
            if attempt == 3:
                raise
            sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main() -> None:
    leaf_map = json.loads(fetch(f"{METADATA}/leaf_grouping/leaf-map.json"))
    rng = random.Random(42)
    selection = {}
    for partition, count in (("train", 100), ("test", 32)):
        listing = fetch(f"{METADATA}/splits/color_{partition}.txt", limit=12_000_000).decode().splitlines()
        for label in CLASSES:
            prefix = f"raw/color/{label}/"
            files = sorted({item.removeprefix(prefix) for item in listing if item.startswith(prefix)})
            for name in rng.sample(files, count):
                if name in (".", "..") or PurePosixPath(name).name != name or "\\" in name or ":" in name:
                    raise ValueError(f"Unsafe upstream filename: {name}")
                key = Path(name).stem.split("___", 1)[-1].lower()
                groups = [g for g in leaf_map.get(key, []) if g.split(":::")[0] == label]
                rel = f"{label}/{name}"
                selection[rel] = {"partition": partition, "group": groups[0] if len(groups) == 1 else rel}

    def save(rel: str) -> None:
        destination = ROOT / rel
        if not destination.exists():
            content = fetch(f"{SOURCE}/raw/color/{quote(rel)}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".part")
            temporary.write_bytes(content)
            temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(save, selection))
    (ROOT / "partitions.json").write_text(
        json.dumps(selection, sort_keys=True, indent=2), encoding="utf-8", newline="\n",
    )
    print(f"Fetched {len(selection)} pinned PlantVillage images into {ROOT}")


if __name__ == "__main__":
    main()
