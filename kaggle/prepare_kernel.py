"""Generate private Kaggle kernel folders tied to an exact Git commit."""
import argparse
import json
from pathlib import Path
import re
import subprocess

MODELS = {
    "banana": ["efficientnet_b0", "mobilenet_v3_small"],
    "fruit": ["efficientnet_b0", "mobilenet_v3_small"],
    "weeds": ["unet16", "unet32"],
    "pests": ["yolov8n", "yolov8s"],
}
DATASETS = {
    "fruit": ["sriramr/fruits-fresh-and-rotten-for-classification"],
    "pests": [],
    "weeds": [],
    "banana": [],
}


def generate(task, username, root, commit, epochs, limit, dataset=None, one_model=False):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Provide the full Git commit SHA")
    target = root / task
    target.mkdir(parents=True, exist_ok=True)
    name = f"plantguard-{task}-training"
    sources = DATASETS[task].copy()
    if dataset:
        sources.append(dataset)
    metadata = {"id": f"{username}/{name}", "title": name.replace("-", " ").title(),
                "code_file": "train.py", "language": "python", "kernel_type": "script",
                "enable_gpu": True, "enable_internet": True, "is_private": True,
                "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    args = ["-m", "ml.vision.train", task, "--work", f"/tmp/plantguard-{task}", "--require-gpu",
            "--output", f"/kaggle/working/{task}", "--size", "416" if task == "pests" else "224",
            "--batch", "16", "--epochs", str(epochs), "--models", *(MODELS[task][:1] if one_model else MODELS[task])]
    if limit:
        args.extend(["--limit", str(limit)])
    script = f"""import os
from pathlib import Path
import subprocess
import sys

repo = Path('/tmp/PlantGuard')
repo.mkdir(exist_ok=True)
subprocess.run(['git', 'init', '-q', str(repo)], check=True)
subprocess.run(['git', 'fetch', '--depth', '1', 'https://github.com/ayebkalil/PlantGuard.git', {commit!r}], cwd=repo, check=True)
subprocess.run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=repo, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'kaggle/requirements.txt'], cwd=repo, check=True)
env = os.environ.copy()
env['MLFLOW_DISABLE_AGENT_HINT'] = '1'
args = {args!r}
"""
    if dataset and task in ("pests", "weeds"):
        script += f"""
input_root = Path('/kaggle/input')
# Derive the attachment folder from the Kaggle dataset slug.
source = next(input_root.rglob({dataset.rsplit('/', 1)[-1]!r}))
args.extend(['--data', str(source)])
"""
    script += "\nsubprocess.run([sys.executable, '-u', *args], cwd=repo, env=env, check=True)\n"
    (target / "train.py").write_text(script, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=MODELS)
    parser.add_argument("--username", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", default=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset", help="Optional owner/slug for the private author archive input.")
    parser.add_argument("--one-model", action="store_true")
    args = parser.parse_args()
    generate(args.task, args.username, args.output, args.commit, args.epochs, args.limit, args.dataset, args.one_model)
