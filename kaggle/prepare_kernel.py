"""Generate private Kaggle kernel folders tied to an exact Git commit."""
import argparse
import json
from pathlib import Path
import re
import subprocess

MODELS = {
    "disease": ["simple_cnn", "mobilenet_v3_small"],
    "fruit": ["efficientnet_b0", "mobilenet_v3_small"],
    "weeds": ["unet16", "unet32"],
    "pests": ["yolov8n", "yolov8s"],
}
DATASETS = {
    "fruit": ["sriramr/fruits-fresh-and-rotten-for-classification"],
    "pests": [],
    "weeds": [],
    "disease": [],
}


def generate(task, username, root, commit, epochs, limit, dataset=None, one_model=False, experiment=False):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Provide the full Git commit SHA")
    target = root / task
    target.mkdir(parents=True, exist_ok=True)
    suffix = "experiment" if experiment else "training"
    name = f"plantguard-{task}-{suffix}"
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
    if task == "disease":
        args = ["-m", "ml.training.train", "--epochs", str(epochs), "--batch-size", "16",
                "--image-size", "128", "--lr", "0.001", "--num-workers", "0", "--models",
                *(MODELS[task][:1] if one_model else MODELS[task])]
        if limit:
            args.extend(["--max-samples-per-class", str(limit)])
    if experiment:
        args = ["-m", "ml.experiments", "--output", f"/kaggle/working/{task}", "--require-gpu"]
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
run_dir = repo
"""
    if dataset:
        script += f"""
input_root = Path('/kaggle/input')
# Derive the attachment folder from the Kaggle dataset slug.
source = next(input_root.rglob({dataset.rsplit('/', 1)[-1]!r}))
"""
    if dataset and task in ("pests", "weeds"):
        script += "args.extend(['--data', str(source)])\n"
    if task == "disease":
        if not dataset:
            script += """
subprocess.run([sys.executable, '-m', 'ml.data.fetch_plantvillage'], cwd=repo, check=True)
source = repo / 'data/raw/plantvillage'
"""
        script += f"""
import torch

if not torch.cuda.is_available():
    raise RuntimeError('A Kaggle GPU must be selected for this run.')
if {not experiment!r}:
    run_dir = Path('/kaggle/working/disease')
    run_dir.mkdir(parents=True, exist_ok=True)
    env['PYTHONPATH'] = str(repo)
args.extend([{('--data' if experiment else '--dataset-root')!r}, str(source)])
"""
    script += "\nsubprocess.run([sys.executable, '-u', *args], cwd=run_dir, env=env, check=True)\n"
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
    parser.add_argument("--experiment", action="store_true", help="run the checked-in experiment recipe")
    args = parser.parse_args()
    if args.experiment:
        recipe = json.loads(Path("experiments/candidate.json").read_text(encoding="utf-8"))
        if args.task != recipe["task"]:
            parser.error(f"This branch contains an experiment for {recipe['task']}.")
    generate(args.task, args.username, args.output, args.commit, args.epochs, args.limit,
             args.dataset, args.one_model, args.experiment)
