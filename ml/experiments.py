"""Train the branch candidate and record its comparison with the published baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline(task):
    if task == "disease":
        report = read_json(Path("reports/benchmark.json"))
        winner = next(run for run in report["experiments"] if run["model_name"] == report["best_model"])
        return winner, winner["split_sha256"], "macro_f1"
    report = read_json(Path(f"reports/vision/{task}.json"))
    winner = next(run for run in report["runs"] if run["bundle"] == report["winner"])
    return winner, report["split_sha256"], report["validation_metric"]


def run_candidate(recipe, output, dataset, tracking_uri, require_gpu):
    import mlflow

    task, model = recipe["task"], recipe["model"]
    baseline, expected_split, primary = load_baseline(task)
    baseline_id = baseline.get("mlflow_run_id") or baseline["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if require_gpu and task == "disease":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A GPU is required for this experiment.")
    if task == "disease":
        command = ["-m", "ml.training.train", "--no-promote", "--output-dir", str(output)]
        if dataset:
            command.extend(["--dataset-root", str(dataset)])
    else:
        command = ["-m", "ml.vision.train", task, "--output", str(output), "--work", str(Path("data/vision").resolve())]
        if require_gpu:
            command.append("--require-gpu")
        if dataset:
            command.extend(["--data", str(dataset)])
    command.extend(["--models", model, "--seed", str(recipe["seed"]), "--experiment", recipe["experiment"],
                    "--expected-split", expected_split, *recipe["args"]])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    subprocess.run([sys.executable, "-u", *command], env=env, check=True)

    if task == "disease":
        candidate = read_json(output / f"model_metadata_{model}.json")
        metrics = candidate
        candidate_id = candidate["run_id"]
        version = candidate["version"]
    else:
        report = read_json(output / "comparison.json")
        candidate = read_json(output / report["winner"] / "bundle.json")
        metrics = candidate["metrics"]
        candidate_id = candidate["source_run_id"]
        version = f"{candidate_id[:12]}-{candidate['checkpoint_sha256'][:12]}"
    if candidate["split_sha256"] != expected_split:
        raise ValueError("Candidate result does not match the baseline split.")

    comparable = sorted(name for name in baseline.keys() & metrics.keys()
                        if name.startswith(("val_", "test_")))
    comparison = {
        "task": task,
        "experiment": recipe["experiment"],
        "git_sha": candidate.get("params", {}).get("git_sha", commit),
        "split_sha256": expected_split,
        "primary_validation_metric": f"val_{primary}",
        "baseline": {key: baseline[key] for key in ("architecture", "version", "checkpoint_sha256")},
        "candidate": {"architecture": model, "version": version, "checkpoint_sha256": candidate["checkpoint_sha256"]},
        "metrics": {name: {"baseline": baseline[name], "candidate": metrics[name],
                           "delta": metrics[name] - baseline[name]} for name in comparable},
        "latency_p95_ms_cpu": {
            "baseline": baseline.get("latency_p95_ms_cpu", baseline.get("cpu_p95_ms")),
            "candidate": metrics.get("latency_p95_ms_cpu", metrics.get("cpu_p95_ms")),
        },
    }
    comparison["baseline"]["mlflow_run_id"] = baseline_id
    comparison["candidate"]["mlflow_run_id"] = candidate_id
    result = output / "baseline-comparison.json"
    result.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    client = mlflow.MlflowClient(tracking_uri)
    client.set_tag(candidate_id, "plantguard.baseline_run_id", baseline_id)
    client.set_tag(candidate_id, "plantguard.candidate_version", version)
    client.set_tag(candidate_id, "plantguard.git_sha", comparison["git_sha"])
    client.set_tag(candidate_id, "plantguard.split_sha256", expected_split)
    for name, values in comparison["metrics"].items():
        client.log_metric(candidate_id, f"baseline_{name}", values["baseline"])
        client.log_metric(candidate_id, f"candidate_{name}", values["candidate"])
        client.log_metric(candidate_id, f"delta_{name}", values["delta"])
    client.log_artifact(candidate_id, str(result))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(output / 'mlflow.db').as_posix()}"
    recipe = read_json(Path("experiments/candidate.json"))
    run_candidate(recipe, output, args.data, tracking_uri, args.require_gpu)
