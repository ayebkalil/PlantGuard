"""Import every candidate of a completed vision comparison and activate the winner."""
import argparse
import json
from pathlib import Path

from ml.registry import save_json
from ml.vision.artifacts import install, verify_bundle


def collect(comparison, source, root, report_dir, promote):
    comparison = comparison.resolve()
    results = json.loads(comparison.read_text(encoding="utf-8"))
    for row in results["runs"]:
        bundle = (comparison.parent / row["bundle"]).resolve()
        if not bundle.is_relative_to(comparison.parent):
            raise ValueError("Bundle path leaves the comparison directory")
        metadata = verify_bundle(bundle)
        if metadata["task"] != results["task"] or metadata["split_sha256"] != results["split_sha256"]:
            raise ValueError("Comparison and bundle provenance disagree")
        metadata = install(bundle, root, promote and row["bundle"] == results["winner"])
        row.update({name: metadata[name] for name in ("version", "source_run_id", "mlflow_run_id", "checkpoint_sha256")})
        row["params"] = metadata["params"]
        results["dataset"] = metadata["dataset"]
    results["execution_source"] = source
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_dir / f"{results['task']}.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--source", required=True, help="Kaggle kernel version URL or local training environment.")
    parser.add_argument("--root", type=Path, default=Path("vision-models"))
    parser.add_argument("--reports", type=Path, default=Path("reports/vision"))
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    collect(args.comparison, args.source, args.root, args.reports, args.promote)
