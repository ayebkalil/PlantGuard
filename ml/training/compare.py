"""Rank every model trained in the MLflow experiment and recommend one.

    python -m ml.training.compare
    python -m ml.training.compare --max-latency-ms 500 --out models/model_comparison.csv

For the interactive view open MLflow, tick the runs and click "Compare":
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from ml.training.train import DEFAULT_TRACKING_URI, EXPERIMENT_NAME, _configure_mlflow

COLUMNS = {
    "tags.owner": "owner",
    "params.model_name": "model",
    "params.epochs": "epochs",
    "params.split_hash": "split",
    "metrics.best_val_macro_f1": "val_macro_f1",
    "metrics.test_accuracy": "test_acc",
    "metrics.test_macro_f1": "test_macro_f1",
    "metrics.latency_ms_cpu": "latency_ms",
    "metrics.model_size_mb": "size_mb",
    "metrics.train_time_s": "train_s",
}


def load_runs(experiment_name: str = EXPERIMENT_NAME) -> pd.DataFrame:
    _configure_mlflow(experiment_name)
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="attributes.status = 'FINISHED'")
    if runs.empty:
        return runs
    runs = runs.rename(columns=COLUMNS)
    keep = [c for c in COLUMNS.values() if c in runs.columns]
    table = runs[["run_id", *keep]].copy()
    table["run_id"] = table["run_id"].str[:8]
    if "train_s" in table:
        table["train_min"] = (table.pop("train_s") / 60).round(1)
    return table.dropna(subset=["val_macro_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLflow runs of the PlantGuard benchmark.")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--max-latency-ms", type=float, default=500.0,
                        help="models slower than this on CPU are not recommended (params.yaml: 500)")
    parser.add_argument("--out", default="models/model_comparison.csv")
    args = parser.parse_args()

    table = load_runs(args.experiment)
    if table.empty:
        print(f"No finished runs in experiment '{args.experiment}' (tracking store: {DEFAULT_TRACKING_URI}).")
        return

    table = table.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(table.to_string(index=False))

    if table["split"].nunique() > 1:
        print("\nWARNING: runs used different train/val/test splits, so they are not directly comparable.")

    eligible = table[table["latency_ms"] <= args.max_latency_ms]
    if eligible.empty:
        print(f"\nNo model meets the {args.max_latency_ms:.0f} ms CPU latency limit.")
    else:
        best = eligible.iloc[0]
        print(f"\nRecommended: {best['model']} (owner: {best['owner']}) - best validation macro F1 "
              f"{best['val_macro_f1']:.4f} among models under {args.max_latency_ms:.0f} ms "
              f"(test macro F1 {best['test_macro_f1']:.4f}, {best['latency_ms']:.1f} ms).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()