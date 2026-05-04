"""Backfill MLflow experiment tracking for GRU4Rec V9 500k and 1M runs.

Does NOT retrain. Reads training_log.json + hparams.json from GCS (or local)
and logs them into MLflow. Run this once after deploying the MLflow server.

Usage:
    python scripts/serving/log_experiments_mlflow.py --mlflow-uri <URL>

    # Or with a local MLflow server:
    mlflow server --backend-store-uri sqlite:///mlruns.db &
    python scripts/serving/log_experiments_mlflow.py --mlflow-uri http://localhost:5000
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# ── GCS paths ─────────────────────────────────────────────────────────────────
GCS_500K_DIR = "gs://recosys-data-bucket/models/gru4rec_session_v9"
GCS_1M_DIR   = "gs://recosys-data-bucket/models/gru4rec_session_v9_1M"

# ── Fallback constants (from reports/07_session_model_results.md) ──────────────
_500K_HPARAMS_FALLBACK: dict = {
    "model":           "GRU4RecModel",
    "schema_version":  "v9_session",
    "n_items":         284524,
    "embed_dim":       128,
    "gru_hidden":      256,
    "n_layers":        2,
    "dropout":         0.3,
    "max_seq_len":     20,
    "batch_size":      512,
    "lr":              3e-4,
    "weight_decay":    1e-5,
    "label_smoothing": 0.1,
    "temperature":     0.07,
    "grad_clip":       1.0,
    "patience":        5,
    "scheduler":       "cosine",
    "lr_min":          1e-5,
    "seed":            42,
    "loss":            "full_softmax",
    "train_sessions":  1450480,
    "dataset":         "REES46_500k",
}

_500K_BEST_METRICS_FALLBACK: dict = {
    "best_val_ndcg_20": 0.2606,
    "best_val_hr_20":   0.449,
    "best_val_ndcg_10": 0.2101,
    "best_val_hr_10":   0.3524,
    "best_epoch":       15,
    "pop_val_ndcg_20":  0.0341,
    "pop_val_hr_20":    0.0777,
}

_1M_HPARAMS_FALLBACK: dict = {
    "model":           "GRU4RecModel",
    "schema_version":  "v9_session",
    "n_items":         222864,
    "embed_dim":       128,
    "gru_hidden":      256,
    "n_layers":        1,
    "dropout":         0.3,
    "max_seq_len":     20,
    "batch_size":      512,
    "lr":              3e-4,
    "weight_decay":    1e-5,
    "label_smoothing": 0.1,
    "temperature":     0.07,
    "grad_clip":       1.0,
    "patience":        5,
    "scheduler":       "cosine",
    "lr_min":          1e-5,
    "seed":            42,
    "loss":            "gBCE",
    "train_sessions":  2884945,
    "dataset":         "REES46_1M",
}


def _try_gcs_json(gcs_path: str) -> dict | list | None:
    """Try to download a JSON file from GCS; return None on any failure."""
    try:
        from google.cloud import storage  # type: ignore
        bucket_name, blob_name = gcs_path.removeprefix("gs://").split("/", 1)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            blob.download_to_filename(f.name)
            return json.loads(Path(f.name).read_text())
    except Exception as exc:
        print(f"  [warn] Could not fetch {gcs_path}: {exc}")
        return None


def _load_run_data(gcs_dir: str, fallback_hparams: dict, dataset_label: str):
    """Return (hparams, training_log_list) for a run, using GCS then fallbacks."""
    hparams = _try_gcs_json(f"{gcs_dir}/hparams.json") or fallback_hparams
    training_log = _try_gcs_json(f"{gcs_dir}/training_log.json")
    if training_log is None:
        print(f"  [warn] No training_log.json for {dataset_label} — using best metrics only")
    return hparams, training_log


def _log_run(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
    hparams: dict,
    training_log: list | None,
    best_metrics: dict,
    dataset_size: str,
    gcs_checkpoint_dir: str,
) -> str:
    """Create one MLflow run, log params + per-epoch metrics + tags. Return run_id."""
    run = client.create_run(experiment_id, run_name=run_name)
    run_id = run.info.run_id

    # Params — everything in hparams except timestamps
    skip = {"started_at", "finished_at", "schema_version"}
    for k, v in hparams.items():
        if k not in skip:
            client.log_param(run_id, k, v)

    # Per-epoch NDCG curves
    if training_log:
        for entry in training_log:
            epoch = int(entry.get("epoch", 0))
            for metric_key in ("val_ndcg_20", "val_hr_20", "val_ndcg_10", "val_hr_10",
                               "train_loss"):
                if metric_key in entry and entry[metric_key] is not None:
                    client.log_metric(run_id, metric_key, float(entry[metric_key]), step=epoch)
    else:
        # Log only the best epoch point
        epoch = int(best_metrics.get("best_epoch", 0))
        for k, v in best_metrics.items():
            if k.startswith(("best_val_", "pop_val_")):
                metric_name = k.replace("best_", "")
                client.log_metric(run_id, metric_name, float(v), step=epoch)

    # Summary metrics at the best epoch
    for k, v in best_metrics.items():
        client.log_metric(run_id, k, float(v))

    # Tags
    client.set_tag(run_id, "dataset_size",        dataset_size)
    client.set_tag(run_id, "model",               "gru4rec_v9")
    client.set_tag(run_id, "checkpoint_dir",      gcs_checkpoint_dir)
    client.set_tag(run_id, "dataset",             "REES46")
    client.set_tag(run_id, "eval_protocol",       "T4Rec_session")
    client.set_tag(run_id, "mlflow.runName",      run_name)

    client.set_terminated(run_id, "FINISHED")
    print(f"  Logged run '{run_name}'  run_id={run_id}")
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", required=True, help="MLflow tracking server URI")
    parser.add_argument("--experiment-name", default="gru4rec_session")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = MlflowClient(args.mlflow_uri)

    # Create (or get existing) experiment
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(
            args.experiment_name,
            artifact_location="gs://recosys-mlflow-artifacts/",
        )
        print(f"Created experiment '{args.experiment_name}' (id={experiment_id})")
    else:
        experiment_id = exp.experiment_id
        print(f"Using existing experiment '{args.experiment_name}' (id={experiment_id})")

    # ── 500k run ──────────────────────────────────────────────────────────────
    print("\n--- Loading 500k run data ---")
    hparams_500k, log_500k = _load_run_data(GCS_500K_DIR, _500K_HPARAMS_FALLBACK, "500k")
    hparams_500k.setdefault("train_sessions", 1450480)
    hparams_500k.setdefault("dataset", "REES46_500k")

    best_500k = {
        "best_val_ndcg_20": 0.2606,
        "best_val_hr_20":   0.449,
        "best_val_ndcg_10": 0.2101,
        "best_val_hr_10":   0.3524,
        "best_epoch":       15,
        "pop_val_ndcg_20":  0.0341,
        "pop_val_hr_20":    0.0777,
    }
    # Override with actual best from training log if available
    if log_500k:
        best_entry = max(log_500k, key=lambda e: e.get("val_ndcg_20", 0))
        best_500k["best_val_ndcg_20"] = best_entry.get("val_ndcg_20", best_500k["best_val_ndcg_20"])
        best_500k["best_val_hr_20"]   = best_entry.get("val_hr_20",   best_500k["best_val_hr_20"])
        best_500k["best_epoch"]       = int(best_entry.get("epoch",   best_500k["best_epoch"]))

    run_id_500k = _log_run(
        client, experiment_id,
        run_name            = "gru4rec_v9_500k",
        hparams             = hparams_500k,
        training_log        = log_500k,
        best_metrics        = best_500k,
        dataset_size        = "500k",
        gcs_checkpoint_dir  = GCS_500K_DIR,
    )

    # ── 1M run ────────────────────────────────────────────────────────────────
    print("\n--- Loading 1M run data ---")
    hparams_1m, log_1m = _load_run_data(GCS_1M_DIR, _1M_HPARAMS_FALLBACK, "1M")
    hparams_1m.setdefault("train_sessions", 2884945)
    hparams_1m.setdefault("dataset", "REES46_1M")

    best_1m = {
        "best_val_ndcg_20": 0.2676,
        "best_val_hr_20":   0.4815,
        "best_val_ndcg_10": 0.2420,
        "best_val_hr_10":   0.3803,
        "best_epoch":       24,
        "pop_val_ndcg_20":  0.0353,
        "pop_val_hr_20":    0.0806,
    }
    if log_1m:
        best_entry = max(log_1m, key=lambda e: e.get("val_ndcg_20", 0))
        best_1m["best_val_ndcg_20"] = best_entry.get("val_ndcg_20", best_1m["best_val_ndcg_20"])
        best_1m["best_val_hr_20"]   = best_entry.get("val_hr_20",   best_1m["best_val_hr_20"])
        best_1m["best_epoch"]       = int(best_entry.get("epoch",   best_1m["best_epoch"]))

    run_id_1m = _log_run(
        client, experiment_id,
        run_name            = "gru4rec_v9_1m",
        hparams             = hparams_1m,
        training_log        = log_1m,
        best_metrics        = best_1m,
        dataset_size        = "1M",
        gcs_checkpoint_dir  = GCS_1M_DIR,
    )
    client.set_tag(run_id_1m, "production", "true")

    # ── Model Registry ────────────────────────────────────────────────────────
    print("\n--- Registering models ---")
    model_name = "GRU4Rec-V9"

    try:
        client.create_registered_model(model_name, description="GRU4Rec V9 session-based recommender trained on REES46 eCommerce dataset.")
        print(f"  Created registered model '{model_name}'")
    except Exception:
        print(f"  Registered model '{model_name}' already exists")

    # Version 1: 500k
    mv1 = client.create_model_version(
        name        = model_name,
        source      = f"{GCS_500K_DIR}/best_checkpoint.pt",
        run_id      = run_id_500k,
        description = "GRU4Rec V9 trained on 500k users. NDCG@20=0.2606.",
    )
    client.set_model_version_tag(model_name, mv1.version, "dataset_size", "500k")

    # Version 2: 1M (mark as Production)
    mv2 = client.create_model_version(
        name        = model_name,
        source      = f"{GCS_1M_DIR}/best_checkpoint.pt",
        run_id      = run_id_1m,
        description = "GRU4Rec V9 trained on 1M users. NDCG@20=0.2676. Production model.",
    )
    client.set_model_version_tag(model_name, mv2.version, "dataset_size", "1M")
    client.set_registered_model_alias(model_name, "production", mv2.version)

    print(f"\n=== Done ===")
    print(f"Experiment : {args.experiment_name}")
    print(f"Runs       : gru4rec_v9_500k ({run_id_500k[:8]}…)  |  gru4rec_v9_1m ({run_id_1m[:8]}…)")
    print(f"Registry   : {model_name}  v{mv1.version} (500k)  v{mv2.version} (1M, production)")
    print(f"UI         : {args.mlflow_uri}/#/experiments/{experiment_id}")


if __name__ == "__main__":
    main()
