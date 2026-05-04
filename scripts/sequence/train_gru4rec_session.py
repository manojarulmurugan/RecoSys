"""GRU4Rec session-based training — V9 / T4Rec reframe.

Trains GRU4Rec with full softmax + label smoothing on the session sequences
built by build_session_sequences.py.  Replaces the user-level V7 run
(train_gru4rec.py) without touching it.

Key changes vs V7:
  - Full softmax over all catalog items (no sampled negatives)
  - Label smoothing 0.1 (T4Rec sec. 3.2)
  - Session-aware training unit: one session per example, not one user
  - Early stopping on val NDCG@20 (session-based, single-item GT)
  - MLflow logging (graceful no-op when MLflow is not configured)

Outputs (all under --checkpoint-dir):
  best_checkpoint.pt      Best val-NDCG@20 checkpoint (model + optimizer state)
  latest_checkpoint.pt    End-of-epoch checkpoint (for resuming)
  training_log.json       Per-epoch metrics list (written after every epoch)
  hparams.json            Hyperparameter snapshot

Usage (Colab A100 recommended):
    python scripts/sequence/train_gru4rec_session.py

  Override defaults:
    python scripts/sequence/train_gru4rec_session.py \\
        --batch-size 128 --epochs 20 --gru-hidden 256

  With MLflow (configure after Day 6):
    python scripts/sequence/train_gru4rec_session.py \\
        --mlflow-tracking-uri http://<mlflow-cloud-run-url> \\
        --run-name gru4rec_session_v9_500k
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.sequence.data.session_dataset import SessionEvalDataset, SessionTrainDataset


# ── GCS helpers (used when running on Vertex AI) ───────────────────────────────

def _ensure_gcp_credentials() -> None:
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing and pathlib.Path(existing).expanduser().is_file():
        return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        _REPO_ROOT / "recosys-service-account.json",
        pathlib.Path.home() / "secrets" / "recosys-service-account.json",
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate.resolve())
            return


def _maybe_download_gcs_artifacts(artifacts_dir_str: str) -> pathlib.Path:
    """If artifacts_dir is a gs:// path, download required files to /tmp/ and return that path.

    On Vertex AI there is no local copy of the data — ADC provides auth automatically.
    For local runs with a gs:// path, GOOGLE_APPLICATION_CREDENTIALS must be set.
    If artifacts_dir is a local path, returns _REPO_ROOT / artifacts_dir_str unchanged.
    """
    if not artifacts_dir_str.startswith("gs://"):
        return _REPO_ROOT / artifacts_dir_str

    from google.cloud import storage as gcs_storage

    local_dir = pathlib.Path("/tmp/gru4rec_artifacts")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Parse gs://bucket/prefix
    without_scheme = artifacts_dir_str[5:]
    bucket_name, prefix = without_scheme.split("/", 1)
    prefix = prefix.rstrip("/")

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)

    files_needed = [
        f"{prefix}/vocabs.pkl",
        f"{prefix}/sequences_v2/train_sessions.parquet",
        f"{prefix}/sequences_v2/val_sessions.parquet",
        f"{prefix}/sequences_v2/metadata.json",
    ]
    for blob_name in files_needed:
        rel = blob_name[len(prefix) + 1:]
        local_path = local_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            print(f"  [GCS] downloading {blob_name} -> {local_path}")
            bucket.blob(blob_name).download_to_filename(str(local_path))
        else:
            print(f"  [GCS] cached     {local_path}")

    return local_dir


def _gcs_checkpoint_upload(local_path: pathlib.Path, gcs_ckpt_dir: str | None) -> None:
    """Upload a checkpoint file to GCS after torch.save() (no-op for local runs)."""
    if not gcs_ckpt_dir or not gcs_ckpt_dir.startswith("gs://"):
        return
    try:
        from google.cloud import storage as gcs_storage
        without_scheme = gcs_ckpt_dir[5:]
        bucket_name, prefix = without_scheme.split("/", 1)
        prefix = prefix.rstrip("/")
        client = gcs_storage.Client()
        blob_name = f"{prefix}/{local_path.name}"
        client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))
        print(f"  [GCS] uploaded   {local_path.name} -> gs://{bucket_name}/{blob_name}")
    except Exception as exc:
        print(f"  [GCS] upload failed for {local_path.name}: {exc}")
from src.sequence.evaluation.evaluate_sequence import evaluate_sessions
from src.sequence.models.gru4rec import GRU4RecModel
from src.sequence.training.train_sequence import get_param_groups, train_epoch_session


# ── Helpers ────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── MLflow (optional — gracefully skipped if not installed / configured) ───────

def _mlflow_start(tracking_uri: str | None, run_name: str) -> Any:
    if not tracking_uri:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("recosys_session_v9")
        run = mlflow.start_run(run_name=run_name)
        print(f"  MLflow run  : {run.info.run_id}  @ {tracking_uri}")
        return run
    except Exception as exc:
        print(f"  MLflow unavailable ({exc}) — skipping tracking.")
        return None


def _mlflow_log_params(run: Any, params: dict) -> None:
    if run is None:
        return
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception:
        pass


def _mlflow_log_metrics(run: Any, metrics: dict, step: int) -> None:
    if run is None:
        return
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def _mlflow_end(run: Any) -> None:
    if run is None:
        return
    try:
        import mlflow
        mlflow.end_run()
    except Exception:
        pass


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRU4Rec session-based training V9 (full softmax, T4Rec reframe)",
    )
    # Paths
    p.add_argument("--artifacts-dir",   default="artifacts/500k")
    p.add_argument("--checkpoint-dir",  default=None,
                   help="Defaults to <artifacts-dir>/sequences_v2/checkpoints_v9_gru4rec_session")
    # Model
    p.add_argument("--embed-dim",    type=int,   default=128)
    p.add_argument("--gru-hidden",   type=int,   default=256)
    p.add_argument("--n-layers",     type=int,   default=1)
    p.add_argument("--dropout",      type=float, default=0.3)
    # Training
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--batch-size",       type=int,   default=256,
                   help="256 for A100 (11 GB peak). Use 64-128 for V100/T4.")
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--temperature",      type=float, default=0.07,
                   help="Logit temperature for full-softmax (cosine sim / temperature). "
                        "0.07 is standard for contrastive cosine-similarity losses.")
    p.add_argument("--lr-min",           type=float, default=1e-5,
                   help="Minimum LR for CosineAnnealingLR scheduler.")
    p.add_argument("--scheduler",        type=str,   default="cosine",
                   choices=["cosine", "none"],
                   help="LR scheduler: cosine (CosineAnnealingLR) or none.")
    p.add_argument("--weight-decay",     type=float, default=1e-5)
    p.add_argument("--label-smoothing",  type=float, default=0.1)
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--patience",         type=int,   default=5,
                   help="Early-stop patience in epochs (on val NDCG@20).")
    p.add_argument("--num-workers",      type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    # Evaluation
    p.add_argument("--val-batch-size",   type=int,   default=512)
    p.add_argument("--n-faiss-cands",    type=int,   default=50,
                   help="FAISS candidates per session (>= 20 for HR@20).")
    # MLflow
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--run-name",            default="gru4rec_session_v9_500k")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    _ensure_gcp_credentials()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Paths ─────────────────────────────────────────────────────────────
    # Download from GCS to /tmp/ when running on Vertex AI (gs:// prefix);
    # behaves identically to before for local paths.
    artifacts_dir = _maybe_download_gcs_artifacts(args.artifacts_dir)
    seq_dir       = artifacts_dir / "sequences_v2"

    # _gcs_ckpt_dir holds the original gs:// target for post-save uploads.
    # Local ckpt_dir is /tmp/ when targeting GCS, otherwise local as before.
    _gcs_ckpt_dir: str | None = None
    if args.checkpoint_dir and args.checkpoint_dir.startswith("gs://"):
        _gcs_ckpt_dir = args.checkpoint_dir
        ckpt_dir = pathlib.Path("/tmp/gru4rec_checkpoints")
    else:
        ckpt_dir = (
            pathlib.Path(args.checkpoint_dir) if args.checkpoint_dir
            else seq_dir / "checkpoints_v9_gru4rec_session"
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _section("GRU4Rec Session Training V9")
    print(f"  Device          : {device}")
    print(f"  Artifacts dir   : {artifacts_dir}")
    print(f"  Sequences dir   : {seq_dir}")
    print(f"  Checkpoint dir  : {ckpt_dir}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Max epochs      : {args.epochs}")
    print(f"  Patience        : {args.patience}")
    print(f"  embed_dim       : {args.embed_dim}")
    print(f"  gru_hidden      : {args.gru_hidden}  (n_layers={args.n_layers})")
    print(f"  dropout         : {args.dropout}")
    print(f"  lr              : {args.lr}  wd={args.weight_decay}")
    print(f"  temperature     : {args.temperature}")
    print(f"  scheduler       : {args.scheduler}  (lr_min={args.lr_min})")
    print(f"  label_smoothing : {args.label_smoothing}")

    # ── Load vocabs ───────────────────────────────────────────────────────
    t0 = time.time()
    print("\n  > Loading vocabs...")
    with open(artifacts_dir / "vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)
    n_users = len(vocabs["user2idx"])
    n_items = len(vocabs["item2idx"]) + 1  # +1 for PAD token at idx 0
    print(f"     {n_users:,} users  /  {n_items:,} items (incl. PAD)")

    # ── Load session parquets ─────────────────────────────────────────────
    print("\n  > Loading session parquets...")
    train_df = pd.read_parquet(seq_dir / "train_sessions.parquet")
    val_df   = pd.read_parquet(seq_dir / "val_sessions.parquet")
    with open(seq_dir / "metadata.json") as f:
        meta = json.load(f)
    max_seq_len = int(meta["max_seq_len"])
    print(f"     train: {len(train_df):,} sessions")
    print(f"     val  : {len(val_df):,} sessions")
    print(f"     max_seq_len: {max_seq_len}")

    # ── Build datasets ────────────────────────────────────────────────────
    _section("Building datasets")
    train_ds = SessionTrainDataset(train_df, max_seq_len=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = device.type == "cuda",
        drop_last   = False,
    )

    # Val eval dataset — built once, reused every epoch.
    val_eval_ds = SessionEvalDataset(val_df, max_seq_len=max_seq_len)
    print(f"\n  Train loader : {len(train_loader):,} batches x {args.batch_size}")

    # ── Build model ───────────────────────────────────────────────────────
    _section("Model")
    model = GRU4RecModel(
        n_items       = n_items,
        n_event_types = 4,
        embed_dim     = args.embed_dim,
        gru_hidden    = args.gru_hidden,
        n_layers      = args.n_layers,
        dropout       = args.dropout,
    ).to(device)
    model.model_summary()

    optimizer = optim.AdamW(
        get_param_groups(model, lr=args.lr, weight_decay=args.weight_decay),
    )

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min,
        )

    # ── Save hyperparameters ──────────────────────────────────────────────
    hparams: dict[str, Any] = {
        "model":            "GRU4RecModel",
        "schema_version":   "v9_session",
        "n_items":          n_items,
        "n_users":          n_users,
        "embed_dim":        args.embed_dim,
        "gru_hidden":       args.gru_hidden,
        "n_layers":         args.n_layers,
        "dropout":          args.dropout,
        "max_seq_len":      max_seq_len,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "weight_decay":     args.weight_decay,
        "label_smoothing":  args.label_smoothing,
        "temperature":      args.temperature,
        "grad_clip":        args.grad_clip,
        "patience":         args.patience,
        "scheduler":        args.scheduler,
        "lr_min":           args.lr_min,
        "seed":             args.seed,
        "loss":             "full_softmax",
        "started_at":       _now_iso(),
    }
    with open(ckpt_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"\n  hparams.json -> {ckpt_dir / 'hparams.json'}")
    _gcs_checkpoint_upload(ckpt_dir / "hparams.json", _gcs_ckpt_dir)

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_run = _mlflow_start(args.mlflow_tracking_uri, args.run_name)
    _mlflow_log_params(mlflow_run, {k: v for k, v in hparams.items()
                                    if k not in ("started_at",)})

    # ── Training loop ─────────────────────────────────────────────────────
    _section("Training")
    training_log: list[dict[str, Any]] = []
    best_val_ndcg20 = 0.0
    patience_ctr    = 0
    best_path       = ckpt_dir / "best_checkpoint.pt"
    latest_path     = ckpt_dir / "latest_checkpoint.pt"
    log_path        = ckpt_dir / "training_log.json"

    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        _section(f"Epoch {epoch}/{args.epochs}")
        t_epoch = time.time()

        # ── Train ──────────────────────────────────────────────────────
        train_loss = train_epoch_session(
            model           = model,
            dataloader      = train_loader,
            optimizer       = optimizer,
            device          = device,
            temperature     = args.temperature,
            label_smoothing = args.label_smoothing,
            grad_clip       = args.grad_clip,
            log_every       = 200,
        )

        if not (train_loss == train_loss):   # NaN guard
            print("  ERROR: train_loss is NaN — aborting.")
            _mlflow_end(mlflow_run)
            sys.exit(1)

        # ── Val eval ───────────────────────────────────────────────────
        val_metrics = evaluate_sessions(
            model            = model,
            prefix_item_arr  = val_eval_ds.prefix_item_arr,
            prefix_event_arr = val_eval_ds.prefix_event_arr,
            target_items     = val_eval_ds.target_items,
            train_sessions_df= train_df,
            n_items          = n_items,
            device           = device,
            batch_size       = args.val_batch_size,
            n_faiss_candidates = args.n_faiss_cands,
            label            = f"val_epoch{epoch}",
        )
        val_ndcg20 = val_metrics["ndcg_20"]

        # ── Checkpoint ─────────────────────────────────────────────────
        elapsed_epoch = time.time() - t_epoch
        is_best       = val_ndcg20 > best_val_ndcg20

        if is_best:
            best_val_ndcg20 = val_ndcg20
            patience_ctr    = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state":      model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "val_ndcg_20":      val_ndcg20,
                    "best_val_ndcg_20": best_val_ndcg20,
                    "hparams":          hparams,
                },
                best_path,
            )
            print(f"  [NEW BEST]  val NDCG@20 = {val_ndcg20:.4f}  -> {best_path.name}")
            _gcs_checkpoint_upload(best_path, _gcs_ckpt_dir)
        else:
            patience_ctr += 1
            print(
                f"  No improvement. patience {patience_ctr}/{args.patience}  "
                f"(best = {best_val_ndcg20:.4f})"
            )

        # Always save latest for easy resuming.
        torch.save(
            {
                "epoch":            epoch,
                "model_state":      model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_ndcg_20":      val_ndcg20,
                "best_val_ndcg_20": best_val_ndcg20,
                "hparams":          hparams,
            },
            latest_path,
        )
        _gcs_checkpoint_upload(latest_path, _gcs_ckpt_dir)

        # ── Epoch log ──────────────────────────────────────────────────
        epoch_entry: dict[str, Any] = {
            "epoch":            epoch,
            "train_loss":       round(train_loss, 6),
            # Model metrics
            "val_hr_10":        round(val_metrics["hr_10"],   4),
            "val_ndcg_10":      round(val_metrics["ndcg_10"], 4),
            "val_hr_20":        round(val_metrics["hr_20"],   4),
            "val_ndcg_20":      round(val_ndcg20,             4),
            # Popularity baseline (session-based, no filter_seen) — reference floor
            "pop_val_hr_10":    round(val_metrics["pop_hr_10"],   4),
            "pop_val_ndcg_10":  round(val_metrics["pop_ndcg_10"], 4),
            "pop_val_hr_20":    round(val_metrics["pop_hr_20"],   4),
            "pop_val_ndcg_20":  round(val_metrics["pop_ndcg_20"], 4),
            "best_val_ndcg_20": round(best_val_ndcg20, 4),
            "is_best":          is_best,
            "elapsed_s":        round(elapsed_epoch, 1),
            "timestamp":        _now_iso(),
        }
        training_log.append(epoch_entry)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        _gcs_checkpoint_upload(log_path, _gcs_ckpt_dir)

        _mlflow_log_metrics(mlflow_run, {
            "train_loss":   train_loss,
            "val_hr_10":    val_metrics["hr_10"],
            "val_ndcg_10":  val_metrics["ndcg_10"],
            "val_hr_20":    val_metrics["hr_20"],
            "val_ndcg_20":  val_ndcg20,
        }, step=epoch)

        # ── LR scheduler step ──────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  LR -> {current_lr:.2e}")

        # ── Early stopping ─────────────────────────────────────────────
        if patience_ctr >= args.patience:
            print(f"\n  Early stopping triggered after {epoch} epochs.")
            break

    # ── Final summary ─────────────────────────────────────────────────────
    total_elapsed = int(time.time() - t_total)
    _section("Training complete")
    print(f"  Total time      : {total_elapsed // 3600}h {(total_elapsed % 3600) // 60}m")
    print(f"  Best val NDCG@20: {best_val_ndcg20:.4f}  (epoch with [NEW BEST])")
    print(f"  Best checkpoint : {best_path}")
    print(f"  Training log    : {log_path}")
    print()
    print("  Per-epoch summary:")
    print(f"  {'epoch':>5}  {'train_loss':>10}  {'val_ndcg@20':>11}  {'best':>6}")
    print("  " + "-" * 38)
    for row in training_log:
        marker = "  <--" if row["is_best"] else ""
        print(
            f"  {row['epoch']:>5}  "
            f"{row['train_loss']:>10.4f}  "
            f"{row['val_ndcg_20']:>11.4f}"
            f"{marker}"
        )

    _mlflow_end(mlflow_run)


if __name__ == "__main__":
    main()
