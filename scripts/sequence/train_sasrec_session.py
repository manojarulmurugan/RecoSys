"""SASRec session-based training -- V10 / T4Rec reframe.

Trains SASRec with full softmax + label smoothing + LR warmup on the session
sequences built by build_session_sequences.py. V8 (train_sasrec.py) is
abandoned -- this is a clean session-based replacement.

Key differences from GRU4Rec V9:
  - SASRecModel instead of GRU4RecModel (2-layer pre-LN transformer)
  - Per-step warmup+cosine LR schedule: linear warmup from lr_start=1e-5 to
    lr=3e-4 over first 1000 steps, then cosine decay to lr_min=1e-5.
    Critical for transformer convergence -- flat LR stalls attention learning.
  - Smaller default batch_size=128 (transformer ~2x activation memory vs GRU)

Outputs (all under --checkpoint-dir):
  best_checkpoint.pt    Best val-NDCG@20 checkpoint (model + optimizer state)
  latest_checkpoint.pt  End-of-epoch checkpoint (for resuming)
  training_log.json     Per-epoch metrics list (written after every epoch)
  hparams.json          Hyperparameter snapshot

Usage (Colab A100 recommended):
    python scripts/sequence/train_sasrec_session.py

  Override defaults:
    python scripts/sequence/train_sasrec_session.py \\
        --batch-size 64 --epochs 20 --n-layers 2 --n-heads 4

  With MLflow (configure after Day 6):
    python scripts/sequence/train_sasrec_session.py \\
        --mlflow-tracking-uri http://<mlflow-cloud-run-url> \\
        --run-name sasrec_session_v10_500k
"""

from __future__ import annotations

import argparse
import json
import math
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
from src.sequence.evaluation.evaluate_sequence import evaluate_sessions
from src.sequence.models.sasrec import SASRecModel
from src.sequence.training.train_sequence import get_param_groups, train_epoch_session


# ── Helpers ────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


# ── MLflow (optional -- gracefully skipped if not installed / configured) ──────

def _mlflow_start(tracking_uri: str | None, run_name: str) -> Any:
    if not tracking_uri:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("recosys_session_v10")
        run = mlflow.start_run(run_name=run_name)
        print(f"  MLflow run  : {run.info.run_id}  @ {tracking_uri}")
        return run
    except Exception as exc:
        print(f"  MLflow unavailable ({exc}) -- skipping tracking.")
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
        description="SASRec session-based training V10 (full softmax, T4Rec reframe)",
    )
    # Paths
    p.add_argument("--artifacts-dir",  default="artifacts/500k")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Defaults to <artifacts-dir>/sequences_v2/checkpoints_v10_sasrec_session")
    # Model
    p.add_argument("--embed-dim",   type=int,   default=128)
    p.add_argument("--n-layers",    type=int,   default=2)
    p.add_argument("--n-heads",     type=int,   default=4)
    p.add_argument("--ffn-dim",     type=int,   default=256)
    p.add_argument("--max-seq-len", type=int,   default=20)
    p.add_argument("--dropout",     type=float, default=0.2)
    # Training
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=128,
                   help="128 for A100. Use 64 for V100/T4 (transformer uses ~2x memory vs GRU).")
    p.add_argument("--lr",              type=float, default=3e-4,
                   help="Peak LR reached after warmup.")
    p.add_argument("--lr-start",        type=float, default=1e-5,
                   help="LR at step 0 before warmup ramp.")
    p.add_argument("--lr-min",          type=float, default=1e-5,
                   help="LR floor after cosine decay.")
    p.add_argument("--warmup-steps",    type=int,   default=1000,
                   help="Steps for linear LR warmup from lr_start to lr.")
    p.add_argument("--temperature",     type=float, default=0.07)
    p.add_argument("--weight-decay",    type=float, default=1e-5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--patience",        type=int,   default=5,
                   help="Early-stop patience in epochs (on val NDCG@20).")
    p.add_argument("--num-workers",     type=int,   default=2)
    p.add_argument("--seed",            type=int,   default=42)
    # Evaluation
    p.add_argument("--val-batch-size",  type=int,   default=512)
    p.add_argument("--n-faiss-cands",   type=int,   default=50,
                   help="FAISS candidates per session (>= 20 for HR@20).")
    # MLflow
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--run-name",            default="sasrec_session_v10_500k")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Paths ─────────────────────────────────────────────────────────────
    artifacts_dir = _REPO_ROOT / args.artifacts_dir
    seq_dir       = artifacts_dir / "sequences_v2"
    ckpt_dir      = (
        pathlib.Path(args.checkpoint_dir) if args.checkpoint_dir
        else seq_dir / "checkpoints_v10_sasrec_session"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _section("SASRec Session Training V10")
    print(f"  Device          : {device}")
    print(f"  Artifacts dir   : {artifacts_dir}")
    print(f"  Sequences dir   : {seq_dir}")
    print(f"  Checkpoint dir  : {ckpt_dir}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Max epochs      : {args.epochs}")
    print(f"  Patience        : {args.patience}")
    print(f"  embed_dim       : {args.embed_dim}")
    print(f"  n_layers        : {args.n_layers}  (heads={args.n_heads}, ffn={args.ffn_dim})")
    print(f"  dropout         : {args.dropout}")
    print(f"  lr              : {args.lr}  (start={args.lr_start}, min={args.lr_min})")
    print(f"  warmup_steps    : {args.warmup_steps}")
    print(f"  temperature     : {args.temperature}")
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

    # Val eval dataset -- built once, reused every epoch.
    val_eval_ds = SessionEvalDataset(val_df, max_seq_len=max_seq_len)
    print(f"\n  Train loader : {len(train_loader):,} batches x {args.batch_size}")

    # ── Build model ───────────────────────────────────────────────────────
    _section("Model")
    model = SASRecModel(
        n_items       = n_items,
        n_event_types = 4,
        embed_dim     = args.embed_dim,
        max_seq_len   = max_seq_len,
        n_layers      = args.n_layers,
        n_heads       = args.n_heads,
        ffn_dim       = args.ffn_dim,
        dropout       = args.dropout,
    ).to(device)
    model.model_summary()

    optimizer = optim.AdamW(
        get_param_groups(model, lr=args.lr, weight_decay=args.weight_decay),
    )

    # Per-step warmup + cosine LR.
    # Warmup: step 0..warmup_steps-1  -> LR linearly from lr_start to lr.
    # Cosine: step warmup_steps..total_steps -> LR from lr to lr_min.
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = min(args.warmup_steps, total_steps)
    lr_start_mult = args.lr_start / args.lr
    lr_min_mult   = args.lr_min   / args.lr

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return lr_start_mult + (1.0 - lr_start_mult) * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_mult + (1.0 - lr_min_mult) * cosine

    step_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    print(
        f"\n  LR schedule : warmup {warmup_steps} steps -> {args.lr:.1e}, "
        f"cosine -> {args.lr_min:.1e} over {total_steps:,} total steps"
    )

    # ── Save hyperparameters ──────────────────────────────────────────────
    hparams: dict[str, Any] = {
        "model":            "SASRecModel",
        "schema_version":   "v10_session",
        "n_items":          n_items,
        "n_users":          n_users,
        "embed_dim":        args.embed_dim,
        "n_layers":         args.n_layers,
        "n_heads":          args.n_heads,
        "ffn_dim":          args.ffn_dim,
        "max_seq_len":      max_seq_len,
        "dropout":          args.dropout,
        "batch_size":       args.batch_size,
        "lr":               args.lr,
        "lr_start":         args.lr_start,
        "lr_min":           args.lr_min,
        "warmup_steps":     warmup_steps,
        "total_steps":      total_steps,
        "weight_decay":     args.weight_decay,
        "label_smoothing":  args.label_smoothing,
        "temperature":      args.temperature,
        "grad_clip":        args.grad_clip,
        "patience":         args.patience,
        "seed":             args.seed,
        "loss":             "full_softmax",
        "started_at":       _now_iso(),
    }
    with open(ckpt_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"\n  hparams.json -> {ckpt_dir / 'hparams.json'}")

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
            model            = model,
            dataloader       = train_loader,
            optimizer        = optimizer,
            device           = device,
            temperature      = args.temperature,
            label_smoothing  = args.label_smoothing,
            grad_clip        = args.grad_clip,
            log_every        = 200,
            step_scheduler   = step_scheduler,
        )

        if not (train_loss == train_loss):  # NaN guard
            print("  ERROR: train_loss is NaN -- aborting.")
            _mlflow_end(mlflow_run)
            sys.exit(1)

        # ── Val eval ───────────────────────────────────────────────────
        val_metrics = evaluate_sessions(
            model             = model,
            prefix_item_arr   = val_eval_ds.prefix_item_arr,
            prefix_event_arr  = val_eval_ds.prefix_event_arr,
            target_items      = val_eval_ds.target_items,
            train_sessions_df = train_df,
            n_items           = n_items,
            device            = device,
            batch_size        = args.val_batch_size,
            n_faiss_candidates= args.n_faiss_cands,
            label             = f"val_epoch{epoch}",
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

        # ── Epoch log ──────────────────────────────────────────────────
        epoch_entry: dict[str, Any] = {
            "epoch":            epoch,
            "train_loss":       round(train_loss, 6),
            "val_hr_10":        round(val_metrics["hr_10"],   4),
            "val_ndcg_10":      round(val_metrics["ndcg_10"], 4),
            "val_hr_20":        round(val_metrics["hr_20"],   4),
            "val_ndcg_20":      round(val_ndcg20,             4),
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

        _mlflow_log_metrics(mlflow_run, {
            "train_loss":  train_loss,
            "val_hr_10":   val_metrics["hr_10"],
            "val_ndcg_10": val_metrics["ndcg_10"],
            "val_hr_20":   val_metrics["hr_20"],
            "val_ndcg_20": val_ndcg20,
        }, step=epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  LR at epoch end : {current_lr:.2e}")

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
