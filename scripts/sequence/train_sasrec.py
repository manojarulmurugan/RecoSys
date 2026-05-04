"""V8 — SASRec training script (sequence model #2).

Same training pipeline as V7 (GRU4Rec) — sampled softmax with K=512 random
negatives, val on Jan 25-31 cart/purchase, test on Feb cart/purchase — but
the recurrent encoder is replaced by a causal pre-LN transformer with
learned positional embeddings (Kang & McAuley, 2018; eSASRec RecSys 2025
for the modern recipe).

Why a near-identical script:
    * Identical evaluator + checkpoint format → V7 vs V8 numbers are
      drop-in comparable.
    * Identical optimizer (split-embedding AdamW), identical loss
      (sampled softmax with K=512), identical val/test splits → the only
      independent variable is the encoder architecture.

Hyperparameters that differ from V7:
    * SASRec encoder (n_layers=2, n_heads=2, ffn_dim=256, dropout=0.2,
      pre-LN, GELU, learned positional embeddings)
    * No GRU hidden dim
    * Otherwise identical to V7

Prerequisites: same as V7 (run scripts/sequence/build_sequences_500k.py
first, plus artifacts/500k/{vocabs.pkl, train_pairs.parquet}).

Usage on Colab:
    !python scripts/sequence/train_sasrec.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── GCS credential + bootstrap helpers ────────────────────────────────────────

def _ensure_gcp_credentials() -> None:
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing and pathlib.Path(existing).expanduser().is_file():
        return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        _REPO_ROOT / "recosys-service-account.json",
        pathlib.Path("/content/recosys-service-account.json"),
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate.resolve())
            print(f"  Using GCP credentials: {candidate}")
            return
    print(
        "WARNING: No service-account JSON found.  Set "
        "GOOGLE_APPLICATION_CREDENTIALS or place "
        "recosys-service-account.json in a known location."
    )


def _gcs_bootstrap(
    bucket: str,
    bootstrap_specs: list[tuple[str, pathlib.Path]],
) -> None:
    """Download (blob_path, local_path) pairs from GCS if missing locally."""
    missing = [(b, p) for (b, p) in bootstrap_specs if not p.is_file()]
    if not missing:
        return

    try:
        from google.cloud import storage
    except ImportError:
        print("google-cloud-storage not installed; skipping GCS bootstrap.")
        return

    print(f"\nGCS bootstrap (bucket={bucket}): {len(missing)} files to download")
    client    = storage.Client()
    bucket_h  = client.bucket(bucket)
    for blob_path, local_path in missing:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob_h = bucket_h.blob(blob_path)
            blob_h.download_to_filename(str(local_path))
            print(f"  ✓ gs://{bucket}/{blob_path}  →  {local_path}")
        except Exception as exc:
            print(f"  ! gs://{bucket}/{blob_path} — {type(exc).__name__}: {exc}")


_ensure_gcp_credentials()

from src.sequence.data.negative_sampler import UniformNegativeSampler
from src.sequence.data.sequence_dataset import (
    SequenceEvalDataset,
    SequenceTrainDataset,
)
from src.sequence.evaluation.evaluate_sequence import (
    evaluate_sequence,
    evaluate_sequence_stratified,
)
from src.sequence.models.sasrec import SASRecModel
from src.sequence.training.train_sequence import (
    get_param_groups,
    train_epoch_sequence,
)


# ── CONFIG ────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "500k"
SEQ_DIR       = ARTIFACTS_DIR / "sequences"
TEST_GCS_PATH = "gs://recosys-data-bucket/samples/users_sample_500k/test/"
GCS_BUCKET    = "recosys-data-bucket"
GCS_MODELS_PREFIX = "models/two_tower_500k"
GCS_SEQ_PREFIX    = "models/sequence_500k/sequences"

MAX_SEQ_LEN     = 50
EMBED_DIM       = 64
N_LAYERS        = 2
N_HEADS         = 2
FFN_DIM         = 256
DROPOUT         = 0.2

BATCH_SIZE      = 256
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5
N_NEG_SAMPLES   = 512
N_EPOCHS        = 30
EVAL_EVERY      = 5
TEMPERATURE     = 0.20  # T=0.07 was too sharp: model over-personalised and scored below
                        # Global Popularity.  T=0.20 keeps strong gradients while letting
                        # some popularity signal survive in the embedding space.
NUM_WORKERS     = 4
GRAD_CLIP       = 1.0

CHECKPOINT_DIR  = ARTIFACTS_DIR / "checkpoints_v8_sasrec"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V8 SASRec trainer (500k sample)")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to resume training from.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help=f"Override total number of epochs (default {N_EPOCHS}).",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _load_test_targets_idx(
    vocabs: dict[str, Any],
) -> pd.DataFrame:
    """Load Feb test events from GCS and convert to idx-space."""
    print(f"\n  Loading test split from {TEST_GCS_PATH}")
    test_df = pd.read_parquet(
        TEST_GCS_PATH,
        columns=["user_id", "product_id", "event_type"],
    )
    print(f"    raw rows : {len(test_df):,}")

    user2idx = vocabs["user2idx"]
    item2idx = vocabs["item2idx"]
    test_df["user_idx"] = test_df["user_id"].map(user2idx)
    test_df["item_idx"] = test_df["product_id"].map(item2idx)
    test_df = test_df.dropna(subset=["user_idx", "item_idx"])
    test_df["user_idx"] = test_df["user_idx"].astype(np.int64)
    test_df["item_idx"] = test_df["item_idx"].astype(np.int64)
    print(f"    after id-mapping : {len(test_df):,} rows "
          f"(over {test_df['user_idx'].nunique():,} users, "
          f"{test_df['item_idx'].nunique():,} items)")
    return test_df[["user_idx", "item_idx", "event_type"]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    _section("V8 — SASRec on the 500k sample")
    print(f"  device       : {DEVICE}")
    print(f"  embed_dim    : {EMBED_DIM}  heads={N_HEADS}  ffn={FFN_DIM}  "
          f"layers={N_LAYERS}")
    print(f"  max_seq_len  : {MAX_SEQ_LEN}")
    print(f"  batch_size   : {BATCH_SIZE}")
    print(f"  lr           : {LEARNING_RATE}  wd={WEIGHT_DECAY} "
          f"(zero on embeddings)")
    print(f"  K negatives  : {N_NEG_SAMPLES}")
    print(f"  temperature  : {TEMPERATURE}")
    print(f"  epochs       : {args.epochs}  eval_every={EVAL_EVERY}")
    print(f"  checkpoints  : {CHECKPOINT_DIR}")

    _gcs_bootstrap(
        bucket = GCS_BUCKET,
        bootstrap_specs = [
            (f"{GCS_MODELS_PREFIX}/vocabs_500k.pkl", ARTIFACTS_DIR / "vocabs.pkl"),
            (f"{GCS_SEQ_PREFIX}/full_train_seqs.parquet",
             SEQ_DIR / "full_train_seqs.parquet"),
            (f"{GCS_SEQ_PREFIX}/metadata.json",
             SEQ_DIR / "metadata.json"),
        ],
    )

    # ── Load vocabs + train pairs ─────────────────────────────────────────
    print("\nLoading artifacts...")
    with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
        vocabs: dict[str, Any] = pickle.load(f)
    n_users_total = len(vocabs["user2idx"])
    n_items_total = len(vocabs["item2idx"])
    print(f"  vocabs       : {n_users_total:,} users / {n_items_total:,} items")

    train_pairs = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")
    print(f"  train_pairs  : {train_pairs.shape}")

    # ── Load sequence artifacts ───────────────────────────────────────────
    # Training uses full_train_seqs (all events through Jan 31) to match the
    # V1–V6 protocol: train on Oct–Jan, evaluate on Feb.
    full_train_seqs_df = pd.read_parquet(SEQ_DIR / "full_train_seqs.parquet")
    with open(SEQ_DIR / "metadata.json") as f:
        metadata: dict = json.load(f)
    print(f"  full_train_seqs  : {len(full_train_seqs_df):,} users")
    print(f"  metadata         : val_end={metadata['val_end']}")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\nBuilding datasets (this materialises the padded user arrays)...")
    train_dataset = SequenceTrainDataset(
        train_seqs_df = full_train_seqs_df,
        n_users       = n_users_total,
        max_seq_len   = MAX_SEQ_LEN,
    )
    print(f"  {train_dataset!r}  (train — all of Oct–Jan)")

    # One encoder for evaluation: same full-train sequences.
    test_eval = SequenceEvalDataset(
        seqs_df     = full_train_seqs_df,
        n_users     = n_users_total,
        max_seq_len = MAX_SEQ_LEN,
    )
    print(f"  {test_eval!r}  (test encoder)")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = NUM_WORKERS,
        pin_memory  = (DEVICE.type == "cuda"),
        drop_last   = False,
    )
    print(f"  batches/epoch : {len(train_loader):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    n_event_types = int(metadata.get("n_event_types", 4))
    model = SASRecModel(
        n_items       = n_items_total,
        n_event_types = n_event_types,
        embed_dim     = EMBED_DIM,
        max_seq_len   = MAX_SEQ_LEN,
        n_layers      = N_LAYERS,
        n_heads       = N_HEADS,
        ffn_dim       = FFN_DIM,
        dropout       = DROPOUT,
    ).to(DEVICE)
    model.model_summary()

    # ── Optimizer / scheduler / sampler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        get_param_groups(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    neg_sampler = UniformNegativeSampler(
        n_items = n_items_total,
        n_neg   = N_NEG_SAMPLES,
        device  = DEVICE,
    )
    print(f"  {neg_sampler!r}")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt_path = pathlib.Path(args.resume)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"\nResuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  resuming at epoch {start_epoch}")

    # ── Test ground truth ─────────────────────────────────────────────────
    test_targets_df = _load_test_targets_idx(vocabs)

    # ── Checkpoint dir + log ──────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHECKPOINT_DIR / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            training_log: list[dict] = json.load(f)
    else:
        training_log = []

    # ── Training loop ─────────────────────────────────────────────────────
    losses:    list[float] = []
    best_r10:  float       = 0.0
    best_epoch: int        = 0
    print()
    print(f"Training on {DEVICE} — {args.epochs} epochs total "
          f"(starting at epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        epoch_loss = train_epoch_sequence(
            model        = model,
            dataloader   = train_loader,
            optimizer    = optimizer,
            neg_sampler  = neg_sampler,
            device       = DEVICE,
            temperature  = TEMPERATURE,
            grad_clip    = GRAD_CLIP,
            log_every    = 200,
        )
        losses.append(epoch_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  epoch {epoch} : loss {epoch_loss:.4f}  lr {current_lr:.2e}  "
              f"({int(time.time() - epoch_start)}s)")
        scheduler.step()

        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss":            epoch_loss,
                "config": {
                    "model":          "SASRec",
                    "n_items":        n_items_total,
                    "n_event_types":  n_event_types,
                    "embed_dim":      EMBED_DIM,
                    "n_layers":       N_LAYERS,
                    "n_heads":        N_HEADS,
                    "ffn_dim":        FFN_DIM,
                    "dropout":        DROPOUT,
                    "max_seq_len":    MAX_SEQ_LEN,
                    "temperature":    TEMPERATURE,
                    "n_neg":          N_NEG_SAMPLES,
                },
            },
            ckpt_path,
        )

        # Periodic evaluation on the Feb test set (same protocol as V1–V6).
        # This is the only available clean eval window when training on
        # full_train_seqs (Oct–Jan).  filter_seen=True matches V1–V6.
        log_entry: dict = {
            "epoch":      epoch,
            "train_loss": round(epoch_loss, 6),
            "lr":         round(current_lr, 8),
        }
        if epoch % EVAL_EVERY == 0:
            print(f"\n  --- Test eval at epoch {epoch} ---")
            ep_metrics = evaluate_sequence(
                model              = model,
                item_seq_arr       = test_eval.item_seq_arr,
                event_seq_arr      = test_eval.event_seq_arr,
                eval_targets_df    = test_targets_df,
                train_pairs_df     = train_pairs,
                n_items            = n_items_total,
                device             = DEVICE,
                batch_size         = 512,
                n_faiss_candidates = 400,
                label              = f"test (epoch {epoch})",
                filter_seen        = True,
            )
            log_entry["test_recall_10"] = round(ep_metrics["recall_10"], 6)
            log_entry["test_ndcg_10"]   = round(ep_metrics["ndcg_10"],   6)
            log_entry["test_recall_20"] = round(ep_metrics["recall_20"], 6)
            log_entry["test_ndcg_20"]   = round(ep_metrics["ndcg_20"],   6)

            # Keep a copy of the best checkpoint by R@10.
            if ep_metrics["recall_10"] > best_r10:
                best_r10   = ep_metrics["recall_10"]
                best_epoch = epoch
                best_path  = CHECKPOINT_DIR / "best_recall10.pt"
                import shutil
                shutil.copy2(ckpt_path, best_path)
                print(f"  ★ New best R@10 = {best_r10:.4f}  → saved best_recall10.pt")

        training_log.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"  training_log.json updated ({len(training_log)} entries)")

    # ── Final test evaluation ─────────────────────────────────────────────
    # filter_seen=True: mirrors V1–V6 so test numbers are directly comparable.
    _section("V8 — Final test evaluation")
    test_metrics = evaluate_sequence(
        model              = model,
        item_seq_arr       = test_eval.item_seq_arr,
        event_seq_arr      = test_eval.event_seq_arr,
        eval_targets_df    = test_targets_df,
        train_pairs_df     = train_pairs,
        n_items            = n_items_total,
        device             = DEVICE,
        batch_size         = 512,
        n_faiss_candidates = 400,
        label              = "test (final)",
        filter_seen        = True,
    )
    test_strat = evaluate_sequence_stratified(
        model              = model,
        item_seq_arr       = test_eval.item_seq_arr,
        event_seq_arr      = test_eval.event_seq_arr,
        eval_targets_df    = test_targets_df,
        train_pairs_df     = train_pairs,
        n_items            = n_items_total,
        device             = DEVICE,
        batch_size         = 512,
        n_faiss_candidates = 400,
        label              = "test (final)",
        filter_seen        = True,
    )

    final_results = {
        "test_recall_10":      test_metrics["recall_10"],
        "test_ndcg_10":        test_metrics["ndcg_10"],
        "test_recall_20":      test_metrics["recall_20"],
        "test_ndcg_20":        test_metrics["ndcg_20"],
        "test_n_eval_users":   test_metrics["n_eval_users"],
        "test_cohort_overall": test_strat["overall"],
        "test_cohort_cold":    test_strat["cold"],
        "test_cohort_medium":  test_strat["medium"],
        "test_cohort_warm":    test_strat["warm"],
    }
    with open(CHECKPOINT_DIR / "final_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n  → {CHECKPOINT_DIR / 'final_test_results.json'}")

    # ── Summary ───────────────────────────────────────────────────────────
    best_loss_epoch = int(np.argmin(losses)) + start_epoch
    best_loss       = float(min(losses))
    _section("V8 training complete")
    print(f"  Best train loss : {best_loss:.4f}  (epoch {best_loss_epoch})")
    print(f"  Best R@10 epoch : {best_epoch}  ({best_r10:.4f})")
    print(f"  Final test R@10 : {test_metrics['recall_10']:.4f}")
    print(f"  Final test N@10 : {test_metrics['ndcg_10']:.4f}")
    print(f"  Best checkpoint : {CHECKPOINT_DIR / 'best_recall10.pt'}")
    print(f"  Checkpoints     : {CHECKPOINT_DIR}")
    print(f"  Training log    : {log_path}")


if __name__ == "__main__":
    main()
