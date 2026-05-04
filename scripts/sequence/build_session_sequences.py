"""Build session-grouped chronological sequences for the V9 reframe.

This is the **session-based** replacement for ``build_sequences_500k.py``.
Where the old script collapsed every user's events into one giant
chronological sequence (ignoring session boundaries), this one groups
by ``(user_id, user_session)`` to match the Transformers4Rec paper §4.1
convention used by every published REES46 sequence model.

Why this matters
----------------
The old user-level sequences caused two problems documented in
``reports/07_sequence_model_results.md``:
  * Models trained to predict any next item from a 50-event history
    of mostly random views; tested on cart/purchase events months later.
  * Last-position pooling captured a single recent view as the user
    embedding — uninformative for long-horizon ranking.

Session sequences fix both by treating each shopping session as a
self-contained next-item-prediction problem (T4Rec §4.1.3).

Inputs (already cleaned by Spark — see reports/03_dataproc_preprocessing_run.md):
    gs://recosys-data-bucket/samples/users_sample_500k/train/*.parquet
        Cleaned events Oct 2019 – Jan 2020.
        Columns include: event_time, event_type, product_id, user_id,
        user_session.
    gs://recosys-data-bucket/samples/users_sample_500k/test/*.parquet
        Cleaned events Feb 2020.
    artifacts/500k/vocabs.pkl
        {user2idx, item2idx, idx2user, idx2item}.

Outputs (artifacts/500k/sequences_v2/):
    train_sessions.parquet
        One row per session with event_time < VAL_START.  Columns:
        session_idx (int64), user_idx (int64), item_seq (list[int64]),
        event_seq (list[int64]), seq_len (int64).

    val_sessions.parquet
        One row per session in [VAL_START, TEST_START).  Used for
        early-stopping in training.  Same schema as train_sessions.

    test_sessions.parquet
        One row per session in Feb 2020.  Same schema.  Last item per
        session is the held-out ground truth at eval time (not pre-split
        here — eval code does the split).

    metadata.json
        Build parameters and corpus statistics.

Preprocessing applied (T4Rec paper §4.1.4):
  * Sessions of length < 2 are dropped (no signal for next-item prediction).
  * **Consecutive repeated interactions within a session are removed**
    (paper convention).  E.g. [view-A, view-A, view-A, cart-A] → [view-A, cart-A].
  * Sessions truncated to last MAX_SEQ_LEN=20 events.
  * Padding token (item_idx=0, event_idx=0) is NOT inserted here — applied
    by the dataset at load time so MAX_SEQ_LEN can be tuned.

Usage:
    python scripts/sequence/build_session_sequences.py
    python scripts/sequence/build_session_sequences.py \\
        --sample-name users_sample_1m \\
        --artifacts-dir artifacts/1m  # for the Phase B 1M scale-up
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

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── GCP credentials helper ────────────────────────────────────────────────────

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
            print(f"  GCP credentials: {candidate}")
            return
    print(
        "WARNING: No GCP credentials found.  GCS reads will fail unless the "
        "default application credentials are configured for this environment."
    )


# ── Configuration ─────────────────────────────────────────────────────────────

# Validation window = last 7 days of January.  Held out from training so
# early-stopping has a clean signal that doesn't bleed into the Feb test.
VAL_START   = pd.Timestamp("2020-01-25", tz="UTC")  # inclusive
TEST_START  = pd.Timestamp("2020-02-01", tz="UTC")  # exclusive (training upper bound)

# T4Rec paper §4.1.4: max_seq_len=20 for session-based recommendation.
MAX_SEQ_LEN = 20
MIN_SEQ_LEN = 2

EVENT_TYPE_MAP: dict[str, int] = {"view": 1, "cart": 2, "purchase": 3}
# event_idx 0 is reserved for padding


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def _step(msg: str) -> float:
    print(f"\n  >  {msg}")
    return time.time()


def _done(t0: float, label: str = "") -> None:
    elapsed = int(time.time() - t0)
    suffix  = f" — {label}" if label else ""
    print(f"     done in {elapsed // 60}m {elapsed % 60}s{suffix}")


def _remove_consecutive_repeats(items: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop consecutive duplicate items within a session (T4Rec convention).

    "Consecutive repeats" = same item id at adjacent positions, regardless
    of event_type.  Keeps the first occurrence.  E.g.:
        items=[A, A, A, B, B, C]  events=[v, v, c, v, c, v]
        →  items=[A, B, C]        events=[v, v, v]
    """
    if items.shape[0] <= 1:
        return items, events
    # Mask: True at position 0 OR where item differs from previous position.
    keep = np.ones(items.shape[0], dtype=bool)
    keep[1:] = items[1:] != items[:-1]
    return items[keep], events[keep]


def _build_session_sequences(
    df: pd.DataFrame,
    label: str,
    max_seq_len: int = MAX_SEQ_LEN,
    min_seq_len: int = MIN_SEQ_LEN,
) -> pd.DataFrame:
    """Group events by ``(user_idx, user_session)`` into per-session sequences.

    Assumes df is already sorted by (user_idx, user_session, event_time).

    Returns a frame with columns: session_idx (int64), user_idx (int64),
    item_seq (list[int64]), event_seq (list[int64]), seq_len (int64).
    """
    t0 = _step(f"Aggregating per-session sequences ({label})")

    # Aggregate items + events per (user, session) pair.
    grouped = (
        df.groupby(["user_idx", "user_session"], sort=False, as_index=False)
          .agg(
              item_seq  = ("item_idx",   list),
              event_seq = ("event_idx",  list),
          )
    )

    # Convert to numpy, remove consecutive repeats per session, then truncate.
    new_items: list[list[int]]  = []
    new_events: list[list[int]] = []
    for items, events in zip(grouped["item_seq"].to_numpy(), grouped["event_seq"].to_numpy()):
        items_arr  = np.asarray(items,  dtype=np.int64)
        events_arr = np.asarray(events, dtype=np.int64)
        items_arr, events_arr = _remove_consecutive_repeats(items_arr, events_arr)
        # Truncate: keep the most recent max_seq_len events.
        if items_arr.shape[0] > max_seq_len:
            items_arr  = items_arr [-max_seq_len:]
            events_arr = events_arr[-max_seq_len:]
        new_items.append(items_arr.tolist())
        new_events.append(events_arr.tolist())
    grouped["item_seq"]  = new_items
    grouped["event_seq"] = new_events

    grouped["seq_len"] = grouped["item_seq"].apply(len).astype(np.int64)
    n_before = len(grouped)
    grouped  = grouped[grouped["seq_len"] >= min_seq_len].reset_index(drop=True)
    n_after  = len(grouped)

    # Assign a stable contiguous session_idx.
    grouped.insert(0, "session_idx", np.arange(n_after, dtype=np.int64))
    grouped["user_idx"] = grouped["user_idx"].astype(np.int64)

    _done(
        t0,
        label=(f"{n_after:,} sessions (dropped {n_before - n_after:,} "
               f"with < {min_seq_len} events post-dedup)"),
    )

    # Length distribution diagnostic.
    lens = grouped["seq_len"].to_numpy()
    print(
        f"     session length: min={lens.min()}  median={int(np.median(lens))}  "
        f"mean={lens.mean():.2f}  p90={int(np.percentile(lens, 90))}  "
        f"p99={int(np.percentile(lens, 99))}  max={int(lens.max())}"
    )
    n_unique_users = grouped["user_idx"].nunique()
    sessions_per_user = n_after / max(n_unique_users, 1)
    print(
        f"     {n_unique_users:,} unique users → "
        f"{sessions_per_user:.2f} sessions/user on average"
    )

    return grouped[["session_idx", "user_idx", "item_seq", "event_seq", "seq_len"]]


def _load_and_map(
    gcs_path: str,
    user2idx: dict[int, int],
    item2idx: dict[int, int],
    label: str,
) -> pd.DataFrame:
    """Read parquet from GCS, map ids → idxs, sort, return event-level frame."""
    t0 = _step(f"Reading {gcs_path}*.parquet ({label})")
    events = pd.read_parquet(
        gcs_path,
        columns=["event_time", "event_type", "product_id", "user_id", "user_session"],
    )
    _done(t0, label=f"{len(events):,} rows")

    t0 = _step(f"Mapping ids and event types ({label})")
    events["user_idx"]  = events["user_id"].map(user2idx)
    events["item_idx"]  = events["product_id"].map(item2idx)
    events["event_idx"] = events["event_type"].map(EVENT_TYPE_MAP)

    n_pre = len(events)
    events = events.dropna(subset=["user_idx", "item_idx", "event_idx", "user_session"])
    n_post = len(events)
    if n_pre != n_post:
        print(f"     dropped {n_pre - n_post:,} rows with unknown id / event_type / null session")

    events["user_idx"]  = events["user_idx"].astype(np.int64)
    events["item_idx"]  = events["item_idx"].astype(np.int64)
    events["event_idx"] = events["event_idx"].astype(np.int64)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    _done(t0, label=f"{n_post:,} rows after id mapping")

    t0 = _step(f"Sorting by (user_idx, user_session, event_time) ({label})")
    events = events.sort_values(
        ["user_idx", "user_session", "event_time"], kind="mergesort"
    ).reset_index(drop=True)
    _done(t0)

    return events


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build session-grouped sequences for V9 (T4Rec-aligned)",
    )
    p.add_argument(
        "--sample-name",
        type=str,
        default="users_sample_500k",
        help="GCS sample directory name (default: users_sample_500k). "
             "Use 'users_sample_1m' for the Phase B 1M scale-up.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/500k",
        help="Local artifacts directory (default: artifacts/500k). "
             "Vocabs are loaded from <artifacts-dir>/vocabs.pkl; outputs go to "
             "<artifacts-dir>/sequences_v2/.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Max session length (default: {MAX_SEQ_LEN}, T4Rec paper convention).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t_total = time.time()

    artifacts_dir = _REPO_ROOT / args.artifacts_dir
    seq_dir       = artifacts_dir / "sequences_v2"
    vocabs_path   = artifacts_dir / "vocabs.pkl"

    train_gcs = f"gs://recosys-data-bucket/samples/{args.sample_name}/train/"
    test_gcs  = f"gs://recosys-data-bucket/samples/{args.sample_name}/test/"

    _section("RecoSys — Build session-based sequences (V9 / T4Rec reframe)")
    print(f"  Sample        : {args.sample_name}")
    print(f"  Train GCS     : {train_gcs}")
    print(f"  Test GCS      : {test_gcs}")
    print(f"  Artifacts dir : {artifacts_dir}")
    print(f"  Output dir    : {seq_dir}")
    print(f"  VAL window    : [{VAL_START.isoformat()}, {TEST_START.isoformat()})")
    print(f"  Max seq len   : {args.max_seq_len}")
    print(f"  Min seq len   : {MIN_SEQ_LEN}")

    _ensure_gcp_credentials()
    seq_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vocabs ──────────────────────────────────────────────────────────
    t0 = _step(f"Loading vocabs from {vocabs_path}")
    with open(vocabs_path, "rb") as f:
        vocabs: dict[str, Any] = pickle.load(f)
    user2idx: dict[int, int] = vocabs["user2idx"]
    item2idx: dict[int, int] = vocabs["item2idx"]
    _done(t0, label=f"{len(user2idx):,} users  /  {len(item2idx):,} items")

    # ── Load + map TRAIN-window events (Oct 2019 – Jan 2020) ─────────────────
    train_events = _load_and_map(train_gcs, user2idx, item2idx, label="train window")

    # ── Split into pre-VAL_START (training) and val window ───────────────────
    train_mask = train_events["event_time"] < VAL_START
    val_mask   = (~train_mask) & (train_events["event_time"] < TEST_START)

    n_train_events = int(train_mask.sum())
    n_val_events   = int(val_mask.sum())
    print(
        f"\n  Event-level split:"
        f"\n     train (< {VAL_START.date()})           : {n_train_events:,} events"
        f"\n     val   [{VAL_START.date()}, {TEST_START.date()}) : {n_val_events:,} events"
    )

    # ── Build train_sessions ─────────────────────────────────────────────────
    train_sessions = _build_session_sequences(
        train_events.loc[train_mask,
                         ["user_idx", "user_session", "item_idx", "event_idx"]],
        label="train", max_seq_len=args.max_seq_len,
    )
    out_train = seq_dir / "train_sessions.parquet"
    t0 = _step(f"Writing {out_train.name}")
    train_sessions.to_parquet(out_train, index=False)
    _done(t0, label=f"{out_train.stat().st_size / 1e6:.1f} MB")

    # ── Build val_sessions ───────────────────────────────────────────────────
    val_sessions = _build_session_sequences(
        train_events.loc[val_mask,
                         ["user_idx", "user_session", "item_idx", "event_idx"]],
        label="val", max_seq_len=args.max_seq_len,
    )
    out_val = seq_dir / "val_sessions.parquet"
    t0 = _step(f"Writing {out_val.name}")
    val_sessions.to_parquet(out_val, index=False)
    _done(t0, label=f"{out_val.stat().st_size / 1e6:.1f} MB")

    # Free the train-window frame before loading test.
    del train_events

    # ── Load + map TEST-window events (Feb 2020) ─────────────────────────────
    test_events = _load_and_map(test_gcs, user2idx, item2idx, label="test window")

    # ── Build test_sessions ──────────────────────────────────────────────────
    test_sessions = _build_session_sequences(
        test_events[["user_idx", "user_session", "item_idx", "event_idx"]],
        label="test", max_seq_len=args.max_seq_len,
    )
    out_test = seq_dir / "test_sessions.parquet"
    t0 = _step(f"Writing {out_test.name}")
    test_sessions.to_parquet(out_test, index=False)
    _done(t0, label=f"{out_test.stat().st_size / 1e6:.1f} MB")

    # ── metadata.json ────────────────────────────────────────────────────────
    metadata: dict[str, Any] = {
        "schema_version":     "v9_session",
        "sample_name":        args.sample_name,
        "max_seq_len":        int(args.max_seq_len),
        "min_seq_len":        int(MIN_SEQ_LEN),
        "n_items":            int(len(item2idx)),
        "n_users":            int(len(user2idx)),
        "n_event_types":      int(len(EVENT_TYPE_MAP) + 1),  # +1 for PAD
        "event_type_map":     EVENT_TYPE_MAP,
        "padding_idx":        0,
        "val_start":          VAL_START.isoformat(),
        "test_start":         TEST_START.isoformat(),
        "n_train_sessions":   int(len(train_sessions)),
        "n_val_sessions":     int(len(val_sessions)),
        "n_test_sessions":    int(len(test_sessions)),
        "n_train_events":     n_train_events,
        "n_val_events":       n_val_events,
        "n_test_events":      int(len(test_events)),
        "consecutive_repeats_removed": True,
    }
    out_meta = seq_dir / "metadata.json"
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Wrote {out_meta}")

    _section("Summary")
    print(f"  train_sessions : {len(train_sessions):>8,} sessions  ({out_train})")
    print(f"  val_sessions   : {len(val_sessions):>8,} sessions  ({out_val})")
    print(f"  test_sessions  : {len(test_sessions):>8,} sessions  ({out_test})")
    print(f"  metadata       : {out_meta}")
    elapsed = int(time.time() - t_total)
    print(f"\n  Total time     : {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
