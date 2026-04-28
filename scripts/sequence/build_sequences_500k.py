"""Build per-user chronological sequences from the cleaned 500k train events.

Inputs (already cleaned by the Spark pipeline — k-core 3, dedup, price floor,
bot filter; see reports/03_dataproc_preprocessing_run.md):
    gs://recosys-data-bucket/samples/users_sample_500k/train/*.parquet
        Columns: event_time TIMESTAMP, event_type, product_id, user_id, ...

    artifacts/500k/vocabs.pkl
        {user2idx, item2idx, idx2user, idx2item, ...}

Outputs (artifacts/500k/sequences/):
    train_seqs.parquet
        One row per user.  Columns: user_idx (int64),
        item_seq (list[int64]), event_seq (list[int64]).
        Sequences contain only events with event_time < TRAIN_END (Jan 25),
        sorted chronologically.  Used for both training and validation-time
        user encoding.

    full_train_seqs.parquet
        Same schema as train_seqs but covers events with event_time < TEST_START
        (Feb 1).  Used for final test-time user encoding (mirrors the V1-V6
        setup so V7/V8 numbers slot directly into the comparison table).

    val_targets.parquet
        Cart + purchase events in [TRAIN_END, TEST_START), deduplicated by
        (user_idx, item_idx).  Schema = [user_idx, item_idx, event_type,
        product_id, event_time, user_id] so it slots into evaluate(...) as
        the ``test_df`` argument.

    metadata.json
        Build parameters and corpus statistics.

Usage on Colab:
    !python scripts/sequence/build_sequences_500k.py

Notes:
    - Sequences are stored full length; ``MAX_SEQ_LEN`` truncation happens
      in the dataset at load time, so the truncation length can be tuned
      later without re-running this script.
    - Padding token (item_idx = 0, event_idx = 0) is never produced here;
      it is applied by the dataset.
    - Test ground truth is NOT rebuilt — the existing
      ``artifacts/500k/test_pairs.parquet`` is reused for final eval.
"""

from __future__ import annotations

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


# ── GCS credential helper ──────────────────────────────────────────────────────

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


# ── Configuration ──────────────────────────────────────────────────────────────

ARTIFACTS_DIR  = _REPO_ROOT / "artifacts" / "500k"
SEQ_DIR        = ARTIFACTS_DIR / "sequences"

TRAIN_GCS_PATH = "gs://recosys-data-bucket/samples/users_sample_500k/train/"

VOCABS_PATH    = ARTIFACTS_DIR / "vocabs.pkl"

TRAIN_END   = pd.Timestamp("2020-01-25", tz="UTC")  # exclusive
TEST_START  = pd.Timestamp("2020-02-01", tz="UTC")  # exclusive (training upper bound)

MAX_SEQ_LEN = 50  # informational only; dataset truncates at load time

EVENT_TYPE_MAP: dict[str, int] = {"view": 1, "cart": 2, "purchase": 3}
# event_idx 0 is reserved for padding


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _build_per_user_sequences(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Group an event-level frame into per-user chronological sequences.

    Assumes df is already sorted by (user_idx, event_time).

    Returns a frame with columns: user_idx (int64), item_seq (list[int64]),
    event_seq (list[int64]).  Users with < 2 events are dropped (no signal
    for next-item prediction).
    """
    t0 = _step(f"Aggregating per-user sequences ({label})")

    # groupby(sort=False) preserves user_idx ordering from the input frame,
    # which is already sorted.
    grouped = df.groupby("user_idx", sort=False, as_index=False).agg(
        item_seq  = ("item_idx",   list),
        event_seq = ("event_idx",  list),
    )

    grouped["seq_len"] = grouped["item_seq"].apply(len).astype(np.int64)
    n_before = len(grouped)
    grouped  = grouped[grouped["seq_len"] >= 2].reset_index(drop=True)
    n_after  = len(grouped)
    grouped["user_idx"] = grouped["user_idx"].astype(np.int64)

    _done(
        t0,
        label=(f"{n_after:,} users (dropped {n_before - n_after:,} "
               f"with < 2 events)"),
    )

    # Length distribution diagnostic
    lens = grouped["seq_len"].values
    print(
        f"     seq length: min={lens.min()}  median={int(np.median(lens))}  "
        f"mean={lens.mean():.1f}  p90={int(np.percentile(lens, 90))}  "
        f"p99={int(np.percentile(lens, 99))}  max={int(lens.max())}"
    )
    pct_over_50 = float((lens > MAX_SEQ_LEN).mean() * 100)
    print(
        f"     {pct_over_50:.1f}% of users have > {MAX_SEQ_LEN} events "
        f"(will be truncated at load time)"
    )

    return grouped[["user_idx", "item_seq", "event_seq", "seq_len"]]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    _section("RecoSys — Build sequences (500k)")
    print(f"  Input  : {TRAIN_GCS_PATH}")
    print(f"  Output : {SEQ_DIR}")
    print(f"  Train end (exclusive) : {TRAIN_END.isoformat()}")
    print(f"  Test  start (excl.)   : {TEST_START.isoformat()}")
    print(f"  Validation window     : [{TRAIN_END.isoformat()}, {TEST_START.isoformat()})")

    _ensure_gcp_credentials()
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load vocabs ──────────────────────────────────────────────────────────
    t0 = _step(f"Loading vocabs from {VOCABS_PATH}")
    with open(VOCABS_PATH, "rb") as f:
        vocabs: dict[str, Any] = pickle.load(f)
    user2idx: dict[int, int] = vocabs["user2idx"]
    item2idx: dict[int, int] = vocabs["item2idx"]
    _done(
        t0,
        label=f"{len(user2idx):,} users  /  {len(item2idx):,} items",
    )

    # ── Load cleaned train events from GCS ───────────────────────────────────
    t0 = _step(f"Reading {TRAIN_GCS_PATH}*.parquet (event-level, cleaned)")
    train_events = pd.read_parquet(
        TRAIN_GCS_PATH,
        columns=["event_time", "event_type", "product_id", "user_id"],
    )
    _done(t0, label=f"{len(train_events):,} rows")

    # ── Map ids → idxs and event_type → event_idx ────────────────────────────
    t0 = _step("Mapping user_id / product_id → idx and event_type → event_idx")
    train_events["user_idx"]  = train_events["user_id"].map(user2idx)
    train_events["item_idx"]  = train_events["product_id"].map(item2idx)
    train_events["event_idx"] = train_events["event_type"].map(EVENT_TYPE_MAP)

    n_pre = len(train_events)
    # Drop rows with unknown user, item, or event_type (defensive — should be ~0
    # given the vocab was built from this exact sample).
    train_events = train_events.dropna(subset=["user_idx", "item_idx", "event_idx"])
    n_post = len(train_events)
    if n_pre != n_post:
        print(f"     dropped {n_pre - n_post:,} rows with unknown id/event_type")

    train_events["user_idx"]  = train_events["user_idx"].astype(np.int64)
    train_events["item_idx"]  = train_events["item_idx"].astype(np.int64)
    train_events["event_idx"] = train_events["event_idx"].astype(np.int64)
    train_events["event_time"] = pd.to_datetime(train_events["event_time"], utc=True)
    _done(t0, label=f"{n_post:,} rows after id mapping")

    # ── Sort once (chronological per user) ───────────────────────────────────
    # mergesort is stable so events with identical timestamps keep their
    # original ingest order from upstream.
    t0 = _step("Sorting by (user_idx, event_time)")
    train_events = train_events.sort_values(
        ["user_idx", "event_time"], kind="mergesort"
    ).reset_index(drop=True)
    _done(t0)

    # ── Split into train (< TRAIN_END) and val window [TRAIN_END, TEST_START) ─
    train_mask = train_events["event_time"] < TRAIN_END
    val_mask   = (~train_mask) & (train_events["event_time"] < TEST_START)

    n_train_events     = int(train_mask.sum())
    n_val_window_events = int(val_mask.sum())
    n_full_train_events = int((train_events["event_time"] < TEST_START).sum())
    print(
        f"\n  Event-level split:"
        f"\n     train (< {TRAIN_END.date()})           : {n_train_events:,} events"
        f"\n     val window [{TRAIN_END.date()}, {TEST_START.date()}): "
        f"{n_val_window_events:,} events"
        f"\n     full train (< {TEST_START.date()})    : {n_full_train_events:,} events"
    )

    # ── Build train sequences (events before Jan 25) ─────────────────────────
    train_seqs = _build_per_user_sequences(
        train_events.loc[train_mask, ["user_idx", "item_idx", "event_idx"]],
        label="train (before Jan 25)",
    )

    out_train = SEQ_DIR / "train_seqs.parquet"
    t0 = _step(f"Writing {out_train.name}")
    train_seqs.to_parquet(out_train, index=False)
    _done(t0, label=f"{out_train.stat().st_size / 1e6:.1f} MB")

    # ── Build full train sequences (events before Feb 1) ─────────────────────
    full_train_seqs = _build_per_user_sequences(
        train_events.loc[
            train_events["event_time"] < TEST_START,
            ["user_idx", "item_idx", "event_idx"],
        ],
        label="full train (before Feb 1)",
    )

    out_full = SEQ_DIR / "full_train_seqs.parquet"
    t0 = _step(f"Writing {out_full.name}")
    full_train_seqs.to_parquet(out_full, index=False)
    _done(t0, label=f"{out_full.stat().st_size / 1e6:.1f} MB")

    # ── Build val_targets (cart + purchase in window) ────────────────────────
    t0 = _step("Building val_targets (cart + purchase in window, dedup)")
    val_window = train_events.loc[val_mask].copy()
    val_targets = val_window[val_window["event_type"].isin({"cart", "purchase"})]
    val_targets = val_targets.drop_duplicates(subset=["user_idx", "item_idx"])

    val_targets = val_targets[
        ["user_idx", "item_idx", "event_type", "product_id", "event_time", "user_id"]
    ].reset_index(drop=True)
    val_targets["user_idx"] = val_targets["user_idx"].astype(np.int64)
    val_targets["item_idx"] = val_targets["item_idx"].astype(np.int64)
    _done(
        t0,
        label=(f"{len(val_targets):,} rows over "
               f"{val_targets['user_idx'].nunique():,} users"),
    )

    out_val = SEQ_DIR / "val_targets.parquet"
    t0 = _step(f"Writing {out_val.name}")
    val_targets.to_parquet(out_val, index=False)
    _done(t0, label=f"{out_val.stat().st_size / 1e6:.1f} MB")

    # ── metadata.json ────────────────────────────────────────────────────────
    metadata: dict[str, Any] = {
        "max_seq_len":      MAX_SEQ_LEN,
        "n_items":          int(len(item2idx)),
        "n_event_types":    int(len(EVENT_TYPE_MAP) + 1),  # +1 for PAD
        "event_type_map":   EVENT_TYPE_MAP,
        "padding_idx":      0,
        "train_end":        TRAIN_END.isoformat(),
        "val_start":        TRAIN_END.isoformat(),
        "val_end":          TEST_START.isoformat(),
        "test_start":       TEST_START.isoformat(),
        "n_users_train":    int(len(train_seqs)),
        "n_users_full":     int(len(full_train_seqs)),
        "n_users_val":      int(val_targets["user_idx"].nunique()),
        "n_train_events":   n_train_events,
        "n_val_events":     n_val_window_events,
        "n_full_events":    n_full_train_events,
    }
    out_meta = SEQ_DIR / "metadata.json"
    with open(out_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Wrote {out_meta}")

    _section("Summary")
    print(f"  train_seqs       : {len(train_seqs):,} users  ({out_train})")
    print(f"  full_train_seqs  : {len(full_train_seqs):,} users  ({out_full})")
    print(f"  val_targets      : {len(val_targets):,} rows / "
          f"{val_targets['user_idx'].nunique():,} users  ({out_val})")
    print(f"  metadata         : {out_meta}")
    elapsed = int(time.time() - t_total)
    print(f"\n  Total time       : {elapsed // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
