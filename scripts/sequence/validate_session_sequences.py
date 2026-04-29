"""Day 2 validation: assert all session-sequence invariants hold.

Checks (per split: train / val / test):
  1. max seq_len <= 20
  2. min seq_len >= 2   (no length-1 sessions)
  3. no PAD token (0) as the last item of any session
  4. no consecutive repeated items within any session

Checks (metadata.json):
  5. n_train/val/test_sessions match parquet row counts
  6. max_seq_len, min_seq_len, consecutive_repeats_removed, schema_version

Smoke test (dataset API):
  7. SessionTrainDataset builds and returns correct-shape tensors
  8. SessionEvalDataset builds, shapes correct, target_items all > 0
  9. evaluate_sessions importable

Exit 0 if all assertions pass, exit 1 otherwise.

Usage:
    python scripts/sequence/validate_session_sequences.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

SEQ_DIR = _REPO_ROOT / "artifacts" / "500k" / "sequences_v2"
MAX_SEQ_LEN = 20
MIN_SEQ_LEN = 2


# -- Assertion helpers ----------------------------------------------------------

_results: list[tuple[bool, str]] = []


def _assert(condition: bool, message: str) -> None:
    tag = "[PASS]" if condition else "[FAIL]"
    print(f"  {tag}  {message}")
    _results.append((condition, message))


# -- Per-split checks -----------------------------------------------------------

def _check_split(df: pd.DataFrame, label: str, max_len: int, min_len: int) -> None:
    print(f"\n  --- Data integrity: {label} ---")

    obs_max = int(df["seq_len"].max())
    _assert(obs_max <= max_len, f"{label} | max seq_len <= {max_len} (observed: {obs_max})")

    obs_min = int(df["seq_len"].min())
    _assert(obs_min >= min_len, f"{label} | min seq_len >= {min_len} (observed: {obs_min})")

    # No PAD (0) as last item - check the last element of every item_seq.
    t0 = time.time()
    bad_pad = sum(
        1 for items in df["item_seq"].to_numpy()
        if len(items) == 0 or int(np.asarray(items, dtype=np.int64)[-1]) == 0
    )
    _assert(
        bad_pad == 0,
        f"{label} | no PAD (0) as last item  ({bad_pad} violations, {time.time()-t0:.1f}s)"
    )

    # No consecutive repeated items.
    t0 = time.time()
    consec_violations = sum(
        1 for items in df["item_seq"].to_numpy()
        if (arr := np.asarray(items, dtype=np.int64)).shape[0] > 1
        and bool(np.any(arr[1:] == arr[:-1]))
    )
    _assert(
        consec_violations == 0,
        f"{label} | no consecutive repeated items  "
        f"({consec_violations} violations, {time.time()-t0:.1f}s)"
    )


# -- Main -----------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 64)
    print("  Validate Session Sequences - V9 / T4Rec reframe")
    print("=" * 64)
    print(f"  Sequences dir : {SEQ_DIR}")

    # -- File existence -----------------------------------------------------
    print("\n  --- File system ---")
    train_path = SEQ_DIR / "train_sessions.parquet"
    val_path   = SEQ_DIR / "val_sessions.parquet"
    test_path  = SEQ_DIR / "test_sessions.parquet"
    meta_path  = SEQ_DIR / "metadata.json"

    for path in (train_path, val_path, test_path, meta_path):
        _assert(path.exists(), f"{path.name} exists")

    if not all(ok for ok, _ in _results):
        print("\n  Aborting - required files are missing.")
        sys.exit(1)

    # -- Load splits --------------------------------------------------------
    print("\n  Loading splits...")
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"     train_sessions.parquet : {len(train_df):>10,} rows")
    print(f"     val_sessions.parquet   : {len(val_df):>10,} rows")
    print(f"     test_sessions.parquet  : {len(test_df):>10,} rows")

    # -- Data integrity per split -------------------------------------------
    _check_split(train_df, "train", MAX_SEQ_LEN, MIN_SEQ_LEN)
    _check_split(val_df,   "val",   MAX_SEQ_LEN, MIN_SEQ_LEN)
    _check_split(test_df,  "test",  MAX_SEQ_LEN, MIN_SEQ_LEN)

    # -- Metadata consistency -----------------------------------------------
    print("\n  --- Metadata consistency ---")
    _assert(
        meta.get("n_train_sessions") == len(train_df),
        f"metadata.n_train_sessions = {meta.get('n_train_sessions'):,} matches parquet ({len(train_df):,})"
    )
    _assert(
        meta.get("n_val_sessions") == len(val_df),
        f"metadata.n_val_sessions   = {meta.get('n_val_sessions'):,} matches parquet ({len(val_df):,})"
    )
    _assert(
        meta.get("n_test_sessions") == len(test_df),
        f"metadata.n_test_sessions  = {meta.get('n_test_sessions'):,} matches parquet ({len(test_df):,})"
    )
    _assert(
        meta.get("max_seq_len") == MAX_SEQ_LEN,
        f"metadata.max_seq_len = {meta.get('max_seq_len')}"
    )
    _assert(
        meta.get("min_seq_len") == MIN_SEQ_LEN,
        f"metadata.min_seq_len = {meta.get('min_seq_len')}"
    )
    _assert(
        meta.get("consecutive_repeats_removed") is True,
        f"metadata.consecutive_repeats_removed = {meta.get('consecutive_repeats_removed')}"
    )
    _assert(
        meta.get("schema_version") == "v9_session",
        f"metadata.schema_version = {meta.get('schema_version')!r}"
    )

    # -- Dataset API smoke test ---------------------------------------------
    print("\n  --- Dataset API smoke test ---")

    try:
        from src.sequence.data.session_dataset import (
            SessionTrainDataset,
            SessionEvalDataset,
            load_session_artifacts,
        )
        from src.sequence.evaluation.evaluate_sequence import evaluate_sessions
        _assert(True, "session_dataset + evaluate_sessions import successfully")
    except Exception as exc:
        _assert(False, f"Import failed: {exc}")
        print("\n  Aborting - cannot import required modules.")
        _print_summary()
        sys.exit(1)

    # SessionTrainDataset: build on a small slice, check shapes and keys.
    n_smoke = 100
    train_small = train_df.head(n_smoke).reset_index(drop=True)
    t0 = time.time()
    train_ds = SessionTrainDataset(train_small, max_seq_len=MAX_SEQ_LEN)
    _assert(
        len(train_ds) == n_smoke,
        f"SessionTrainDataset.__len__ = {len(train_ds)} (expected {n_smoke})"
    )

    batch = train_ds[0]
    expected_keys = {"session_idx", "user_idx", "input_seq", "input_event_seq",
                     "target_seq", "target_mask"}
    _assert(
        expected_keys.issubset(batch.keys()),
        f"SessionTrainDataset.__getitem__ keys correct: {sorted(batch.keys())}"
    )

    expected_L = MAX_SEQ_LEN - 1
    _assert(
        tuple(batch["input_seq"].shape) == (expected_L,),
        f"SessionTrainDataset input_seq shape = {tuple(batch['input_seq'].shape)} (expected ({expected_L},))"
    )
    _assert(
        tuple(batch["target_seq"].shape) == (expected_L,),
        f"SessionTrainDataset target_seq shape = {tuple(batch['target_seq'].shape)} (expected ({expected_L},))"
    )
    _assert(
        str(batch["target_mask"].dtype) == "torch.bool",
        f"SessionTrainDataset target_mask dtype = {batch['target_mask'].dtype} (expected torch.bool)"
    )

    # Verify at least one real (non-PAD) target position exists per session.
    n_real_targets = int(batch["target_mask"].sum().item())
    _assert(
        n_real_targets >= 1,
        f"SessionTrainDataset batch[0] has {n_real_targets} real target positions (>= 1)"
    )

    # SessionEvalDataset: build on test slice, check shapes + no-PAD target.
    test_small = test_df.head(n_smoke).reset_index(drop=True)
    eval_ds = SessionEvalDataset(test_small, max_seq_len=MAX_SEQ_LEN)
    _assert(
        eval_ds.n_sessions == n_smoke,
        f"SessionEvalDataset.n_sessions = {eval_ds.n_sessions} (expected {n_smoke})"
    )
    _assert(
        eval_ds.prefix_item_arr.shape == (n_smoke, MAX_SEQ_LEN),
        f"SessionEvalDataset.prefix_item_arr shape = {eval_ds.prefix_item_arr.shape}"
        f" (expected ({n_smoke}, {MAX_SEQ_LEN}))"
    )
    _assert(
        eval_ds.prefix_event_arr.shape == (n_smoke, MAX_SEQ_LEN),
        f"SessionEvalDataset.prefix_event_arr shape = {eval_ds.prefix_event_arr.shape}"
    )
    _assert(
        bool(np.all(eval_ds.target_items > 0)),
        f"SessionEvalDataset.target_items all > 0 - no PAD as target"
        f" (min={int(eval_ds.target_items.min())})"
    )

    # Full-dataset target check (all three splits).
    print("\n  --- Full-dataset target item check ---")
    for df, lbl in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        last_items = np.array(
            [np.asarray(items, dtype=np.int64)[-1] for items in df["item_seq"].to_numpy()],
            dtype=np.int64
        )
        n_zero = int((last_items == 0).sum())
        _assert(
            n_zero == 0,
            f"{lbl} | full dataset: no PAD (0) as last item  (0/{len(df):,})"
        )

    _print_summary()


def _print_summary() -> None:
    n_pass = sum(1 for ok, _ in _results if ok)
    n_fail = sum(1 for ok, _ in _results if not ok)
    total  = len(_results)
    print()
    print("=" * 64)
    if n_fail == 0:
        print(f"  ALL {total} ASSERTIONS PASSED - Day 2 validation complete.")
    else:
        print(f"  FAILED: {n_fail}/{total} assertions.")
        print("  Fix the issues above before proceeding to Day 3.")
    print("=" * 64)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
