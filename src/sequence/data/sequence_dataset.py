"""PyTorch Dataset wrappers for the sequence-recommendation pipeline.

Two classes:

  ``SequenceTrainDataset``
      One sequence per user (their pre-Jan-25 history, truncated and
      left-padded to ``max_seq_len``).  ``__getitem__`` yields the shifted
      next-item training target.

  ``SequenceEvalDataset``
      Dense ``(n_users, max_seq_len)`` numpy arrays (item_seq, event_seq)
      indexable by ``user_idx``.  Used by the evaluator to encode users
      with their full pre-test history.

Both load from parquets produced by
``scripts/sequence/build_sequences_500k.py``.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Padding token (must match build_sequences_500k.py).
PAD_ITEM_IDX:  int = 0
PAD_EVENT_IDX: int = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_padded_arrays(
    seqs_df: pd.DataFrame,
    max_seq_len: int,
    n_users: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise two left-padded ``(n_users, max_seq_len)`` numpy arrays.

    Sequences longer than ``max_seq_len`` are right-truncated (we keep the
    most recent items).  Sequences shorter than ``max_seq_len`` are
    left-padded with ``PAD_ITEM_IDX`` / ``PAD_EVENT_IDX`` so the rightmost
    real item is always at column ``max_seq_len - 1``.

    Args:
        seqs_df:     DataFrame with columns user_idx, item_seq, event_seq.
        max_seq_len: Sequence truncation / padding length.
        n_users:     Length of the output arrays (typically ``len(user2idx)``).
        label:       Short string for the progress print line.

    Returns:
        item_seq_arr:  int64 ndarray, shape (n_users, max_seq_len)
        event_seq_arr: int64 ndarray, shape (n_users, max_seq_len)
    """
    t0 = time.time()
    item_arr  = np.full((n_users, max_seq_len), PAD_ITEM_IDX,  dtype=np.int64)
    event_arr = np.full((n_users, max_seq_len), PAD_EVENT_IDX, dtype=np.int64)

    user_idxs = seqs_df["user_idx"].to_numpy(dtype=np.int64)
    item_seqs = seqs_df["item_seq"].to_numpy(dtype=object)
    event_seqs = seqs_df["event_seq"].to_numpy(dtype=object)

    for u, items, events in zip(user_idxs, item_seqs, event_seqs):
        items_arr  = np.asarray(items,  dtype=np.int64)
        events_arr = np.asarray(events, dtype=np.int64)
        # Keep the most recent max_seq_len events.
        if items_arr.shape[0] > max_seq_len:
            items_arr  = items_arr[-max_seq_len:]
            events_arr = events_arr[-max_seq_len:]
        n = items_arr.shape[0]
        # Left-pad: place the real items at the end of the row.
        item_arr [u, max_seq_len - n:] = items_arr
        event_arr[u, max_seq_len - n:] = events_arr

    elapsed = time.time() - t0
    print(
        f"  Padded {label}: {len(seqs_df):,} users → "
        f"({n_users:,}, {max_seq_len}) in {elapsed:.1f}s "
        f"(memory ~{2 * item_arr.nbytes / 1e6:.0f} MB)"
    )
    return item_arr, event_arr


# ── Training dataset ───────────────────────────────────────────────────────────

class SequenceTrainDataset(Dataset):
    """Per-user shifted next-item training instances for sequence models.

    Each ``__getitem__(idx)`` returns a single user's left-padded sequence
    expressed as four ``(L,)`` tensors where ``L = max_seq_len - 1``:

        ``input_seq``       (long)  — items at positions [0 .. L-1]
        ``input_event_seq`` (long)  — event types,        same length
        ``target_seq``      (long)  — items at positions [1 .. L]   (shifted)
        ``target_mask``     (bool) — True where target is a real item

    ``user_idx`` is also returned so the train script can verify shuffling
    and (later) attach per-user metadata if needed.

    Args:
        train_seqs_df: DataFrame with columns user_idx (int64),
                       item_seq (list[int]), event_seq (list[int]).
                       Loaded from ``train_seqs.parquet``.
        n_users:       Allocate the dense lookup array of this length.
                       Pass ``len(vocabs["user2idx"])`` so user_idx → row
                       lookups work even for users not in this frame.
        max_seq_len:   Truncation / padding length (default 50).
    """

    def __init__(
        self,
        train_seqs_df: pd.DataFrame,
        n_users: int,
        max_seq_len: int = 50,
    ) -> None:
        if not {"user_idx", "item_seq", "event_seq"}.issubset(train_seqs_df.columns):
            raise ValueError(
                "train_seqs_df must have columns: user_idx, item_seq, event_seq"
            )

        self.max_seq_len = int(max_seq_len)
        self.n_users     = int(n_users)

        # Materialise dense arrays once so __getitem__ is a pure numpy slice.
        self._item_arr, self._event_arr = _build_padded_arrays(
            train_seqs_df,
            max_seq_len = self.max_seq_len,
            n_users     = self.n_users,
            label       = "train_seqs",
        )

        # Index → user_idx mapping (so __len__ matches the training pool, not
        # n_users — users with < 2 events are absent from train_seqs_df).
        self._train_user_idxs = train_seqs_df["user_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return int(self._train_user_idxs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        u = int(self._train_user_idxs[idx])
        item_row  = self._item_arr [u]    # (max_seq_len,)
        event_row = self._event_arr[u]    # (max_seq_len,)

        # Shifted next-item training: input = [:-1], target = [1:].
        # With left-padding the leading positions are PAD; the mask drops them.
        input_seq        = item_row [:-1]
        input_event_seq  = event_row[:-1]
        target_seq       = item_row [1:]
        target_mask      = target_seq != PAD_ITEM_IDX

        return {
            "user_idx":        torch.tensor(u, dtype=torch.long),
            "input_seq":       torch.from_numpy(input_seq.copy()).long(),
            "input_event_seq": torch.from_numpy(input_event_seq.copy()).long(),
            "target_seq":      torch.from_numpy(target_seq.copy()).long(),
            "target_mask":     torch.from_numpy(target_mask.copy()).bool(),
        }

    def __repr__(self) -> str:
        return (
            f"SequenceTrainDataset(n_users_with_history={len(self):,}, "
            f"max_seq_len={self.max_seq_len}, "
            f"n_users_total={self.n_users:,})"
        )


# ── Evaluation dataset ─────────────────────────────────────────────────────────

class SequenceEvalDataset:
    """Dense per-user padded sequences for the evaluator.

    Not a ``torch.utils.data.Dataset`` because the evaluator iterates
    over arbitrary ``user_idx`` slices in its own batch loop, so direct
    numpy indexing is the simplest interface.

    Args:
        seqs_df:     DataFrame with columns user_idx, item_seq, event_seq
                     (typically loaded from ``full_train_seqs.parquet`` for
                     test-time encoding, or ``train_seqs.parquet`` for
                     val-time encoding).
        n_users:     Allocate dense arrays of this length so user_idx
                     lookups never go out of bounds.
        max_seq_len: Truncation / padding length.
    """

    def __init__(
        self,
        seqs_df: pd.DataFrame,
        n_users: int,
        max_seq_len: int = 50,
    ) -> None:
        if not {"user_idx", "item_seq", "event_seq"}.issubset(seqs_df.columns):
            raise ValueError(
                "seqs_df must have columns: user_idx, item_seq, event_seq"
            )

        self.max_seq_len = int(max_seq_len)
        self.n_users     = int(n_users)

        self._item_arr, self._event_arr = _build_padded_arrays(
            seqs_df,
            max_seq_len = self.max_seq_len,
            n_users     = self.n_users,
            label       = "eval_seqs",
        )
        self._users_with_history = set(seqs_df["user_idx"].astype(int).tolist())

    @property
    def item_seq_arr(self) -> np.ndarray:
        """Read-only handle to the (n_users, max_seq_len) item array."""
        return self._item_arr

    @property
    def event_seq_arr(self) -> np.ndarray:
        """Read-only handle to the (n_users, max_seq_len) event array."""
        return self._event_arr

    def has_history(self, user_idx: int) -> bool:
        """True iff this user contributed at least one event."""
        return int(user_idx) in self._users_with_history

    def __repr__(self) -> str:
        return (
            f"SequenceEvalDataset(n_users_with_history="
            f"{len(self._users_with_history):,}, "
            f"max_seq_len={self.max_seq_len}, "
            f"n_users_total={self.n_users:,})"
        )


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_sequence_artifacts(
    seq_dir: pathlib.Path | str,
) -> dict[str, Any]:
    """Read all four artifacts produced by ``build_sequences_500k.py``.

    Returns a dict with keys: ``train_seqs_df``, ``full_train_seqs_df``,
    ``val_targets_df``, ``metadata``.
    """
    import json

    seq_dir = pathlib.Path(seq_dir)
    train_seqs_df       = pd.read_parquet(seq_dir / "train_seqs.parquet")
    full_train_seqs_df  = pd.read_parquet(seq_dir / "full_train_seqs.parquet")
    val_targets_df      = pd.read_parquet(seq_dir / "val_targets.parquet")
    with open(seq_dir / "metadata.json") as f:
        metadata = json.load(f)
    return {
        "train_seqs_df":      train_seqs_df,
        "full_train_seqs_df": full_train_seqs_df,
        "val_targets_df":     val_targets_df,
        "metadata":           metadata,
    }
