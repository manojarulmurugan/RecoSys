"""PyTorch Dataset wrappers for session-based sequence recommendation (V9).

Two classes:

  ``SessionTrainDataset``
      One training instance per session.  Each ``__getitem__`` returns the
      shifted next-item targets for every position in the session - same
      convention as ``SequenceTrainDataset`` but indexed positionally by
      session rather than by ``user_idx``.

  ``SessionEvalDataset``
      Per-session (prefix, target_item) pairs for evaluation.
      Prefix = item_seq[:-1] left-padded to max_seq_len.
      Target = item_seq[-1]  (held-out last item, T4Rec sec.4.1.3).

Both load from parquets produced by
``scripts/sequence/build_session_sequences.py``.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_ITEM_IDX:  int = 0
PAD_EVENT_IDX: int = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_session_padded_arrays(
    sessions_df: pd.DataFrame,
    max_seq_len: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise two left-padded ``(n_sessions, max_seq_len)`` numpy arrays.

    Positional: row i <-> sessions_df row i (not user_idx-based).
    Sessions longer than max_seq_len are right-truncated to keep the most
    recent events; shorter sessions are left-padded with PAD.
    """
    t0 = time.time()
    n = len(sessions_df)
    item_arr  = np.full((n, max_seq_len), PAD_ITEM_IDX,  dtype=np.int64)
    event_arr = np.full((n, max_seq_len), PAD_EVENT_IDX, dtype=np.int64)

    for i, (items, events) in enumerate(zip(
        sessions_df["item_seq"].to_numpy(),
        sessions_df["event_seq"].to_numpy(),
    )):
        items_arr  = np.asarray(items,  dtype=np.int64)
        events_arr = np.asarray(events, dtype=np.int64)
        if items_arr.shape[0] > max_seq_len:
            items_arr  = items_arr[-max_seq_len:]
            events_arr = events_arr[-max_seq_len:]
        L = items_arr.shape[0]
        item_arr [i, max_seq_len - L:] = items_arr
        event_arr[i, max_seq_len - L:] = events_arr

    elapsed = time.time() - t0
    print(
        f"  Padded {label}: {n:,} sessions -> "
        f"({n:,} x {max_seq_len}) in {elapsed:.1f}s "
        f"(~{2 * item_arr.nbytes / 1e6:.0f} MB)"
    )
    return item_arr, event_arr


# ── Training dataset ───────────────────────────────────────────────────────────

class SessionTrainDataset(Dataset):
    """Per-session shifted next-item training instances.

    Each ``__getitem__(i)`` corresponds to one session and returns:

        ``input_seq``       (max_seq_len-1,) long  - items [0..L-2], left-padded.
        ``input_event_seq`` (max_seq_len-1,) long  - event types, same shape.
        ``target_seq``      (max_seq_len-1,) long  - items [1..L-1], left-padded.
        ``target_mask``     (max_seq_len-1,) bool  - True where target != PAD.
        ``session_idx``     scalar long.
        ``user_idx``        scalar long.

    The DataLoader from this dataset plugs directly into the existing
    ``train_epoch_sequence`` loop (keys match what it expects).

    Args:
        sessions_df: DataFrame with columns session_idx (int64), user_idx (int64),
                     item_seq (list[int]), event_seq (list[int]).
                     Loaded from ``train_sessions.parquet``.
        max_seq_len: Truncation / padding length (default 20, T4Rec sec.4.1.4).
    """

    def __init__(
        self,
        sessions_df: pd.DataFrame,
        max_seq_len: int = 20,
    ) -> None:
        required = {"session_idx", "user_idx", "item_seq", "event_seq"}
        if not required.issubset(sessions_df.columns):
            raise ValueError(f"sessions_df must have columns: {required}")

        self.max_seq_len = int(max_seq_len)

        self._item_arr, self._event_arr = _build_session_padded_arrays(
            sessions_df, self.max_seq_len, label="train_sessions"
        )
        self._session_idxs = sessions_df["session_idx"].to_numpy(dtype=np.int64)
        self._user_idxs    = sessions_df["user_idx"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return self._item_arr.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item_row  = self._item_arr [idx]
        event_row = self._event_arr[idx]

        input_seq       = item_row [:-1]
        input_event_seq = event_row[:-1]
        target_seq      = item_row [1:]
        target_mask     = target_seq != PAD_ITEM_IDX

        return {
            "session_idx":     torch.tensor(self._session_idxs[idx], dtype=torch.long),
            "user_idx":        torch.tensor(self._user_idxs[idx],    dtype=torch.long),
            "input_seq":       torch.from_numpy(input_seq.copy()).long(),
            "input_event_seq": torch.from_numpy(input_event_seq.copy()).long(),
            "target_seq":      torch.from_numpy(target_seq.copy()).long(),
            "target_mask":     torch.from_numpy(target_mask.copy()).bool(),
        }

    def __repr__(self) -> str:
        return (
            f"SessionTrainDataset(n_sessions={len(self):,}, "
            f"max_seq_len={self.max_seq_len})"
        )


# ── Evaluation dataset ─────────────────────────────────────────────────────────

class SessionEvalDataset:
    """Per-session (prefix, target_item) pairs for next-item-prediction eval.

    For each session of length L:
        prefix  = item_seq[:-1]  left-padded to ``max_seq_len``
        target  = item_seq[-1]   held-out ground-truth (T4Rec sec.4.1.3)

    Not a ``torch.utils.data.Dataset`` - the evaluator accesses the dense
    numpy arrays directly in its own batch loop.

    Args:
        sessions_df: DataFrame with columns session_idx, user_idx, item_seq,
                     event_seq.  Typically loaded from ``test_sessions.parquet``
                     or ``val_sessions.parquet``.
        max_seq_len: Truncation / padding length.
    """

    def __init__(
        self,
        sessions_df: pd.DataFrame,
        max_seq_len: int = 20,
    ) -> None:
        required = {"session_idx", "user_idx", "item_seq", "event_seq"}
        if not required.issubset(sessions_df.columns):
            raise ValueError(f"sessions_df must have columns: {required}")

        self.max_seq_len = int(max_seq_len)
        n = len(sessions_df)

        self._session_idxs = sessions_df["session_idx"].to_numpy(dtype=np.int64)
        self._user_idxs    = sessions_df["user_idx"].to_numpy(dtype=np.int64)
        self._target_items = np.zeros(n, dtype=np.int64)

        t0 = time.time()
        prefix_item_arr  = np.full((n, max_seq_len), PAD_ITEM_IDX,  dtype=np.int64)
        prefix_event_arr = np.full((n, max_seq_len), PAD_EVENT_IDX, dtype=np.int64)

        for i, (items, events) in enumerate(zip(
            sessions_df["item_seq"].to_numpy(),
            sessions_df["event_seq"].to_numpy(),
        )):
            items_arr  = np.asarray(items,  dtype=np.int64)
            events_arr = np.asarray(events, dtype=np.int64)

            self._target_items[i] = int(items_arr[-1])

            prefix_items  = items_arr[:-1]
            prefix_events = events_arr[:-1]

            if prefix_items.shape[0] > max_seq_len:
                prefix_items  = prefix_items[-max_seq_len:]
                prefix_events = prefix_events[-max_seq_len:]

            L = prefix_items.shape[0]
            if L > 0:
                prefix_item_arr [i, max_seq_len - L:] = prefix_items
                prefix_event_arr[i, max_seq_len - L:] = prefix_events

        elapsed = time.time() - t0
        print(
            f"  SessionEvalDataset: {n:,} sessions, "
            f"prefix shape ({n:,} x {max_seq_len}) in {elapsed:.1f}s"
        )

        self._prefix_item_arr  = prefix_item_arr
        self._prefix_event_arr = prefix_event_arr

    @property
    def n_sessions(self) -> int:
        return int(self._session_idxs.shape[0])

    @property
    def session_idxs(self) -> np.ndarray:
        return self._session_idxs

    @property
    def user_idxs(self) -> np.ndarray:
        return self._user_idxs

    @property
    def target_items(self) -> np.ndarray:
        """(n_sessions,) int64 - held-out last item per session."""
        return self._target_items

    @property
    def prefix_item_arr(self) -> np.ndarray:
        """(n_sessions, max_seq_len) int64 - left-padded session prefixes."""
        return self._prefix_item_arr

    @property
    def prefix_event_arr(self) -> np.ndarray:
        """(n_sessions, max_seq_len) int64 - left-padded event prefixes."""
        return self._prefix_event_arr

    def __repr__(self) -> str:
        return (
            f"SessionEvalDataset(n_sessions={self.n_sessions:,}, "
            f"max_seq_len={self.max_seq_len})"
        )


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_session_artifacts(
    seq_dir: pathlib.Path | str,
) -> dict[str, Any]:
    """Read all artifacts produced by ``build_session_sequences.py``.

    Returns a dict with keys:
        ``train_sessions_df``, ``val_sessions_df``, ``test_sessions_df``,
        ``metadata``.
    """
    seq_dir = pathlib.Path(seq_dir)
    train_df = pd.read_parquet(seq_dir / "train_sessions.parquet")
    val_df   = pd.read_parquet(seq_dir / "val_sessions.parquet")
    test_df  = pd.read_parquet(seq_dir / "test_sessions.parquet")
    with open(seq_dir / "metadata.json") as f:
        metadata = json.load(f)
    return {
        "train_sessions_df": train_df,
        "val_sessions_df":   val_df,
        "test_sessions_df":  test_df,
        "metadata":          metadata,
    }
