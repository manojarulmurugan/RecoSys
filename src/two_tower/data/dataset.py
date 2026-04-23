"""PyTorch Dataset for the Two-Tower training loop.

Wraps pre-encoded parquet artifacts into fast numpy lookups so that
DataLoader workers spend no time in pandas during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    """Maps (user_idx, item_idx, confidence_score) training pairs to
    feature tensors via O(1) numpy array lookups.

    Args:
        train_pairs_df:    DataFrame with columns [user_idx, item_idx, confidence_score].
        users_encoded_df:  DataFrame produced by FeatureBuilder.build() for users.
        items_encoded_df:  DataFrame produced by FeatureBuilder.build() for items.
    """

    def __init__(
        self,
        train_pairs_df: pd.DataFrame,
        users_encoded_df: pd.DataFrame,
        items_encoded_df: pd.DataFrame,
    ) -> None:
        self.pairs = train_pairs_df.reset_index(drop=True)

        # ── User feature lookups keyed by user_idx ────────────────────────────
        # Allocate arrays large enough to hold the highest user_idx.
        n_users = int(users_encoded_df["user_idx"].max()) + 1

        # int64: [top_cat_idx, peak_hour_bucket, preferred_dow, has_purchase_history]
        user_cat_arr = np.zeros((n_users, 4), dtype=np.int64)
        user_cat_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
            ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
        ].values.astype(np.int64)

        # float32: [log_total_events, months_active, purchase_rate, cart_rate,
        #           log_n_sessions, avg_purchase_price_scaled]
        user_dense_arr = np.zeros((n_users, 6), dtype=np.float32)
        user_dense_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
            ["log_total_events", "months_active", "purchase_rate", "cart_rate",
             "log_n_sessions", "avg_purchase_price_scaled"]
        ].values.astype(np.float32)

        self._user_cat   = user_cat_arr
        self._user_dense = user_dense_arr

        # ── Item feature lookups keyed by item_idx ────────────────────────────
        n_items = int(items_encoded_df["item_idx"].max()) + 1

        # int64: [item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket]
        item_cat_arr = np.zeros((n_items, 5), dtype=np.int64)
        item_cat_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
        ].values.astype(np.int64)

        # float32: [avg_price_scaled, log_confidence_scaled, purchase_rate_scaled]
        item_dense_arr = np.zeros((n_items, 3), dtype=np.float32)
        item_dense_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
        ].values.astype(np.float32)

        self._item_cat   = item_cat_arr
        self._item_dense = item_dense_arr

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row      = self.pairs.iloc[idx]
        user_idx = int(row.user_idx)
        item_idx = int(row.item_idx)

        return {
            "user_idx":    torch.tensor(user_idx,                    dtype=torch.long),
            "item_idx":    torch.tensor(item_idx,                    dtype=torch.long),
            "user_cat":    torch.tensor(self._user_cat[user_idx],    dtype=torch.long),
            "user_dense":  torch.tensor(self._user_dense[user_idx],  dtype=torch.float32),
            "item_cat":    torch.tensor(self._item_cat[item_idx],    dtype=torch.long),
            "item_dense":  torch.tensor(self._item_dense[item_idx],  dtype=torch.float32),
            "confidence":  torch.tensor(float(row.confidence_score), dtype=torch.float32),
        }


class TwoTowerDatasetWithHardNegs(Dataset):
    """Two-Tower dataset with pre-mined hard negatives.

    Hard negative strategy (decided from data investigation):

    Primary  — cat_l2 negatives (3 per positive):
        Sample 3 items sharing the positive item's cat_l2, excluding all
        items the user has interacted with.

    Fallback — price_bucket negatives (3 per positive):
        Used when the positive item has cat_l2_idx == 0 (unknown category).
        Sample 3 items from the same price_bucket instead.

    All mining is done once in __init__ and stored in a (N, 3) int64 tensor
    so that __getitem__ is a simple O(1) lookup.

    Args:
        train_pairs_df:    DataFrame with columns [user_idx, item_idx, confidence_score].
        users_encoded_df:  DataFrame produced by FeatureBuilder.build() for users.
        items_encoded_df:  DataFrame produced by FeatureBuilder.build() for items.
        n_hard_negs:       Hard negatives per positive (default: 3).
        seed:              NumPy random seed for reproducible mining (default: 42).
    """

    _N_HARD: int = 3   # hard negatives per positive pair

    def __init__(
        self,
        train_pairs_df: pd.DataFrame,
        users_encoded_df: pd.DataFrame,
        items_encoded_df: pd.DataFrame,
        n_hard_negs: int = 3,
        seed: int = 42,
    ) -> None:
        self.pairs = train_pairs_df.reset_index(drop=True)
        self._n_hard = n_hard_negs

        # ── User feature lookups (identical to TwoTowerDataset) ───────────────
        n_users = int(users_encoded_df["user_idx"].max()) + 1

        user_cat_arr = np.zeros((n_users, 4), dtype=np.int64)
        user_cat_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
            ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
        ].values.astype(np.int64)

        user_dense_arr = np.zeros((n_users, 6), dtype=np.float32)
        user_dense_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
            ["log_total_events", "months_active", "purchase_rate", "cart_rate",
             "log_n_sessions", "avg_purchase_price_scaled"]
        ].values.astype(np.float32)

        self._user_cat   = user_cat_arr
        self._user_dense = user_dense_arr

        # ── Item feature lookups (identical to TwoTowerDataset) ───────────────
        n_items = int(items_encoded_df["item_idx"].max()) + 1

        item_cat_arr = np.zeros((n_items, 5), dtype=np.int64)
        item_cat_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
        ].values.astype(np.int64)

        item_dense_arr = np.zeros((n_items, 3), dtype=np.float32)
        item_dense_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
        ].values.astype(np.float32)

        self._item_cat   = item_cat_arr
        self._item_dense = item_dense_arr

        # ── Hard-negative candidate pools ─────────────────────────────────────
        # col 2 of item_cat_arr = cat_l2_idx; col 4 = price_bucket
        items_dedup = items_encoded_df.drop_duplicates("item_idx")

        cat_l2_to_items: dict[int, np.ndarray] = {}
        for cat_l2, grp in items_dedup.groupby("cat_l2_idx"):
            cat_l2_to_items[int(cat_l2)] = grp["item_idx"].values.astype(np.int64)

        price_bucket_to_items: dict[int, np.ndarray] = {}
        for bucket, grp in items_dedup.groupby("price_bucket"):
            price_bucket_to_items[int(bucket)] = grp["item_idx"].values.astype(np.int64)

        # ── Per-user positive item sets (for exclusion during mining) ─────────
        user_positives: dict[int, set[int]] = {}
        for uid, grp in self.pairs.groupby("user_idx"):
            user_positives[int(uid)] = set(grp["item_idx"].values.tolist())

        # ── Pre-mine all hard negatives ────────────────────────────────────────
        rng = np.random.default_rng(seed)

        n_pairs = len(self.pairs)
        hard_neg_idxs = np.empty((n_pairs, self._n_hard), dtype=np.int64)

        n_cat_l2   = 0   # pairs satisfied by primary (cat_l2) strategy
        n_fallback = 0   # pairs using price_bucket fallback

        pair_user_idxs = self.pairs["user_idx"].values.astype(np.int64)
        pair_item_idxs = self.pairs["item_idx"].values.astype(np.int64)

        for i in range(n_pairs):
            user_idx = int(pair_user_idxs[i])
            item_idx = int(pair_item_idxs[i])

            cat_l2   = int(item_cat_arr[item_idx, 2])   # col 2 = cat_l2_idx
            bucket   = int(item_cat_arr[item_idx, 4])   # col 4 = price_bucket
            pos_set  = user_positives.get(user_idx, set())

            # Choose pool: cat_l2 primary, price_bucket fallback for unknown
            if cat_l2 != 0:
                pool = cat_l2_to_items.get(cat_l2, np.empty(0, dtype=np.int64))
                n_cat_l2 += 1
            else:
                pool = price_bucket_to_items.get(bucket, np.empty(0, dtype=np.int64))
                n_fallback += 1

            # Exclude the user's own positive items from the candidate pool
            if len(pos_set) > 0:
                pool = pool[~np.isin(pool, list(pos_set))]

            if len(pool) == 0:
                # Degenerate: pool is empty after exclusion — fall back to any item
                pool = items_dedup["item_idx"].values.astype(np.int64)
                pool = pool[pool != item_idx]

            replace = len(pool) < self._n_hard
            chosen  = rng.choice(pool, size=self._n_hard, replace=replace)
            hard_neg_idxs[i] = chosen

        self._hard_neg_idxs = torch.from_numpy(hard_neg_idxs)   # (N, 3) int64

        # Stats for __repr__
        self._n_cat_l2_pairs   = n_cat_l2
        self._n_fallback_pairs = n_fallback

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row      = self.pairs.iloc[idx]
        user_idx = int(row.user_idx)
        item_idx = int(row.item_idx)

        return {
            "user_idx":      torch.tensor(user_idx,                    dtype=torch.long),
            "item_idx":      torch.tensor(item_idx,                    dtype=torch.long),
            "user_cat":      torch.tensor(self._user_cat[user_idx],    dtype=torch.long),
            "user_dense":    torch.tensor(self._user_dense[user_idx],  dtype=torch.float32),
            "item_cat":      torch.tensor(self._item_cat[item_idx],    dtype=torch.long),
            "item_dense":    torch.tensor(self._item_dense[item_idx],  dtype=torch.float32),
            "confidence":    torch.tensor(float(row.confidence_score), dtype=torch.float32),
            "hard_neg_idxs": self._hard_neg_idxs[idx],   # (3,) int64
        }

    def __repr__(self) -> str:
        n      = len(self.pairs)
        n_l2   = self._n_cat_l2_pairs
        n_fb   = self._n_fallback_pairs
        ratio  = f"{100 * n_l2 / n:.1f}% cat_l2 / {100 * n_fb / n:.1f}% price_bucket"
        return (
            f"TwoTowerDatasetWithHardNegs("
            f"pairs={n:,}, "
            f"cat_l2={n_l2:,}, "
            f"price_bucket_fallback={n_fb:,}, "
            f"ratio=[{ratio}], "
            f"hard_negs_per_pair={self._n_hard})"
        )


def build_full_item_tensors(
    items_encoded_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense tensors over ALL items for FAISS index construction at inference time.

    Items are sorted by item_idx ascending so that tensor row i corresponds to item_idx i.

    Args:
        items_encoded_df: DataFrame produced by FeatureBuilder.build() for items.

    Returns:
        item_cat_tensor:   LongTensor of shape (n_items, 5)
                           columns: [item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket]
        item_dense_tensor: FloatTensor of shape (n_items, 3)
                           columns: [avg_price_scaled, log_confidence_scaled, purchase_rate_scaled]
    """
    df = items_encoded_df.sort_values("item_idx").reset_index(drop=True)

    item_cat_tensor = torch.tensor(
        df[["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]].values,
        dtype=torch.long,
    )
    item_dense_tensor = torch.tensor(
        df[["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]].values,
        dtype=torch.float32,
    )
    return item_cat_tensor, item_dense_tensor
