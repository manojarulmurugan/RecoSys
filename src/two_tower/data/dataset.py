"""PyTorch Dataset for the Two-Tower training loop.

Wraps pre-encoded parquet artifacts into fast numpy lookups so that
DataLoader workers spend no time in pandas during training.
"""

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
