import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureBuilder:
    """Loads raw interaction and feature parquets from GCS, encodes all
    categorical and continuous columns into model-ready tensors, and saves
    vocabs, scalers, encoded DataFrames, and training pairs to a local
    output directory."""

    def __init__(
        self,
        interactions_path: str,
        item_features_path: str,
        user_features_path: str,
        output_dir: str,
    ) -> None:
        self.interactions_path = interactions_path
        self.item_features_path = item_features_path
        self.user_features_path = user_features_path
        self.output_dir = Path(output_dir)

    def build(self) -> dict:
        # ── Step 1: Load parquets ──────────────────────────────────────────────
        print("Loading parquets...")
        interactions = pd.read_parquet(self.interactions_path)
        print(f"  interactions   : {interactions.shape}")

        item_features = pd.read_parquet(self.item_features_path)
        print(f"  item_features  : {item_features.shape}")

        user_features = pd.read_parquet(self.user_features_path)
        print(f"  user_features  : {user_features.shape}")

        # ── Step 2: Build vocab mappings ──────────────────────────────────────
        print("\nBuilding vocab mappings...")

        user2idx: dict[int, int] = {
            uid: idx for idx, uid in enumerate(interactions["user_id"].unique())
        }
        idx2user: dict[int, int] = {v: k for k, v in user2idx.items()}

        item2idx: dict[int, int] = {
            pid: idx for idx, pid in enumerate(item_features["product_id"].unique())
        }
        idx2item: dict[int, int] = {v: k for k, v in item2idx.items()}

        def _build_str_vocab(values) -> dict[str, int]:
            vocab: dict[str, int] = {"unknown": 0}
            for val in sorted(values):
                if val != "unknown" and val not in vocab:
                    vocab[val] = len(vocab)
            return vocab

        cat_l1_vocab = _build_str_vocab(item_features["cat_l1"].unique())
        cat_l2_vocab = _build_str_vocab(item_features["cat_l2"].unique())
        brand_vocab  = _build_str_vocab(item_features["brand"].unique())
        top_cat_vocab = _build_str_vocab(user_features["top_category"].unique())

        print(f"  users        : {len(user2idx):,}")
        print(f"  items        : {len(item2idx):,}")
        print(f"  cat_l1 vocab : {len(cat_l1_vocab):,}")
        print(f"  cat_l2 vocab : {len(cat_l2_vocab):,}")
        print(f"  brand vocab  : {len(brand_vocab):,}")
        print(f"  top_cat vocab: {len(top_cat_vocab):,}")

        # ── Step 3: Build items_encoded ───────────────────────────────────────
        print("\nEncoding items...")
        items = item_features.copy()

        items["item_idx"]   = items["product_id"].map(item2idx)
        items["cat_l1_idx"] = items["cat_l1"].map(cat_l1_vocab).fillna(0).astype(int)
        items["cat_l2_idx"] = items["cat_l2"].map(cat_l2_vocab).fillna(0).astype(int)
        items["brand_idx"]  = items["brand"].map(brand_vocab).fillna(0).astype(int)

        scaler_avg_price = StandardScaler()
        scaler_log_conf  = StandardScaler()
        scaler_pur_rate  = StandardScaler()

        items["avg_price_scaled"] = scaler_avg_price.fit_transform(
            items[["avg_price"]]
        )
        items["log_confidence_scaled"] = scaler_log_conf.fit_transform(
            np.log1p(items[["item_total_confidence"]])
        )
        items["purchase_rate_scaled"] = scaler_pur_rate.fit_transform(
            items[["item_purchase_rate"]]
        )

        item_scalers = {
            "avg_price":      scaler_avg_price,
            "log_confidence": scaler_log_conf,
            "purchase_rate":  scaler_pur_rate,
        }

        items_encoded = items[[
            "product_id",
            "item_idx",
            "cat_l1_idx",
            "cat_l2_idx",
            "brand_idx",
            "price_bucket",
            "avg_price_scaled",
            "log_confidence_scaled",
            "purchase_rate_scaled",
        ]].reset_index(drop=True)

        print(f"  items_encoded shape: {items_encoded.shape}")

        # ── Step 4: Build users_encoded ───────────────────────────────────────
        print("\nEncoding users...")
        users = user_features.copy()

        users["user_idx"]      = users["user_id"].map(user2idx)
        users["top_cat_idx"]   = users["top_category"].map(top_cat_vocab).fillna(0).astype(int)
        users["preferred_dow"] = users["preferred_dow"] - 1  # remap 1-7 → 0-6

        log_total_events = np.log1p(users[["total_events"]].values)
        log_n_sessions   = np.log1p(users[["n_sessions"]].values)

        dense_matrix = np.hstack([
            log_total_events,
            users[["months_active"]].values,
            users[["purchase_rate"]].values,
            users[["cart_rate"]].values,
            log_n_sessions,
        ])

        user_dense_scaler = StandardScaler()
        dense_scaled = user_dense_scaler.fit_transform(dense_matrix)

        user_price_scaler = StandardScaler()
        price_scaled = user_price_scaler.fit_transform(users[["avg_purchase_price"]])

        user_scalers = {
            "dense": user_dense_scaler,
            "price": user_price_scaler,
        }

        users["log_total_events"]        = dense_scaled[:, 0]
        users["months_active_scaled"]    = dense_scaled[:, 1]
        users["purchase_rate_scaled"]    = dense_scaled[:, 2]
        users["cart_rate_scaled"]        = dense_scaled[:, 3]
        users["log_n_sessions"]          = dense_scaled[:, 4]
        users["avg_purchase_price_scaled"] = price_scaled[:, 0]

        users_encoded = users[[
            "user_id",
            "user_idx",
            "top_cat_idx",
            "peak_hour_bucket",
            "preferred_dow",
            "has_purchase_history",
            "log_total_events",
            "months_active_scaled",
            "purchase_rate_scaled",
            "cart_rate_scaled",
            "log_n_sessions",
            "avg_purchase_price_scaled",
        ]].rename(columns={
            "months_active_scaled": "months_active",
            "purchase_rate_scaled": "purchase_rate",
            "cart_rate_scaled":     "cart_rate",
        }).reset_index(drop=True)

        print(f"  users_encoded shape: {users_encoded.shape}")

        # ── Step 5: Build train_pairs ──────────────────────────────────────────
        print("\nBuilding train_pairs...")
        pairs = interactions[["user_id", "product_id", "confidence_score"]].copy()
        pairs["user_idx"] = pairs["user_id"].map(user2idx)
        pairs["item_idx"] = pairs["product_id"].map(item2idx)

        n_before = len(pairs)
        pairs.dropna(subset=["user_idx", "item_idx"], inplace=True)
        n_after  = len(pairs)
        n_dropped = n_before - n_after
        print(f"  Rows kept   : {n_after:,}")
        print(f"  Rows dropped: {n_dropped:,}")

        train_pairs = pairs[["user_idx", "item_idx", "confidence_score"]].copy()
        train_pairs["user_idx"] = train_pairs["user_idx"].astype(int)
        train_pairs["item_idx"] = train_pairs["item_idx"].astype(int)
        train_pairs = train_pairs.reset_index(drop=True)

        # ── Step 6: Save artifacts ────────────────────────────────────────────
        print(f"\nSaving artifacts to {self.output_dir} ...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        vocabs = {
            "user2idx":     user2idx,
            "idx2user":     idx2user,
            "item2idx":     item2idx,
            "idx2item":     idx2item,
            "cat_l1_vocab": cat_l1_vocab,
            "cat_l2_vocab": cat_l2_vocab,
            "brand_vocab":  brand_vocab,
            "top_cat_vocab": top_cat_vocab,
        }
        with open(self.output_dir / "vocabs.pkl", "wb") as f:
            pickle.dump(vocabs, f)

        with open(self.output_dir / "item_scalers.pkl", "wb") as f:
            pickle.dump(item_scalers, f)

        with open(self.output_dir / "user_scalers.pkl", "wb") as f:
            pickle.dump(user_scalers, f)

        items_encoded.to_parquet(self.output_dir / "items_encoded.parquet", index=False)
        users_encoded.to_parquet(self.output_dir / "users_encoded.parquet", index=False)
        train_pairs.to_parquet(self.output_dir / "train_pairs.parquet", index=False)

        print("[ok] All artifacts saved")

        # ── Step 7: Summary ───────────────────────────────────────────────────
        summary = {
            "n_users":       len(user2idx),
            "n_items":       len(item2idx),
            "n_cat_l1":      len(cat_l1_vocab),
            "n_cat_l2":      len(cat_l2_vocab),
            "n_brands":      len(brand_vocab),
            "n_top_cats":    len(top_cat_vocab),
            "n_train_pairs": len(train_pairs),
            "items_encoded_shape":   items_encoded.shape,
            "items_encoded_columns": list(items_encoded.columns),
            "users_encoded_shape":   users_encoded.shape,
            "users_encoded_columns": list(users_encoded.columns),
            "train_pairs_shape":     train_pairs.shape,
        }

        print("\n── Summary ──────────────────────────────────────────────────────")
        print(f"  n_users      : {summary['n_users']:,}")
        print(f"  n_items      : {summary['n_items']:,}")
        print(f"  n_cat_l1     : {summary['n_cat_l1']}")
        print(f"  n_cat_l2     : {summary['n_cat_l2']}")
        print(f"  n_brands     : {summary['n_brands']:,}")
        print(f"  n_top_cats   : {summary['n_top_cats']}")
        print(f"  n_train_pairs: {summary['n_train_pairs']:,}")
        print(f"\n  items_encoded {items_encoded.shape}")
        print(f"    columns: {list(items_encoded.columns)}")
        print(items_encoded.head(2).to_string(index=False))
        print(f"\n  users_encoded {users_encoded.shape}")
        print(f"    columns: {list(users_encoded.columns)}")
        print(users_encoded.head(2).to_string(index=False))
        print(f"\n  train_pairs {train_pairs.shape}")
        print(train_pairs.head(2).to_string(index=False))

        return summary
