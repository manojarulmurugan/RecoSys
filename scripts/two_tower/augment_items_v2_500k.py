"""Compute two new item features and save items_encoded_v2.parquet.

New columns added on top of items_encoded.parquet:
  price_relative_to_cat_avg_scaled : (avg_price - cat_l2_mean_price) / cat_l2_mean_price,
                                      then StandardScaler.  Captures whether an item is
                                      expensive or cheap relative to its sub-category peers
                                      (per NVIDIA Merlin / T4Rec recommendation).
  product_recency_log_scaled       : log1p(days from item's first appearance to training
                                      cutoff date 2020-02-01).  Captures catalog staleness.

Usage on Colab:
    !python scripts/two_tower/augment_items_v2_500k.py

Outputs:
    artifacts/500k/items_encoded_v2.parquet   — 11 columns (9 original + 2 new)
    artifacts/500k/item_scalers_v2.pkl        — scalers for the two new features
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

ARTIFACTS_DIR      = _REPO_ROOT / "artifacts" / "500k"
ITEM_FEATURES_GCS  = "gs://recosys-data-bucket/features/item_features/"
INTERACTIONS_GCS   = "gs://recosys-data-bucket/samples/users_sample_500k/interactions/"
TRAINING_END_DATE  = "2020-02-01"   # day the test split begins


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
    print("WARNING: No GCP credentials found — GCS reads may fail.")


_ensure_gcp_credentials()


# ── Load base artifacts ────────────────────────────────────────────────────────

print("Loading items_encoded.parquet...")
items_enc = pd.read_parquet(ARTIFACTS_DIR / "items_encoded.parquet")
print(f"  shape: {items_enc.shape}")
print(f"  columns: {list(items_enc.columns)}")

# ── Load raw item_features from GCS (for avg_price) ──────────────────────────

print("\nLoading item_features from GCS (for raw avg_price)...")
item_features = pd.read_parquet(ITEM_FEATURES_GCS)
print(f"  shape: {item_features.shape}")
print(f"  columns: {list(item_features.columns)}")

# ── Feature 1: price_relative_to_cat_avg ─────────────────────────────────────
# (avg_price - cat_l2_group_mean) / cat_l2_group_mean
# We use cat_l2_idx from items_enc and raw avg_price from item_features.

print("\nComputing price_relative_to_cat_avg...")

# Merge raw avg_price into items_enc (align by product_id)
merged = items_enc[["product_id", "item_idx", "cat_l2_idx"]].merge(
    item_features[["product_id", "avg_price"]].drop_duplicates("product_id"),
    on="product_id",
    how="left",
)
merged["avg_price"] = merged["avg_price"].fillna(0.0).clip(lower=0.0)

# Group mean by cat_l2_idx
cat_mean = merged.groupby("cat_l2_idx")["avg_price"].transform("mean")
denom    = cat_mean.replace(0.0, 1.0)   # avoid divide-by-zero for empty cats
merged["price_relative_to_cat_avg"] = (merged["avg_price"] - cat_mean) / denom

scaler_price_rel = StandardScaler()
price_rel_scaled = scaler_price_rel.fit_transform(
    merged[["price_relative_to_cat_avg"]]
).ravel().astype(np.float32)
print(f"  min={price_rel_scaled.min():.3f}  max={price_rel_scaled.max():.3f}"
      f"  mean={price_rel_scaled.mean():.3f}")

# ── Feature 2: product_recency_log ────────────────────────────────────────────
# log1p(days from item's first appearance in training interactions to 2020-02-01).
# Items seen later in the dataset (e.g. new products) get lower recency values.

print("\nLoading interactions from GCS (for first-seen timestamps)...")
# The aggregated interactions parquet has first_interaction / last_interaction
# per (user_id, product_id) pair rather than individual event_time rows.
# We take the minimum first_interaction per product_id as its catalog debut date.
interactions = pd.read_parquet(
    INTERACTIONS_GCS,
    columns=["product_id", "first_interaction"],
)
print(f"  shape: {interactions.shape}")
print(f"  columns: {list(interactions.columns)}")

print("Computing product_recency_log...")
interactions["first_interaction"] = pd.to_datetime(interactions["first_interaction"], utc=True)
cutoff_utc = pd.Timestamp(TRAINING_END_DATE, tz="UTC")

first_seen = (
    interactions.groupby("product_id")["first_interaction"]
    .min()
    .reset_index()
    .rename(columns={"first_interaction": "first_seen"})
)
first_seen["days_to_cutoff"] = (
    (cutoff_utc - first_seen["first_seen"]).dt.total_seconds() / 86400.0
).clip(lower=0.0)
first_seen["product_recency_log"] = np.log1p(first_seen["days_to_cutoff"])

# Merge into merged DataFrame (already aligned with items_enc row order)
merged = merged.merge(
    first_seen[["product_id", "product_recency_log"]],
    on="product_id",
    how="left",
)
# Items not in interactions (shouldn't happen, but guard)
merged["product_recency_log"] = merged["product_recency_log"].fillna(0.0)

scaler_recency = StandardScaler()
recency_scaled = scaler_recency.fit_transform(
    merged[["product_recency_log"]]
).ravel().astype(np.float32)
print(f"  min={recency_scaled.min():.3f}  max={recency_scaled.max():.3f}"
      f"  mean={recency_scaled.mean():.3f}")

# ── Sanity: merged rows must be identical to items_enc rows ───────────────────
assert len(merged) == len(items_enc), (
    f"Row count mismatch after merge: {len(merged)} vs {len(items_enc)}"
)
assert (merged["item_idx"].values == items_enc["item_idx"].values).all(), (
    "item_idx alignment broken after merge"
)

# ── Assemble items_encoded_v2 ─────────────────────────────────────────────────
items_v2 = items_enc.copy()
items_v2["price_relative_to_cat_avg_scaled"] = price_rel_scaled
items_v2["product_recency_log_scaled"]       = recency_scaled

print(f"\nitems_encoded_v2 shape : {items_v2.shape}")
print(f"  columns: {list(items_v2.columns)}")
assert not items_v2.isna().any().any(), "NaN found in items_encoded_v2!"

out_path = ARTIFACTS_DIR / "items_encoded_v2.parquet"
items_v2.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")

scalers_v2 = {
    "price_relative_to_cat_avg": scaler_price_rel,
    "product_recency_log":       scaler_recency,
}
scalers_path = ARTIFACTS_DIR / "item_scalers_v2.pkl"
with open(scalers_path, "wb") as f:
    pickle.dump(scalers_v2, f)
print(f"Saved → {scalers_path}")
print("\nDone. Upload items_encoded_v2.parquet to GCS or keep locally for training.")
