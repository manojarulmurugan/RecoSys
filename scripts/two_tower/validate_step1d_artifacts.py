"""Step 1d: load local artifacts and print validation stats (vocab sizes, ranges, round-trip)."""

import os
import pickle
import pathlib

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)

artifacts_dir = _REPO_ROOT / "artifacts" / "50k"

# Load everything
with open(artifacts_dir / "vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)
with open(artifacts_dir / "item_scalers.pkl", "rb") as f:
    item_scalers = pickle.load(f)
with open(artifacts_dir / "user_scalers.pkl", "rb") as f:
    user_scalers = pickle.load(f)

items_enc   = pd.read_parquet(artifacts_dir / "items_encoded.parquet")
users_enc   = pd.read_parquet(artifacts_dir / "users_encoded.parquet")
train_pairs = pd.read_parquet(artifacts_dir / "train_pairs.parquet")

print("=" * 55)
print("VOCAB SIZES")
print("=" * 55)
print(f"  n_users      : {len(vocabs['user2idx']):>8,}")
print(f"  n_items      : {len(vocabs['item2idx']):>8,}")
print(f"  n_cat_l1     : {len(vocabs['cat_l1_vocab']):>8,}  (expect ~13-15)")
print(f"  n_cat_l2     : {len(vocabs['cat_l2_vocab']):>8,}  (expect ~80-120)")
print(f"  n_brands     : {len(vocabs['brand_vocab']):>8,}  (expect ~2000-4000)")
print(f"  n_top_cats   : {len(vocabs['top_cat_vocab']):>8,}  (expect same as n_cat_l1)")

print("\n" + "=" * 55)
print("TRAINING PAIRS")
print("=" * 55)
print(f"  Total pairs  : {len(train_pairs):>8,}")
print(f"  Unique users : {train_pairs.user_idx.nunique():>8,}")
print(f"  Unique items : {train_pairs.item_idx.nunique():>8,}")
print(f"  Conf score range: {train_pairs.confidence_score.min()} – {train_pairs.confidence_score.max()}")

print("\n" + "=" * 55)
print("SCALED FEATURE RANGES (should be roughly -3 to +3)")
print("=" * 55)
item_scaled_cols = ['avg_price_scaled', 'log_confidence_scaled', 'purchase_rate_scaled']
user_dense_cols  = ['log_total_events', 'months_active', 'purchase_rate',
                    'cart_rate', 'log_n_sessions']
print("Item scaled features:")
print(items_enc[item_scaled_cols].describe().round(3).to_string())
print("\nUser dense features (post-scaling):")
print(users_enc[user_dense_cols].describe().round(3).to_string())

print("\n" + "=" * 55)
print("CATEGORICAL INDEX RANGES")
print("=" * 55)
print(f"  cat_l1_idx   : {items_enc.cat_l1_idx.min()} – {items_enc.cat_l1_idx.max()}  "
      f"(expect 0 – {len(vocabs['cat_l1_vocab'])-1})")
print(f"  cat_l2_idx   : {items_enc.cat_l2_idx.min()} – {items_enc.cat_l2_idx.max()}")
print(f"  brand_idx    : {items_enc.brand_idx.min()} – {items_enc.brand_idx.max()}")
print(f"  top_cat_idx  : {users_enc.top_cat_idx.min()} – {users_enc.top_cat_idx.max()}")
print(f"  peak_hour    : {users_enc.peak_hour_bucket.min()} – {users_enc.peak_hour_bucket.max()}  (expect 0-3)")
print(f"  preferred_dow: {users_enc.preferred_dow.min()} – {users_enc.preferred_dow.max()}  (expect 0-6)")

print("\n" + "=" * 55)
print("ROUND-TRIP ENCODING CHECK")
print("=" * 55)
sample = train_pairs.sample(5, random_state=42)
all_passed = True
for _, row in sample.iterrows():
    u_idx   = int(row.user_idx)
    i_idx   = int(row.item_idx)
    user_id = vocabs['idx2user'][u_idx]
    prod_id = vocabs['idx2item'][i_idx]
    u_back  = vocabs['user2idx'][user_id]
    i_back  = vocabs['item2idx'][prod_id]
    ok = (u_back == u_idx) and (i_back == i_idx)
    if not ok:
        all_passed = False
    mark = "OK" if ok else "FAIL"
    print(f"  user_idx {u_idx} → user_id {user_id} → {u_back}  |  "
          f"item_idx {i_idx} → product_id {prod_id} → {i_back}  {mark}")

print("\n" + "=" * 55)
print("PURCHASE RATE DISTRIBUTION IN items_encoded")
print("(93% should be at the scaled minimum — verify zero-inflated)")
print("=" * 55)
zero_pr = (items_enc.purchase_rate_scaled == items_enc.purchase_rate_scaled.min()).mean()
print(f"  Items at min purchase_rate_scaled: {zero_pr:.1%}  (expect ~93%)")

print("\n" + "=" * 55)
print("HAS_PURCHASE_HISTORY SPLIT IN users_encoded")
print("=" * 55)
phist = users_enc.has_purchase_history.value_counts()
print(f"  has_purchase_history=1 : {phist.get(1, 0):,}  ({phist.get(1,0)/len(users_enc):.1%})")
print(f"  has_purchase_history=0 : {phist.get(0, 0):,}  ({phist.get(0,0)/len(users_enc):.1%})")

print("\n" + "=" * 55)
if all_passed:
    print("Step 1d COMPLETE — all checks passed.")
    print("Proceed to Step 2: PyTorch model implementation.")
else:
    print("Step 1d FAILED — fix round-trip encoding before proceeding.")
print("=" * 55)
