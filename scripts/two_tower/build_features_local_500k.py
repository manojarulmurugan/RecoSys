import os
import sys
import pathlib

# Same key location as scripts/two_tower/build_item_features.py (repo secrets/, not ~/secrets/)
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)
import pickle

sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from src.data.feature_builder import FeatureBuilder

# ── Run ────────────────────────────────────────────────────────────────────────
builder = FeatureBuilder(
    interactions_path  = "gs://recosys-data-bucket/samples/users_sample_500k/interactions/",
    item_features_path = "gs://recosys-data-bucket/features/item_features/",
    user_features_path = "gs://recosys-data-bucket/features/user_features_500k/",
    output_dir         = str(_REPO_ROOT / "artifacts" / "500k"),
)
summary = builder.build()

# ── Validation checks ──────────────────────────────────────────────────────────
ARTIFACTS = _REPO_ROOT / "artifacts" / "500k"

passed = 0
failed = 0

def check(n: int, label: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] Check {n:>2}: {label}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Check {n:>2}: {label} — {e}")
        failed += 1

print("\n── Validation ────────────────────────────────────────────────────────────")

# Pre-load artifacts once so individual checks don't re-read from disk repeatedly
_vocabs        = None
_item_scalers  = None
_user_scalers  = None
_items_enc     = None
_users_enc     = None
_train_pairs   = None

def _load_all():
    global _vocabs, _item_scalers, _user_scalers, _items_enc, _users_enc, _train_pairs
    with open(ARTIFACTS / "vocabs.pkl",        "rb") as f: _vocabs       = pickle.load(f)
    with open(ARTIFACTS / "item_scalers.pkl",  "rb") as f: _item_scalers = pickle.load(f)
    with open(ARTIFACTS / "user_scalers.pkl",  "rb") as f: _user_scalers = pickle.load(f)
    _items_enc   = pd.read_parquet(ARTIFACTS / "items_encoded.parquet")
    _users_enc   = pd.read_parquet(ARTIFACTS / "users_encoded.parquet")
    _train_pairs = pd.read_parquet(ARTIFACTS / "train_pairs.parquet")

try:
    _load_all()
except Exception as e:
    print(f"  [FAIL] Could not load artifacts — {e}")
    print("  Cannot run further checks.")
    sys.exit(1)

# Check 1
VOCAB_KEYS = {"user2idx", "idx2user", "item2idx", "idx2item",
              "cat_l1_vocab", "cat_l2_vocab", "brand_vocab", "top_cat_vocab"}
def _c1():
    assert (ARTIFACTS / "vocabs.pkl").exists(), "vocabs.pkl not found"
    assert set(_vocabs.keys()) == VOCAB_KEYS, \
        f"unexpected keys: {set(_vocabs.keys()) ^ VOCAB_KEYS}"
check(1, "vocabs.pkl exists with correct keys", _c1)

# Check 2
def _c2():
    assert (ARTIFACTS / "item_scalers.pkl").exists(), "item_scalers.pkl not found"
    assert set(_item_scalers.keys()) == {"avg_price", "log_confidence", "purchase_rate"}, \
        f"unexpected keys: {set(_item_scalers.keys())}"
check(2, "item_scalers.pkl exists with correct keys", _c2)

# Check 3
def _c3():
    assert (ARTIFACTS / "user_scalers.pkl").exists(), "user_scalers.pkl not found"
    assert set(_user_scalers.keys()) == {"dense", "price"}, \
        f"unexpected keys: {set(_user_scalers.keys())}"
check(3, "user_scalers.pkl exists with correct keys", _c3)

# Check 4
ITEM_COLS = {"product_id", "item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx",
             "price_bucket", "avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"}
def _c4():
    actual = set(_items_enc.columns)
    assert actual == ITEM_COLS, f"column mismatch: {actual ^ ITEM_COLS}"
check(4, "items_encoded.parquet has correct columns", _c4)

# Check 5
USER_COLS = {"user_id", "user_idx", "top_cat_idx", "peak_hour_bucket", "preferred_dow",
             "has_purchase_history", "log_total_events", "months_active", "purchase_rate",
             "cart_rate", "log_n_sessions", "avg_purchase_price_scaled"}
def _c5():
    actual = set(_users_enc.columns)
    assert actual == USER_COLS, f"column mismatch: {actual ^ USER_COLS}"
check(5, "users_encoded.parquet has correct columns", _c5)

# Check 6
def _c6():
    actual = set(_train_pairs.columns)
    expected = {"user_idx", "item_idx", "confidence_score"}
    assert actual == expected, f"column mismatch: {actual ^ expected}"
check(6, "train_pairs.parquet has correct columns", _c6)

# Check 7
def _c7():
    assert not _items_enc.isna().any().any(), \
        f"NaN found in columns: {list(_items_enc.columns[_items_enc.isna().any()])}"
check(7, "items_encoded has no NaN values", _c7)

# Check 8
def _c8():
    assert not _users_enc.isna().any().any(), \
        f"NaN found in columns: {list(_users_enc.columns[_users_enc.isna().any()])}"
check(8, "users_encoded has no NaN values", _c8)

# Check 9
def _c9():
    n_items = len(_vocabs["item2idx"])
    bad = _train_pairs["item_idx"]
    assert bad.ge(0).all() and bad.lt(n_items).all(), \
        f"item_idx out of range [0, {n_items}): min={bad.min()}, max={bad.max()}"
check(9, "train_pairs item_idx values in valid range", _c9)

# Check 10
def _c10():
    n_users = len(_vocabs["user2idx"])
    bad = _train_pairs["user_idx"]
    assert bad.ge(0).all() and bad.lt(n_users).all(), \
        f"user_idx out of range [0, {n_users}): min={bad.min()}, max={bad.max()}"
check(10, "train_pairs user_idx values in valid range", _c10)

# Check 11
def _c11():
    n = len(_train_pairs)
    assert 5_000_000 <= n <= 15_000_000, \
        f"train_pairs row count {n:,} not in [5_000_000, 15_000_000]"
check(11, "train_pairs row count in expected range (500k-user sample)", _c11)

# Check 12
def _c12():
    assert _vocabs["cat_l1_vocab"]["unknown"] == 0, "cat_l1_vocab['unknown'] != 0"
    assert _vocabs["brand_vocab"]["unknown"]  == 0, "brand_vocab['unknown'] != 0"
check(12, "cat_l1_vocab and brand_vocab reserve index 0 for 'unknown'", _c12)

# Check 13
def _c13():
    n_vocab  = len(_vocabs["item2idx"])
    n_unique = _items_enc["item_idx"].nunique()
    assert n_unique == n_vocab, \
        f"items_encoded has {n_unique} unique item_idx but vocab has {n_vocab}"
check(13, "items_encoded.item_idx matches vocab size", _c13)

# Check 14
def _c14():
    n_unique = _users_enc["user_idx"].nunique()
    assert 400_000 < n_unique < 500_000, \
        f"users_encoded unique user count {n_unique:,} not in (400_000, 500_000)"
check(14, "users_encoded user count in expected range (500k sample)", _c14)

# ── Final summary ──────────────────────────────────────────────────────────────
total = passed + failed
print(f"\n{passed}/{total} checks passed")
if failed:
    print("Review failures above before proceeding.")
else:
    print("Step 1c complete. Ready for Step 1d validation.")
