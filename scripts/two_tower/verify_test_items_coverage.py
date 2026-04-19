"""Verify overlap between Feb test purchase items and training item_idx set."""

import os
import pathlib
import pickle
import sys

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)

ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "50k"
TEST_GCS_PATH = "gs://recosys-data-bucket/samples/users_sample_50k/test/"

with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)

train_pairs = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")
test_df = pd.read_parquet(TEST_GCS_PATH)

# Items that appeared in training
trained_items = set(train_pairs["item_idx"].unique())

# Ground truth items from Feb purchases
purchases = test_df[test_df["event_type"] == "purchase"]
gt_item_idxs = {
    vocabs["item2idx"][p]
    for p in purchases["product_id"].unique()
    if p in vocabs["item2idx"]
}

in_trained     = gt_item_idxs & trained_items
not_in_trained = gt_item_idxs - trained_items

n_gt = len(gt_item_idxs)
print(f"Ground truth items total         : {n_gt:,}")
if n_gt > 0:
    print(f"Of those, seen in training       : {len(in_trained):,}  "
          f"({len(in_trained) / n_gt:.1%})")
    print(f"Of those, NEVER seen in training : {len(not_in_trained):,}  "
          f"({len(not_in_trained) / n_gt:.1%})")
else:
    print("  (no ground-truth items mapped to item2idx)")
