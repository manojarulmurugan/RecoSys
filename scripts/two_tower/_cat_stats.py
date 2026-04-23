"""Print cat_l1 / cat_l2 item-count statistics for the 500k artifact set."""
import pathlib
import pickle

import numpy as np
import pandas as pd

ARTIFACTS = pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / "500k"
SEP  = "=" * 72
THIN = "-" * 72


def main() -> None:
    items = pd.read_parquet(ARTIFACTS / "items_encoded.parquet")
    with open(ARTIFACTS / "vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)

    l1_inv = {v: k for k, v in vocabs["cat_l1_vocab"].items()}
    l2_inv = {v: k for k, v in vocabs["cat_l2_vocab"].items()}

    items_dedup = items.drop_duplicates("item_idx")
    n_catalog   = len(items_dedup)

    # ── CAT_L1 ────────────────────────────────────────────────────────────────
    l1 = (
        items_dedup.groupby("cat_l1_idx")["item_idx"].nunique()
        .rename("item_count").reset_index()
        .sort_values("item_count", ascending=False)
    )
    l1["pct"]   = 100 * l1["item_count"] / n_catalog
    l1["label"] = l1["cat_l1_idx"].map(l1_inv)

    print()
    print(SEP)
    print(f"  CAT_L1  —  {len(l1)} top-level categories  |  {n_catalog:,} catalog items")
    print(SEP)
    print(f"  {'label':<42}  {'idx':>4}  {'items':>8}  {'% catalog':>10}")
    print(f"  {THIN}")
    for _, row in l1.iterrows():
        print(f"  {str(row['label']):<42}  {int(row['cat_l1_idx']):>4}"
              f"  {int(row['item_count']):>8,}  {row['pct']:>9.2f}%")

    vals = l1["item_count"].values
    print()
    print(f"  Summary stats across {len(l1)} cat_l1 bins:")
    print(f"  {'mean':<10} {np.mean(vals):>10,.1f}")
    print(f"  {'median':<10} {np.median(vals):>10,.0f}")
    print(f"  {'min':<10} {np.min(vals):>10,}")
    print(f"  {'max':<10} {np.max(vals):>10,}")
    print(f"  {'p10':<10} {np.percentile(vals, 10):>10,.0f}")
    print(f"  {'p90':<10} {np.percentile(vals, 90):>10,.0f}")

    # ── CAT_L2 ────────────────────────────────────────────────────────────────
    l2 = (
        items_dedup.groupby("cat_l2_idx")["item_idx"].nunique()
        .rename("item_count").reset_index()
        .sort_values("item_count", ascending=False)
    )
    l2["pct"]   = 100 * l2["item_count"] / n_catalog
    l2["label"] = l2["cat_l2_idx"].map(l2_inv)

    print()
    print(SEP)
    print(f"  CAT_L2  —  {len(l2)} sub-categories  |  {n_catalog:,} catalog items")
    print(SEP)

    hdr = f"  {'label':<52}  {'idx':>4}  {'items':>8}  {'% catalog':>10}"
    print(f"  TOP 20 by item count")
    print(hdr)
    print(f"  {THIN}")
    for _, row in l2.head(20).iterrows():
        print(f"  {str(row['label']):<52}  {int(row['cat_l2_idx']):>4}"
              f"  {int(row['item_count']):>8,}  {row['pct']:>9.2f}%")

    print()
    print(f"  BOTTOM 20 by item count  (smallest pools)")
    print(hdr)
    print(f"  {THIN}")
    for _, row in l2.tail(20).sort_values("item_count").iterrows():
        print(f"  {str(row['label']):<52}  {int(row['cat_l2_idx']):>4}"
              f"  {int(row['item_count']):>8,}  {row['pct']:>9.4f}%")

    vals2 = l2["item_count"].values
    print()
    print(f"  Summary stats across {len(l2)} cat_l2 bins:")
    print(f"  {'mean':<10} {np.mean(vals2):>10,.1f}")
    print(f"  {'median':<10} {np.median(vals2):>10,.0f}")
    print(f"  {'min':<10} {np.min(vals2):>10,}")
    print(f"  {'max':<10} {np.max(vals2):>10,}")
    print(f"  {'p10':<10} {np.percentile(vals2, 10):>10,.0f}")
    print(f"  {'p25':<10} {np.percentile(vals2, 25):>10,.0f}")
    print(f"  {'p75':<10} {np.percentile(vals2, 75):>10,.0f}")
    print(f"  {'p90':<10} {np.percentile(vals2, 90):>10,.0f}")
    print(f"  {'p99':<10} {np.percentile(vals2, 99):>10,.0f}")

    n_under50 = int((vals2 < 50).sum())
    n_under10 = int((vals2 < 10).sum())
    print(f"\n  bins with < 50 items : {n_under50}  ({100*n_under50/len(l2):.1f}%)")
    print(f"  bins with < 10 items : {n_under10}  ({100*n_under10/len(l2):.1f}%)")
    print()
    print(SEP)


if __name__ == "__main__":
    main()
