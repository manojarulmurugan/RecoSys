import argparse
import json
import os
import pathlib
import pickle
import sys
from collections import Counter

import pandas as pd
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Optional service-account JSON for reading test parquet from GCS (gcloud ADC also works)
_gac = _REPO_ROOT / "secrets" / "recosys-service-account.json"
if _gac.is_file():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_gac)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Two-Tower checkpoints (Recall / NDCG) against GCS test split."
    )
    p.add_argument(
        "--artifacts-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "artifacts" / "500k",
        help="Directory with vocabs.pkl, *encoded.parquet, train_pairs.parquet (default: repo artifacts/500k)",
    )
    p.add_argument(
        "--checkpoint-subdir",
        type=str,
        default="checkpoints",
        help="Subfolder of artifacts-dir with epoch_*.pt files (default: checkpoints)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[5, 30],
        help="Which epoch_*.pt files to run (default: 5 30)",
    )
    p.add_argument(
        "--test-gcs-path",
        type=str,
        default="gs://recosys-data-bucket/samples/users_sample_500k/test/",
        help="Parquet test split on GCS (default: 500k sample)",
    )
    return p.parse_args()


# ── CONFIG (defaults suitable for 500k local checkpoints under artifacts/500k/checkpoints/) ──
_args = _parse_args()
ARTIFACTS_DIR: pathlib.Path = _args.artifacts_dir.resolve()
CHECKPOINT_DIR = ARTIFACTS_DIR / _args.checkpoint_subdir
EVAL_EPOCHS: list[int] = list(_args.epochs)
TEST_GCS_PATH: str = _args.test_gcs_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 10

# ── Load shared artifacts (once) ──────────────────────────────────────────────
print("Loading artifacts...")

with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)

items_enc   = pd.read_parquet(ARTIFACTS_DIR / "items_encoded.parquet")
users_enc   = pd.read_parquet(ARTIFACTS_DIR / "users_encoded.parquet")
train_pairs = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")

print(f"  items_encoded  : {items_enc.shape}")
print(f"  users_encoded  : {users_enc.shape}")
print(f"  train_pairs    : {train_pairs.shape}")

print(f"Loading test split from {TEST_GCS_PATH} ...")
test_df = pd.read_parquet(TEST_GCS_PATH)
print(f"  test_df        : {test_df.shape}")
print(f"  device         : {DEVICE}")

# ── Build model architecture once (weights reloaded per checkpoint) ───────────
from src.two_tower.models.two_tower import UserTower, ItemTower, TwoTowerModel
from src.two_tower.evaluation.evaluate import evaluate

user_tower = UserTower(
    n_users    = len(vocabs["user2idx"]),
    n_top_cats = len(vocabs["top_cat_vocab"]),
)
item_tower = ItemTower(
    n_items  = len(vocabs["item2idx"]),
    n_cat_l1 = len(vocabs["cat_l1_vocab"]),
    n_cat_l2 = len(vocabs["cat_l2_vocab"]),
    n_brands = len(vocabs["brand_vocab"]),
)
model = TwoTowerModel(user_tower, item_tower, temperature=0.05)
model.to(DEVICE)

# ── Popularity Collapse Diagnostic ───────────────────────────────────────────

def check_popularity_collapse(
    all_recommendations: dict,
    k: int = 10,
) -> None:
    """Print a popularity-collapse diagnostic for a set of per-user top-K lists.

    Flattens all recommendation lists, counts how often each item appears
    across all users, then reports what share of total recommendation slots
    is accounted for by the top-1 / 5 / 10 / 50 / 100 most-recommended items.

    A healthy retrieval model should spread recommendations across hundreds or
    thousands of distinct items.  If the top-50 items absorb more than ~20% of
    all slots across tens of thousands of users, the model is collapsing onto
    a small set of popular items regardless of user preferences.

    Args:
        all_recommendations: dict mapping user_idx -> list of top-K item_idxs.
        k:                   List length per user (used only for display).
    """
    flat        = [item for recs in all_recommendations.values() for item in recs[:k]]
    total_slots = len(flat)
    n_users     = len(all_recommendations)
    n_unique    = len(set(flat))

    if total_slots == 0:
        print("  [check_popularity_collapse] No recommendations to analyse.")
        return

    most_common = Counter(flat).most_common()   # sorted by count descending

    thresholds = [1, 5, 10, 50, 100]

    print(f"\n{'=' * 58}")
    print(f"{'POPULARITY COLLAPSE DIAGNOSTIC':^58}")
    print(f"{'=' * 58}")
    print(f"  Users evaluated              : {n_users:>10,}")
    print(f"  Total recommendation slots   : {total_slots:>10,}  (users x {k})")
    print(f"  Unique items ever recommended: {n_unique:>10,}")
    print()
    print(f"  {'Top-N items':<20}  {'Cumulative slots':>16}  {'% of total':>10}")
    print(f"  {'-'*20}  {'-'*16}  {'-'*10}")

    cumulative = 0
    prev       = 0
    top50_pct  = 0.0
    for t in thresholds:
        cumulative += sum(count for _, count in most_common[prev:t])
        pct         = 100.0 * cumulative / total_slots
        print(f"  Top-{t:<16}  {cumulative:>16,}  {pct:>9.2f}%")
        if t == 50:
            top50_pct = pct
        prev = t

    print(f"{'=' * 58}")
    if top50_pct > 20.0:
        print(f"  COLLAPSE WARNING : top-50 items cover {top50_pct:.1f}% of slots "
              f"(threshold > 20%)")
    else:
        print(f"  OK               : top-50 items cover {top50_pct:.1f}% of slots "
              f"(threshold > 20%)")
    print(f"{'=' * 58}\n")


# ── Evaluate each checkpoint ──────────────────────────────────────────────────
all_results: list[dict] = []

for epoch in EVAL_EPOCHS:
    ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
    if not ckpt_path.exists():
        print(f"\n[SKIP] epoch_{epoch}.pt not found — skipping")
        continue

    print(f"\n{'=' * 55}")
    print(f"Evaluating checkpoint: epoch_{epoch}.pt")
    print(f"{'=' * 55}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  train loss at epoch {epoch}: {checkpoint['loss']:.4f}")

    metrics = evaluate(
        model            = model,
        items_encoded_df = items_enc,
        users_encoded_df = users_enc,
        test_df          = test_df,
        train_pairs_df   = train_pairs,
        vocabs           = vocabs,
        device           = torch.device(DEVICE),
    )

    # Extract per-user recommendations before the float-conversion below
    all_recs = metrics.pop("recommendations", {})
    check_popularity_collapse(all_recs, k=K)

    all_results.append({
        "checkpoint": str(ckpt_path),
        "epoch":      epoch,
        "train_loss": float(checkpoint["loss"]),
        **{key: float(val) for key, val in metrics.items()},
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 74}")
print(f"{'CHECKPOINT SWEEP SUMMARY':^74}")
print(f"{'=' * 74}")
print(f"  {'Epoch':<8}  {'Train Loss':>10}  {'Recall@10':>10}  {'NDCG@10':>9}"
      f"  {'Recall@20':>10}  {'NDCG@20':>9}")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*9}")

best = max(all_results, key=lambda r: r.get("recall_10", 0)) if all_results else None

for r in all_results:
    marker = "  ← best Recall@10" if r is best else ""
    print(f"  {r['epoch']:<8}  {r['train_loss']:>10.4f}  "
          f"{r.get('recall_10', 0):>10.4f}  "
          f"{r.get('ndcg_10', 0):>9.4f}  "
          f"{r.get('recall_20', 0):>10.4f}  "
          f"{r.get('ndcg_20', 0):>9.4f}"
          f"{marker}")
print(f"{'=' * 74}")

# ── Save all results ──────────────────────────────────────────────────────────
eval_results_path = ARTIFACTS_DIR / "eval_results.json"
with open(eval_results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to {eval_results_path}")
if best:
    print(f"Best checkpoint : epoch_{best['epoch']}.pt  "
          f"Recall@10={best['recall_10']:.4f}  "
          f"Recall@20={best['recall_20']:.4f}")
