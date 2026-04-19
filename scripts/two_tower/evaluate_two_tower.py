import json
import os
import pathlib
import pickle
import sys

import pandas as pd
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR   = _REPO_ROOT / "artifacts" / "50k"
CHECKPOINT_DIR  = ARTIFACTS_DIR / "checkpoints_v2"
EVAL_EPOCHS     = [5, 10, 15, 20, 25, 30]   # checkpoints to evaluate
TEST_GCS_PATH   = "gs://recosys-data-bucket/samples/users_sample_50k/test/"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
K               = 10

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
        k                = K,
    )

    all_results.append({
        "checkpoint": str(ckpt_path),
        "epoch":      epoch,
        "train_loss": float(checkpoint["loss"]),
        **{key: float(val) for key, val in metrics.items()},
    })

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print(f"{'CHECKPOINT SWEEP SUMMARY':^62}")
print(f"{'=' * 62}")
print(f"  {'Epoch':<8}  {'Train Loss':>10}  {'Recall@10':>10}  {'NDCG@10':>9}  {'Prec@10':>8}")
print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*8}")

best = max(all_results, key=lambda r: r.get(f"recall@{K}", 0)) if all_results else None

for r in all_results:
    marker = "  ← best" if r is best else ""
    print(f"  {r['epoch']:<8}  {r['train_loss']:>10.4f}  "
          f"{r.get(f'recall@{K}', 0):>10.4f}  "
          f"{r.get(f'ndcg@{K}', 0):>9.4f}  "
          f"{r.get(f'precision@{K}', 0):>8.4f}"
          f"{marker}")
print(f"{'=' * 62}")

# ── Save all results ──────────────────────────────────────────────────────────
eval_results_path = ARTIFACTS_DIR / "eval_results.json"
with open(eval_results_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to {eval_results_path}")
if best:
    print(f"Best checkpoint : epoch_{best['epoch']}.pt  "
          f"Recall@{K}={best[f'recall@{K}']:.4f}")
