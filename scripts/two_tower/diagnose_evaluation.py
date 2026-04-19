import os, sys, pathlib, pickle
import pandas as pd
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _ensure_gcp_credentials() -> None:
    """Match evaluate_two_tower.py: prefer repo secrets, then ~/secrets, else ADC.

    Only set GOOGLE_APPLICATION_CREDENTIALS when the file exists; a missing
    path breaks PyArrow GCS (it will not fall back to gcloud ADC).
    """
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing:
        p = pathlib.Path(existing).expanduser()
        if p.is_file():
            return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        pathlib.Path(os.path.expanduser("~/secrets/recosys-service-account.json")),
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
            return


_ensure_gcp_credentials()

from src.two_tower.models.two_tower import UserTower, ItemTower, TwoTowerModel
from src.two_tower.data.dataset import build_full_item_tensors

ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "50k"
CHECKPOINT    = _REPO_ROOT / "artifacts" / "50k" / "checkpoints" / "epoch_10.pt"
TEST_GCS_PATH = "gs://recosys-data-bucket/samples/users_sample_50k/test/"

# ── Load everything ────────────────────────────────────────────────────────────
with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)

items_enc   = pd.read_parquet(ARTIFACTS_DIR / "items_encoded.parquet")
users_enc   = pd.read_parquet(ARTIFACTS_DIR / "users_encoded.parquet")
train_pairs = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")
test_df     = pd.read_parquet(TEST_GCS_PATH)

user_tower = UserTower(n_users=len(vocabs['user2idx']),
                       n_top_cats=len(vocabs['top_cat_vocab']))
item_tower = ItemTower(n_items=len(vocabs['item2idx']),
                       n_cat_l1=len(vocabs['cat_l1_vocab']),
                       n_cat_l2=len(vocabs['cat_l2_vocab']),
                       n_brands=len(vocabs['brand_vocab']))
model = TwoTowerModel(user_tower, item_tower, temperature=0.07)
ckpt  = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(ckpt['model_state'])
model.eval()

print("=" * 60)
print("CHECK 1 — Ground truth quality")
print("=" * 60)
purchases = test_df[test_df.event_type == 'purchase']
print(f"  Total Feb purchases          : {len(purchases):,}")
print(f"  Unique users with purchases  : {purchases.user_id.nunique():,}")
print(f"  Users also in training       : {purchases.user_id.isin(vocabs['user2idx']).sum():,}")

known_purchases = purchases[purchases.user_id.isin(vocabs['user2idx'])]
gt_per_user = known_purchases.groupby('user_id')['product_id'].apply(set)
print(f"  Avg purchases per eval user  : {gt_per_user.apply(len).mean():.2f}")
print(f"  Users with 1 purchase        : {(gt_per_user.apply(len)==1).sum():,}")
print(f"  Users with 2+ purchases      : {(gt_per_user.apply(len)>=2).sum():,}")

print()
print("=" * 60)
print("CHECK 2 — Are ground truth items in the item index?")
print("=" * 60)
all_gt_items = set(known_purchases.product_id.unique())
items_in_index = set(vocabs['item2idx'].keys())
covered = all_gt_items & items_in_index
print(f"  Unique items purchased in Feb : {len(all_gt_items):,}")
print(f"  Of those in item index        : {len(covered):,}  "
      f"({len(covered)/len(all_gt_items):.1%})")
print(f"  Of those NOT in index         : {len(all_gt_items - items_in_index):,}")
if len(all_gt_items - items_in_index) > 0:
    print("  !! Items purchased in Feb but absent from training vocab.")
    print("     These can NEVER be retrieved — hard ceiling on Recall@10.")

print()
print("=" * 60)
print("CHECK 3 — Seen-items filter: are we over-filtering?")
print("=" * 60)
seen_per_user = train_pairs.groupby('user_idx')['item_idx'].apply(set)
sample_users  = list(vocabs['user2idx'][uid]
                     for uid in gt_per_user.index[:5]
                     if uid in vocabs['user2idx'])
for u_idx in sample_users[:3]:
    n_seen = len(seen_per_user.get(u_idx, set()))
    print(f"  user_idx {u_idx}: {n_seen} training items filtered out of 284,523")

print()
print("=" * 60)
print("CHECK 4 — Embedding sanity on 5 random eval users")
print("=" * 60)
import faiss

item_cat_t, item_dense_t = build_full_item_tensors(items_enc)
all_embs = []
with torch.no_grad():
    for i in range(0, len(item_cat_t), 512):
        emb = model.get_item_embeddings(
            item_cat_t[i:i+512], item_dense_t[i:i+512]
        ).numpy()
        all_embs.append(emb)
item_embs = np.vstack(all_embs).astype('float32')
faiss.normalize_L2(item_embs)
index = faiss.IndexFlatIP(64)
index.add(item_embs)

user_cat_cols   = ['top_cat_idx','peak_hour_bucket','preferred_dow','has_purchase_history']
user_dense_cols = ['log_total_events','months_active','purchase_rate',
                   'cart_rate','log_n_sessions','avg_purchase_price_scaled']

users_enc_indexed = users_enc.set_index('user_idx')

print(f"  {'user_idx':>10}  {'gt_item_rank':>14}  {'gt_in_top10':>12}  "
      f"{'gt_in_top100':>13}  {'gt_in_top1000':>14}")
print("  " + "-"*70)

for uid in vocabs['user2idx']:
    u_idx = vocabs['user2idx'][uid]
    if u_idx not in gt_per_user.reindex(
            [uid], fill_value=set()).index: continue
    if uid not in gt_per_user.index: continue

    gt_products = gt_per_user[uid]
    gt_item_idxs = {vocabs['item2idx'][p]
                    for p in gt_products if p in vocabs['item2idx']}
    if not gt_item_idxs: continue
    if u_idx not in users_enc_indexed.index: continue

    row = users_enc_indexed.loc[u_idx]
    u_cat   = torch.tensor([[row[c] for c in user_cat_cols]], dtype=torch.long)
    u_dense = torch.tensor([[row[c] for c in user_dense_cols]], dtype=torch.float32)
    u_idx_t = torch.tensor([u_idx], dtype=torch.long)

    with torch.no_grad():
        u_emb = model.get_user_embedding(u_idx_t, u_cat, u_dense).numpy().astype('float32')
    faiss.normalize_L2(u_emb)

    D, I = index.search(u_emb, 1000)
    retrieved_item_idxs = I[0].tolist()

    first_hit_rank = None
    for rank, item_pos in enumerate(retrieved_item_idxs):
        if item_pos in gt_item_idxs:
            first_hit_rank = rank + 1
            break

    in_top10   = first_hit_rank is not None and first_hit_rank <= 10
    in_top100  = first_hit_rank is not None and first_hit_rank <= 100
    in_top1000 = first_hit_rank is not None and first_hit_rank <= 1000

    print(f"  {u_idx:>10}  {str(first_hit_rank) if first_hit_rank else '>1000':>14}  "
          f"{'YES' if in_top10 else 'no':>12}  "
          f"{'YES' if in_top100 else 'no':>13}  "
          f"{'YES' if in_top1000 else 'no':>14}")

    # only show 5 users
    if sum(1 for _ in [None]) >= 5: break

print()
print("=" * 60)
print("CHECK 5 — Score distribution for one user")
print("=" * 60)
uid = list(gt_per_user.index)[0]
u_idx = vocabs['user2idx'][uid]
row   = users_enc_indexed.loc[u_idx]
u_cat   = torch.tensor([[row[c] for c in user_cat_cols]], dtype=torch.long)
u_dense = torch.tensor([[row[c] for c in user_dense_cols]], dtype=torch.float32)
u_idx_t = torch.tensor([u_idx], dtype=torch.long)

with torch.no_grad():
    u_emb = model.get_user_embedding(u_idx_t, u_cat, u_dense).numpy().astype('float32')
faiss.normalize_L2(u_emb)

D_all, _ = index.search(u_emb, 284523)
scores = D_all[0]
print(f"  Score distribution across all 284,523 items:")
print(f"    max score  : {scores.max():.4f}")
print(f"    top-10 avg : {scores[:10].mean():.4f}")
print(f"    top-100 avg: {scores[:100].mean():.4f}")
print(f"    median     : {np.median(scores):.4f}")
print(f"    min score  : {scores.min():.4f}")
print(f"    std dev    : {scores.std():.4f}")
print()
gt_products = gt_per_user[uid]
gt_item_idxs = [vocabs['item2idx'][p] for p in gt_products if p in vocabs['item2idx']]
if gt_item_idxs:
    gt_scores = [float(D_all[0][i]) for i in gt_item_idxs
                 if i < len(D_all[0])]
    print(f"  Ground truth item scores for this user: {[round(s,4) for s in gt_scores]}")
    print(f"  Compared to top-10 avg: {scores[:10].mean():.4f}")
