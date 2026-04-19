import os
import sys
import pathlib
import pickle

import pandas as pd
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)

from src.two_tower.data.dataset import TwoTowerDataset
from src.two_tower.models.two_tower import UserTower, ItemTower, TwoTowerModel
from src.two_tower.training.train import in_batch_loss

ARTIFACTS = _REPO_ROOT / "artifacts" / "50k"

# Load artifacts
with open(ARTIFACTS / "vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)

items_enc   = pd.read_parquet(ARTIFACTS / "items_encoded.parquet")
users_enc   = pd.read_parquet(ARTIFACTS / "users_encoded.parquet")
train_pairs = pd.read_parquet(ARTIFACTS / "train_pairs.parquet")

# Build dataset and grab one batch
dataset = TwoTowerDataset(train_pairs, users_enc, items_enc)
loader  = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
batch   = next(iter(loader))

print("Batch keys:", list(batch.keys()))
print("user_idx shape  :", batch["user_idx"].shape)
print("user_cat shape  :", batch["user_cat"].shape)
print("user_dense shape:", batch["user_dense"].shape)
print("item_cat shape  :", batch["item_cat"].shape)
print("item_dense shape:", batch["item_dense"].shape)

# Build model
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
# temperature=1.0 for the sanity check: the theoretical random-init loss is log(B)
# only when temperature=1.0. Lower temperatures (e.g. 0.05 used in training) scale
# scores by 1/τ before the softmax, amplifying noise and pushing the loss above log(B).
model = TwoTowerModel(user_tower, item_tower, temperature=1.0)
model.model_summary()

# Forward pass
model.eval()
with torch.no_grad():
    user_emb, item_emb, scores = model(
        batch["user_idx"], batch["user_cat"], batch["user_dense"],
        batch["item_cat"], batch["item_dense"],
    )

print("\nForward pass output shapes:")
print("  user_emb :", user_emb.shape,  "— expect (8, 64)")
print("  item_emb :", item_emb.shape,  "— expect (8, 64)")
print("  scores   :", scores.shape,    "— expect (8, 8)")
print("  user_emb norms:", user_emb.norm(dim=1).round(decimals=4))
print("  item_emb norms:", item_emb.norm(dim=1).round(decimals=4))
print("  (all norms should be 1.0 — L2 normalized)")

# Loss check — with temperature=1.0, random init → uniform logits → E[loss] = log(B)
loss     = in_batch_loss(scores)
expected = torch.log(torch.tensor(8.0)).item()
print(f"\n  Initial loss: {loss.item():.4f}  (expect ~{expected:.4f} = log(batch_size))")
print("  (With temperature=1.0, random-init loss should be close to log(B).)")
