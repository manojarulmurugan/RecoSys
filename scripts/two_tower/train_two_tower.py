import functools
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
ARTIFACTS_DIR  = _REPO_ROOT / "artifacts" / "50k"
CHECKPOINT_DIR           = _REPO_ROOT / "artifacts" / "50k" / "checkpoints_v2"
TEST_GCS_PATH            = "gs://recosys-data-bucket/samples/users_sample_50k/test/"
N_EPOCHS                 = 30
BATCH_SIZE               = 1024
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-5
TEMPERATURE              = 0.05
EVAL_EVERY               = 5
USE_CONFIDENCE_WEIGHTING = False
DEVICE                   = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load artifacts ────────────────────────────────────────────────────────────
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
print(f"  Confidence weighting : {USE_CONFIDENCE_WEIGHTING}")

# ── Build model ───────────────────────────────────────────────────────────────
from src.two_tower.models.two_tower import UserTower, ItemTower, TwoTowerModel

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
model = TwoTowerModel(user_tower, item_tower, temperature=TEMPERATURE)
model.model_summary()

# ── Evaluation callback (runs every EVAL_EVERY epochs during training) ────────
from src.two_tower.evaluation.evaluate import evaluate as run_eval

eval_callback = functools.partial(
    run_eval,
    items_encoded_df = items_enc,
    users_encoded_df = users_enc,
    test_df          = test_df,
    train_pairs_df   = train_pairs,
    vocabs           = vocabs,
    device           = torch.device(DEVICE),
    k                = 10,
)

# ── Train ─────────────────────────────────────────────────────────────────────
from src.two_tower.training.train import train

history = train(
    model            = model,
    train_pairs_df   = train_pairs,
    users_encoded_df = users_enc,
    items_encoded_df = items_enc,
    n_epochs         = N_EPOCHS,
    batch_size       = BATCH_SIZE,
    learning_rate    = LEARNING_RATE,
    weight_decay     = WEIGHT_DECAY,
    device_str       = DEVICE,
    checkpoint_dir   = str(CHECKPOINT_DIR),
    eval_every               = EVAL_EVERY,
    eval_fn                  = eval_callback,
    use_confidence_weighting = USE_CONFIDENCE_WEIGHTING,
)

# ── Final summary ─────────────────────────────────────────────────────────────
losses     = history["train_loss"]
best_loss  = min(losses)
best_epoch = losses.index(best_loss) + 1
final_ckpt = CHECKPOINT_DIR / f"epoch_{N_EPOCHS}.pt"

print("\n" + "=" * 55)
print("Training complete.")
print("=" * 55)
print(f"  {'Epoch':<8}  {'Loss':>8}")
print(f"  {'-'*8}  {'-'*8}")
for i, loss in enumerate(losses, start=1):
    marker = "  ← best" if i == best_epoch else ""
    print(f"  {i:<8}  {loss:>8.4f}{marker}")
print("=" * 55)
print(f"  Best loss  : {best_loss:.4f}  (epoch {best_epoch}/{N_EPOCHS})")
print(f"  Final ckpt : {final_ckpt}")
print("=" * 55)
print("Run evaluate_two_tower.py to get full metrics.")
