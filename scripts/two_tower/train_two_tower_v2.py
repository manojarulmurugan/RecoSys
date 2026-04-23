"""Two-Tower training script v2 — hard negatives + stratified evaluation.

Changes vs train_two_tower.py:
  - Uses TwoTowerDatasetWithHardNegs (cat_l2 primary / price_bucket fallback)
    when USE_HARD_NEGATIVES=True; falls back to TwoTowerDataset otherwise.
  - Calls train_epoch_with_hard_negs / train_epoch accordingly.
  - Split AdamW param groups: embeddings get weight_decay=0, MLP layers keep it.
  - Runs both evaluate() and evaluate_stratified() every EVAL_EVERY epochs.
  - Checkpoints are zero-padded (epoch_05.pt, epoch_10.pt, …).
  - Appends a row to checkpoints_v2/training_log.json after every eval.
"""

import json
import os
import pathlib
import pickle
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR  = _REPO_ROOT / "artifacts" / "500k"
TEST_GCS_PATH  = "gs://recosys-data-bucket/samples/users_sample_500k/test/"

USE_HARD_NEGATIVES = True
HARD_NEG_RATIO     = 3
TEMPERATURE        = 0.05
BATCH_SIZE         = 2048
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-5
N_EPOCHS           = 40
EVAL_EVERY         = 5
CHECKPOINT_DIR     = _REPO_ROOT / "artifacts" / "500k" / "checkpoints_v2"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


# ── Optimizer: split embedding vs MLP params ──────────────────────────────────

def get_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> list[dict]:
    """Return two AdamW param groups.

    Embedding parameters receive weight_decay=0 (L2 on sparse lookup tables
    hurts recall without reducing overfitting meaningfully).  All other
    parameters (MLP weights/biases) keep the supplied weight_decay.
    """
    emb_params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            emb_params.extend(list(module.parameters()))

    emb_param_ids = {id(p) for p in emb_params}
    other_params = [p for p in model.parameters() if id(p) not in emb_param_ids]

    return [
        {"params": emb_params,   "lr": lr, "weight_decay": 0.0},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]


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
print(f"  hard negatives : {USE_HARD_NEGATIVES}  (ratio: {HARD_NEG_RATIO})")

# ── Build model ───────────────────────────────────────────────────────────────
from src.two_tower.models.two_tower import ItemTower, TwoTowerModel, UserTower

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

device = torch.device(DEVICE)
model.to(device)

# ── Dataset ───────────────────────────────────────────────────────────────────
from src.two_tower.data.dataset import TwoTowerDataset, TwoTowerDatasetWithHardNegs

if USE_HARD_NEGATIVES:
    dataset = TwoTowerDatasetWithHardNegs(
        train_pairs_df   = train_pairs,
        users_encoded_df = users_enc,
        items_encoded_df = items_enc,
        n_hard_negs      = HARD_NEG_RATIO,
        seed             = 42,
    )
    print(dataset)
else:
    dataset = TwoTowerDataset(train_pairs, users_enc, items_enc)
    print(f"TwoTowerDataset: {len(dataset):,} pairs")

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
print(f"  Batches/epoch : {len(dataloader):,}")

# ── Optimizer + scheduler ─────────────────────────────────────────────────────
from src.two_tower.training.train import (
    in_batch_loss,
    train_epoch,
    train_epoch_with_hard_negs,
)

optimizer = torch.optim.AdamW(
    get_param_groups(model, LEARNING_RATE, WEIGHT_DECAY)
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-5
)

# ── Checkpoint dir + training log ─────────────────────────────────────────────
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
log_path = CHECKPOINT_DIR / "training_log.json"

# Load existing log so re-runs append rather than overwrite
if log_path.exists():
    with open(log_path) as f:
        training_log: list[dict] = json.load(f)
else:
    training_log = []

# ── Evaluation helpers ────────────────────────────────────────────────────────
from src.two_tower.evaluation.evaluate import evaluate, evaluate_stratified

_eval_kwargs = dict(
    items_encoded_df = items_enc,
    users_encoded_df = users_enc,
    test_df          = test_df,
    train_pairs_df   = train_pairs,
    vocabs           = vocabs,
    device           = device,
)

# ── Training loop ─────────────────────────────────────────────────────────────
history: dict[str, list[float]] = {"train_loss": []}

print(f"\nTraining on {device} — {N_EPOCHS} epochs, batch {BATCH_SIZE}, "
      f"lr {LEARNING_RATE} → 1e-5 (cosine), wd {WEIGHT_DECAY}")

for epoch in range(1, N_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{N_EPOCHS}")

    if USE_HARD_NEGATIVES:
        epoch_loss = train_epoch_with_hard_negs(
            model       = model,
            dataloader  = dataloader,
            optimizer   = optimizer,
            temperature = TEMPERATURE,
            device      = device,
            log_every   = 100,
        )
    else:
        epoch_loss = train_epoch(
            model       = model,
            dataloader  = dataloader,
            optimizer   = optimizer,
            device      = device,
            log_every   = 100,
        )

    history["train_loss"].append(epoch_loss)
    current_lr = scheduler.get_last_lr()[0]
    print(f"  Epoch {epoch}/{N_EPOCHS} — loss: {epoch_loss:.4f}  lr: {current_lr:.2e}")

    scheduler.step()

    # ── Zero-padded checkpoint ─────────────────────────────────────────────
    ckpt_name = f"epoch_{epoch:02d}.pt"
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss":            epoch_loss,
        },
        CHECKPOINT_DIR / ckpt_name,
    )

    # ── Evaluation every EVAL_EVERY epochs ────────────────────────────────
    if epoch % EVAL_EVERY == 0:
        print(f"\n  --- Evaluation at epoch {epoch} ---")

        metrics = evaluate(model=model, **_eval_kwargs)

        print(f"\n  --- Stratified evaluation at epoch {epoch} ---")
        evaluate_stratified(model=model, **_eval_kwargs)

        # Append to training log
        log_entry = {
            "epoch":       epoch,
            "train_loss":  round(epoch_loss, 6),
            "recall_10":   round(metrics["recall_10"],  6),
            "ndcg_10":     round(metrics["ndcg_10"],    6),
            "recall_20":   round(metrics["recall_20"],  6),
            "ndcg_20":     round(metrics["ndcg_20"],    6),
            "lr":          round(current_lr,            8),
        }
        training_log.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"  training_log.json updated ({len(training_log)} entries)")

# ── Final summary ─────────────────────────────────────────────────────────────
losses     = history["train_loss"]
best_loss  = min(losses)
best_epoch = losses.index(best_loss) + 1
final_ckpt = CHECKPOINT_DIR / f"epoch_{N_EPOCHS:02d}.pt"

print("\n" + "=" * 55)
print("Training complete.")
print("=" * 55)
print(f"  {'Epoch':<8}  {'Loss':>8}")
print(f"  {'-'*8}  {'-'*8}")
for i, loss in enumerate(losses, start=1):
    marker = "  <- best" if i == best_epoch else ""
    print(f"  {i:<8}  {loss:>8.4f}{marker}")
print("=" * 55)
print(f"  Best loss  : {best_loss:.4f}  (epoch {best_epoch}/{N_EPOCHS})")
print(f"  Final ckpt : {final_ckpt}")
print(f"  Train log  : {log_path}")
print("=" * 55)
