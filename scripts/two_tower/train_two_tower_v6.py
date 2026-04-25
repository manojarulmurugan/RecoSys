"""Two-Tower training script v6 — Distribution-aligned training + item centroid.

V6 makes three targeted changes to close the gap to the popularity baseline:

  1. **High-intent pairs only** (the killer change):
     Filter ``train_pairs`` to ``confidence_score >= 2.0`` (cart + purchase).
     This aligns the training objective with the evaluation target, which
     grades on cart/purchase recall. Views are ~5–10× more frequent but carry
     no signal for the eval metric; they dilute every gradient step.
     Pair count drops from ~7.47M → ~1–2M; epochs run ~4× faster.

  2. **Item centroid as user feature** (YouTube Two-Tower §3.1):
     For each user, the mean 32-dim ``item_emb`` vector over their
     cart/purchase history is precomputed and stored in
     ``users_encoded_v2.parquet`` (run ``build_users_v2_500k.py`` once).
     ``UserTowerV3`` appends this centroid to the dense vector (8 → 40 dims),
     giving the static user tower direct access to the item embedding space.
     Cold users receive a zero centroid and fall back to their other features.

  3. **No confidence weighting** (dropped):
     All remaining pairs are high-intent (cart + purchase); there is nothing
     to up-weight.  Uniform loss per pair is correct.

All other V3 settings are preserved: LogQ correction, label smoothing 0.1,
temperature 0.07, batch 4096, cosine LR, split-embedding AdamW, stratified
eval every 5 epochs.

Prerequisites:
  - artifacts/500k/items_encoded_v2.parquet   (from augment_items_v2_500k.py)
  - artifacts/500k/users_encoded_v2.parquet   (from build_users_v2_500k.py)
"""

import json
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _ensure_gcp_credentials() -> None:
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing and pathlib.Path(existing).expanduser().is_file():
        return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        _REPO_ROOT / "recosys-service-account.json",
        pathlib.Path("/content/recosys-service-account.json"),
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate.resolve())
            print(f"  Using GCP credentials: {candidate}")
            return
    print(
        "WARNING: No service-account JSON found. Set GOOGLE_APPLICATION_CREDENTIALS "
        "or place recosys-service-account.json in a known location."
    )


_ensure_gcp_credentials()

from src.two_tower.evaluation.evaluate import evaluate, evaluate_stratified

# ── CONFIG ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR  = _REPO_ROOT / "artifacts" / "500k"
TEST_GCS_PATH  = "gs://recosys-data-bucket/samples/users_sample_500k/test/"

MIN_CONFIDENCE           = 2.0    # keep cart (2) + purchase (4) only
USE_CONFIDENCE_WEIGHTING = False  # all pairs are high-intent — no need to weight
LABEL_SMOOTHING          = 0.1
TEMPERATURE              = 0.07
BATCH_SIZE               = 4096
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-5
N_EPOCHS                 = 30
EVAL_EVERY               = 5
CHECKPOINT_DIR           = _REPO_ROOT / "artifacts" / "500k" / "checkpoints_v6"
DEVICE                   = "cuda" if torch.cuda.is_available() else "cpu"


# ── Optimizer: split embedding vs MLP params ──────────────────────────────────

def get_param_groups(model: nn.Module, lr: float, weight_decay: float) -> list[dict]:
    emb_params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            emb_params.extend(list(module.parameters()))
    emb_param_ids = {id(p) for p in emb_params}
    other_params  = [p for p in model.parameters() if id(p) not in emb_param_ids]
    return [
        {"params": emb_params,   "lr": lr, "weight_decay": 0.0},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]


# ── Load artifacts ────────────────────────────────────────────────────────────
print("Loading artifacts...")

with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
    vocabs = pickle.load(f)

items_enc_v2 = pd.read_parquet(ARTIFACTS_DIR / "items_encoded_v2.parquet")
users_enc_v2 = pd.read_parquet(ARTIFACTS_DIR / "users_encoded_v2.parquet")
train_pairs  = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")

print(f"  items_encoded_v2 : {items_enc_v2.shape}")
print(f"  users_encoded_v2 : {users_enc_v2.shape}  "
      f"(centroid cols: {'item_centroid_0' in users_enc_v2.columns})")
print(f"  train_pairs      : {train_pairs.shape}")

# ── Filter to high-intent pairs ───────────────────────────────────────────────
hi_pairs = train_pairs[train_pairs["confidence_score"] >= MIN_CONFIDENCE].copy()
print(f"\nHigh-intent pairs (confidence >= {MIN_CONFIDENCE}): "
      f"{len(hi_pairs):,} of {len(train_pairs):,} "
      f"({100 * len(hi_pairs) / len(train_pairs):.1f}%)")
print(f"  Unique users  : {hi_pairs['user_idx'].nunique():,}")
print(f"  Unique items  : {hi_pairs['item_idx'].nunique():,}")

print(f"\nLoading test split from {TEST_GCS_PATH} ...")
test_df = pd.read_parquet(TEST_GCS_PATH)
print(f"  test_df          : {test_df.shape}")
print(f"  device           : {DEVICE}")
print(f"  label smoothing  : {LABEL_SMOOTHING}")
print(f"  temperature      : {TEMPERATURE}")
print(f"  batch size       : {BATCH_SIZE}")

# ── LogQ correction array ─────────────────────────────────────────────────────
# Compute from HIGH-INTENT pairs so the correction matches the new training dist.
print("\nBuilding LogQ correction array (from high-intent pairs)...")
item_counts   = hi_pairs["item_idx"].value_counts()
n_total_pairs = len(hi_pairs)
n_items_max   = int(items_enc_v2["item_idx"].max()) + 1

log_q_arr = np.full(n_items_max, fill_value=np.log(1.0 / n_items_max), dtype=np.float32)
for item_idx_val, count in item_counts.items():
    log_q_arr[int(item_idx_val)] = float(np.log(count / n_total_pairs))

print(
    f"  LogQ array  : {n_items_max:,} items  "
    f"min={log_q_arr.min():.3f}  max={log_q_arr.max():.3f}  "
    f"mean={log_q_arr.mean():.3f}"
)

# ── Build V6 model ─────────────────────────────────────────────────────────────
from src.two_tower.models.two_tower import ItemTowerV2, TwoTowerModel, UserTowerV3

user_tower = UserTowerV3(
    n_users    = len(vocabs["user2idx"]),
    n_top_cats = len(vocabs["top_cat_vocab"]),
    # dense_input_dim=40 is the default (8 V2 + 32 centroid)
)
item_tower = ItemTowerV2(
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
from src.two_tower.data.dataset import TwoTowerDataset

dataset = TwoTowerDataset(hi_pairs, users_enc_v2, items_enc_v2)
print(f"\nTwoTowerDataset (V6 — high-intent only): {len(dataset):,} pairs")
print(f"  item_dense dim  : {dataset._item_dense.shape[1]}  (V2 features: {dataset._use_v2_items})")
print(f"  user_dense dim  : {dataset._user_dense.shape[1]}  (centroid: {dataset._use_centroid})")

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
print(f"  Batches/epoch   : {len(dataloader):,}")

# ── Optimizer + scheduler ─────────────────────────────────────────────────────
from src.two_tower.training.train import train_epoch

optimizer = torch.optim.AdamW(get_param_groups(model, LEARNING_RATE, WEIGHT_DECAY))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_EPOCHS, eta_min=1e-5
)

# ── Checkpoint dir + training log ─────────────────────────────────────────────
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
log_path = CHECKPOINT_DIR / "training_log.json"

if log_path.exists():
    with open(log_path) as f:
        training_log: list[dict] = json.load(f)
else:
    training_log = []

# ── Evaluation helpers ────────────────────────────────────────────────────────
_eval_kwargs = dict(
    items_encoded_df = items_enc_v2,
    users_encoded_df = users_enc_v2,
    test_df          = test_df,
    train_pairs_df   = train_pairs,   # full pairs for stratified cohort sizing
    vocabs           = vocabs,
    device           = device,
)

# ── Training loop ─────────────────────────────────────────────────────────────
history: dict[str, list[float]] = {"train_loss": []}

print(f"\nTraining on {device} — {N_EPOCHS} epochs, batch {BATCH_SIZE}, "
      f"lr {LEARNING_RATE} → 1e-5 (cosine), wd {WEIGHT_DECAY}")
print(f"  High-intent pairs only (conf >= {MIN_CONFIDENCE}): "
      f"{len(hi_pairs):,} pairs per epoch")

for epoch in range(1, N_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{N_EPOCHS}")

    epoch_loss = train_epoch(
        model                    = model,
        dataloader               = dataloader,
        optimizer                = optimizer,
        device                   = device,
        log_every                = 100,
        use_confidence_weighting = USE_CONFIDENCE_WEIGHTING,
        log_q_correction_arr     = log_q_arr,
        label_smoothing          = LABEL_SMOOTHING,
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

        log_entry = {
            "epoch":      epoch,
            "train_loss": round(epoch_loss, 6),
            "recall_10":  round(metrics["recall_10"],  6),
            "ndcg_10":    round(metrics["ndcg_10"],    6),
            "recall_20":  round(metrics["recall_20"],  6),
            "ndcg_20":    round(metrics["ndcg_20"],    6),
            "lr":         round(current_lr,            8),
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
