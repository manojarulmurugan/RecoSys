"""Embedding-collapse diagnostic for a trained Two-Tower checkpoint.

Loads a checkpoint and runs three analyses:
  1. User-user pairwise cosine similarity  (3 000 random users)
  2. Item-item pairwise cosine similarity  (3 000 random items)
  3. Random user-item cosine similarity    (3 000 unrelated pairs)

Since embeddings are L2-normalised, cosine similarity == dot product, so
the full (3000, 3000) similarity matrix costs one matrix multiply.

Healthy signal ranges
─────────────────────
  User-user mean  :  0.01 – 0.10   (> 0.30 → likely collapse)
  Item-item mean  :  0.01 – 0.10   (> 0.30 → likely collapse)
  Random user-item:  ~0.00          (large deviation → cross-tower misalignment)

Usage
─────
  python scripts/two_tower/diagnose_embeddings.py \\
      --checkpoint artifacts/500k/checkpoints/epoch_5.pt

  python scripts/two_tower/diagnose_embeddings.py \\
      --checkpoint artifacts/500k/checkpoints/epoch_30.pt \\
      --sample-size 5000 --seed 99
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
    _REPO_ROOT / "secrets" / "recosys-service-account.json"
)

from src.two_tower.models.two_tower import UserTower, ItemTower, TwoTowerModel


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding-collapse diagnostic for a Two-Tower checkpoint."
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint file, e.g. artifacts/500k/checkpoints/epoch_5.pt",
    )
    p.add_argument(
        "--artifacts-dir", default=None,
        help=(
            "Directory containing vocabs.pkl, *_encoded.parquet, etc.  "
            "Defaults to the parent of the checkpoint's grandparent "
            "(i.e. artifacts/500k/ when checkpoint is inside checkpoints/)."
        ),
    )
    p.add_argument(
        "--sample-size", type=int, default=3000,
        help="Number of users/items to sample for the similarity matrices (default: 3000).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    p.add_argument(
        "--temperature", type=float, default=0.05,
        help="Temperature used when the model was trained (default: 0.05).",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pairwise_stats(embs: np.ndarray) -> dict[str, float]:
    """Compute off-diagonal cosine similarity statistics for an (N, D) embedding matrix.

    Because embeddings are L2-normalised, cosine_sim(i, j) == dot(i, j),
    so the full similarity matrix is a single matmul.

    Args:
        embs: float32 ndarray of shape (N, D), each row L2-normalised.

    Returns:
        Dict with keys: mean, std, p10, p50, p90.
    """
    sim: np.ndarray = embs @ embs.T           # (N, N)
    n = sim.shape[0]

    # Mask diagonal (self-similarity == 1.0 by construction)
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim[mask]                      # (N*N - N,) flattened

    return {
        "mean": float(np.mean(off_diag)),
        "std":  float(np.std(off_diag)),
        "p10":  float(np.percentile(off_diag, 10)),
        "p50":  float(np.percentile(off_diag, 50)),
        "p90":  float(np.percentile(off_diag, 90)),
    }


def _cross_stats(user_embs: np.ndarray, item_embs: np.ndarray) -> dict[str, float]:
    """Compute per-pair cosine similarity between paired rows of two matrices.

    Each row i of user_embs is paired with row i of item_embs (random,
    unrelated pairs).  Returns mean, std, p10, p50, p90.

    Args:
        user_embs: float32 (N, D), L2-normalised.
        item_embs: float32 (N, D), L2-normalised.
    """
    # Element-wise dot product across the 64 dims, summed → scalar per pair
    pair_sims = (user_embs * item_embs).sum(axis=1)   # (N,)
    return {
        "mean": float(np.mean(pair_sims)),
        "std":  float(np.std(pair_sims)),
        "p10":  float(np.percentile(pair_sims, 10)),
        "p50":  float(np.percentile(pair_sims, 50)),
        "p90":  float(np.percentile(pair_sims, 90)),
    }


def _print_stats(label: str, stats: dict[str, float], warn_threshold: float | None = None) -> None:
    width = 52
    print(f"  {'Statistic':<14}  {'Value':>10}")
    print(f"  {'-'*14}  {'-'*10}")
    for key, val in stats.items():
        print(f"  {key:<14}  {val:>10.6f}")
    if warn_threshold is not None:
        mean = stats["mean"]
        if mean > warn_threshold:
            print(f"\n  COLLAPSE WARNING : mean {mean:.4f} exceeds threshold {warn_threshold:.2f}")
        else:
            print(f"\n  OK               : mean {mean:.4f} is below threshold {warn_threshold:.2f}")


def _encode_users(
    model: TwoTowerModel,
    users_encoded_df: pd.DataFrame,
    user_idxs: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run a set of user_idxs through the user tower.

    Builds the same feature lookup arrays as the training dataset so the
    model receives exactly the same input format.

    Returns:
        float32 ndarray of shape (len(user_idxs), 64), L2-normalised.
    """
    n_users = int(users_encoded_df["user_idx"].max()) + 1

    user_cat_arr = np.zeros((n_users, 4), dtype=np.int64)
    user_cat_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
        ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
    ].values.astype(np.int64)

    user_dense_arr = np.zeros((n_users, 6), dtype=np.float32)
    user_dense_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
        ["log_total_events", "months_active", "purchase_rate", "cart_rate",
         "log_n_sessions", "avg_purchase_price_scaled"]
    ].values.astype(np.float32)

    all_embs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(user_idxs), batch_size):
            batch = user_idxs[start : start + batch_size]
            uid_t   = torch.tensor(batch,                  dtype=torch.long,    device=device)
            cat_t   = torch.tensor(user_cat_arr[batch],    dtype=torch.long,    device=device)
            dense_t = torch.tensor(user_dense_arr[batch],  dtype=torch.float32, device=device)
            emb = model.get_user_embedding(uid_t, cat_t, dense_t)
            all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs).astype(np.float32)


def _encode_items(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    item_idxs: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run a set of item_idxs through the item tower.

    Returns:
        float32 ndarray of shape (len(item_idxs), 64), L2-normalised.
    """
    # Build a sub-dataframe for the sampled items only
    idx_set = set(item_idxs.tolist())
    sub_df  = items_encoded_df[items_encoded_df["item_idx"].isin(idx_set)].copy()
    sub_df  = sub_df.set_index("item_idx").loc[item_idxs].reset_index()

    item_cat_t = torch.tensor(
        sub_df[["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
               ].values.astype(np.int64),
        dtype=torch.long,
    )
    item_dense_t = torch.tensor(
        sub_df[["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
               ].values.astype(np.float32),
        dtype=torch.float32,
    )

    all_embs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(item_idxs), batch_size):
            cat_b   = item_cat_t[start : start + batch_size].to(device)
            dense_b = item_dense_t[start : start + batch_size].to(device)
            emb = model.get_item_embeddings(cat_b, dense_b)
            all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    ckpt_path = pathlib.Path(args.checkpoint).resolve()

    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")

    # Derive artifacts dir from checkpoint path if not explicitly supplied.
    # Expected layout: artifacts/<name>/checkpoints/epoch_N.pt
    if args.artifacts_dir is not None:
        artifacts_dir = pathlib.Path(args.artifacts_dir).resolve()
    else:
        artifacts_dir = ckpt_path.parent.parent
    print(f"Artifacts dir : {artifacts_dir}")

    if not artifacts_dir.exists():
        sys.exit(f"ERROR: artifacts directory not found: {artifacts_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)
    N      = args.sample_size

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading artifacts...")

    with open(artifacts_dir / "vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)

    items_enc = pd.read_parquet(artifacts_dir / "items_encoded.parquet")
    users_enc = pd.read_parquet(artifacts_dir / "users_encoded.parquet")

    print(f"  items_encoded  : {items_enc.shape}")
    print(f"  users_encoded  : {users_enc.shape}")
    print(f"  device         : {device}")

    # ── Build model ───────────────────────────────────────────────────────────
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
    model = TwoTowerModel(user_tower, item_tower, temperature=args.temperature)
    model.to(device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Epoch      : {ckpt.get('epoch', 'unknown')}")
    print(f"  Train loss : {ckpt.get('loss', float('nan')):.4f}")

    # ── Sample users and items ────────────────────────────────────────────────
    available_user_idxs = users_enc["user_idx"].values.astype(np.int64)
    available_item_idxs = items_enc["item_idx"].values.astype(np.int64)

    n_users_sample = min(N, len(available_user_idxs))
    n_items_sample = min(N, len(available_item_idxs))

    sampled_user_idxs  = rng.choice(available_user_idxs, size=n_users_sample, replace=False)
    sampled_item_idxs  = rng.choice(available_item_idxs, size=n_items_sample, replace=False)
    # Independent resample for user-item cross pairs (unrelated pairs)
    cross_user_idxs    = rng.choice(available_user_idxs, size=min(N, n_users_sample), replace=False)
    cross_item_idxs    = rng.choice(available_item_idxs, size=min(N, n_items_sample), replace=False)
    # Trim to equal length for element-wise pairing
    cross_n            = min(len(cross_user_idxs), len(cross_item_idxs))
    cross_user_idxs    = cross_user_idxs[:cross_n]
    cross_item_idxs    = cross_item_idxs[:cross_n]

    print(f"\nSampling {n_users_sample:,} users  and  {n_items_sample:,} items  "
          f"(seed={args.seed})")

    # ── Encode ────────────────────────────────────────────────────────────────
    print("Encoding user sample through user tower ...")
    user_embs = _encode_users(model, users_enc, sampled_user_idxs, device)

    print("Encoding item sample through item tower ...")
    item_embs = _encode_items(model, items_enc, sampled_item_idxs, device)

    print("Encoding cross pairs ...")
    cross_user_embs = _encode_users(model, users_enc, cross_user_idxs, device)
    cross_item_embs = _encode_items(model, items_enc, cross_item_idxs, device)

    # ── Compute statistics ────────────────────────────────────────────────────
    print("Computing similarity matrices ...")
    user_stats  = _pairwise_stats(user_embs)
    item_stats  = _pairwise_stats(item_embs)
    cross_stats = _cross_stats(cross_user_embs, cross_item_embs)

    # ── Print report ──────────────────────────────────────────────────────────
    sep      = "=" * 58
    thin_sep = "-" * 58

    print(f"\n{sep}")
    print(f"  EMBEDDING COLLAPSE DIAGNOSTIC")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Epoch      : {ckpt.get('epoch', '?')}   |   "
          f"Train loss : {ckpt.get('loss', float('nan')):.4f}")
    print(f"  Sample size: {N:,}   |   Seed: {args.seed}")
    print(sep)

    print(f"\n  [1] USER-USER PAIRWISE COSINE SIMILARITY")
    print(f"      ({n_users_sample:,} users, {n_users_sample**2 - n_users_sample:,} off-diagonal pairs)")
    print(f"      Healthy range: mean 0.01 – 0.10  |  Collapse threshold: > 0.30")
    print()
    _print_stats("user-user", user_stats, warn_threshold=0.30)

    print(f"\n  {thin_sep}")

    print(f"\n  [2] ITEM-ITEM PAIRWISE COSINE SIMILARITY")
    print(f"      ({n_items_sample:,} items, {n_items_sample**2 - n_items_sample:,} off-diagonal pairs)")
    print(f"      Healthy range: mean 0.01 – 0.10  |  Collapse threshold: > 0.30")
    print()
    _print_stats("item-item", item_stats, warn_threshold=0.30)

    print(f"\n  {thin_sep}")

    print(f"\n  [3] RANDOM USER-ITEM COSINE SIMILARITY  (unrelated pairs)")
    print(f"      ({cross_n:,} random user-item pairs — NOT positive pairs)")
    print(f"      Healthy range: mean ~ 0.00  |  Large deviation → cross-tower misalignment")
    print()
    _print_stats("user-item", cross_stats, warn_threshold=None)
    mean_ui = cross_stats["mean"]
    if abs(mean_ui) > 0.05:
        print(f"\n  ALIGNMENT WARNING : mean {mean_ui:.4f} deviates from 0 by > 0.05")
    else:
        print(f"\n  OK               : mean {mean_ui:.4f} is near zero")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
