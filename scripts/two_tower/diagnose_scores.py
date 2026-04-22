"""Score-distribution diagnostic for a trained Two-Tower checkpoint.

Two independent checks are run:

PART A — TRAINING SCORE DISTRIBUTION
  Samples N random batches from train_pairs.  For each batch, computes the
  (B × B) raw cosine similarity matrix (pre-temperature) and separates:
    • Diagonal   = positive pair scores  (user_i ↔ item_i)
    • Off-diagonal = in-batch negative scores  (user_i ↔ item_j, i≠j)
  Aggregates across all batches and reports mean, std, and the gap.

PART B — EVALUATION SCORE DISTRIBUTION
  Samples random users from the eligible eval pool.  For each user:
    • Computes cosine similarity to their held-out ground-truth item(s).
    • Computes cosine similarity to 100 random items from the trained pool.
  Reports mean GT score, mean random score, and the gap.

KEY INSIGHT
  If the train gap is large (e.g. 0.8) but the eval gap is small (e.g. 0.1),
  the model learned to separate in-batch negatives but not to retrieve
  genuinely relevant held-out items — the training objective is decoupled
  from the evaluation task.

Usage
─────
  # 50k experiment
  python scripts/two_tower/diagnose_scores.py \\
      --checkpoint artifacts/50k/checkpoints_v2/epoch_5.pt \\
      --test-path  gs://recosys-data-bucket/samples/users_sample_50k/test/

  # 500k experiment
  python scripts/two_tower/diagnose_scores.py \\
      --checkpoint artifacts/500k/checkpoints/epoch_30.pt \\
      --test-path  gs://recosys-data-bucket/samples/users_sample_500k/test/

  # compare two epochs side by side (run twice, compare output)
  python scripts/two_tower/diagnose_scores.py --checkpoint ... --test-path ...
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


# ── GCP credentials (same logic as diagnose_evaluation.py) ───────────────────

def _ensure_gcp_credentials() -> None:
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing:
        if pathlib.Path(existing).expanduser().is_file():
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score-distribution diagnostic for a Two-Tower checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt checkpoint file.")
    p.add_argument("--test-path", required=True,
                   help="GCS or local path to the test-split parquet "
                        "(e.g. gs://…/users_sample_50k/test/).")
    p.add_argument("--artifacts-dir", default=None,
                   help="Directory containing vocabs.pkl, *_encoded.parquet, "
                        "train_pairs.parquet.  Defaults to checkpoint's "
                        "grandparent directory.")
    p.add_argument("--n-train-batches", type=int, default=5,
                   help="Number of random train batches to score (default: 5).")
    p.add_argument("--train-batch-size", type=int, default=2048,
                   help="Pairs per train batch (default: 2048).")
    p.add_argument("--n-eval-users", type=int, default=500,
                   help="Eval users to sample for Part B (default: 500).")
    p.add_argument("--n-random-items", type=int, default=100,
                   help="Random items per eval user for negative baseline "
                        "(default: 100).")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Training temperature (default: 0.05).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42).")
    return p.parse_args()


# ── Feature lookup builders ───────────────────────────────────────────────────

def _build_user_lookups(
    users_encoded_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (user_cat_arr, user_dense_arr) indexed by user_idx."""
    n = int(users_encoded_df["user_idx"].max()) + 1

    cat = np.zeros((n, 4), dtype=np.int64)
    cat[users_encoded_df["user_idx"].values] = users_encoded_df[
        ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
    ].values.astype(np.int64)

    dense = np.zeros((n, 6), dtype=np.float32)
    dense[users_encoded_df["user_idx"].values] = users_encoded_df[
        ["log_total_events", "months_active", "purchase_rate", "cart_rate",
         "log_n_sessions", "avg_purchase_price_scaled"]
    ].values.astype(np.float32)

    return cat, dense


def _build_item_lookups(
    items_encoded_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (item_cat_arr, item_dense_arr) indexed by item_idx."""
    n = int(items_encoded_df["item_idx"].max()) + 1

    cat = np.zeros((n, 5), dtype=np.int64)
    cat[items_encoded_df["item_idx"].values] = items_encoded_df[
        ["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
    ].values.astype(np.int64)

    dense = np.zeros((n, 3), dtype=np.float32)
    dense[items_encoded_df["item_idx"].values] = items_encoded_df[
        ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
    ].values.astype(np.float32)

    return cat, dense


# ── Batch encoders ────────────────────────────────────────────────────────────

def _encode_users(
    model: TwoTowerModel,
    user_idxs: np.ndarray,
    user_cat_arr: np.ndarray,
    user_dense_arr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Encode a batch of users; returns (N, 64) float32 L2-normalised array."""
    with torch.no_grad():
        uid_t   = torch.tensor(user_idxs,             dtype=torch.long,    device=device)
        cat_t   = torch.tensor(user_cat_arr[user_idxs], dtype=torch.long,  device=device)
        dense_t = torch.tensor(user_dense_arr[user_idxs], dtype=torch.float32, device=device)
        emb = model.get_user_embedding(uid_t, cat_t, dense_t)
    return emb.cpu().numpy().astype(np.float32)


def _encode_items(
    model: TwoTowerModel,
    item_idxs: np.ndarray,
    item_cat_arr: np.ndarray,
    item_dense_arr: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Encode a batch of items; returns (N, 64) float32 L2-normalised array."""
    with torch.no_grad():
        cat_t   = torch.tensor(item_cat_arr[item_idxs], dtype=torch.long,    device=device)
        dense_t = torch.tensor(item_dense_arr[item_idxs], dtype=torch.float32, device=device)
        emb = model.get_item_embeddings(cat_t, dense_t)
    return emb.cpu().numpy().astype(np.float32)


# ── Part A — training score distribution ─────────────────────────────────────

def run_training_distribution(
    model: TwoTowerModel,
    train_pairs_df: pd.DataFrame,
    user_cat_arr: np.ndarray,
    user_dense_arr: np.ndarray,
    item_cat_arr: np.ndarray,
    item_dense_arr: np.ndarray,
    device: torch.device,
    n_batches: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample random training batches and report positive vs negative score stats.

    For each batch of B (user, item) positive pairs:
      • user_embs @ item_embs.T  →  (B, B) raw cosine similarity matrix
      • Diagonal   = positive pair scores  (the signal the model is trained on)
      • Off-diagonal = in-batch negative scores  (511 or 2047 per positive)

    Returns aggregated stats across all batches.
    """
    all_pos_scores: list[float] = []
    all_neg_scores: list[float] = []

    n_pairs = len(train_pairs_df)

    print(f"  Sampling {n_batches} batches of {batch_size:,} pairs from "
          f"{n_pairs:,} training pairs ...")

    model.eval()
    for b in range(n_batches):
        indices   = rng.choice(n_pairs, size=batch_size, replace=False)
        batch_df  = train_pairs_df.iloc[indices]

        user_idxs = batch_df["user_idx"].values.astype(np.int64)
        item_idxs = batch_df["item_idx"].values.astype(np.int64)

        user_embs = _encode_users(model, user_idxs, user_cat_arr, user_dense_arr, device)
        item_embs = _encode_items(model, item_idxs, item_cat_arr, item_dense_arr, device)

        # Raw cosine similarity matrix — pre-temperature
        sim: np.ndarray = user_embs @ item_embs.T   # (B, B)

        B    = sim.shape[0]
        diag = np.diag(sim)                         # (B,)  positive pairs
        mask = ~np.eye(B, dtype=bool)
        off  = sim[mask]                            # (B*B - B,) negatives

        all_pos_scores.extend(diag.tolist())
        all_neg_scores.extend(off.tolist())

        print(f"    batch {b+1}/{n_batches}  |  "
              f"pos mean: {diag.mean():.4f}  |  neg mean: {off.mean():.4f}")

    pos = np.array(all_pos_scores, dtype=np.float32)
    neg = np.array(all_neg_scores, dtype=np.float32)

    return {
        "n_positive_pairs":  len(pos),
        "n_negative_pairs":  len(neg),
        "pos_mean":          float(pos.mean()),
        "pos_std":           float(pos.std()),
        "pos_p10":           float(np.percentile(pos, 10)),
        "pos_p50":           float(np.percentile(pos, 50)),
        "pos_p90":           float(np.percentile(pos, 90)),
        "neg_mean":          float(neg.mean()),
        "neg_std":           float(neg.std()),
        "neg_p10":           float(np.percentile(neg, 10)),
        "neg_p50":           float(np.percentile(neg, 50)),
        "neg_p90":           float(np.percentile(neg, 90)),
        "gap":               float(pos.mean() - neg.mean()),
    }


# ── Part B — evaluation score distribution ───────────────────────────────────

def _build_item_pool(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    trained_item_idxs: set[int],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all trained items; return (pool_embs, pool_item_idxs).

    pool_embs     : float32 (n_trained, 64), L2-normalised.
    pool_item_idxs: int64   (n_trained,), item_idx per row.
    """
    df = items_encoded_df[
        items_encoded_df["item_idx"].isin(trained_item_idxs)
    ].sort_values("item_idx").reset_index(drop=True)

    item_cat_t, item_dense_t = build_full_item_tensors(df)
    item_idx_arr = item_cat_t[:, 0].numpy().astype(np.int64)

    all_embs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            emb = model.get_item_embeddings(
                item_cat_t[start : start + batch_size].to(device),
                item_dense_t[start : start + batch_size].to(device),
            )
            all_embs.append(emb.cpu().numpy())

    pool_embs = np.vstack(all_embs).astype(np.float32)
    return pool_embs, item_idx_arr


def run_eval_distribution(
    model: TwoTowerModel,
    test_df: pd.DataFrame,
    users_encoded_df: pd.DataFrame,
    items_encoded_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    vocabs: dict,
    device: torch.device,
    n_eval_users: int,
    n_random_items: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample eval users and compare GT item scores vs random item scores.

    For each sampled user:
      gt_score   = mean cosine similarity between user and their held-out GT items.
      rand_score = mean cosine similarity between user and N_RANDOM random items
                   drawn from the trained item pool (unrelated to the user).

    Returns aggregated stats across all sampled users.
    """
    user2idx = vocabs["user2idx"]
    item2idx = vocabs["item2idx"]

    # ── Build ground truth (cart + purchase events from test split) ───────────
    relevant = test_df[test_df["event_type"].isin({"cart", "purchase"})]
    valid_encoded_idxs = set(users_encoded_df["user_idx"].astype(int).tolist())

    gt_map: dict[int, set[int]] = {}   # user_idx → set of item_idxs
    for user_id, grp in relevant.groupby("user_id"):
        if user_id not in user2idx:
            continue
        u_idx = user2idx[user_id]
        if u_idx not in valid_encoded_idxs:
            continue
        item_idxs = {
            item2idx[pid]
            for pid in grp["product_id"].tolist()
            if pid in item2idx
        }
        if item_idxs:
            gt_map[u_idx] = item_idxs

    eligible_user_idxs = list(gt_map.keys())
    print(f"  Eligible eval users with GT items in item vocab: "
          f"{len(eligible_user_idxs):,}")

    if not eligible_user_idxs:
        print("  ERROR: no eligible eval users found — check test_path and vocabs.")
        return {}

    n_sample = min(n_eval_users, len(eligible_user_idxs))
    sampled  = rng.choice(eligible_user_idxs, size=n_sample, replace=False).astype(np.int64)
    print(f"  Sampling {n_sample} users for eval score distribution ...")

    # ── Build trained item pool ───────────────────────────────────────────────
    trained_item_idxs = set(train_pairs_df["item_idx"].unique().tolist())
    print(f"  Building item pool ({len(trained_item_idxs):,} trained items) ...")
    pool_embs, pool_item_idxs = _build_item_pool(
        model, items_encoded_df, trained_item_idxs, device
    )
    # Fast lookup: item_idx → row index in pool_embs
    item_to_pool_row: dict[int, int] = {
        int(iidx): row for row, iidx in enumerate(pool_item_idxs)
    }
    n_pool = len(pool_embs)
    print(f"  Item pool ready: {n_pool:,} items encoded.")

    # ── Encode all sampled users in one batched call ──────────────────────────
    user_cat_arr, user_dense_arr = _build_user_lookups(users_encoded_df)
    user_embs = _encode_users(
        model, sampled, user_cat_arr, user_dense_arr, device
    )   # (n_sample, 64)

    # ── Score per user ────────────────────────────────────────────────────────
    per_user_gt_scores:   list[float] = []
    per_user_rand_scores: list[float] = []
    n_skipped = 0

    pool_indices = np.arange(n_pool, dtype=np.int64)

    for i, user_idx in enumerate(sampled):
        u_emb = user_embs[i]   # (64,)

        # Ground-truth item embeddings
        gt_item_idxs = gt_map[int(user_idx)]
        gt_rows = [
            item_to_pool_row[iidx]
            for iidx in gt_item_idxs
            if iidx in item_to_pool_row
        ]
        if not gt_rows:
            # GT items not in trained pool (cold-start items) — skip
            n_skipped += 1
            continue

        gt_embs   = pool_embs[gt_rows]            # (n_gt, 64)
        gt_scores = (u_emb @ gt_embs.T)           # (n_gt,) cosine sims

        # Random item embeddings (n_random_items drawn without replacement)
        n_draw      = min(n_random_items, n_pool)
        rand_rows   = rng.choice(pool_indices, size=n_draw, replace=False)
        rand_embs   = pool_embs[rand_rows]         # (n_draw, 64)
        rand_scores = (u_emb @ rand_embs.T)        # (n_draw,) cosine sims

        per_user_gt_scores.append(float(gt_scores.mean()))
        per_user_rand_scores.append(float(rand_scores.mean()))

    if n_skipped:
        print(f"  Note: {n_skipped} user(s) skipped — GT items absent from trained pool.")

    gt   = np.array(per_user_gt_scores,   dtype=np.float32)
    rand = np.array(per_user_rand_scores, dtype=np.float32)

    return {
        "n_users_scored":        len(gt),
        "gt_mean":               float(gt.mean()),
        "gt_std":                float(gt.std()),
        "gt_p10":                float(np.percentile(gt, 10)),
        "gt_p50":                float(np.percentile(gt, 50)),
        "gt_p90":                float(np.percentile(gt, 90)),
        "rand_mean":             float(rand.mean()),
        "rand_std":              float(rand.std()),
        "rand_p10":              float(np.percentile(rand, 10)),
        "rand_p50":              float(np.percentile(rand, 50)),
        "rand_p90":              float(np.percentile(rand, 90)),
        "gap":                   float(gt.mean() - rand.mean()),
    }


# ── Pretty printers ───────────────────────────────────────────────────────────

def _print_two_col(
    label_a: str, stats_a: dict[str, float],
    label_b: str, stats_b: dict[str, float],
    gap: float,
) -> None:
    """Print a two-column stat table: positives/GT on left, negatives/random on right."""
    rows = [("mean", "mean"), ("std", "std"),
            ("p10", "p10"),   ("p50", "p50"), ("p90", "p90")]
    print(f"  {'Statistic':<10}  {label_a:>12}  {label_b:>14}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*14}")
    for ka, kb in rows:
        va = stats_a.get(ka, float("nan"))
        vb = stats_b.get(kb, float("nan"))
        print(f"  {ka:<10}  {va:>12.6f}  {vb:>14.6f}")
    print(f"\n  Gap ({label_a.strip()} − {label_b.strip()})  :  {gap:+.6f}")


def _interpret_gap(train_gap: float, eval_gap: float) -> None:
    """Print a human-readable diagnosis of the train/eval gap comparison."""
    print()
    print("  DIAGNOSIS")
    print("  " + "-" * 54)
    if train_gap >= 0.5 and eval_gap < 0.15:
        print(f"  DECOUPLING DETECTED")
        print(f"  Train gap {train_gap:.3f} is large; eval gap {eval_gap:.3f} is small.")
        print(f"  The model learned to separate in-batch negatives but does not")
        print(f"  generalise to retrieving held-out items.")
        print(f"  Possible causes: in-batch negatives too easy, temperature too")
        print(f"  low, no hard negatives, or insufficient training epochs.")
    elif train_gap >= 0.5 and eval_gap >= 0.15:
        print(f"  BOTH GAPS HEALTHY")
        print(f"  Train gap {train_gap:.3f}  |  Eval gap {eval_gap:.3f}")
        print(f"  Training separation generalises to held-out retrieval.")
    elif train_gap < 0.2:
        print(f"  TRAINING COLLAPSE: train gap {train_gap:.3f} — model is not")
        print(f"  separating positives from in-batch negatives at all.")
    else:
        print(f"  Train gap {train_gap:.3f}  |  Eval gap {eval_gap:.3f}")
        print(f"  Partial separation — check individual percentiles above.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = _parse_args()
    ckpt_path = pathlib.Path(args.checkpoint).resolve()

    if not ckpt_path.exists():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")

    artifacts_dir = (
        pathlib.Path(args.artifacts_dir).resolve()
        if args.artifacts_dir
        else ckpt_path.parent.parent
    )
    if not artifacts_dir.exists():
        sys.exit(f"ERROR: artifacts directory not found: {artifacts_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  Score Distribution Diagnostic")
    print(f"  Checkpoint   : {ckpt_path.name}")
    print(f"  Artifacts dir: {artifacts_dir}")
    print(f"  Test path    : {args.test_path}")
    print(f"  Device       : {device}")
    print("=" * 62)
    print("\nLoading artifacts ...")

    with open(artifacts_dir / "vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)

    items_enc   = pd.read_parquet(artifacts_dir / "items_encoded.parquet")
    users_enc   = pd.read_parquet(artifacts_dir / "users_encoded.parquet")
    train_pairs = pd.read_parquet(artifacts_dir / "train_pairs.parquet")

    print(f"  items_encoded  : {items_enc.shape}")
    print(f"  users_encoded  : {users_enc.shape}")
    print(f"  train_pairs    : {train_pairs.shape}")

    print(f"\nLoading test split from {args.test_path} ...")
    test_df = pd.read_parquet(args.test_path)
    print(f"  test_df        : {test_df.shape}")

    # ── Build model and load checkpoint ───────────────────────────────────────
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

    print(f"\nLoading checkpoint: {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Epoch      : {ckpt.get('epoch', 'unknown')}")
    print(f"  Train loss : {ckpt.get('loss', float('nan')):.4f}")

    # ── Pre-build feature lookup arrays (shared by both parts) ────────────────
    user_cat_arr, user_dense_arr = _build_user_lookups(users_enc)
    item_cat_arr, item_dense_arr = _build_item_lookups(items_enc)

    sep  = "=" * 62
    thin = "-" * 62

    # ════════════════════════════════════════════════════════════
    # PART A — TRAINING SCORE DISTRIBUTION
    # ════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print(f"  PART A — TRAINING SCORE DISTRIBUTION")
    print(f"  {args.n_train_batches} batches × {args.train_batch_size:,} pairs  "
          f"(raw cosine sim, pre-temperature)")
    print(sep)

    train_stats = run_training_distribution(
        model         = model,
        train_pairs_df = train_pairs,
        user_cat_arr  = user_cat_arr,
        user_dense_arr = user_dense_arr,
        item_cat_arr  = item_cat_arr,
        item_dense_arr = item_dense_arr,
        device        = device,
        n_batches     = args.n_train_batches,
        batch_size    = args.train_batch_size,
        rng           = rng,
    )

    pos_stats  = {k[4:]: v for k, v in train_stats.items() if k.startswith("pos_")}
    neg_stats  = {k[4:]: v for k, v in train_stats.items() if k.startswith("neg_")}
    train_gap  = train_stats["gap"]

    print(f"\n  {thin}")
    print(f"  Total positive pairs : {train_stats['n_positive_pairs']:>10,}")
    print(f"  Total negative pairs : {train_stats['n_negative_pairs']:>10,}")
    print()
    _print_two_col(
        "positive (diag)", pos_stats,
        "negative (off-diag)", neg_stats,
        train_gap,
    )
    print()
    if train_gap >= 0.5:
        print(f"  OK : train gap {train_gap:.4f} — model separates positives "
              f"from in-batch negatives.")
    else:
        print(f"  WARNING : train gap {train_gap:.4f} is low — model may not be "
              f"learning useful separations.")
    print(f"  {thin}")

    # ════════════════════════════════════════════════════════════
    # PART B — EVALUATION SCORE DISTRIBUTION
    # ════════════════════════════════════════════════════════════
    print(f"\n{sep}")
    print(f"  PART B — EVALUATION SCORE DISTRIBUTION")
    print(f"  {args.n_eval_users} eval users  ×  {args.n_random_items} random items per user")
    print(sep)

    eval_stats = run_eval_distribution(
        model            = model,
        test_df          = test_df,
        users_encoded_df = users_enc,
        items_encoded_df = items_enc,
        train_pairs_df   = train_pairs,
        vocabs           = vocabs,
        device           = device,
        n_eval_users     = args.n_eval_users,
        n_random_items   = args.n_random_items,
        rng              = rng,
    )

    if not eval_stats:
        print("  Skipping eval stats — no eligible users found.")
    else:
        gt_stats   = {k[3:]: v for k, v in eval_stats.items() if k.startswith("gt_")}
        rand_stats = {k[5:]: v for k, v in eval_stats.items() if k.startswith("rand_")}
        eval_gap   = eval_stats["gap"]

        print(f"\n  {thin}")
        print(f"  Users scored         : {eval_stats['n_users_scored']:>10,}")
        print()
        _print_two_col(
            "GT items", gt_stats,
            "random items", rand_stats,
            eval_gap,
        )
        print()
        if eval_gap >= 0.15:
            print(f"  OK : eval gap {eval_gap:.4f} — model scores GT items above random.")
        else:
            print(f"  WARNING : eval gap {eval_gap:.4f} — GT items barely above random.")
        print(f"  {thin}")

    # ════════════════════════════════════════════════════════════
    # SUMMARY & DIAGNOSIS
    # ════════════════════════════════════════════════════════════
    if eval_stats:
        print(f"\n{sep}")
        print(f"  SUMMARY — {ckpt_path.name}")
        print(f"  {'':32}  {'Train':>8}  {'Eval':>8}")
        print(f"  {'-'*32}  {'-'*8}  {'-'*8}")
        print(f"  {'Positive/GT mean score':<32}  "
              f"{train_stats['pos_mean']:>8.4f}  {eval_stats['gt_mean']:>8.4f}")
        print(f"  {'Negative/random mean score':<32}  "
              f"{train_stats['neg_mean']:>8.4f}  {eval_stats['rand_mean']:>8.4f}")
        print(f"  {'Gap (pos/GT − neg/random)':<32}  "
              f"{train_gap:>8.4f}  {eval_gap:>8.4f}")
        print(f"  {'Ratio (eval gap / train gap)':<32}  "
              f"{'':>8}  "
              f"{eval_gap / train_gap:>8.4f}"
              if train_gap != 0 else "  (train gap is zero)")
        print(sep)

        _interpret_gap(train_gap, eval_gap)
        print()


if __name__ == "__main__":
    main()
