"""FAISS-based evaluation pipeline for the trained Two-Tower model.

Builds an item embedding index, retrieves top-k candidates per user,
filters training-seen items, and computes Recall@k, NDCG@k, Precision@k.
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch

from src.two_tower.data.dataset import build_full_item_tensors
from src.two_tower.models.two_tower import TwoTowerModel


# ── Diagnostic: user coverage through each filter stage ──────────────────────

def diagnose_user_coverage(
    test_df: pd.DataFrame,
    users_encoded_df: pd.DataFrame,
    vocabs: dict[str, Any],
) -> dict[str, int]:
    """Print a breakdown of where users are lost between test split and eval.

    Stages:
        1. Unique users with purchases in test_df.
        2. Of those, how many are in vocabs['user2idx'] (have training history).
        3. Of those, how many have a feature row in users_encoded_df.
        4. How many end up in the eligible evaluation pool.

    Args:
        test_df:          Raw test events DataFrame.
        users_encoded_df: DataFrame produced by FeatureBuilder for users.
        vocabs:           Vocab dict containing user2idx.

    Returns:
        Counts dict keyed by stage name, for programmatic use in callers.
    """
    user2idx = vocabs["user2idx"]

    relevant             = test_df[test_df["event_type"].isin({"cart", "purchase"})]
    unique_buyers        = set(relevant["user_id"].unique().tolist())
    buyers_in_user2idx   = {u for u in unique_buyers if u in user2idx}
    buyers_user_idxs     = {user2idx[u] for u in buyers_in_user2idx}

    encoded_user_idxs    = set(users_encoded_df["user_idx"].astype(int).tolist())
    encoded_user_ids     = set(users_encoded_df["user_id"].astype(int).tolist()) \
                           if "user_id" in users_encoded_df.columns else set()

    eligible_user_idxs   = buyers_user_idxs & encoded_user_idxs
    lost_no_history      = len(unique_buyers) - len(buyers_in_user2idx)
    lost_no_features     = len(buyers_user_idxs) - len(eligible_user_idxs)

    print("=" * 60)
    print("USER COVERAGE DIAGNOSTIC")
    print("=" * 60)
    print(f"  Step 1 | Unique users with cart/purchase in test split : {len(unique_buyers):>7,}")
    print(f"  Step 2 | Of those, in vocabs['user2idx']               : {len(buyers_in_user2idx):>7,}  "
          f"(lost: {lost_no_history:,} cold-start Feb users)")
    print(f"  Step 3 | Of those, in users_encoded_df                 : {len(eligible_user_idxs):>7,}  "
          f"(lost: {lost_no_features:,} users in vocab but missing feature row)")
    print(f"  Step 4 | Eligible eval pool (final)                    : {len(eligible_user_idxs):>7,}")
    if encoded_user_ids:
        print(f"  (users_encoded_df unique user_id count                 : {len(encoded_user_ids):>7,})")
    print(f"  (users_encoded_df unique user_idx count                : {len(encoded_user_idxs):>7,})")
    print("=" * 60)

    return {
        "unique_buyers":      len(unique_buyers),
        "buyers_in_user2idx": len(buyers_in_user2idx),
        "eligible_user_pool": len(eligible_user_idxs),
        "lost_no_history":    lost_no_history,
        "lost_no_features":   lost_no_features,
    }


# ── FAISS Index Builder ───────────────────────────────────────────────────────

def build_faiss_index(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    device: torch.device,
    trained_item_idxs: set | np.ndarray | None = None,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode items through the item tower and return L2-normalised embeddings.

    Items are sorted by item_idx ascending so that the returned arrays are
    aligned (row i of embeddings corresponds to item_idx_array[i]).

    Args:
        model:              Trained TwoTowerModel.
        items_encoded_df:   Full item feature DataFrame.
        device:             Torch device for inference.
        trained_item_idxs:  Optional set/array of item_idx values to include.
                            When provided, only items whose item_idx is in this
                            set are embedded — uninitialised items are excluded.
                            Pass None to embed all items (full-scale deployment).
        batch_size:         Number of items per forward pass.

    Returns:
        embeddings:     float32 ndarray, shape (n_indexed, 64), L2-normalised.
        item_idx_array: int64 ndarray, shape (n_indexed,) — item_idx per row.
    """
    n_total = len(items_encoded_df)

    if trained_item_idxs is not None:
        df = items_encoded_df[
            items_encoded_df["item_idx"].isin(trained_item_idxs)
        ].copy()
        n_trained = len(df)
        print(f"Building item index with {n_trained:,} trained items only "
              f"(filtered from {n_total:,} total)")
    else:
        df = items_encoded_df
        print(f"Building item index with all {n_total:,} items")

    item_cat_tensor, item_dense_tensor = build_full_item_tensors(df)
    n_items = item_cat_tensor.size(0)

    # item_idx values in ascending order (build_full_item_tensors sorts by item_idx)
    item_idx_array = item_cat_tensor[:, 0].numpy().astype(np.int64)

    all_embeddings: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            cat_batch   = item_cat_tensor[start : start + batch_size].to(device)
            dense_batch = item_dense_tensor[start : start + batch_size].to(device)
            emb = model.get_item_embeddings(cat_batch, dense_batch)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    faiss.normalize_L2(embeddings)

    print(f"Item index ready: {n_items:,} items, 64 dims")
    return embeddings, item_idx_array


# ── Ground Truth ──────────────────────────────────────────────────────────────

def build_ground_truth(
    test_df: pd.DataFrame,
    user2idx: dict[int, int],
    valid_user_idxs: set[int] | None = None,
) -> dict[int, set]:
    """Build per-user purchase ground truth from the test split.

    Filters applied in order:
        1. event_type == 'purchase'.
        2. user_id ∈ user2idx (i.e. has training history).
        3. If valid_user_idxs is given, user_idx ∈ valid_user_idxs
           (i.e. has a feature row in users_encoded_df).

    Each stage's remaining count is printed so silent drops are visible.

    Args:
        test_df:         Raw test events DataFrame.
        user2idx:        Mapping raw user_id → user_idx from training.
        valid_user_idxs: Optional set of user_idx values that have feature
                         rows in users_encoded_df. If None, step 3 is skipped.

    Returns:
        Dict mapping user_idx → set of product_ids (raw int) purchased in test.
    """
    relevant       = test_df[test_df["event_type"].isin({"cart", "purchase"})]
    relevant_users = set(relevant["user_id"].unique().tolist())
    print(f"Ground truth pipeline:")
    print(f"  {len(relevant_users):>7,} users have Feb cart+purchase events")

    gt_after_vocab: dict[int, set] = {}
    for user_id, grp in relevant.groupby("user_id"):
        if user_id not in user2idx:
            continue
        gt_after_vocab[user2idx[user_id]] = set(grp["product_id"].tolist())
    print(f"  {len(gt_after_vocab):>7,} of those are in user2idx "
          f"(have training history)")

    if valid_user_idxs is None:
        print(f"  {len(gt_after_vocab):>7,} will be evaluated "
              f"(feature-row check skipped)")
        return gt_after_vocab

    gt_final = {uidx: prods for uidx, prods in gt_after_vocab.items()
                if uidx in valid_user_idxs}
    print(f"  {len(gt_final):>7,} of those are in users_encoded "
          f"(have feature rows) — evaluating these")

    return gt_final


# ── Seen Items ────────────────────────────────────────────────────────────────

def build_seen_items(
    train_pairs_df: pd.DataFrame,
) -> dict[int, set]:
    """Build per-user set of item_idx values seen during training."""
    seen: dict[int, set] = {}
    for user_idx, grp in train_pairs_df.groupby("user_idx"):
        seen[int(user_idx)] = set(grp["item_idx"].tolist())
    return seen


# ── Full Evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    users_encoded_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    vocabs: dict[str, Any],
    device: torch.device,
    k: int = 10,
    batch_size: int = 512,
    n_faiss_candidates: int = 50,
    trained_item_idxs: set | np.ndarray | None = None,
) -> dict[str, Any]:
    """End-to-end retrieval evaluation using FAISS nearest-neighbour search.

    Only users that satisfy *all* of the following are evaluated:
      - Had a cart or purchase event in the test split.
      - Present in vocabs['user2idx'] (trained user embedding exists).
      - Present in users_encoded_df (real feature row, not a zero vector).

    The FAISS index is restricted to items that appeared in training when
    trained_item_idxs is provided (or auto-derived from train_pairs_df),
    removing noise from uninitialised item embeddings.
    """

    # ── Metric helpers ────────────────────────────────────────────────────────
    def recall_at_k(predicted: list, ground_truth: set, k: int) -> float:
        hits = len(set(predicted[:k]) & ground_truth)
        return hits / min(len(ground_truth), k)

    def ndcg_at_k(predicted: list, ground_truth: set, k: int) -> float:
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, item in enumerate(predicted[:k])
            if item in ground_truth
        )
        ideal_hits = min(len(ground_truth), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    def precision_at_k(predicted: list, ground_truth: set, k: int) -> float:
        return len(set(predicted[:k]) & ground_truth) / k

    # ── Step 0: Coverage diagnostic ───────────────────────────────────────────
    diagnose_user_coverage(test_df, users_encoded_df, vocabs)

    # ── Step 1: Build FAISS index (trained items only) ────────────────────────
    if trained_item_idxs is None:
        trained_item_idxs = set(train_pairs_df["item_idx"].unique().tolist())
        print(f"Auto-derived trained_item_idxs: {len(trained_item_idxs):,} items")

    embeddings, item_idx_array = build_faiss_index(
        model, items_encoded_df, device,
        trained_item_idxs=trained_item_idxs,
        batch_size=batch_size,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # ── Step 2: Ground truth (intersected with encoded users) ─────────────────
    valid_user_idxs: set[int] = set(users_encoded_df["user_idx"].astype(int).tolist())
    ground_truth = build_ground_truth(test_df, vocabs["user2idx"], valid_user_idxs)
    eval_users   = list(ground_truth.keys())
    print(f"Evaluating on {len(eval_users):,} users")

    # ── Step 3: Seen items ────────────────────────────────────────────────────
    seen_items = build_seen_items(train_pairs_df)

    # ── Step 4: User feature lookup arrays ────────────────────────────────────
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

    # Safety check: no eval user should fall outside the feature matrix
    if eval_users:
        max_eval_idx = max(eval_users)
        if max_eval_idx >= n_users:
            raise RuntimeError(
                f"eval user_idx {max_eval_idx} exceeds users_encoded_df max "
                f"user_idx {n_users - 1}; intersection filter failed."
            )

    # ── Step 5: Batch retrieval and metric computation ────────────────────────
    idx2item: dict[int, int] = vocabs["idx2item"]

    recall_scores:    list[float] = []
    ndcg_scores:      list[float] = []
    precision_scores: list[float] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_users), batch_size):
            batch_user_idxs = eval_users[start : start + batch_size]

            uid_tensor   = torch.tensor(batch_user_idxs, dtype=torch.long, device=device)
            cat_tensor   = torch.tensor(
                user_cat_arr[batch_user_idxs], dtype=torch.long, device=device
            )
            dense_tensor = torch.tensor(
                user_dense_arr[batch_user_idxs], dtype=torch.float32, device=device
            )

            user_embs = model.get_user_embedding(uid_tensor, cat_tensor, dense_tensor)
            user_embs_np = user_embs.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(user_embs_np)

            _, faiss_indices = index.search(user_embs_np, n_faiss_candidates)

            for i, user_idx in enumerate(batch_user_idxs):
                raw_positions = faiss_indices[i]

                recommended_product_ids: list[int] = []
                user_seen = seen_items.get(user_idx, set())

                for pos in raw_positions:
                    if pos < 0:
                        continue
                    iidx = int(item_idx_array[pos])
                    if iidx in user_seen:
                        continue
                    prod_id = idx2item.get(iidx)
                    if prod_id is None:
                        continue
                    recommended_product_ids.append(prod_id)

                gt = ground_truth[user_idx]
                recall_scores.append(recall_at_k(recommended_product_ids, gt, k))
                ndcg_scores.append(ndcg_at_k(recommended_product_ids, gt, k))
                precision_scores.append(precision_at_k(recommended_product_ids, gt, k))

    # ── Step 6: Aggregate ─────────────────────────────────────────────────────
    mean_recall    = float(np.mean(recall_scores))   if recall_scores    else 0.0
    mean_ndcg      = float(np.mean(ndcg_scores))     if ndcg_scores      else 0.0
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    n_eval_users   = len(eval_users)

    # ── Step 7: Print ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 38}")
    print(f"Evaluation Results (k={k})")
    print(f"{'═' * 38}")
    print(f"  Eval users          : {n_eval_users:,}")
    print(f"  Recall@{k:<3}          : {mean_recall:.4f}")
    print(f"  NDCG@{k:<5}          : {mean_ndcg:.4f}")
    print(f"  Precision@{k:<1}         : {mean_precision:.4f}")
    print(f"{'═' * 38}")

    return {
        f"recall@{k}":    mean_recall,
        f"ndcg@{k}":      mean_ndcg,
        f"precision@{k}": mean_precision,
        "n_eval_users":   n_eval_users,
    }
