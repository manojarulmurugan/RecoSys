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


# ── Per-user diagnostic printer ───────────────────────────────────────────────

def _print_per_user_diagnostics(
    eval_users:    list[int],
    gt_sizes:      list[int],
    recall_scores: list[float],
    seen_items:    dict[int, set],
    k:             int,
) -> None:
    """Print four per-user diagnostic blocks after the aggregate metrics.

    (1) Ground truth density  — distribution of how many GT items each user has.
    (2) Hit rate              — % of users with ≥1 correct item in top-k.
    (3) Recall distribution   — for users who *do* get a hit, spread of Recall@k.
    (4) Zero-hit profile      — avg GT size and training interactions for users
                                with zero hits (cold-start vs warm failure).

    Args:
        eval_users:    Ordered list of user_idx values that were evaluated.
        gt_sizes:      Per-user GT item counts, aligned with eval_users.
        recall_scores: Per-user Recall@k values, aligned with eval_users.
        seen_items:    Dict user_idx → set of item_idxs seen during training.
        k:             Recommendation list length (for display only).
    """
    if not eval_users:
        return

    gt_arr     = np.array(gt_sizes,       dtype=np.float32)
    recall_arr = np.array(recall_scores,  dtype=np.float32)
    hit_arr    = (recall_arr > 0)          # bool mask: True iff ≥1 hit

    n_users   = len(eval_users)
    n_hits    = int(hit_arr.sum())
    n_zero    = n_users - n_hits

    sep  = "─" * 52
    head = "═" * 52

    print(f"\n{head}")
    print(f"  PER-USER DIAGNOSTICS")
    print(head)

    # ── (1) Ground truth density ───────────────────────────────────────────
    n_one_gt  = int((gt_arr == 1).sum())
    pct_one   = 100.0 * n_one_gt / n_users

    print(f"\n  (1) GROUND TRUTH DENSITY  (GT items per eval user)")
    print(f"  {sep}")
    print(f"  {'min':<18}: {int(gt_arr.min()):>6}")
    print(f"  {'median':<18}: {np.median(gt_arr):>6.1f}")
    print(f"  {'mean':<18}: {gt_arr.mean():>6.2f}")
    print(f"  {'90th percentile':<18}: {np.percentile(gt_arr, 90):>6.1f}")
    print(f"  {'max':<18}: {int(gt_arr.max()):>6}")
    print(f"  {'users with 1 GT item':<18}: {n_one_gt:>6,}  ({pct_one:.1f}% of eval users)")

    # ── (2) Hit rate ────────────────────────────────────────────────────────
    hit_rate = 100.0 * n_hits / n_users

    print(f"\n  (2) HIT RATE  (users with ≥1 hit in top-{k})")
    print(f"  {sep}")
    print(f"  {'Users with ≥1 hit':<28}: {n_hits:>6,}  ({hit_rate:.1f}%)")
    print(f"  {'Users with zero hits':<28}: {n_zero:>6,}  ({100.0 - hit_rate:.1f}%)")
    print(f"  Note: Recall@{k} averages over all users; hit rate counts binary success.")

    # ── (3) Recall distribution — hit users only ────────────────────────────
    print(f"\n  (3) RECALL@{k} DISTRIBUTION  (hit users only, n={n_hits:,})")
    print(f"  {sep}")
    if n_hits == 0:
        print(f"  No users achieved any hits.")
    else:
        hit_recall = recall_arr[hit_arr]
        print(f"  {'mean':<18}: {hit_recall.mean():>6.4f}")
        print(f"  {'median':<18}: {np.median(hit_recall):>6.4f}")
        print(f"  {'75th percentile':<18}: {np.percentile(hit_recall, 75):>6.4f}")
        n_perfect = int((hit_recall >= (1.0 / k - 1e-9)).sum())
        print(f"  {'users @ max recall':<18}: {n_perfect:>6,}  "
              f"(Recall@{k} = {1.0/min(1, k):.2f}, i.e. ≥1 hit in 1 GT item)")

    # ── (4) Zero-hit user profile ────────────────────────────────────────────
    print(f"\n  (4) ZERO-HIT USER PROFILE  (n={n_zero:,})")
    print(f"  {sep}")
    if n_zero == 0:
        print(f"  All eval users achieved at least one hit.")
    else:
        zero_mask = ~hit_arr
        zero_gt   = gt_arr[zero_mask]

        # Count training interactions per zero-hit user
        zero_train_counts = np.array(
            [len(seen_items.get(u, set())) for u, is_zero
             in zip(eval_users, zero_mask) if is_zero],
            dtype=np.float32,
        )

        print(f"  {'Avg GT items':<30}: {zero_gt.mean():>6.2f}  "
              f"(all-user avg: {gt_arr.mean():.2f})")
        print(f"  {'Avg training interactions':<30}: {zero_train_counts.mean():>6.1f}")
        print(f"  {'Median training interactions':<30}: {np.median(zero_train_counts):>6.1f}")

        cold_start = int((zero_train_counts <= 5).sum())
        pct_cold   = 100.0 * cold_start / n_zero
        warm_fail  = n_zero - cold_start
        pct_warm   = 100.0 * warm_fail / n_zero
        print(f"  {'≤5 train interactions (cold)':<30}: {cold_start:>6,}  ({pct_cold:.1f}%)")
        print(f"  {'>5 train interactions (warm)':<30}: {warm_fail:>6,}  ({pct_warm:.1f}%)")
        if pct_warm > 50:
            print(f"\n  Note: majority of zero-hit users are WARM — the model has")
            print(f"  training signal for them but fails to retrieve their GT items.")
            print(f"  This points to model quality issues, not cold-start data gaps.")
        else:
            print(f"\n  Note: majority of zero-hit users are COLD-START — limited")
            print(f"  training signal is the primary driver of zero-hit failures.")

    print(f"\n{head}\n")


# ── Full Evaluation ───────────────────────────────────────────────────────────

_CENTROID_DIM  = 32
_CENTROID_COLS = [f"item_centroid_{i}" for i in range(_CENTROID_DIM)]


def _build_user_feature_arrays(
    users_encoded_df: pd.DataFrame,
    model: "TwoTowerModel",
) -> tuple[np.ndarray, np.ndarray]:
    """Build user_cat and user_dense numpy lookup arrays for all eval users.

    Auto-detects the user tower variant and adjusts the dense vector:
      - V1 UserTower (has dow_emb):        dense 6-dim
      - V2 UserTowerV2 / SequentialUserTower (no dow_emb, no centroid cols):
                                            dense 8-dim (+ sin/cos DOW)
      - V3 UserTowerV3 (no dow_emb, centroid cols present in df):
                                            dense 40-dim (8 + 32 centroid)

    Returns:
        user_cat_arr:   int64 array (n_users, 4)
        user_dense_arr: float32 array (n_users, 6 | 8 | 40)
    """
    n_users = int(users_encoded_df["user_idx"].max()) + 1

    user_cat_arr = np.zeros((n_users, 4), dtype=np.int64)
    user_cat_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
        ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
    ].values.astype(np.int64)

    base_dense = users_encoded_df[
        ["log_total_events", "months_active", "purchase_rate", "cart_rate",
         "log_n_sessions", "avg_purchase_price_scaled"]
    ].values.astype(np.float32)

    use_v2_user = not hasattr(model.user_tower, "dow_emb")
    use_centroid = (
        use_v2_user
        and all(c in users_encoded_df.columns for c in _CENTROID_COLS)
    )

    if use_centroid:
        # V3: 8-dim V2 features + 32-dim item centroid
        dow_vals = users_encoded_df["preferred_dow"].values.astype(np.float32)
        sin_dow  = np.sin(2.0 * np.pi * dow_vals / 7.0).reshape(-1, 1)
        cos_dow  = np.cos(2.0 * np.pi * dow_vals / 7.0).reshape(-1, 1)
        centroid = users_encoded_df[_CENTROID_COLS].values.astype(np.float32)
        dense_to_assign = np.hstack([base_dense, sin_dow, cos_dow, centroid])
        user_dense_arr  = np.zeros((n_users, 40), dtype=np.float32)
    elif use_v2_user:
        # V2: sin/cos DOW in dense (8-dim)
        dow_vals = users_encoded_df["preferred_dow"].values.astype(np.float32)
        sin_dow  = np.sin(2.0 * np.pi * dow_vals / 7.0).reshape(-1, 1)
        cos_dow  = np.cos(2.0 * np.pi * dow_vals / 7.0).reshape(-1, 1)
        dense_to_assign = np.hstack([base_dense, sin_dow, cos_dow])
        user_dense_arr  = np.zeros((n_users, 8), dtype=np.float32)
    else:
        # V1: plain 6-dim dense
        dense_to_assign = base_dense
        user_dense_arr  = np.zeros((n_users, 6), dtype=np.float32)

    user_dense_arr[users_encoded_df["user_idx"].values] = dense_to_assign
    return user_cat_arr, user_dense_arr


def evaluate(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    users_encoded_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    vocabs: dict[str, Any],
    device: torch.device,
    batch_size: int = 512,
    n_faiss_candidates: int = 100,
    trained_item_idxs: set | np.ndarray | None = None,
    user_seq_arr: np.ndarray | None = None,
) -> dict[str, Any]:
    """End-to-end retrieval evaluation using FAISS nearest-neighbour search.

    Always computes Recall@10, NDCG@10, Recall@20, and NDCG@20 from a single
    FAISS search pass.  FAISS retrieves `n_faiss_candidates` neighbours per
    user; seen-item filtering is applied once; metrics at both cutoffs are
    derived from the same filtered list — no second forward pass or index
    search is needed.

    Only users that satisfy *all* of the following are evaluated:
      - Had a cart or purchase event in the test split.
      - Present in vocabs['user2idx'] (trained user embedding exists).
      - Present in users_encoded_df (real feature row, not a zero vector).

    The FAISS index is restricted to items that appeared in training when
    trained_item_idxs is provided (or auto-derived from train_pairs_df),
    removing noise from uninitialised item embeddings.

    Returns:
        Dict with keys: recall_10, ndcg_10, recall_20, ndcg_20,
        n_eval_users, recommendations.
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
    user_cat_arr, user_dense_arr = _build_user_feature_arrays(users_encoded_df, model)
    n_users = user_cat_arr.shape[0]

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

    recall_scores_10: list[float] = []
    ndcg_scores_10:   list[float] = []
    recall_scores_20: list[float] = []
    ndcg_scores_20:   list[float] = []
    gt_sizes:         list[int]   = []   # per-user GT item count, aligned with eval_users
    all_recommendations: dict[int, list[int]] = {}

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

            seq_tensor: torch.Tensor | None = None
            if user_seq_arr is not None:
                seq_tensor = torch.tensor(
                    user_seq_arr[batch_user_idxs], dtype=torch.long, device=device
                )

            user_embs = model.get_user_embedding(
                uid_tensor, cat_tensor, dense_tensor, user_seq=seq_tensor
            )
            user_embs_np = user_embs.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(user_embs_np)

            _, faiss_indices = index.search(user_embs_np, n_faiss_candidates)

            for i, user_idx in enumerate(batch_user_idxs):
                raw_positions = faiss_indices[i]

                recommended_item_idxs:  list[int] = []
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
                    recommended_item_idxs.append(iidx)
                    recommended_product_ids.append(prod_id)

                all_recommendations[user_idx] = recommended_item_idxs[:20]

                gt = ground_truth[user_idx]
                gt_sizes.append(len(gt))
                recall_scores_10.append(recall_at_k(recommended_product_ids, gt, 10))
                ndcg_scores_10.append(ndcg_at_k(recommended_product_ids, gt, 10))
                recall_scores_20.append(recall_at_k(recommended_product_ids, gt, 20))
                ndcg_scores_20.append(ndcg_at_k(recommended_product_ids, gt, 20))

    # ── Step 6: Aggregate ─────────────────────────────────────────────────────
    mean_recall_10 = float(np.mean(recall_scores_10)) if recall_scores_10 else 0.0
    mean_ndcg_10   = float(np.mean(ndcg_scores_10))   if ndcg_scores_10   else 0.0
    mean_recall_20 = float(np.mean(recall_scores_20)) if recall_scores_20 else 0.0
    mean_ndcg_20   = float(np.mean(ndcg_scores_20))   if ndcg_scores_20   else 0.0
    n_eval_users   = len(eval_users)

    # ── Step 7: Print aggregate metrics ──────────────────────────────────────
    print(f"\n{'═' * 44}")
    print(f"Evaluation Results")
    print(f"{'═' * 44}")
    print(f"  Eval users          : {n_eval_users:,}")
    print(f"  Recall@10           : {mean_recall_10:.4f}")
    print(f"  NDCG@10             : {mean_ndcg_10:.4f}")
    print(f"  Recall@20           : {mean_recall_20:.4f}")
    print(f"  NDCG@20             : {mean_ndcg_20:.4f}")
    print(f"{'═' * 44}")

    # ── Step 8: Per-user diagnostics (based on @10) ────────────────────────
    _print_per_user_diagnostics(
        eval_users    = eval_users,
        gt_sizes      = gt_sizes,
        recall_scores = recall_scores_10,
        seen_items    = seen_items,
        k             = 10,
    )

    return {
        "recall_10":       mean_recall_10,
        "ndcg_10":         mean_ndcg_10,
        "recall_20":       mean_recall_20,
        "ndcg_20":         mean_ndcg_20,
        "n_eval_users":    n_eval_users,
        "recommendations": all_recommendations,
    }


# ── Stratified Evaluation ─────────────────────────────────────────────────────

def evaluate_stratified(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    users_encoded_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    vocabs: dict[str, Any],
    device: torch.device,
    batch_size: int = 512,
    n_faiss_candidates: int = 100,
    trained_item_idxs: set | np.ndarray | None = None,
    user_seq_arr: np.ndarray | None = None,
) -> dict[str, Any]:
    """End-to-end retrieval evaluation broken down by training-interaction cohort.

    Runs a single FAISS search pass (identical to evaluate) and partitions
    results into three cohorts defined by each user's number of training
    interactions:

        Cold   :  3 – 10  interactions
        Medium : 11 – 50  interactions
        Warm   : 51+       interactions

    For each cohort and overall, reports:
        Recall@10, NDCG@10, Recall@20, NDCG@20, hit rate (% users ≥1 hit @10),
        and cohort size.

    The FAISS index and ground-truth dict are built once — no per-cohort rebuild.

    Args:
        model:              Trained TwoTowerModel.
        items_encoded_df:   Full item feature DataFrame.
        users_encoded_df:   User feature DataFrame from FeatureBuilder.
        test_df:            Raw test-split events DataFrame.
        train_pairs_df:     Training pairs with columns [user_idx, item_idx].
                            Used to compute per-user training interaction counts
                            and to build the seen-items filter.
        vocabs:             Vocab dict containing user2idx / idx2item.
        device:             Torch device for inference.
        batch_size:         Users and items per forward pass.
        n_faiss_candidates: Neighbours retrieved from FAISS before filtering.
        trained_item_idxs:  Optional set of item_idx values to include in the
                            index. Auto-derived from train_pairs_df if None.

    Returns:
        Dict with keys:
            'overall'  → {recall_10, ndcg_10, recall_20, ndcg_20,
                          hit_rate_10, n_users}
            'cold'     → same structure
            'medium'   → same structure
            'warm'     → same structure
            'recommendations' → {user_idx: [item_idx, …]}
    """

    # ── Cohort boundaries ─────────────────────────────────────────────────────
    COHORTS: list[tuple[str, int, int]] = [
        ("cold",   3,  10),
        ("medium", 11, 50),
        ("warm",   51, int(1e9)),
    ]

    # ── Metric helpers (mirrors evaluate) ─────────────────────────────────────
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

    # ── Step 0: Training interaction counts ───────────────────────────────────
    train_interaction_counts: dict[int, int] = (
        train_pairs_df.groupby("user_idx").size().to_dict()
    )

    # ── Step 1: Build FAISS index once ────────────────────────────────────────
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

    # ── Step 2: Ground truth ──────────────────────────────────────────────────
    valid_user_idxs: set[int] = set(users_encoded_df["user_idx"].astype(int).tolist())
    ground_truth = build_ground_truth(test_df, vocabs["user2idx"], valid_user_idxs)
    eval_users   = list(ground_truth.keys())
    print(f"Evaluating on {len(eval_users):,} users")

    # ── Step 3: Seen items ────────────────────────────────────────────────────
    seen_items = build_seen_items(train_pairs_df)

    # ── Step 4: User feature lookup arrays ────────────────────────────────────
    user_cat_arr, user_dense_arr = _build_user_feature_arrays(users_encoded_df, model)

    # ── Step 5: Inference — single pass over all eval users ───────────────────
    idx2item: dict[int, int] = vocabs["idx2item"]

    # Per-user storage (aligned with eval_users ordering)
    per_user_recall_10: list[float] = []
    per_user_ndcg_10:   list[float] = []
    per_user_recall_20: list[float] = []
    per_user_ndcg_20:   list[float] = []
    all_recommendations: dict[int, list[int]] = {}

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

            seq_tensor: torch.Tensor | None = None
            if user_seq_arr is not None:
                seq_tensor = torch.tensor(
                    user_seq_arr[batch_user_idxs], dtype=torch.long, device=device
                )

            user_embs    = model.get_user_embedding(
                uid_tensor, cat_tensor, dense_tensor, user_seq=seq_tensor
            )
            user_embs_np = user_embs.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(user_embs_np)

            _, faiss_indices = index.search(user_embs_np, n_faiss_candidates)

            for i, user_idx in enumerate(batch_user_idxs):
                raw_positions = faiss_indices[i]

                recommended_item_idxs:   list[int] = []
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
                    recommended_item_idxs.append(iidx)
                    recommended_product_ids.append(prod_id)

                all_recommendations[user_idx] = recommended_item_idxs[:20]

                gt = ground_truth[user_idx]
                per_user_recall_10.append(recall_at_k(recommended_product_ids, gt, 10))
                per_user_ndcg_10.append(ndcg_at_k(recommended_product_ids, gt, 10))
                per_user_recall_20.append(recall_at_k(recommended_product_ids, gt, 20))
                per_user_ndcg_20.append(ndcg_at_k(recommended_product_ids, gt, 20))

    # ── Step 6: Partition into cohorts ────────────────────────────────────────
    def _cohort_metrics(mask: np.ndarray) -> dict[str, float]:
        """Aggregate metrics for the users selected by boolean mask."""
        n = int(mask.sum())
        if n == 0:
            return {
                "recall_10": 0.0, "ndcg_10": 0.0,
                "recall_20": 0.0, "ndcg_20": 0.0,
                "hit_rate_10": 0.0, "n_users": 0,
            }
        r10 = np.array(per_user_recall_10, dtype=np.float32)[mask]
        n10 = np.array(per_user_ndcg_10,   dtype=np.float32)[mask]
        r20 = np.array(per_user_recall_20, dtype=np.float32)[mask]
        n20 = np.array(per_user_ndcg_20,   dtype=np.float32)[mask]
        return {
            "recall_10":   float(r10.mean()),
            "ndcg_10":     float(n10.mean()),
            "recall_20":   float(r20.mean()),
            "ndcg_20":     float(n20.mean()),
            "hit_rate_10": float((r10 > 0).mean()) * 100.0,
            "n_users":     n,
        }

    # Build a per-user training count array aligned with eval_users
    train_counts_arr = np.array(
        [train_interaction_counts.get(u, 0) for u in eval_users], dtype=np.int64
    )

    overall_mask = np.ones(len(eval_users), dtype=bool)
    results: dict[str, Any] = {
        "overall":         _cohort_metrics(overall_mask),
        "recommendations": all_recommendations,
    }
    for name, lo, hi in COHORTS:
        mask = (train_counts_arr >= lo) & (train_counts_arr <= hi)
        results[name] = _cohort_metrics(mask)

    # ── Step 7: Print side-by-side cohort table ───────────────────────────────
    col_names = ["overall"] + [c[0] for c in COHORTS]
    col_w     = 12   # width of each data column

    metrics_rows: list[tuple[str, str]] = [
        ("n_users",    "N users"),
        ("recall_10",  "Recall@10"),
        ("ndcg_10",    "NDCG@10"),
        ("recall_20",  "Recall@20"),
        ("ndcg_20",    "NDCG@20"),
        ("hit_rate_10", "Hit rate@10 %"),
    ]
    cohort_labels = {
        "overall": "Overall",
        "cold":    "Cold (3-10)",
        "medium":  "Med (11-50)",
        "warm":    "Warm (51+)",
    }
    label_w = max(len(v) for _, v in metrics_rows) + 2

    sep   = "═" * (label_w + (col_w + 2) * len(col_names) + 1)
    thin  = "─" * (label_w + (col_w + 2) * len(col_names) + 1)

    def _fmt(key: str, val: float | int) -> str:
        if key == "n_users":
            return f"{int(val):>{col_w},}"
        if key == "hit_rate_10":
            return f"{val:>{col_w}.1f}"
        return f"{val:>{col_w}.4f}"

    print(f"\n{sep}")
    print(f"  STRATIFIED EVALUATION RESULTS")
    print(sep)

    # Header row
    header = f"  {'Metric':<{label_w}}"
    for c in col_names:
        header += f"  {cohort_labels[c]:>{col_w}}"
    print(header)
    print(thin)

    # Cohort size row (interactions range)
    range_row = f"  {'Interactions':<{label_w}}"
    ranges = {"overall": "all", "cold": "3–10", "medium": "11–50", "warm": "51+"}
    for c in col_names:
        range_row += f"  {ranges[c]:>{col_w}}"
    print(range_row)
    print(thin)

    # Metric rows
    for key, label in metrics_rows:
        row = f"  {label:<{label_w}}"
        for c in col_names:
            val = results[c][key]
            row += f"  {_fmt(key, val)}"
        print(row)

    print(sep)

    return results
