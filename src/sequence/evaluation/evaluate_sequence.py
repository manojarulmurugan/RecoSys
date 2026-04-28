"""FAISS-based evaluation for sequence models (V7 / V8).

Key design difference from the two-tower evaluator:
─────────────────────────────────────────────────────────────────────────────
Sequential recommendation solves *next-item prediction*, not *novel item
discovery*.  The two-tower pipeline filters every item the user touched in
training (train_pairs_df) out of the recommendation list.  This is correct
for Feb test evaluation (items seen in Oct–Jan 24 vs Feb purchases — good
temporal gap) but **kills the validation signal for the Jan 25–31 window**:
a user who carts something on Jan 27 almost certainly viewed/carted it
before Jan 25 → it is in train_pairs → it is filtered from recs →
ground-truth items can never appear → Recall = 0.0000 no matter how good
the model is.

The fix: ``filter_seen: bool`` parameter.
  - ``filter_seen=False``  (default for validation runs)
    No filtering.  Measures raw next-item prediction quality.
    Use this for the periodic val eval (Jan 25-31).
  - ``filter_seen=True``   (use for the final Feb test eval)
    Mirrors the V1–V6 two-tower protocol so numbers are comparable.
    For the Feb window the overlap between training history and ground
    truth is lower (5+ week gap) so some hits survive the filter.

Both public functions expose a ``filter_seen`` parameter.  The train scripts
set it explicitly for each call so the log is unambiguous.
─────────────────────────────────────────────────────────────────────────────

Two public functions:

    ``evaluate_sequence(...)``            — overall Recall@10/20, NDCG@10/20.
    ``evaluate_sequence_stratified(...)`` — same + Cold/Med/Warm cohorts.
"""

from __future__ import annotations

from typing import Any, Protocol

import faiss
import numpy as np
import pandas as pd
import torch

from src.two_tower.evaluation.evaluate import build_seen_items


# ── Sequence model interface ──────────────────────────────────────────────────

class SequenceModel(Protocol):
    """Minimal interface required of any sequence model."""

    def encode_sequence(
        self,
        item_seq:  torch.Tensor,
        event_seq: torch.Tensor,
    ) -> torch.Tensor: ...

    def get_item_embeddings(self) -> torch.Tensor: ...

    def eval(self) -> "SequenceModel": ...


# ── Metric helpers (inlined from two-tower evaluator) ─────────────────────────

def _recall_at_k(predicted: list, ground_truth: set, k: int) -> float:
    hits = len(set(predicted[:k]) & ground_truth)
    return hits / min(len(ground_truth), k)


def _ndcg_at_k(predicted: list, ground_truth: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(predicted[:k])
        if item in ground_truth
    )
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Ground-truth builder (idx-space) ──────────────────────────────────────────

def _build_ground_truth_idx(
    targets_df: pd.DataFrame,
) -> dict[int, set]:
    """user_idx → set of item_idx for cart + purchase rows."""
    if "user_idx" not in targets_df.columns or "item_idx" not in targets_df.columns:
        raise ValueError(
            "targets_df must have user_idx and item_idx columns; "
            f"got {list(targets_df.columns)}"
        )

    df = targets_df
    if "event_type" in df.columns:
        df = df[df["event_type"].isin({"cart", "purchase"})]

    gt: dict[int, set] = {}
    for u, grp in df.groupby("user_idx"):
        gt[int(u)] = set(int(i) for i in grp["item_idx"].tolist())
    return gt


# ── Seen-item filter diagnostic ───────────────────────────────────────────────

def _print_filter_diagnostic(
    ground_truth: dict[int, set],
    seen_items:   dict[int, set],
    eval_users:   list[int],
    label:        str,
) -> None:
    """Show what fraction of GT items would be hidden by the seen-item filter.

    If this number is close to 100% for the val split (Jan 25–31) it means
    the filter would make Recall impossible — users re-interacted with the
    same items in the val window that they already had in training.
    """
    total_gt  = 0
    total_hit = 0
    for u in eval_users:
        gt   = ground_truth.get(u, set())
        seen = seen_items.get(u, set())
        total_gt  += len(gt)
        total_hit += len(gt & seen)
    pct = 100.0 * total_hit / max(total_gt, 1)
    print(
        f"  [filter diag | {label}] "
        f"{total_hit:,}/{total_gt:,} GT items are in seen_items ({pct:.1f}%) "
        f"— these would be invisible if filter_seen=True"
    )
    if pct > 80:
        print(
            f"  ⚠  > 80 % of GT items are filtered. Using filter_seen=True "
            f"for this eval window would produce near-zero recall even with a "
            f"perfect model.  Use filter_seen=False for this window."
        )


# ── FAISS index from item-embedding table ─────────────────────────────────────

def _build_item_faiss_index(
    model:             SequenceModel,
    n_items:           int,
    trained_item_idxs: set | np.ndarray | None,
    device:            torch.device,
) -> tuple[np.ndarray, np.ndarray, faiss.Index]:
    """Build a FAISS IndexFlatIP over trained item embeddings.

    Returns:
        embeddings:      (n_indexed, D) float32, L2-normalised.
        item_idx_array:  (n_indexed,)   int64 — item_idx per row.
        index:           Ready FAISS index.
    """
    model.eval()
    with torch.no_grad():
        all_embs = model.get_item_embeddings().detach().cpu().numpy().astype(np.float32)
    if all_embs.shape[0] != n_items:
        raise RuntimeError(
            f"Item embedding count {all_embs.shape[0]} != expected n_items "
            f"{n_items}; check vocab vs. model n_items config."
        )

    if trained_item_idxs is None:
        keep_mask = np.ones(n_items, dtype=bool)
        keep_mask[0] = False  # never index PAD
    else:
        if isinstance(trained_item_idxs, set):
            trained_item_idxs = np.fromiter(trained_item_idxs, dtype=np.int64)
        keep_mask = np.zeros(n_items, dtype=bool)
        keep_mask[np.asarray(trained_item_idxs, dtype=np.int64)] = True
        keep_mask[0] = False

    item_idx_array = np.where(keep_mask)[0].astype(np.int64)
    embeddings     = all_embs[item_idx_array].copy()
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(
        f"  FAISS index : {embeddings.shape[0]:,} items × {embeddings.shape[1]} dims "
        f"(filtered from {n_items:,} catalog items)"
    )
    return embeddings, item_idx_array, index


# ── Per-user retrieval ────────────────────────────────────────────────────────

def _retrieve_recommendations(
    model:               SequenceModel,
    item_seq_arr:        np.ndarray,
    event_seq_arr:       np.ndarray,
    eval_users:          list[int],
    seen_items:          dict[int, set],
    index:               faiss.Index,
    item_idx_array:      np.ndarray,
    device:              torch.device,
    batch_size:          int,
    n_faiss_candidates:  int,
    filter_seen:         bool,
) -> dict[int, list[int]]:
    """Encode users, run FAISS, and return per-user top-20 item_idx lists.

    When ``filter_seen=False``, ``seen_items`` is not consulted and the
    raw FAISS result (top-20 by score) is returned.
    When ``filter_seen=True``, items whose item_idx ∈ ``seen_items[u]``
    are removed before capping at 20.

    Returns:
        recs_idx_by_user: user_idx → list[item_idx], length ≤ 20.
    """
    recs_idx_by_user: dict[int, list[int]] = {}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(eval_users), batch_size):
            batch_user_idxs = eval_users[start : start + batch_size]

            item_batch = torch.tensor(
                item_seq_arr[batch_user_idxs], dtype=torch.long, device=device
            )
            event_batch = torch.tensor(
                event_seq_arr[batch_user_idxs], dtype=torch.long, device=device
            )

            user_embs    = model.encode_sequence(item_batch, event_batch)
            user_embs_np = user_embs.detach().cpu().numpy().astype(np.float32)
            faiss.normalize_L2(user_embs_np)

            _, faiss_indices = index.search(user_embs_np, n_faiss_candidates)

            for i, u in enumerate(batch_user_idxs):
                positions = faiss_indices[i]

                if filter_seen:
                    user_seen = seen_items.get(u, set())
                    kept: list[int] = []
                    for pos in positions:
                        if pos < 0:
                            continue
                        iidx = int(item_idx_array[pos])
                        if iidx not in user_seen:
                            kept.append(iidx)
                    recs_idx_by_user[u] = kept[:20]
                else:
                    raw: list[int] = [
                        int(item_idx_array[pos])
                        for pos in positions
                        if pos >= 0
                    ]
                    recs_idx_by_user[u] = raw[:20]

    return recs_idx_by_user


# ── Main: overall evaluation ──────────────────────────────────────────────────

def evaluate_sequence(
    model:               SequenceModel,
    item_seq_arr:        np.ndarray,
    event_seq_arr:       np.ndarray,
    eval_targets_df:     pd.DataFrame,
    train_pairs_df:      pd.DataFrame,
    n_items:             int,
    device:              torch.device,
    batch_size:          int = 512,
    n_faiss_candidates:  int = 100,
    trained_item_idxs:   set | np.ndarray | None = None,
    label:               str = "eval",
    filter_seen:         bool = False,
) -> dict[str, Any]:
    """End-to-end retrieval evaluation for a sequence model.

    Args:
        model:              Sequence model with ``encode_sequence`` +
                            ``get_item_embeddings`` interface.
        item_seq_arr:       (n_users, max_seq_len) int64 padded items.
                            Use train-window array for val eval, full-train
                            array for test eval.
        event_seq_arr:      (n_users, max_seq_len) int64 padded event types.
        eval_targets_df:    (user_idx, item_idx [, event_type]) ground-truth
                            pairs.  cart + purchase only if event_type present.
        train_pairs_df:     (user_idx, item_idx) training interactions.
                            Used to derive the FAISS index item set and (when
                            filter_seen=True) the seen-item exclusion set.
        n_items:            Catalog size including PAD token 0.
        device:             Torch device for inference.
        batch_size:         Users per forward + search batch.
        n_faiss_candidates: Top-N retrieved before optional seen filtering.
                            Raise this (e.g. 200) when filter_seen=True so
                            enough candidates survive the filter.
        trained_item_idxs:  Optional explicit set of item_idx to index.
                            Auto-derived from train_pairs_df if None.
        label:              Short tag printed in headers (e.g. 'val', 'test').
        filter_seen:        If False (default), return the raw top-20 FAISS
                            results — correct for next-item prediction eval.
                            If True, remove items seen in train_pairs_df
                            before capping at 20 — matches V1–V6 protocol.

    Returns:
        Dict: recall_10, ndcg_10, recall_20, ndcg_20, n_eval_users,
        recommendations.
    """
    filter_tag = "filtered" if filter_seen else "unfiltered"
    print(f"\n  ── {label.upper()} EVALUATION  [{filter_tag}] ──")

    # ── Step 1: FAISS index ────────────────────────────────────────────────
    if trained_item_idxs is None:
        trained_item_idxs = set(train_pairs_df["item_idx"].unique().tolist())
        print(f"  Auto-derived trained_item_idxs: {len(trained_item_idxs):,} items")

    _, item_idx_array, index = _build_item_faiss_index(
        model, n_items, trained_item_idxs, device
    )

    # ── Step 2: Ground truth ───────────────────────────────────────────────
    ground_truth = _build_ground_truth_idx(eval_targets_df)
    n_users_arr  = item_seq_arr.shape[0]
    eval_users   = [u for u in ground_truth.keys() if 0 <= u < n_users_arr]
    dropped      = len(ground_truth) - len(eval_users)
    if dropped:
        print(f"  Dropped {dropped:,} users out of seq array bounds (cold/OOV)")
    print(f"  Eval users : {len(eval_users):,}")

    # ── Step 3: Seen-item filter (build even when filter_seen=False to log diag)
    seen_items = build_seen_items(train_pairs_df)
    _print_filter_diagnostic(ground_truth, seen_items, eval_users, label)

    # ── Step 4: Encode + retrieve ──────────────────────────────────────────
    recs_idx_by_user = _retrieve_recommendations(
        model              = model,
        item_seq_arr       = item_seq_arr,
        event_seq_arr      = event_seq_arr,
        eval_users         = eval_users,
        seen_items         = seen_items,
        index              = index,
        item_idx_array     = item_idx_array,
        device             = device,
        batch_size         = batch_size,
        n_faiss_candidates = n_faiss_candidates,
        filter_seen        = filter_seen,
    )

    # ── Step 5: Metrics ───────────────────────────────────────────────────
    recall_10: list[float] = []
    ndcg_10:   list[float] = []
    recall_20: list[float] = []
    ndcg_20:   list[float] = []

    for u in eval_users:
        recs = recs_idx_by_user.get(u, [])
        gt   = ground_truth[u]
        if not gt:
            continue
        recall_10.append(_recall_at_k(recs, gt, 10))
        ndcg_10  .append(_ndcg_at_k  (recs, gt, 10))
        recall_20.append(_recall_at_k(recs, gt, 20))
        ndcg_20  .append(_ndcg_at_k  (recs, gt, 20))

    mean_r10 = float(np.mean(recall_10)) if recall_10 else 0.0
    mean_n10 = float(np.mean(ndcg_10  )) if ndcg_10   else 0.0
    mean_r20 = float(np.mean(recall_20)) if recall_20 else 0.0
    mean_n20 = float(np.mean(ndcg_20  )) if ndcg_20   else 0.0

    sep = "═" * 48
    print(f"\n  {sep}")
    print(f"  {label.upper()} RESULTS  [{filter_tag}]")
    print(f"  {sep}")
    print(f"    Eval users : {len(eval_users):,}")
    print(f"    Recall@10  : {mean_r10:.4f}")
    print(f"    NDCG@10    : {mean_n10:.4f}")
    print(f"    Recall@20  : {mean_r20:.4f}")
    print(f"    NDCG@20    : {mean_n20:.4f}")
    print(f"  {sep}")

    return {
        "recall_10":       mean_r10,
        "ndcg_10":         mean_n10,
        "recall_20":       mean_r20,
        "ndcg_20":         mean_n20,
        "n_eval_users":    len(eval_users),
        "recommendations": recs_idx_by_user,
        "filter_seen":     filter_seen,
    }


# ── Stratified evaluation ─────────────────────────────────────────────────────

_COHORTS: list[tuple[str, int, int]] = [
    ("cold",   3,  10),
    ("medium", 11, 50),
    ("warm",   51, int(1e9)),
]


def evaluate_sequence_stratified(
    model:               SequenceModel,
    item_seq_arr:        np.ndarray,
    event_seq_arr:       np.ndarray,
    eval_targets_df:     pd.DataFrame,
    train_pairs_df:      pd.DataFrame,
    n_items:             int,
    device:              torch.device,
    batch_size:          int = 512,
    n_faiss_candidates:  int = 100,
    trained_item_idxs:   set | np.ndarray | None = None,
    label:               str = "eval",
    filter_seen:         bool = False,
) -> dict[str, Any]:
    """Same as ``evaluate_sequence`` plus Cold/Med/Warm cohort breakdown.

    Cohort = number of rows in ``train_pairs_df`` per ``user_idx``:
        Cold   :  3 – 10
        Medium : 11 – 50
        Warm   : 51 +

    ``filter_seen`` has the same semantics as in ``evaluate_sequence``.
    Pass ``filter_seen=False`` for val (Jan 25–31) and ``filter_seen=True``
    for the final Feb test eval.

    Returns:
        Dict keyed by ``overall``, ``cold``, ``medium``, ``warm``,
        ``recommendations``, ``filter_seen``.
    """
    filter_tag = "filtered" if filter_seen else "unfiltered"
    print(f"\n  ── {label.upper()} STRATIFIED EVALUATION  [{filter_tag}] ──")

    # ── Step 1: FAISS + ground truth ──────────────────────────────────────
    if trained_item_idxs is None:
        trained_item_idxs = set(train_pairs_df["item_idx"].unique().tolist())
        print(f"  Auto-derived trained_item_idxs: {len(trained_item_idxs):,} items")

    _, item_idx_array, index = _build_item_faiss_index(
        model, n_items, trained_item_idxs, device
    )

    ground_truth = _build_ground_truth_idx(eval_targets_df)
    n_users_arr  = item_seq_arr.shape[0]
    eval_users   = [u for u in ground_truth.keys() if 0 <= u < n_users_arr]
    dropped      = len(ground_truth) - len(eval_users)
    if dropped:
        print(f"  Dropped {dropped:,} users out of seq array bounds (cold/OOV)")
    print(f"  Eval users : {len(eval_users):,}")

    seen_items = build_seen_items(train_pairs_df)
    _print_filter_diagnostic(ground_truth, seen_items, eval_users, label)

    # ── Step 2: Retrieve for all users ────────────────────────────────────
    recs_idx_by_user = _retrieve_recommendations(
        model              = model,
        item_seq_arr       = item_seq_arr,
        event_seq_arr      = event_seq_arr,
        eval_users         = eval_users,
        seen_items         = seen_items,
        index              = index,
        item_idx_array     = item_idx_array,
        device             = device,
        batch_size         = batch_size,
        n_faiss_candidates = n_faiss_candidates,
        filter_seen        = filter_seen,
    )

    # ── Step 3: Per-user metrics ──────────────────────────────────────────
    per_r10 = np.zeros(len(eval_users), dtype=np.float32)
    per_n10 = np.zeros(len(eval_users), dtype=np.float32)
    per_r20 = np.zeros(len(eval_users), dtype=np.float32)
    per_n20 = np.zeros(len(eval_users), dtype=np.float32)
    for i, u in enumerate(eval_users):
        recs = recs_idx_by_user.get(u, [])
        gt   = ground_truth[u]
        if not gt:
            continue
        per_r10[i] = _recall_at_k(recs, gt, 10)
        per_n10[i] = _ndcg_at_k  (recs, gt, 10)
        per_r20[i] = _recall_at_k(recs, gt, 20)
        per_n20[i] = _ndcg_at_k  (recs, gt, 20)

    # ── Step 4: Cohort partitioning ────────────────────────────────────────
    train_counts: dict[int, int] = train_pairs_df.groupby("user_idx").size().to_dict()
    counts_arr = np.array(
        [train_counts.get(u, 0) for u in eval_users], dtype=np.int64
    )

    def _cohort_metrics(mask: np.ndarray) -> dict[str, float]:
        n = int(mask.sum())
        if n == 0:
            return {
                "recall_10": 0.0, "ndcg_10": 0.0,
                "recall_20": 0.0, "ndcg_20": 0.0,
                "hit_rate_10": 0.0, "n_users": 0,
            }
        return {
            "recall_10":   float(per_r10[mask].mean()),
            "ndcg_10":     float(per_n10[mask].mean()),
            "recall_20":   float(per_r20[mask].mean()),
            "ndcg_20":     float(per_n20[mask].mean()),
            "hit_rate_10": float((per_r10[mask] > 0).mean()) * 100.0,
            "n_users":     n,
        }

    overall_mask = np.ones(len(eval_users), dtype=bool)
    results: dict[str, Any] = {
        "overall":         _cohort_metrics(overall_mask),
        "recommendations": recs_idx_by_user,
        "filter_seen":     filter_seen,
    }
    for name, lo, hi in _COHORTS:
        mask = (counts_arr >= lo) & (counts_arr <= hi)
        results[name] = _cohort_metrics(mask)

    # ── Step 5: Pretty-print ──────────────────────────────────────────────
    col_names = ["overall"] + [c[0] for c in _COHORTS]
    col_w     = 12
    metrics_rows: list[tuple[str, str]] = [
        ("n_users",     "N users"),
        ("recall_10",   "Recall@10"),
        ("ndcg_10",     "NDCG@10"),
        ("recall_20",   "Recall@20"),
        ("ndcg_20",     "NDCG@20"),
        ("hit_rate_10", "Hit rate@10 %"),
    ]
    cohort_labels = {
        "overall": "Overall",
        "cold":    "Cold (3-10)",
        "medium":  "Med (11-50)",
        "warm":    "Warm (51+)",
    }
    label_w = max(len(v) for _, v in metrics_rows) + 2
    sep  = "═" * (label_w + (col_w + 2) * len(col_names) + 1)
    thin = "─" * (label_w + (col_w + 2) * len(col_names) + 1)

    def _fmt(key: str, val: float | int) -> str:
        if key == "n_users":
            return f"{int(val):>{col_w},}"
        if key == "hit_rate_10":
            return f"{val:>{col_w}.1f}"
        return f"{val:>{col_w}.4f}"

    print(f"\n  {sep}")
    print(f"  {label.upper()} STRATIFIED RESULTS  [{filter_tag}]")
    print(f"  {sep}")

    header = f"  {'Metric':<{label_w}}"
    for c in col_names:
        header += f"  {cohort_labels[c]:>{col_w}}"
    print(header)
    print(thin)

    range_row = f"  {'Interactions':<{label_w}}"
    ranges = {"overall": "all", "cold": "3-10", "medium": "11-50", "warm": "51+"}
    for c in col_names:
        range_row += f"  {ranges[c]:>{col_w}}"
    print(range_row)
    print(thin)

    for key, name in metrics_rows:
        row = f"  {name:<{label_w}}"
        for c in col_names:
            val = results[c][key]
            row += f"  {_fmt(key, val)}"
        print(row)

    print(f"  {sep}")
    return results
