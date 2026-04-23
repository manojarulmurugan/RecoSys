"""Category granularity, session-negative, and in-batch difficulty analysis.

analyze_category_granularity  — answers three questions about catalog density:

  (1) CAT-L1 COVERAGE
      For each top-level category (cat_l1_idx), how many unique items exist in
      the full catalog?  Reports item_count and % of catalog.

  (2) CAT-L2 COVERAGE
      Same analysis at the finer cat_l2 level.  Shows the top-20 and bottom-20
      categories by item count so we can spot both dense and sparse bins.

  (3) HARD-NEGATIVE POOL SIZE PER POSITIVE PAIR
      For a random sample of 500 positive pairs from train_pairs, resolves the
      positive item's category and counts how many *other* catalog items share
      that category — i.e. the pool of candidate hard negatives at each level.
      Reports median and 10th-percentile pool sizes.

      Decision rule:
        • cat_l2 p10 < 50  → cat_l2 is too fine-grained for reliable hard
          negative sampling (some items will have almost no hard negatives).
        • Otherwise        → cat_l2 is safe to use.

analyze_session_negatives  — evaluates session-coview signals:

  For every user session that contains at least one cart/purchase event AND at
  least one view event, the items that were VIEWED but NOT carted/purchased are
  candidate session-based hard negatives (the user saw them but rejected them).

  Reports:
    • How many qualifying sessions exist (≥1 view + ≥1 cart/purchase)?
    • Median number of view-only items per qualifying session.
    • % of sessions with ≥3 view-only items (minimum pool for reliable sampling).
    • Category overlap: what fraction of session negatives share cat_l2 with the
      positive item vs. coming from a different sub-category?  Cross-category
      session negatives are complementary to category-based hard negatives.

assess_false_negative_risk  — estimates how often a proposed hard negative is actually
  an item the user would want (a false negative):

  For a random sample of 1000 users, the function inspects each user's positive items
  and classifies their proposed category hard negatives (same cat_l2, different item):

    High-risk  — same cat_l2 AND same brand as one of the user's positives.
    Safer      — same cat_l2, brand differs from all of the user's positives.

  Also measures what fraction of the top-2000 most popular items fall into a cat_l2
  the user has already interacted with (making them risky as popularity negatives).

  Decision rules:
    • same-cat-same-brand > 20%  → false negative filtering is necessary.
    • same-cat-same-brand <  5%  → risk is low enough to ignore.
    • 5–20%                      → borderline; worth a lightweight brand-exclusion heuristic.

analyze_intra_category_scores  — scores the trained model against same-category items:

  Loads a checkpoint, samples 200 positive (user, item) pairs, and for each pair
  draws 50 same-cat_l2 items and 50 random different-cat items from the catalog.
  Runs both the item tower and the user tower and reports mean cosine similarities
  across three groups — positive item, same-cat hard negatives, diff-cat easy negs.

  The "critical number" is the gap between user→positive and user→same-cat:
    • small gap (< 0.05) → the model barely discriminates within categories;
      hard negatives will inject strong gradient signal.
    • large gap (> 0.15) → the model already separates within-category items;
      hard negatives still help but the model is less broken at this level.

analyze_negative_difficulty  — measures how hard current in-batch negatives are:

  (1) ITEM POPULARITY
      Counts how often each item appears as a positive in train_pairs (training
      frequency).  Reports the popularity distribution and what fraction of all
      training pairs are covered by the top-1000 most popular items.

  (2) IN-BATCH NEGATIVE POPULARITY
      For 5 random batches of 2048 pairs (mirroring real training), computes
      what fraction of in-batch negatives come from the top-1000 most popular
      items and what fraction come from very rare items (< 10 appearances).
      Popular items are over-represented as in-batch negatives by construction
      (they are sampled proportionally to their frequency in train_pairs).

  (3) IN-BATCH NEGATIVE CATEGORY HARDNESS
      For the same 5 batches, computes the fraction of off-diagonal negative
      pairs that share cat_l2 / cat_l1 with their positive item.  Uses an
      efficient O(B) groupby method rather than iterating over all B×(B-1) pairs.

      Interpretation:
        • same-cat_l2 fraction ≈ 0.1–1%  → nearly all in-batch negatives are
          easy (random items from unrelated categories).  Hard negative mining
          will make a large difference.
        • same-cat_l2 fraction ≈ 5–15%   → the batch is already diverse enough
          to include a meaningful fraction of same-category negatives.  Hard
          negatives still help but marginal gain is smaller.

Usage
─────
  python scripts/two_tower/investigate_negatives.py

  Artifacts are loaded from artifacts/500k/ by default.
  Pass --artifacts-dir to override.
  Pass --skip-session to skip analyze_session_negatives (no GCS access needed).
  Pass --skip-difficulty to skip analyze_negative_difficulty.
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

from src.two_tower.models.two_tower import UserTower, ItemTower, TwoTowerModel


# ── GCS credentials ────────────────────────────────────────────────────────────

def _ensure_gcp_credentials() -> None:
    """Set GOOGLE_APPLICATION_CREDENTIALS if a service-account key can be found.

    Mirrors the pattern used throughout this repo's diagnostic scripts.
    Only sets the env var when the file actually exists — a missing path breaks
    PyArrow's GCS filesystem and won't fall back to gcloud ADC.
    """
    existing = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if existing and pathlib.Path(existing).expanduser().is_file():
        return
    for candidate in (
        _REPO_ROOT / "secrets" / "recosys-service-account.json",
        pathlib.Path(os.path.expanduser("~/secrets/recosys-service-account.json")),
    ):
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
            return

_DEFAULT_ARTIFACTS = _REPO_ROOT / "artifacts" / "500k"

_SEP  = "=" * 64
_THIN = "-" * 64


# ── CLI ────────────────────────────────────────────────────────────────────────

_DEFAULT_TRAIN_PATH = "gs://recosys-data-bucket/samples/users_sample_500k/train/"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Category granularity and session-negative analysis for hard-negative mining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--artifacts-dir",
        default=str(_DEFAULT_ARTIFACTS),
        help=f"Directory with items_encoded.parquet and train_pairs.parquet "
             f"(default: {_DEFAULT_ARTIFACTS})",
    )
    p.add_argument(
        "--train-path",
        default=_DEFAULT_TRAIN_PATH,
        help=f"GCS or local path to the raw train-split parquet "
             f"(default: {_DEFAULT_TRAIN_PATH})",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of positive pairs to sample for pool-size analysis (default: 500).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--skip-session",
        action="store_true",
        help="Skip analyze_session_negatives (avoids GCS access).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .pt checkpoint for analyze_intra_category_scores.  "
             "Defaults to artifacts/500k/checkpoints/epoch_5.pt.",
    )
    p.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip analyze_intra_category_scores (no checkpoint needed).",
    )
    p.add_argument(
        "--skip-fn-risk",
        action="store_true",
        help="Skip assess_false_negative_risk.",
    )
    p.add_argument(
        "--n-fn-users",
        type=int,
        default=1000,
        help="Users to sample for false-negative risk assessment (default: 1000).",
    )
    p.add_argument(
        "--top-popular",
        type=int,
        default=2000,
        help="Top-K popular items used as popularity-based negatives (default: 2000).",
    )
    p.add_argument(
        "--skip-difficulty",
        action="store_true",
        help="Skip analyze_negative_difficulty.",
    )
    p.add_argument(
        "--n-batches",
        type=int,
        default=5,
        help="Number of random training batches to simulate (default: 5).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Pairs per simulated training batch (default: 2048).",
    )
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_artifacts(
    artifacts_dir: pathlib.Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """Return (items_encoded_df, train_pairs_df, vocabs_or_None)."""
    items_path  = artifacts_dir / "items_encoded.parquet"
    pairs_path  = artifacts_dir / "train_pairs.parquet"
    vocabs_path = artifacts_dir / "vocabs.pkl"

    if not items_path.exists():
        sys.exit(f"ERROR: {items_path} not found.")
    if not pairs_path.exists():
        sys.exit(f"ERROR: {pairs_path} not found.")

    print(f"  Loading {items_path.name} ...")
    items_df = pd.read_parquet(items_path)
    print(f"  Loading {pairs_path.name} ...")
    pairs_df = pd.read_parquet(pairs_path)

    vocabs: dict | None = None
    if vocabs_path.exists():
        with open(vocabs_path, "rb") as f:
            vocabs = pickle.load(f)
        print(f"  Loaded vocabs.pkl — category labels will be decoded.")
    else:
        print(f"  vocabs.pkl not found — reporting raw index values.")

    return items_df, pairs_df, vocabs


def _build_idx_to_label(vocab: dict[str, int] | None) -> dict[int, str]:
    """Invert a name→index vocab dict into an index→name lookup."""
    if vocab is None:
        return {}
    return {v: k for k, v in vocab.items()}


def _fmt_label(idx: int, idx2label: dict[int, str]) -> str:
    if idx2label:
        return idx2label.get(idx, str(idx))
    return str(idx)


# ── Analysis functions ─────────────────────────────────────────────────────────

def analyze_category_granularity(
    items_encoded_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    *,
    sample_size: int = 500,
    seed: int = 42,
    vocabs: dict | None = None,
) -> None:
    """Run all three category-granularity analyses and print results.

    Args:
        items_encoded_df: Full item catalog with cat_l1_idx / cat_l2_idx columns.
        train_pairs_df:   Training pairs with item_idx column linking into the catalog.
        sample_size:      How many positive pairs to sample for Part 3.
        seed:             Random seed for reproducible sampling.
        vocabs:           Optional vocabs dict (from vocabs.pkl) used to decode
                          category indices into human-readable names.
    """
    rng = np.random.default_rng(seed)

    n_catalog = items_encoded_df["item_idx"].nunique()

    # Reverse-map indices → human-readable labels (if vocabs available)
    l1_idx2label = _build_idx_to_label(vocabs.get("cat_l1_vocab") if vocabs else None)
    l2_idx2label = _build_idx_to_label(vocabs.get("cat_l2_vocab") if vocabs else None)

    # ── Pre-compute per-category item counts over the full catalog ────────────
    #    Use product_id as the unique item identifier (item_idx is equivalent).
    l1_counts = (
        items_encoded_df.groupby("cat_l1_idx")["item_idx"]
        .nunique()
        .rename("item_count")
        .reset_index()
        .sort_values("item_count", ascending=False)
    )
    l2_counts = (
        items_encoded_df.groupby("cat_l2_idx")["item_idx"]
        .nunique()
        .rename("item_count")
        .reset_index()
        .sort_values("item_count", ascending=False)
    )

    # ════════════════════════════════════════════════════════════════
    # PART 1 — CAT-L1 COVERAGE
    # ════════════════════════════════════════════════════════════════
    print(f"\n{_SEP}")
    print(f"  PART 1 — CAT-L1 COVERAGE  ({l1_counts.shape[0]} unique categories)")
    print(f"  Full catalog: {n_catalog:,} unique items")
    print(_SEP)

    l1_counts["pct_catalog"] = 100.0 * l1_counts["item_count"] / n_catalog

    label_col = "cat_l1_label" if l1_idx2label else "cat_l1_idx"
    l1_display = l1_counts.copy()
    if l1_idx2label:
        l1_display[label_col] = l1_display["cat_l1_idx"].map(
            lambda x: _fmt_label(x, l1_idx2label)
        )
        cols_order = [label_col, "cat_l1_idx", "item_count", "pct_catalog"]
    else:
        cols_order = ["cat_l1_idx", "item_count", "pct_catalog"]

    l1_display = l1_display[cols_order].reset_index(drop=True)

    _print_table(l1_display, float_cols={"pct_catalog": "{:.2f}%"})

    # ════════════════════════════════════════════════════════════════
    # PART 2 — CAT-L2 COVERAGE  (top 20 and bottom 20)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{_SEP}")
    print(f"  PART 2 — CAT-L2 COVERAGE  ({l2_counts.shape[0]} unique sub-categories)")
    print(_SEP)

    l2_counts["pct_catalog"] = 100.0 * l2_counts["item_count"] / n_catalog

    l2_label_col = "cat_l2_label" if l2_idx2label else "cat_l2_idx"
    l2_display = l2_counts.copy()
    if l2_idx2label:
        l2_display[l2_label_col] = l2_display["cat_l2_idx"].map(
            lambda x: _fmt_label(x, l2_idx2label)
        )
        l2_cols = [l2_label_col, "cat_l2_idx", "item_count", "pct_catalog"]
    else:
        l2_cols = ["cat_l2_idx", "item_count", "pct_catalog"]

    l2_display = l2_display[l2_cols].reset_index(drop=True)

    top20 = l2_display.head(20)
    bot20 = l2_display.tail(20).sort_values("item_count")

    print(f"\n  TOP 20 cat_l2 by item count:")
    _print_table(top20, float_cols={"pct_catalog": "{:.2f}%"}, indent=4)

    print(f"\n  BOTTOM 20 cat_l2 by item count (smallest pools):")
    _print_table(bot20, float_cols={"pct_catalog": "{:.4f}%"}, indent=4)

    # ════════════════════════════════════════════════════════════════
    # PART 3 — HARD-NEGATIVE POOL SIZE PER POSITIVE PAIR
    # ════════════════════════════════════════════════════════════════
    print(f"\n{_SEP}")
    n_sample = min(sample_size, len(train_pairs_df))
    print(f"  PART 3 — HARD-NEGATIVE POOL SIZE PER POSITIVE PAIR")
    print(f"  Sampling {n_sample:,} positive pairs from {len(train_pairs_df):,} total")
    print(_SEP)

    sample_pairs = train_pairs_df.sample(n=n_sample, random_state=int(rng.integers(0, 2**31)))

    # Join with items_encoded to get category for each positive item
    item_cats = items_encoded_df[["item_idx", "cat_l1_idx", "cat_l2_idx"]].drop_duplicates("item_idx")
    sample_with_cats = sample_pairs.merge(item_cats, on="item_idx", how="left")

    n_unmatched = sample_with_cats["cat_l1_idx"].isna().sum()
    if n_unmatched > 0:
        print(f"  Warning: {n_unmatched} sampled pairs have no matching item in catalog.")
    sample_with_cats = sample_with_cats.dropna(subset=["cat_l1_idx", "cat_l2_idx"])

    # Pre-build lookup: cat_l1_idx → count of catalog items with that category
    # "Candidate hard negatives" = items in same category but DIFFERENT item_idx
    l1_pool_map = l1_counts.set_index("cat_l1_idx")["item_count"].to_dict()
    l2_pool_map = l2_counts.set_index("cat_l2_idx")["item_count"].to_dict()

    # For each sampled pair, pool size = (total items in same cat) - 1  (exclude the item itself)
    sample_with_cats["l1_pool"] = (
        sample_with_cats["cat_l1_idx"].map(l1_pool_map).fillna(0).astype(int) - 1
    ).clip(lower=0)
    sample_with_cats["l2_pool"] = (
        sample_with_cats["cat_l2_idx"].map(l2_pool_map).fillna(0).astype(int) - 1
    ).clip(lower=0)

    l1_pools = sample_with_cats["l1_pool"].values
    l2_pools = sample_with_cats["l2_pool"].values

    l1_median = float(np.median(l1_pools))
    l1_p10    = float(np.percentile(l1_pools, 10))
    l1_mean   = float(np.mean(l1_pools))
    l1_min    = int(np.min(l1_pools))
    l1_max    = int(np.max(l1_pools))

    l2_median = float(np.median(l2_pools))
    l2_p10    = float(np.percentile(l2_pools, 10))
    l2_mean   = float(np.mean(l2_pools))
    l2_min    = int(np.min(l2_pools))
    l2_max    = int(np.max(l2_pools))

    print(f"\n  {'Metric':<24}  {'cat_l1':>12}  {'cat_l2':>12}")
    print(f"  {_THIN}")
    print(f"  {'Mean pool size':<24}  {l1_mean:>12,.1f}  {l2_mean:>12,.1f}")
    print(f"  {'Median pool size':<24}  {l1_median:>12,.0f}  {l2_median:>12,.0f}")
    print(f"  {'10th percentile (p10)':<24}  {l1_p10:>12,.0f}  {l2_p10:>12,.0f}")
    print(f"  {'Min pool size':<24}  {l1_min:>12,}  {l2_min:>12,}")
    print(f"  {'Max pool size':<24}  {l1_max:>12,}  {l2_max:>12,}")

    # ── Decision ──────────────────────────────────────────────────────────────
    THRESHOLD = 50
    print(f"\n  {_THIN}")
    print(f"  DECISION  (threshold: cat_l2 p10 ≥ {THRESHOLD} = safe for hard negatives)")
    print(f"  {_THIN}")

    if l2_p10 < THRESHOLD:
        print(f"  CAUTION  cat_l2 p10 = {l2_p10:.0f}  < {THRESHOLD}")
        print(f"  cat_l2 is TOO FINE-GRAINED for reliable hard negative sampling.")
        print(f"  Some positive items sit in near-empty sub-categories and will")
        print(f"  have fewer than {THRESHOLD} hard-negative candidates at the cat_l2 level.")
        print(f"  Recommendation: use cat_l1 for hard negatives, OR filter to")
        print(f"  cat_l2 bins with item_count ≥ {THRESHOLD} and fall back to cat_l1 otherwise.")
    else:
        print(f"  OK  cat_l2 p10 = {l2_p10:.0f}  ≥ {THRESHOLD}")
        print(f"  cat_l2 provides sufficient hard-negative candidates for all")
        print(f"  sampled positive pairs.  Safe to use cat_l2-level hard negatives.")

    if l1_p10 < THRESHOLD:
        print(f"\n  Note: cat_l1 p10 = {l1_p10:.0f} < {THRESHOLD} as well — even the")
        print(f"  coarser level is sparse for some items (e.g. very rare categories).")

    print(f"\n{_SEP}\n")


# ── Session-negative analysis ──────────────────────────────────────────────────

def analyze_session_negatives(
    train_df_path: str,
    *,
    items_encoded_df: pd.DataFrame | None = None,
    artifacts_dir: pathlib.Path | None = None,
    seed: int = 42,
) -> None:
    """Evaluate session-coview signals as a source of hard negatives.

    For every user session that contains at least one cart or purchase event AND
    at least one view event, items that were VIEWED but NOT carted/purchased are
    treated as candidate session-based hard negatives — the user considered them
    but ultimately rejected them, making them semantically plausible yet negative.

    Reports:
      • Session counts: total, qualifying (≥1 view + ≥1 positive), usable (≥1
        view-only item), and the subset with ≥3 view-only items.
      • Distribution of view-only pool size per qualifying session (median, p10,
        p25, p75, p90).
      • Category overlap analysis: for each (positive_item, session_negative)
        pair, whether they share cat_l2.  High overlap means session negatives
        are redundant with category-based negatives; low overlap means they are
        complementary (cross-category signal).

    Args:
        train_df_path:    GCS or local path to the raw train-split parquet
                          (e.g. "gs://recosys-data-bucket/…/train/").
        items_encoded_df: Pre-loaded items_encoded DataFrame.  If None, the
                          function attempts to load it from artifacts_dir.
        artifacts_dir:    Fallback directory for items_encoded.parquet when
                          items_encoded_df is not provided.
        seed:             Random seed for reproducible sampling.
    """
    _ensure_gcp_credentials()

    rng = np.random.default_rng(seed)

    # ── Load raw train events ─────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print(f"  ANALYZE SESSION NEGATIVES")
    print(f"  Train path: {train_df_path}")
    print(_SEP)

    print(f"\n  Loading raw train events from {train_df_path} ...")
    train_df = pd.read_parquet(
        train_df_path,
        columns=["user_id", "user_session", "product_id", "event_type"],
    )
    print(f"  Loaded {len(train_df):,} rows  |  "
          f"{train_df['user_session'].nunique():,} unique sessions  |  "
          f"{train_df['user_id'].nunique():,} unique users")

    # ── Resolve category map: product_id → (cat_l1_idx, cat_l2_idx) ──────────
    cat_map: pd.DataFrame | None = None

    if items_encoded_df is not None:
        cat_map = (
            items_encoded_df[["product_id", "cat_l1_idx", "cat_l2_idx"]]
            .drop_duplicates("product_id")
        )
    elif artifacts_dir is not None:
        enc_path = pathlib.Path(artifacts_dir) / "items_encoded.parquet"
        if enc_path.exists():
            print(f"  Loading {enc_path.name} for category lookups ...")
            cat_map = pd.read_parquet(
                enc_path, columns=["product_id", "cat_l1_idx", "cat_l2_idx"]
            ).drop_duplicates("product_id")

    if cat_map is None:
        print("  Warning: items_encoded not available — skipping category "
              "overlap analysis.")

    # ── Build per-session positive and view sets ──────────────────────────────
    print("\n  Aggregating events by session ...")

    positives_flag = train_df["event_type"].isin({"cart", "purchase"})

    # Aggregate per session: sets of positive product_ids and all viewed product_ids
    sess_pos = (
        train_df[positives_flag]
        .groupby("user_session")["product_id"]
        .agg(set)
        .rename("positive_items")
    )
    sess_view = (
        train_df[train_df["event_type"] == "view"]
        .groupby("user_session")["product_id"]
        .agg(set)
        .rename("viewed_items")
    )

    # Combine: only sessions that have BOTH views and at least one positive
    sess = pd.concat([sess_pos, sess_view], axis=1).dropna()
    n_qualifying = len(sess)

    total_sessions = train_df["user_session"].nunique()

    print(f"\n  Total sessions in train split              : {total_sessions:,}")
    print(f"  Sessions with ≥1 view + ≥1 cart/purchase  : {n_qualifying:,}  "
          f"({100*n_qualifying/total_sessions:.1f}% of all sessions)")

    if n_qualifying == 0:
        print("  No qualifying sessions found — exiting.")
        return

    # ── Compute view-only (session-negative) pool per session ────────────────
    sess["session_negatives"] = sess.apply(
        lambda r: r["viewed_items"] - r["positive_items"], axis=1
    )
    sess["n_session_neg"] = sess["session_negatives"].apply(len)
    sess["n_positives"]   = sess["positive_items"].apply(len)

    # Restrict to sessions that actually have at least one view-only item
    sess_usable = sess[sess["n_session_neg"] > 0]
    n_usable    = len(sess_usable)

    pct_usable     = 100 * n_usable / n_qualifying
    pct_ge3        = 100 * (sess_usable["n_session_neg"] >= 3).sum() / n_qualifying
    pct_ge3_usable = 100 * (sess_usable["n_session_neg"] >= 3).sum() / n_usable if n_usable else 0.0

    pool_sizes = sess_usable["n_session_neg"].values

    median_pool = float(np.median(pool_sizes))
    p10_pool    = float(np.percentile(pool_sizes, 10))
    p25_pool    = float(np.percentile(pool_sizes, 25))
    p75_pool    = float(np.percentile(pool_sizes, 75))
    p90_pool    = float(np.percentile(pool_sizes, 90))
    mean_pool   = float(np.mean(pool_sizes))

    print(f"\n  {_THIN}")
    print(f"  VIEW-ONLY POOL SIZE PER QUALIFYING SESSION")
    print(f"  {_THIN}")
    print(f"  Sessions with ≥1 view-only item            : {n_usable:,}  "
          f"({pct_usable:.1f}% of qualifying sessions)")
    print(f"  Sessions with ≥3 view-only items           : "
          f"{(sess_usable['n_session_neg'] >= 3).sum():,}  "
          f"({pct_ge3:.1f}% of qualifying, "
          f"{pct_ge3_usable:.1f}% of usable)")
    print()
    print(f"  {'Metric':<28}  {'view-only items':>16}")
    print(f"  {'-'*28}  {'-'*16}")
    print(f"  {'Mean':<28}  {mean_pool:>16.1f}")
    print(f"  {'Median (p50)':<28}  {median_pool:>16.1f}")
    print(f"  {'p10':<28}  {p10_pool:>16.1f}")
    print(f"  {'p25':<28}  {p25_pool:>16.1f}")
    print(f"  {'p75':<28}  {p75_pool:>16.1f}")
    print(f"  {'p90':<28}  {p90_pool:>16.1f}")

    # Viability verdict
    MIN_POOL = 3
    print(f"\n  {_THIN}")
    if pct_ge3 >= 50:
        print(f"  OK  {pct_ge3:.1f}% of qualifying sessions have ≥{MIN_POOL} view-only items.")
        print(f"  Session-based negatives are VIABLE for the majority of positive pairs.")
    elif pct_ge3 >= 20:
        print(f"  PARTIAL  {pct_ge3:.1f}% of qualifying sessions have ≥{MIN_POOL} view-only items.")
        print(f"  Session negatives are viable for a minority of pairs — use as a supplement,")
        print(f"  not the primary negative strategy.")
    else:
        print(f"  CAUTION  Only {pct_ge3:.1f}% of qualifying sessions have ≥{MIN_POOL} view-only")
        print(f"  items.  Session negatives are too sparse to be a primary strategy.")

    # ── Category overlap analysis ─────────────────────────────────────────────
    if cat_map is None:
        print(f"\n  (Skipping category overlap analysis — no items_encoded available.)")
        print(f"\n{_SEP}\n")
        return

    print(f"\n  {_THIN}")
    print(f"  CATEGORY OVERLAP: SESSION NEGATIVES vs. CATEGORY-BASED NEGATIVES")
    print(f"  {_THIN}")
    print(f"  (Sampling up to 2,000 qualifying sessions for efficiency ...)")

    # Sample sessions for the overlap computation
    n_overlap_sample = min(2_000, n_usable)
    sampled_sessions = sess_usable.sample(
        n=n_overlap_sample,
        random_state=int(rng.integers(0, 2**31)),
    )

    prod2cat = cat_map.set_index("product_id")[["cat_l1_idx", "cat_l2_idx"]]

    # For each sampled session, compare each (positive, session_negative) pair
    n_pairs_total       = 0
    n_same_cat2         = 0   # session negative shares cat_l2 with positive
    n_diff_cat2         = 0   # session negative has different cat_l2 (complementary)
    n_same_cat1         = 0   # session negative shares cat_l1 with positive
    n_diff_cat1         = 0
    n_pairs_no_cat      = 0   # product_id not in items_encoded (cold-start items)

    # Breakdown: sessions where ALL session negs are same-cat2 vs. any diff-cat2
    sessions_any_diff_cat2 = 0

    for _, row in sampled_sessions.iterrows():
        pos_items  = row["positive_items"]
        neg_items  = row["session_negatives"]

        session_has_diff_cat2 = False

        for pos_pid in pos_items:
            if pos_pid not in prod2cat.index:
                continue
            pos_l1 = prod2cat.at[pos_pid, "cat_l1_idx"]
            pos_l2 = prod2cat.at[pos_pid, "cat_l2_idx"]

            for neg_pid in neg_items:
                if neg_pid not in prod2cat.index:
                    n_pairs_no_cat += 1
                    continue

                neg_l1 = prod2cat.at[neg_pid, "cat_l1_idx"]
                neg_l2 = prod2cat.at[neg_pid, "cat_l2_idx"]

                n_pairs_total += 1

                if neg_l2 == pos_l2:
                    n_same_cat2 += 1
                else:
                    n_diff_cat2 += 1
                    session_has_diff_cat2 = True

                if neg_l1 == pos_l1:
                    n_same_cat1 += 1
                else:
                    n_diff_cat1 += 1

        if session_has_diff_cat2:
            sessions_any_diff_cat2 += 1

    if n_pairs_total == 0:
        print("  No (positive, session_negative) pairs could be categorised.")
        print(f"\n{_SEP}\n")
        return

    pct_same_cat2  = 100 * n_same_cat2 / n_pairs_total
    pct_diff_cat2  = 100 * n_diff_cat2 / n_pairs_total
    pct_same_cat1  = 100 * n_same_cat1 / n_pairs_total
    pct_diff_cat1  = 100 * n_diff_cat1 / n_pairs_total
    pct_no_cat     = 100 * n_pairs_no_cat / (n_pairs_total + n_pairs_no_cat)
    pct_sess_diff2 = 100 * sessions_any_diff_cat2 / n_overlap_sample

    print(f"\n  Analysis over {n_overlap_sample:,} sampled sessions  "
          f"→  {n_pairs_total:,} (positive, session_neg) pairs scored")
    if n_pairs_no_cat:
        print(f"  ({n_pairs_no_cat:,} pairs skipped — product_id absent from catalog "
              f"[{pct_no_cat:.1f}% of raw pairs])")

    print(f"\n  {'Category relationship':<40}  {'pairs':>8}  {'%':>7}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*7}")
    print(f"  {'Same cat_l2 as positive (redundant w/ cat neg)':<40}  "
          f"{n_same_cat2:>8,}  {pct_same_cat2:>6.1f}%")
    print(f"  {'Diff cat_l2, same cat_l1 (partial overlap)':<40}  "
          f"{n_same_cat1 - n_same_cat2:>8,}  "
          f"{100*(n_same_cat1 - n_same_cat2)/n_pairs_total:>6.1f}%")
    print(f"  {'Diff cat_l1 entirely (purely complementary)':<40}  "
          f"{n_diff_cat1:>8,}  {pct_diff_cat1:>6.1f}%")
    print(f"  {'─'*40}  {'─'*8}  {'─'*7}")
    print(f"  {'Total cat_l2 DIFFERENT (complementary)':<40}  "
          f"{n_diff_cat2:>8,}  {pct_diff_cat2:>6.1f}%")
    print(f"\n  Sessions containing ≥1 cross-cat_l2 session neg : "
          f"{sessions_any_diff_cat2:,}  ({pct_sess_diff2:.1f}% of sampled)")

    # Interpretation
    print(f"\n  {_THIN}")
    print(f"  INTERPRETATION")
    print(f"  {_THIN}")

    if pct_diff_cat2 >= 60:
        print(f"  COMPLEMENTARY  {pct_diff_cat2:.1f}% of session negatives have a DIFFERENT")
        print(f"  cat_l2 from the positive item.  Session negatives and category-based")
        print(f"  negatives capture mostly DIFFERENT signals — combining both strategies")
        print(f"  will meaningfully enrich the negative training distribution.")
    elif pct_diff_cat2 >= 30:
        print(f"  MIXED  {pct_diff_cat2:.1f}% of session negatives are cross-cat_l2.")
        print(f"  Session and category negatives are partially complementary.  Using")
        print(f"  both still adds diversity, but the marginal gain is moderate.")
    else:
        print(f"  REDUNDANT  Only {pct_diff_cat2:.1f}% of session negatives are cross-cat_l2.")
        print(f"  Most session negatives are already within the same sub-category as")
        print(f"  the positive — they largely overlap with category-based negatives.")
        print(f"  Session negatives may not add much beyond cat_l2 sampling.")

    if pct_sess_diff2 >= 70:
        print(f"\n  Cross-category session negatives are WIDESPREAD: {pct_sess_diff2:.1f}% of")
        print(f"  sessions have at least one cross-cat_l2 negative — this signal is")
        print(f"  available for most training pairs.")
    elif pct_sess_diff2 >= 40:
        print(f"\n  Cross-category session negatives exist in {pct_sess_diff2:.1f}% of sessions —")
        print(f"  available for a meaningful subset of training pairs.")
    else:
        print(f"\n  Cross-category session negatives appear in only {pct_sess_diff2:.1f}% of")
        print(f"  sessions — this signal would cover few training pairs.")

    print(f"\n{_SEP}\n")


# ── Intra-category score distribution ─────────────────────────────────────────

def _load_model(
    checkpoint_path: pathlib.Path,
    vocabs: dict,
    device: torch.device,
) -> TwoTowerModel:
    """Instantiate TwoTowerModel from vocabs and load checkpoint weights."""
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
    model = TwoTowerModel(user_tower, item_tower)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    return model


def _encode_items_batched(
    model: TwoTowerModel,
    item_idxs: np.ndarray,
    item_cat_arr: np.ndarray,
    item_dense_arr: np.ndarray,
    device: torch.device,
    encode_batch: int = 512,
) -> np.ndarray:
    """Encode a flat array of item_idxs; returns float32 (N, 64) L2-normalised."""
    all_embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(item_idxs), encode_batch):
            chunk = item_idxs[start : start + encode_batch]
            cat_t   = torch.tensor(item_cat_arr[chunk],   dtype=torch.long,    device=device)
            dense_t = torch.tensor(item_dense_arr[chunk], dtype=torch.float32, device=device)
            emb = model.get_item_embeddings(cat_t, dense_t)
            all_embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(all_embs)


def _encode_users_batched(
    model: TwoTowerModel,
    user_idxs: np.ndarray,
    user_cat_arr: np.ndarray,
    user_dense_arr: np.ndarray,
    device: torch.device,
    encode_batch: int = 512,
) -> np.ndarray:
    """Encode a flat array of user_idxs; returns float32 (N, 64) L2-normalised."""
    all_embs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(user_idxs), encode_batch):
            chunk = user_idxs[start : start + encode_batch]
            uid_t   = torch.tensor(chunk,                  dtype=torch.long,    device=device)
            cat_t   = torch.tensor(user_cat_arr[chunk],    dtype=torch.long,    device=device)
            dense_t = torch.tensor(user_dense_arr[chunk],  dtype=torch.float32, device=device)
            emb = model.get_user_embedding(uid_t, cat_t, dense_t)
            all_embs.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(all_embs)


def _sim_stats(sims: np.ndarray) -> dict[str, float]:
    """Return mean, std, p10, p50, p90 for a 1-D similarity array."""
    return {
        "mean": float(np.mean(sims)),
        "std":  float(np.std(sims)),
        "p10":  float(np.percentile(sims, 10)),
        "p50":  float(np.percentile(sims, 50)),
        "p90":  float(np.percentile(sims, 90)),
    }


def _print_sim_table(groups: dict[str, dict[str, float]]) -> None:
    """Print a mean/std/p10/p50/p90 table for multiple similarity groups."""
    cols  = ["mean", "std", "p10", "p50", "p90"]
    label_w = max(len(k) for k in groups) + 2

    header = f"  {'Group':<{label_w}}  " + "  ".join(f"{c:>8}" for c in cols)
    rule   = f"  {'-'*label_w}  " + "  ".join("-" * 8 for _ in cols)
    print(header)
    print(rule)
    for label, stats in groups.items():
        row = "  ".join(f"{stats[c]:>8.4f}" for c in cols)
        print(f"  {label:<{label_w}}  {row}")


def analyze_intra_category_scores(
    checkpoint_path: str | pathlib.Path,
    items_encoded_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    *,
    n_pairs: int = 200,
    n_same_cat: int = 50,
    n_diff_cat: int = 50,
    artifacts_dir: pathlib.Path | None = None,
    seed: int = 42,
) -> None:
    """Score a trained checkpoint against same-category vs. different-category items.

    For each of n_pairs randomly sampled positive (user, item) pairs, the function:
      (a) Draws n_same_cat items that share the positive's cat_l2 (hard negatives).
      (b) Draws n_diff_cat items with a different cat_l2 (easy random negatives).
      (c) Runs the item tower to get cosine similarity: positive ↔ same-cat and
          positive ↔ diff-cat.  High same-cat similarity means the item tower
          embeds items of the same sub-category close together.
      (d) Runs the user tower and computes cosine similarity of the user embedding
          to the positive item, same-cat items, and diff-cat items.

    The key diagnostic number is the gap between user→positive and user→same-cat:
      • Small gap (< 0.05): the model barely discriminates within a category.
        Hard negatives will inject strong gradient signal.
      • Large gap (> 0.15): the model already has meaningful within-category
        discrimination.  Hard negatives still help at the margin.

    Args:
        checkpoint_path: Path to a saved .pt checkpoint.
        items_encoded_df: Full item catalog DataFrame.
        train_pairs_df:   Training pairs DataFrame (user_idx, item_idx, …).
        n_pairs:          Positive pairs to sample (default: 200).
        n_same_cat:       Same-cat items to draw per pair (default: 50).
        n_diff_cat:       Different-cat items to draw per pair (default: 50).
        artifacts_dir:    Directory containing vocabs.pkl and users_encoded.parquet.
                          Inferred from checkpoint_path.parent.parent if None.
        seed:             Random seed.
    """
    checkpoint_path = pathlib.Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        print(f"  ERROR: checkpoint not found: {checkpoint_path}")
        print(f"  Skipping analyze_intra_category_scores.")
        return

    if artifacts_dir is None:
        artifacts_dir = checkpoint_path.parent.parent

    vocabs_path    = pathlib.Path(artifacts_dir) / "vocabs.pkl"
    users_enc_path = pathlib.Path(artifacts_dir) / "users_encoded.parquet"

    if not vocabs_path.exists():
        print(f"  ERROR: vocabs.pkl not found at {vocabs_path}")
        print(f"  Skipping analyze_intra_category_scores.")
        return
    if not users_enc_path.exists():
        print(f"  ERROR: users_encoded.parquet not found at {users_enc_path}")
        print(f"  Skipping analyze_intra_category_scores.")
        return

    rng    = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{_SEP}")
    print(f"  ANALYZE INTRA-CATEGORY SCORE DISTRIBUTION")
    print(f"  Checkpoint : {checkpoint_path.name}")
    print(f"  Artifacts  : {artifacts_dir}")
    print(f"  Device     : {device}")
    print(f"  {n_pairs} positive pairs | {n_same_cat} same-cat items | "
          f"{n_diff_cat} diff-cat items per pair")
    print(_SEP)

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("\n  Loading vocabs and users_encoded ...")
    with open(vocabs_path, "rb") as f:
        vocabs = pickle.load(f)

    users_enc = pd.read_parquet(users_enc_path)

    print(f"  Loading checkpoint {checkpoint_path.name} ...")
    model = _load_model(checkpoint_path, vocabs, device)
    ckpt_epoch = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    ).get("epoch", "?")
    print(f"  Checkpoint epoch : {ckpt_epoch}")
    print(f"  Model in eval mode on {device}")

    # ── Build feature lookup arrays ───────────────────────────────────────────
    max_item_idx = int(items_encoded_df["item_idx"].max()) + 1
    max_user_idx = int(users_enc["user_idx"].max()) + 1

    # Item features: cat array (N, 5) and dense array (N, 3)
    item_cat_arr   = np.zeros((max_item_idx, 5), dtype=np.int64)
    item_dense_arr = np.zeros((max_item_idx, 3), dtype=np.float32)
    item_cat_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
        ["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
    ].values.astype(np.int64)
    item_dense_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
        ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
    ].values.astype(np.float32)

    # User features: cat array (N, 4) and dense array (N, 6)
    user_cat_arr   = np.zeros((max_user_idx, 4), dtype=np.int64)
    user_dense_arr = np.zeros((max_user_idx, 6), dtype=np.float32)
    user_cat_arr[users_enc["user_idx"].values] = users_enc[
        ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
    ].values.astype(np.int64)
    user_dense_arr[users_enc["user_idx"].values] = users_enc[
        ["log_total_events", "months_active", "purchase_rate", "cart_rate",
         "log_n_sessions", "avg_purchase_price_scaled"]
    ].values.astype(np.float32)

    # ── Build cat_l2 → item_idx lookup ───────────────────────────────────────
    # Deduplicate so each item_idx appears once per cat_l2 group
    item_cat_dedup = items_encoded_df[["item_idx", "cat_l2_idx"]].drop_duplicates("item_idx")
    cat2items: dict[int, np.ndarray] = {}
    for cat_l2, grp in item_cat_dedup.groupby("cat_l2_idx"):
        cat2items[int(cat_l2)] = grp["item_idx"].values.astype(np.int64)

    all_item_idxs_catalog = item_cat_dedup["item_idx"].values.astype(np.int64)

    # ── Sample positive pairs ─────────────────────────────────────────────────
    n_sample = min(n_pairs, len(train_pairs_df))
    sampled  = train_pairs_df.sample(n=n_sample, random_state=int(rng.integers(0, 2**31)))

    print(f"\n  Sampling {n_sample} positive pairs and building item sets ...")

    # ── Build per-pair item index sets (vectorised over pairs) ────────────────
    # We collect everything into flat arrays for batch encoding.
    # Layout per pair: [pos_item | n_same_cat items | n_diff_cat items]
    # Total items per pair = 1 + n_same_cat + n_diff_cat
    items_per_pair = 1 + n_same_cat + n_diff_cat

    flat_item_idxs: list[int] = []   # all item_idxs to encode, in order
    flat_user_idxs: list[int] = []   # one user_idx per pair
    valid_pair_mask: list[bool] = [] # pairs skipped due to insufficient same-cat items

    for _, row in sampled.iterrows():
        pos_item_idx = int(row["item_idx"])
        user_idx     = int(row["user_idx"])
        pos_cat_l2   = int(item_cat_arr[pos_item_idx, 2])   # col 2 = cat_l2_idx

        # Same-cat candidates: same cat_l2, excluding the positive itself
        same_pool = cat2items.get(pos_cat_l2, np.empty(0, dtype=np.int64))
        same_pool = same_pool[same_pool != pos_item_idx]

        if len(same_pool) < n_same_cat:
            # Not enough same-cat items — skip this pair
            valid_pair_mask.append(False)
            continue

        # Different-cat candidates: any item with different cat_l2
        # Fast approach: sample from the full catalog and filter
        diff_drawn: list[int] = []
        attempts = 0
        max_attempts = n_diff_cat * 20
        while len(diff_drawn) < n_diff_cat and attempts < max_attempts:
            candidates = rng.choice(all_item_idxs_catalog,
                                    size=n_diff_cat * 3, replace=False)
            for c in candidates:
                if item_cat_arr[int(c), 2] != pos_cat_l2:
                    diff_drawn.append(int(c))
                    if len(diff_drawn) == n_diff_cat:
                        break
            attempts += n_diff_cat * 3

        if len(diff_drawn) < n_diff_cat:
            valid_pair_mask.append(False)
            continue

        same_drawn = rng.choice(same_pool, size=n_same_cat, replace=False).tolist()
        diff_drawn = diff_drawn[:n_diff_cat]

        flat_item_idxs.append(pos_item_idx)
        flat_item_idxs.extend(same_drawn)
        flat_item_idxs.extend(diff_drawn)
        flat_user_idxs.append(user_idx)
        valid_pair_mask.append(True)

    n_valid = sum(valid_pair_mask)
    n_skipped = n_sample - n_valid
    print(f"  Valid pairs     : {n_valid}  "
          f"({'all' if n_skipped == 0 else f'{n_skipped} skipped — insufficient same-cat items'})")

    if n_valid == 0:
        print("  No valid pairs — cannot continue.")
        print(f"\n{_SEP}\n")
        return

    flat_item_arr = np.array(flat_item_idxs, dtype=np.int64)
    flat_user_arr = np.array(flat_user_idxs, dtype=np.int64)

    # ── Batch-encode all items and users in one pass ──────────────────────────
    print(f"  Encoding {len(flat_item_arr):,} item feature vectors ...")
    all_item_embs = _encode_items_batched(
        model, flat_item_arr, item_cat_arr, item_dense_arr, device
    )   # (N_total_items, 64)

    print(f"  Encoding {n_valid} user feature vectors ...")
    all_user_embs = _encode_users_batched(
        model, flat_user_arr, user_cat_arr, user_dense_arr, device
    )   # (n_valid, 64)

    # ── Compute per-pair similarities ─────────────────────────────────────────
    # Item tower: pos_item ↔ same-cat, pos_item ↔ diff-cat
    item_same_sims: list[float] = []   # mean sim per pair: pos_item → same-cat
    item_diff_sims: list[float] = []   # mean sim per pair: pos_item → diff-cat

    # User tower: user ↔ pos, user ↔ same-cat, user ↔ diff-cat
    user_pos_sims:  list[float] = []
    user_same_sims: list[float] = []
    user_diff_sims: list[float] = []

    for pair_i in range(n_valid):
        base = pair_i * items_per_pair

        pos_emb  = all_item_embs[base]                              # (64,)
        same_embs = all_item_embs[base + 1 : base + 1 + n_same_cat] # (n_same_cat, 64)
        diff_embs = all_item_embs[base + 1 + n_same_cat : base + items_per_pair]  # (n_diff_cat, 64)

        u_emb = all_user_embs[pair_i]   # (64,)

        # Item tower: cosine sim from positive item embedding to the two groups
        # Embeddings are already L2-normalised → dot product = cosine similarity
        item_same_sims.append(float(np.dot(same_embs, pos_emb).mean()))
        item_diff_sims.append(float(np.dot(diff_embs, pos_emb).mean()))

        # User tower: cosine sim from user embedding to each group
        user_pos_sims.append(float(np.dot(u_emb, pos_emb)))
        user_same_sims.append(float(np.dot(same_embs, u_emb).mean()))
        user_diff_sims.append(float(np.dot(diff_embs, u_emb).mean()))

    # Convert to arrays for statistics
    item_same_arr  = np.array(item_same_sims,  dtype=np.float32)
    item_diff_arr  = np.array(item_diff_sims,  dtype=np.float32)
    user_pos_arr   = np.array(user_pos_sims,   dtype=np.float32)
    user_same_arr  = np.array(user_same_sims,  dtype=np.float32)
    user_diff_arr  = np.array(user_diff_sims,  dtype=np.float32)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  {_THIN}")
    print(f"  ITEM TOWER  —  cosine similarity from positive item to candidate groups")
    print(f"  (same cat_l2 items should cluster near the positive; diff-cat items far)")
    print(f"  {_THIN}\n")
    _print_sim_table({
        "Same cat_l2 (hard neg pool)": _sim_stats(item_same_arr),
        "Diff cat_l2 (easy neg pool)": _sim_stats(item_diff_arr),
    })
    item_gap = float(np.mean(item_same_arr) - np.mean(item_diff_arr))
    print(f"\n  Item-space gap  (same − diff)  : {item_gap:+.4f}")
    if item_gap > 0.20:
        print(f"  The item tower DOES cluster same-cat items together ({item_gap:.3f} gap).")
        print(f"  Hard negatives will be meaningfully harder than random negatives.")
    elif item_gap > 0.05:
        print(f"  The item tower shows modest same-cat clustering ({item_gap:.3f} gap).")
    else:
        print(f"  The item tower barely clusters same-cat items ({item_gap:.3f} gap).")
        print(f"  Category structure is not yet encoded in the item embedding space.")

    print(f"\n  {_THIN}")
    print(f"  USER TOWER  —  cosine similarity from user to each item group")
    print(f"  (user→positive should be highest; user→diff-cat should be lowest)")
    print(f"  {_THIN}\n")
    _print_sim_table({
        "Positive item   (true match)": _sim_stats(user_pos_arr),
        "Same cat_l2     (hard negs) ": _sim_stats(user_same_arr),
        "Diff cat_l2     (easy negs) ": _sim_stats(user_diff_arr),
    })

    # ── Key gap metrics ───────────────────────────────────────────────────────
    gap_pos_vs_same = float(np.mean(user_pos_arr)  - np.mean(user_same_arr))
    gap_same_vs_diff = float(np.mean(user_same_arr) - np.mean(user_diff_arr))
    gap_pos_vs_diff  = float(np.mean(user_pos_arr)  - np.mean(user_diff_arr))

    print(f"\n  {_THIN}")
    print(f"  KEY GAPS  (all computed over {n_valid} pairs × mean group similarities)")
    print(f"  {_THIN}")
    print(f"\n  {'Gap':<48}  {'value':>8}")
    print(f"  {'-'*48}  {'-'*8}")
    print(f"  {'user→positive  −  user→same_cat  [intra-cat]':<48}  "
          f"{gap_pos_vs_same:>+8.4f}")
    print(f"  {'user→same_cat  −  user→diff_cat  [category-level]':<48}  "
          f"{gap_same_vs_diff:>+8.4f}")
    print(f"  {'user→positive  −  user→diff_cat  [total]':<48}  "
          f"{gap_pos_vs_diff:>+8.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  {_THIN}")
    print(f"  VERDICT — EXPECTED IMPACT OF HARD NEGATIVE MINING")
    print(f"  {_THIN}\n")

    pos_mean  = float(np.mean(user_pos_arr))
    same_mean = float(np.mean(user_same_arr))
    diff_mean = float(np.mean(user_diff_arr))

    print(f"  user→positive   : {pos_mean:+.4f}")
    print(f"  user→same_cat   : {same_mean:+.4f}   ← key: how hard are same-cat items?")
    print(f"  user→diff_cat   : {diff_mean:+.4f}")
    print()

    if gap_pos_vs_same < 0.05:
        print(f"  STRONG SIGNAL  intra-category gap = {gap_pos_vs_same:.4f}")
        print(f"  The model barely distinguishes the positive item from same-category")
        print(f"  items ({same_mean:.4f} vs {pos_mean:.4f}).  These are GENUINELY HARD")
        print(f"  negatives — the model assigns them nearly the same score as the true")
        print(f"  positive.  Hard negative mining will inject strong gradient signal")
        print(f"  and is expected to significantly improve within-category ranking.")
    elif gap_pos_vs_same < 0.15:
        print(f"  MODERATE SIGNAL  intra-category gap = {gap_pos_vs_same:.4f}")
        print(f"  The model has some within-category discrimination but the same-cat")
        print(f"  similarity ({same_mean:.4f}) is close to the positive ({pos_mean:.4f}).")
        print(f"  Hard negatives will still provide useful gradient — the model is not")
        print(f"  yet reliably separating within-category items for retrieval.")
    else:
        print(f"  WEAK SIGNAL  intra-category gap = {gap_pos_vs_same:.4f}")
        print(f"  The model already discriminates within-category items reasonably well")
        print(f"  ({pos_mean:.4f} vs {same_mean:.4f} for same-cat).  Hard negatives")
        print(f"  may fine-tune precision at rank but the largest gains have likely")
        print(f"  already been captured by the current training regime.")

    print()
    frac_same_near_pos = float(
        np.mean(np.abs(user_same_arr - user_pos_arr) < 0.05)
    )
    print(f"  Fraction of pairs where |user→same_cat − user→positive| < 0.05 : "
          f"{frac_same_near_pos:.1%}")
    print(f"  (pairs where the model is nearly indifferent to same-cat hard negatives)")

    print(f"\n{_SEP}\n")


# ── False-negative risk assessment ────────────────────────────────────────────

def assess_false_negative_risk(
    train_pairs_df: pd.DataFrame,
    items_encoded_df: pd.DataFrame,
    *,
    n_users: int = 1000,
    top_k_popular: int = 2000,
    seed: int = 42,
) -> None:
    """Estimate how often a category hard negative is actually a false negative.

    A category hard negative for a (user, pos_item) pair is any catalog item that
    shares pos_item's cat_l2 but is a different item.  Such a negative is a
    **false negative** if the user would actually want it — which we proxy as:
    same cat_l2 AND the same brand as one of the user's already-positive items.

    For each of n_users randomly sampled users, the function:
      (1) Collects the user's cat_l2 and brand_idx values from their positive items.
      (2) For each positive item, counts how many same-cat catalog items share a brand
          with any of the user's positives (high-risk false negatives) vs. how many
          have a brand the user has never engaged with (safer negatives).
          Uses O(1) lookups into pre-built (cat_l2, brand) → count dicts — no
          per-item iteration.
      (3) Also counts, for the top-K most popular items used as popularity-based
          negatives, how many fall into a cat_l2 the user has interacted with.

    Brand index 0 ("unknown") is excluded from same-brand matching because "unknown"
    is a catalog-level sentinel that does not represent a real brand relationship.

    Args:
        train_pairs_df:   Training pairs with columns [user_idx, item_idx, …].
        items_encoded_df: Full item catalog with cat_l2_idx and brand_idx columns.
        n_users:          Number of users to sample (default: 1000).
        top_k_popular:    Size of the popularity-negative pool (default: 2000).
        seed:             Random seed.
    """
    rng = np.random.default_rng(seed)

    print(f"\n{_SEP}")
    print(f"  ASSESS FALSE NEGATIVE RISK")
    print(f"  {n_users} sampled users  |  top-{top_k_popular:,} popularity negatives")
    print(_SEP)

    # ── Pre-compute catalog lookups ───────────────────────────────────────────
    # Deduplicate items_encoded by item_idx so every lookup is unique.
    items_dedup = items_encoded_df.drop_duplicates("item_idx").set_index("item_idx")
    n_catalog   = len(items_dedup)

    # item_idx → (cat_l2_idx, brand_idx) as two numpy arrays for fast indexing
    max_item_idx   = int(items_dedup.index.max()) + 1
    cat2_lookup    = np.zeros(max_item_idx, dtype=np.int64)   # cat_l2_idx per item
    brand_lookup   = np.zeros(max_item_idx, dtype=np.int64)   # brand_idx per item
    cat2_lookup[items_dedup.index.values]  = items_dedup["cat_l2_idx"].values.astype(np.int64)
    brand_lookup[items_dedup.index.values] = items_dedup["brand_idx"].values.astype(np.int64)

    # (cat_l2_idx, brand_idx) → item count  [key pre-computation for O(1) risk calc]
    cat_brand_counts: dict[tuple[int, int], int] = (
        items_encoded_df.drop_duplicates("item_idx")
        .groupby(["cat_l2_idx", "brand_idx"])["item_idx"]
        .nunique()
        .to_dict()
    )
    # cat_l2_idx → total item count  (for denominator)
    cat2_total: dict[int, int] = (
        items_encoded_df.drop_duplicates("item_idx")
        .groupby("cat_l2_idx")["item_idx"]
        .nunique()
        .to_dict()
    )

    # Unknown-brand item count
    n_unknown_brand = int((items_dedup["brand_idx"] == 0).sum())
    pct_unknown_brand = 100 * n_unknown_brand / n_catalog

    print(f"\n  Catalog items       : {n_catalog:,}")
    print(f"  Unique cat_l2 bins  : {len(cat2_total):,}")
    print(f"  Items with unknown brand (brand_idx=0) : {n_unknown_brand:,}  "
          f"({pct_unknown_brand:.1f}% of catalog)")
    print(f"  Note: unknown-brand items are excluded from same-brand matching.")

    # ── Popularity: top-K items by training frequency ─────────────────────────
    item_freq = (
        train_pairs_df.groupby("item_idx")
        .size()
        .rename("freq")
        .sort_values(ascending=False)
    )
    top_k_actual = min(top_k_popular, len(item_freq))
    top_k_item_idxs = item_freq.index[:top_k_actual].values.astype(np.int64)
    top_k_cat2s     = set(int(cat2_lookup[i]) for i in top_k_item_idxs)

    print(f"\n  Top-{top_k_actual:,} popular items span {len(top_k_cat2s):,} unique cat_l2 bins "
          f"({100*len(top_k_cat2s)/len(cat2_total):.1f}% of all bins)")

    # ── Sample users ──────────────────────────────────────────────────────────
    unique_users = train_pairs_df["user_idx"].unique()
    n_sample     = min(n_users, len(unique_users))
    sampled_users = rng.choice(unique_users, size=n_sample, replace=False).astype(np.int64)

    # user_idx → array of item_idxs from train_pairs
    user_to_items: dict[int, np.ndarray] = {
        int(uid): grp["item_idx"].values.astype(np.int64)
        for uid, grp in train_pairs_df.groupby("user_idx")
    }

    print(f"\n  Sampled users : {n_sample:,}  "
          f"(of {len(unique_users):,} unique users in train_pairs)")

    # ── Per-user false-negative risk calculation ───────────────────────────────
    # Aggregates across all (user, positive_item) pairs
    agg_total_hard_neg  = 0   # Σ same-cat candidate pool sizes across all pairs
    agg_high_risk       = 0   # Σ same-cat-same-brand counts
    agg_pairs           = 0   # total (user, positive_item) pairs examined

    # Per-user metrics for distribution reporting
    per_user_fn_rate:    list[float] = []   # per-user high-risk fraction
    per_user_pop_risk:   list[float] = []   # per-user % of top-K in user's cat2s

    n_users_no_items = 0

    for user_idx in sampled_users:
        pos_items = user_to_items.get(int(user_idx))
        if pos_items is None or len(pos_items) == 0:
            n_users_no_items += 1
            continue

        # User's cat_l2 and brand sets across all their positives
        user_cat2s:  set[int] = set()
        user_brands: set[int] = set()   # excludes brand_idx=0 (unknown)
        for it in pos_items:
            user_cat2s.add(int(cat2_lookup[it]))
            b = int(brand_lookup[it])
            if b != 0:
                user_brands.add(b)

        # ── Category hard negative risk for this user ─────────────────────────
        user_total_hn  = 0
        user_high_risk = 0

        for it in pos_items:
            pos_cat2  = int(cat2_lookup[it])
            pos_brand = int(brand_lookup[it])

            # Denominator: all same-cat items except the positive itself
            pool_size = cat2_total.get(pos_cat2, 0) - 1
            if pool_size <= 0:
                continue

            # High-risk count: same (cat2, brand) as any user positive, excluding
            # the positive item itself (which contributes to its own brand bucket)
            hr = 0
            for b in user_brands:
                n_same_cat_brand = cat_brand_counts.get((pos_cat2, b), 0)
                if b == pos_brand:
                    n_same_cat_brand -= 1   # exclude the positive item itself
                hr += max(n_same_cat_brand, 0)

            user_total_hn  += pool_size
            user_high_risk += min(hr, pool_size)  # cap at pool_size (no double counting guard)

        if user_total_hn > 0:
            per_user_fn_rate.append(100 * user_high_risk / user_total_hn)
            agg_total_hard_neg += user_total_hn
            agg_high_risk      += user_high_risk
            agg_pairs          += len(pos_items)

        # ── Popularity negative risk for this user ────────────────────────────
        # Fraction of top-K popular items whose cat_l2 the user has interacted with
        n_pop_in_user_cats = sum(1 for c in (int(cat2_lookup[i]) for i in top_k_item_idxs)
                                 if c in user_cat2s)
        per_user_pop_risk.append(100 * n_pop_in_user_cats / top_k_actual)

    # ── Aggregate reporting ───────────────────────────────────────────────────
    n_valid_users = len(per_user_fn_rate)

    print(f"\n  {_THIN}")
    print(f"  CATEGORY HARD NEGATIVE FALSE-NEGATIVE RISK  (same cat_l2)")
    print(f"  {_THIN}")

    if n_valid_users == 0:
        print("  No valid users found — check that train_pairs and items_encoded overlap.")
        print(f"\n{_SEP}\n")
        return

    overall_fn_rate = 100 * agg_high_risk / agg_total_hard_neg

    fn_arr  = np.array(per_user_fn_rate,  dtype=np.float32)
    pop_arr = np.array(per_user_pop_risk, dtype=np.float32)

    print(f"\n  Aggregate over {n_valid_users:,} users / {agg_pairs:,} positive items:")
    print(f"\n  {'Candidate pool':<44}  {'count':>12}  {'%':>7}")
    print(f"  {'-'*44}  {'-'*12}  {'-'*7}")
    print(f"  {'Total same-cat hard neg candidates':<44}  "
          f"{agg_total_hard_neg:>12,}  {'100.00%':>7}")
    print(f"  {'High-risk (same cat_l2 + same brand)':<44}  "
          f"{agg_high_risk:>12,}  {overall_fn_rate:>6.2f}%")
    print(f"  {'Safer (same cat_l2 + different/no brand match)':<44}  "
          f"{agg_total_hard_neg - agg_high_risk:>12,}  "
          f"{100 - overall_fn_rate:>6.2f}%")

    print(f"\n  Per-user high-risk fraction distribution:")
    print(f"\n  {'Metric':<28}  {'same-cat-same-brand %':>22}")
    print(f"  {'-'*28}  {'-'*22}")
    for label, val in [
        ("Mean",        float(np.mean(fn_arr))),
        ("Median (p50)", float(np.median(fn_arr))),
        ("p10",         float(np.percentile(fn_arr, 10))),
        ("p25",         float(np.percentile(fn_arr, 25))),
        ("p75",         float(np.percentile(fn_arr, 75))),
        ("p90",         float(np.percentile(fn_arr, 90))),
        ("p99",         float(np.percentile(fn_arr, 99))),
    ]:
        print(f"  {label:<28}  {val:>21.2f}%")

    pct_above_20 = 100 * float(np.mean(fn_arr > 20))
    pct_below_5  = 100 * float(np.mean(fn_arr <  5))
    print(f"\n  Users with high-risk rate > 20%  :  {pct_above_20:.1f}%  "
          f"(these users need false-negative filtering)")
    print(f"  Users with high-risk rate <  5%  :  {pct_below_5:.1f}%  "
          f"(low risk — filtering optional)")

    # ── Popularity negative risk ──────────────────────────────────────────────
    print(f"\n  {_THIN}")
    print(f"  POPULARITY HARD NEGATIVE RISK  (top-{top_k_actual:,} items)")
    print(f"  {_THIN}")
    print(f"\n  For each user: what fraction of the top-{top_k_actual:,} popular items")
    print(f"  fall in a cat_l2 the user has already interacted with?")
    print(f"  Such items are potentially false negatives as popularity negatives.")

    print(f"\n  {'Metric':<28}  {'% of top-{k} in user cats':>26}".replace("{k}", str(top_k_actual)))
    print(f"  {'-'*28}  {'-'*26}")
    for label, val in [
        ("Mean",         float(np.mean(pop_arr))),
        ("Median (p50)", float(np.median(pop_arr))),
        ("p10",          float(np.percentile(pop_arr, 10))),
        ("p25",          float(np.percentile(pop_arr, 25))),
        ("p75",          float(np.percentile(pop_arr, 75))),
        ("p90",          float(np.percentile(pop_arr, 90))),
    ]:
        print(f"  {label:<28}  {val:>25.1f}%")

    pct_pop_above_50 = 100 * float(np.mean(pop_arr > 50))
    print(f"\n  Users where >50% of top-{top_k_actual:,} items are in their cat_l2s  :  "
          f"{pct_pop_above_50:.1f}%")
    print(f"  (for these users, popularity negatives are especially risky)")

    # ── Decision ─────────────────────────────────────────────────────────────
    print(f"\n  {_THIN}")
    print(f"  DECISION — FALSE NEGATIVE FILTERING REQUIRED?")
    print(f"  {_THIN}")
    print(f"\n  Overall same-cat-same-brand rate : {overall_fn_rate:.2f}%")
    print(f"  Threshold for required filtering  : > 20%")
    print(f"  Threshold for safe to ignore      : < 5%")
    print()

    if overall_fn_rate > 20:
        print(f"  FILTERING REQUIRED  ({overall_fn_rate:.2f}% > 20%)")
        print(f"  More than 1 in 5 proposed category hard negatives are high-risk false")
        print(f"  negatives (same sub-category AND same brand as a user positive).")
        print(f"  Recommendation: exclude items sharing both cat_l2 and brand_idx with")
        print(f"  any of the user's positive items before adding to the hard-negative pool.")
    elif overall_fn_rate > 5:
        print(f"  BORDERLINE  ({overall_fn_rate:.2f}% is between 5% and 20%)")
        print(f"  Same-cat-same-brand false negatives are non-trivial but not dominant.")
        print(f"  Options:")
        print(f"    (a) Apply lightweight brand-exclusion: remove items whose brand matches")
        print(f"        any of the user's positive brands from the hard-neg pool.")
        print(f"    (b) Accept the noise — {100 - overall_fn_rate:.2f}% of category hard")
        print(f"        negatives are still safer (different-brand) negatives.")
        print(f"    (c) Use a confidence-score discount rather than hard exclusion.")
    else:
        print(f"  LOW RISK  ({overall_fn_rate:.2f}% < 5%)")
        print(f"  Fewer than 1 in 20 category hard negatives are same-brand false positives.")
        print(f"  False negative filtering is NOT necessary.  Proceed with category-based")
        print(f"  hard negative sampling without brand exclusion.")

    mean_pop_risk = float(np.mean(pop_arr))
    print(f"\n  Popularity negative risk: on average {mean_pop_risk:.1f}% of the top-"
          f"{top_k_actual:,} items")
    print(f"  fall in a cat_l2 the user has interacted with.")
    if mean_pop_risk > 40:
        print(f"  HIGH — popular items frequently overlap with user interest categories.")
        print(f"  Consider filtering top-K negatives by cat_l2 as well.")
    elif mean_pop_risk > 20:
        print(f"  MODERATE — some popularity negatives will be in user-relevant categories.")
        print(f"  Acceptable as a secondary negative source but not as the primary one.")
    else:
        print(f"  LOW — popular items mostly fall outside user-interacted categories.")
        print(f"  Popularity-based negatives are relatively safe for this dataset.")

    print(f"\n{_SEP}\n")


# ── In-batch negative difficulty analysis ─────────────────────────────────────

def _count_same_cat_pairs(cat_arr: np.ndarray) -> int:
    """Count ordered same-category (positive, negative) pairs in a batch.

    If a category has k items in the batch, it contributes k*(k-1) ordered pairs
    where both items share that category.  This is O(B) via np.unique.
    """
    _, counts = np.unique(cat_arr, return_counts=True)
    return int(np.sum(counts * (counts - 1)))


def analyze_negative_difficulty(
    train_pairs_df: pd.DataFrame,
    items_encoded_df: pd.DataFrame,
    *,
    n_batches: int = 5,
    batch_size: int = 2048,
    top_k_popular: int = 1000,
    rare_threshold: int = 10,
    seed: int = 42,
) -> None:
    """Measure how semantically hard current in-batch negatives actually are.

    In standard in-batch negative training the off-diagonal items in a batch of B
    positive pairs serve as B-1 negatives for each query.  Because batches are
    sampled uniformly from train_pairs, popular items (those with many training
    appearances) are over-represented as in-batch negatives.  This function
    quantifies both the popularity skew and the category hardness of those
    negatives — answering whether hard negative mining is likely to help
    significantly.

    Args:
        train_pairs_df:   Training pairs with columns [user_idx, item_idx, …].
        items_encoded_df: Full item catalog with cat_l1_idx / cat_l2_idx columns.
        n_batches:        Number of random batches to simulate (default: 5).
        batch_size:       Pairs per simulated batch (default: 2048).
        top_k_popular:    Popularity rank cutoff for "popular" items (default: 1000).
        rare_threshold:   Training-frequency cutoff for "rare" items (default: 10).
        seed:             Random seed for reproducible batch sampling.
    """
    rng = np.random.default_rng(seed)

    n_pairs  = len(train_pairs_df)
    n_items  = items_encoded_df["item_idx"].nunique()

    print(f"\n{_SEP}")
    print(f"  ANALYZE IN-BATCH NEGATIVE DIFFICULTY")
    print(f"  {n_batches} batches × {batch_size:,} pairs  "
          f"(simulating real training; {n_pairs:,} total train pairs)")
    print(_SEP)

    # ════════════════════════════════════════════════════════════════
    # PART 1 — ITEM POPULARITY DISTRIBUTION
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {_THIN}")
    print(f"  PART 1 — ITEM POPULARITY  (training frequency per item)")
    print(f"  {_THIN}")

    item_freq = (
        train_pairs_df.groupby("item_idx")
        .size()
        .rename("freq")
        .reset_index()
        .sort_values("freq", ascending=False)
        .reset_index(drop=True)
    )
    # Assign popularity rank (1 = most popular)
    item_freq["pop_rank"] = np.arange(1, len(item_freq) + 1, dtype=np.int64)

    # Items that appear in train_pairs at all
    n_trained_items = len(item_freq)
    n_zero_items    = n_items - n_trained_items   # catalog items with no training pairs

    freq_vals = item_freq["freq"].values

    print(f"\n  Catalog items (items_encoded)        : {n_items:,}")
    print(f"  Items with ≥1 training pair          : {n_trained_items:,}  "
          f"({100*n_trained_items/n_items:.1f}%)")
    print(f"  Items with 0 training pairs          : {n_zero_items:,}  "
          f"({100*n_zero_items/n_items:.1f}%)")

    # Frequency percentiles
    print(f"\n  Training-frequency distribution (items with ≥1 pair):")
    print(f"  {'Metric':<28}  {'appearances':>12}")
    print(f"  {'-'*28}  {'-'*12}")
    for label, pct in [("p1", 1), ("p10", 10), ("p25", 25), ("p50 (median)", 50),
                        ("p75", 75), ("p90", 90), ("p99", 99), ("max", 100)]:
        val = np.percentile(freq_vals, pct) if pct < 100 else freq_vals.max()
        print(f"  {label:<28}  {val:>12,.0f}")

    # Rarity breakdown
    rare_mask    = freq_vals < rare_threshold
    n_rare       = rare_mask.sum()
    n_freq_1     = (freq_vals == 1).sum()
    pct_rare     = 100 * n_rare / n_trained_items

    print(f"\n  Items with < {rare_threshold} training appearances : "
          f"{n_rare:,}  ({pct_rare:.1f}% of trained items)")
    print(f"  Items with exactly 1 appearance    : "
          f"{n_freq_1:,}  ({100*n_freq_1/n_trained_items:.1f}% of trained items)")

    # Popularity concentration: what % of all training pairs are covered by top-K?
    top_k_actual = min(top_k_popular, n_trained_items)
    top_k_pairs  = item_freq["freq"].iloc[:top_k_actual].sum()
    pct_top_k_pairs = 100 * top_k_pairs / n_pairs

    print(f"\n  Top-{top_k_actual:,} most popular items cover  : "
          f"{top_k_pairs:,} / {n_pairs:,} training pairs  "
          f"({pct_top_k_pairs:.1f}%)")

    # Top-10 most popular items for inspection
    print(f"\n  Top-10 items by training frequency:")
    top10 = item_freq.head(10)[["pop_rank", "item_idx", "freq"]].copy()
    top10["cum_pct"] = 100 * item_freq["freq"].cumsum().iloc[:10].values / n_pairs
    _print_table(top10, float_cols={"cum_pct": "{:.2f}%"})

    # ── Build fast numpy lookups keyed by item_idx for Parts 2 & 3 ───────────

    max_item_idx = int(items_encoded_df["item_idx"].max()) + 1

    # Popularity rank lookup: 0 = item never appeared in training (rank ∞)
    pop_rank_arr = np.zeros(max_item_idx, dtype=np.int64)
    pop_rank_arr[item_freq["item_idx"].values] = item_freq["pop_rank"].values

    # Training frequency lookup
    freq_arr = np.zeros(max_item_idx, dtype=np.int64)
    freq_arr[item_freq["item_idx"].values] = item_freq["freq"].values

    # Category lookup arrays (0 = unknown / not in catalog)
    cat_l1_arr = np.zeros(max_item_idx, dtype=np.int64)
    cat_l2_arr = np.zeros(max_item_idx, dtype=np.int64)
    cat_l1_arr[items_encoded_df["item_idx"].values] = \
        items_encoded_df["cat_l1_idx"].values.astype(np.int64)
    cat_l2_arr[items_encoded_df["item_idx"].values] = \
        items_encoded_df["cat_l2_idx"].values.astype(np.int64)

    all_item_idxs = train_pairs_df["item_idx"].values.astype(np.int64)

    # ════════════════════════════════════════════════════════════════
    # PART 2 — IN-BATCH NEGATIVE POPULARITY
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {_THIN}")
    print(f"  PART 2 — IN-BATCH NEGATIVE POPULARITY  "
          f"({n_batches} batches × {batch_size:,} pairs)")
    print(f"  {_THIN}")
    print(f"\n  Each batch row (user_i, item_i) uses the other {batch_size-1:,} items")
    print(f"  as negatives.  Fraction from top-{top_k_actual:,} = fraction of the")
    print(f"  {batch_size:,} batch items that are in the top-{top_k_actual:,} by popularity.")

    header = (f"  {'batch':>6}  {'top-{k} neg %':>14}  "
              f"{'rare neg %':>12}  {'mean pop rank':>14}  {'median pop rank':>16}")
    header = header.replace("{k}", str(top_k_actual))
    print(f"\n{header}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*16}")

    batch_top_k_pcts:   list[float] = []
    batch_rare_pcts:    list[float] = []
    batch_mean_ranks:   list[float] = []
    batch_median_ranks: list[float] = []

    for b in range(n_batches):
        idx      = rng.choice(n_pairs, size=batch_size, replace=False)
        batch_items = all_item_idxs[idx]               # (B,) positive item_idxs

        ranks   = pop_rank_arr[batch_items]             # (B,) popularity ranks (0 = unseen)
        freqs   = freq_arr[batch_items]                 # (B,) training frequencies

        # Items in top-K: rank in [1, top_k_actual]  (0 means never appeared → not top-K)
        in_top_k  = ((ranks >= 1) & (ranks <= top_k_actual)).sum()
        is_rare   = (freqs < rare_threshold).sum()

        # For mean/median rank, treat rank-0 (unseen) as rank = n_trained_items + 1
        adj_ranks = np.where(ranks == 0, n_trained_items + 1, ranks).astype(np.float64)
        mean_rank   = float(adj_ranks.mean())
        median_rank = float(np.median(adj_ranks))

        pct_top_k = 100 * in_top_k / batch_size
        pct_rare  = 100 * is_rare  / batch_size

        batch_top_k_pcts.append(pct_top_k)
        batch_rare_pcts.append(pct_rare)
        batch_mean_ranks.append(mean_rank)
        batch_median_ranks.append(median_rank)

        print(f"  {b+1:>6}  {pct_top_k:>13.2f}%  {pct_rare:>11.2f}%  "
              f"{mean_rank:>14,.0f}  {median_rank:>16,.0f}")

    # Aggregate across batches
    avg_top_k    = float(np.mean(batch_top_k_pcts))
    avg_rare     = float(np.mean(batch_rare_pcts))
    avg_mean_rk  = float(np.mean(batch_mean_ranks))
    avg_med_rk   = float(np.mean(batch_median_ranks))

    print(f"  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*14}  {'─'*16}")
    print(f"  {'avg':>6}  {avg_top_k:>13.2f}%  {avg_rare:>11.2f}%  "
          f"{avg_mean_rk:>14,.0f}  {avg_med_rk:>16,.0f}")

    print(f"\n  Interpretation:")
    if avg_top_k > 50:
        print(f"  HIGH POPULARITY SKEW  {avg_top_k:.1f}% of in-batch negatives come from")
        print(f"  top-{top_k_actual:,} items.  The model sees mostly popular items as negatives")
        print(f"  and will learn to distinguish popular items from each other, but will")
        print(f"  not be challenged by rare catalog items it must retrieve.")
    elif avg_top_k > 20:
        print(f"  MODERATE POPULARITY SKEW  {avg_top_k:.1f}% from top-{top_k_actual:,}.")
        print(f"  Popular items dominate, but the batch still contains some long-tail items.")
    else:
        print(f"  LOW POPULARITY SKEW  Only {avg_top_k:.1f}% from top-{top_k_actual:,}.")
        print(f"  Batches are relatively diverse across the popularity spectrum.")

    # ════════════════════════════════════════════════════════════════
    # PART 3 — IN-BATCH NEGATIVE CATEGORY HARDNESS
    # ════════════════════════════════════════════════════════════════
    print(f"\n  {_THIN}")
    print(f"  PART 3 — IN-BATCH NEGATIVE CATEGORY HARDNESS  "
          f"({n_batches} batches × {batch_size:,} pairs)")
    print(f"  {_THIN}")
    print(f"\n  For each of the {batch_size:,} × {batch_size-1:,} = "
          f"{batch_size*(batch_size-1):,} off-diagonal (positive, negative) pairs")
    print(f"  per batch, we count how many share cat_l2 / cat_l1 with the positive.")
    print(f"  Uses an O(B) groupby method — exact, not sampled.")

    hdr2 = (f"  {'batch':>6}  {'same cat_l2':>12}  {'same cat_l1':>12}  "
            f"{'diff cat_l2 neg/pos':>20}  {'diff cat_l1 neg/pos':>20}")
    print(f"\n{hdr2}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*20}  {'-'*20}")

    batch_same_l2_pcts: list[float] = []
    batch_same_l1_pcts: list[float] = []

    # Reset rng to the same seed so batch composition is identical to Part 2
    rng2 = np.random.default_rng(seed)

    for b in range(n_batches):
        idx         = rng2.choice(n_pairs, size=batch_size, replace=False)
        batch_items = all_item_idxs[idx]   # (B,) positive item_idxs in this batch

        B     = len(batch_items)
        total = B * (B - 1)                # total ordered off-diagonal pairs

        batch_l1 = cat_l1_arr[batch_items]  # (B,) cat_l1_idx per batch item
        batch_l2 = cat_l2_arr[batch_items]  # (B,) cat_l2_idx per batch item

        same_l2_pairs = _count_same_cat_pairs(batch_l2)
        same_l1_pairs = _count_same_cat_pairs(batch_l1)

        pct_same_l2 = 100 * same_l2_pairs / total
        pct_same_l1 = 100 * same_l1_pairs / total

        batch_same_l2_pcts.append(pct_same_l2)
        batch_same_l1_pcts.append(pct_same_l1)

        print(f"  {b+1:>6}  {pct_same_l2:>11.3f}%  {pct_same_l1:>11.3f}%  "
              f"  {same_l2_pairs:>10,} / {total:>8,}  "
              f"  {same_l1_pairs:>10,} / {total:>8,}")

    avg_same_l2 = float(np.mean(batch_same_l2_pcts))
    avg_same_l1 = float(np.mean(batch_same_l1_pcts))

    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*20}  {'─'*20}")
    print(f"  {'avg':>6}  {avg_same_l2:>11.3f}%  {avg_same_l1:>11.3f}%")

    # ── Theoretical baseline: what same-cat fraction would pure uniform sampling give? ──
    # If each batch item were drawn uniformly from all catalog items (not from train_pairs),
    # the expected same-cat2 fraction in a large batch ≈ Σ (n_k/N)^2 (collision probability).
    n_cat_items = items_encoded_df.groupby("cat_l2_idx")["item_idx"].nunique()
    N = n_cat_items.sum()
    expected_same_l2_uniform = float(((n_cat_items / N) ** 2).sum()) * 100

    n_cat1_items = items_encoded_df.groupby("cat_l1_idx")["item_idx"].nunique()
    N1 = n_cat1_items.sum()
    expected_same_l1_uniform = float(((n_cat1_items / N1) ** 2).sum()) * 100

    print(f"\n  Theoretical same-cat fraction if batch were drawn uniformly from catalog:")
    print(f"    cat_l2 (expected) : {expected_same_l2_uniform:.3f}%")
    print(f"    cat_l1 (expected) : {expected_same_l1_uniform:.3f}%")
    print(f"  Observed / expected ratio:")
    print(f"    cat_l2 : {avg_same_l2 / expected_same_l2_uniform:.2f}×  "
          f"({'over' if avg_same_l2 > expected_same_l2_uniform else 'under'}-represented)")
    print(f"    cat_l1 : {avg_same_l1 / expected_same_l1_uniform:.2f}×  "
          f"({'over' if avg_same_l1 > expected_same_l1_uniform else 'under'}-represented)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  {_THIN}")
    print(f"  VERDICT — IS HARD NEGATIVE MINING LIKELY TO HELP?")
    print(f"  {_THIN}")

    print(f"\n  In-batch same-cat_l2 fraction : {avg_same_l2:.3f}%  "
          f"(hard negatives at cat_l2 level: {avg_same_l2:.3f}% of {batch_size*(batch_size-1):,})")
    print(f"  In-batch same-cat_l1 fraction : {avg_same_l1:.3f}%")

    if avg_same_l2 < 1.0:
        print(f"\n  STRONG SIGNAL FOR HARD NEGATIVES")
        print(f"  Only {avg_same_l2:.3f}% of in-batch negatives share cat_l2 with the positive.")
        print(f"  The model is currently trained almost exclusively on EASY negatives")
        print(f"  (items from completely unrelated categories).  Introducing hard")
        print(f"  negatives (same cat_l2 or cat_l1) will substantially change the")
        print(f"  training signal and is expected to improve retrieval quality.")
    elif avg_same_l2 < 5.0:
        print(f"\n  MODERATE SIGNAL FOR HARD NEGATIVES")
        print(f"  {avg_same_l2:.3f}% of in-batch negatives share cat_l2 — some same-category")
        print(f"  negatives are already present due to batch size {batch_size:,}, but the")
        print(f"  majority are still easy cross-category negatives.  Hard negative mining")
        print(f"  will increase same-category exposure and likely improve precision.")
    else:
        print(f"\n  WEAKER SIGNAL FOR HARD NEGATIVES")
        print(f"  {avg_same_l2:.3f}% of in-batch negatives already share cat_l2 with the")
        print(f"  positive.  The batch is large enough to naturally sample within-")
        print(f"  category negatives at a meaningful rate.  Hard negative mining may")
        print(f"  still help, but the marginal gain over pure in-batch training is smaller.")

    print(f"\n{_SEP}\n")


# ── Table printer ──────────────────────────────────────────────────────────────

def _print_table(
    df: pd.DataFrame,
    float_cols: dict[str, str] | None = None,
    indent: int = 2,
) -> None:
    """Print a DataFrame as a right-aligned fixed-width table."""
    float_cols = float_cols or {}

    pad = " " * indent

    # Compute column widths (max of header and formatted values)
    col_widths: dict[str, int] = {}
    formatted: dict[str, list[str]] = {}

    for col in df.columns:
        fmt = float_cols.get(col)
        if fmt:
            vals = [fmt.format(v) for v in df[col]]
        else:
            vals = [str(v) for v in df[col]]
        formatted[col] = vals
        col_widths[col] = max(len(col), max((len(v) for v in vals), default=0))

    header = "  ".join(col.rjust(col_widths[col]) for col in df.columns)
    rule   = "  ".join("-" * col_widths[col] for col in df.columns)

    print(f"{pad}{header}")
    print(f"{pad}{rule}")
    for row_idx in range(len(df)):
        row = "  ".join(
            formatted[col][row_idx].rjust(col_widths[col]) for col in df.columns
        )
        print(f"{pad}{row}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    artifacts_dir = pathlib.Path(args.artifacts_dir).resolve()

    if not artifacts_dir.exists():
        sys.exit(f"ERROR: artifacts directory not found: {artifacts_dir}")

    print(_SEP)
    print(f"  Hard-Negative Mining Analysis")
    print(f"  Artifacts : {artifacts_dir}")
    if not args.skip_session:
        print(f"  Train path: {args.train_path}")
    print(_SEP)

    items_encoded_df, train_pairs_df, vocabs = _load_artifacts(artifacts_dir)

    print(f"\n  items_encoded shape : {items_encoded_df.shape}")
    print(f"  train_pairs shape   : {train_pairs_df.shape}")

    # ── Part A: category granularity ─────────────────────────────────────────
    analyze_category_granularity(
        items_encoded_df=items_encoded_df,
        train_pairs_df=train_pairs_df,
        sample_size=args.sample_size,
        seed=args.seed,
        vocabs=vocabs,
    )

    # ── Part B: session-based negatives ──────────────────────────────────────
    if args.skip_session:
        print("  (--skip-session set: skipping analyze_session_negatives)")
    else:
        analyze_session_negatives(
            args.train_path,
            items_encoded_df=items_encoded_df,
            artifacts_dir=artifacts_dir,
            seed=args.seed,
        )

    # ── Part C: false negative risk ──────────────────────────────────────────
    if args.skip_fn_risk:
        print("  (--skip-fn-risk set: skipping assess_false_negative_risk)")
    else:
        assess_false_negative_risk(
            train_pairs_df=train_pairs_df,
            items_encoded_df=items_encoded_df,
            n_users=args.n_fn_users,
            top_k_popular=args.top_popular,
            seed=args.seed,
        )

    # ── Part D: intra-category score distribution ────────────────────────────
    if args.skip_scoring:
        print("  (--skip-scoring set: skipping analyze_intra_category_scores)")
    else:
        ckpt_path = (
            pathlib.Path(args.checkpoint).resolve()
            if args.checkpoint
            else artifacts_dir / "checkpoints" / "epoch_5.pt"
        )
        analyze_intra_category_scores(
            checkpoint_path=ckpt_path,
            items_encoded_df=items_encoded_df,
            train_pairs_df=train_pairs_df,
            artifacts_dir=artifacts_dir,
            seed=args.seed,
        )

    # ── Part E: in-batch negative difficulty ─────────────────────────────────
    if args.skip_difficulty:
        print("  (--skip-difficulty set: skipping analyze_negative_difficulty)")
        return

    analyze_negative_difficulty(
        train_pairs_df=train_pairs_df,
        items_encoded_df=items_encoded_df,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
