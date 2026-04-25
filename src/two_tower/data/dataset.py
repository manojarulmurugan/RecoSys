"""PyTorch Dataset for the Two-Tower training loop.

Wraps pre-encoded parquet artifacts into fast numpy lookups so that
DataLoader workers spend no time in pandas during training.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    """Maps (user_idx, item_idx, confidence_score) training pairs to
    feature tensors via O(1) numpy array lookups.

    Auto-detects V2 features:
      - If ``items_encoded_df`` contains ``price_relative_to_cat_avg_scaled``
        and ``product_recency_log_scaled``, those are included as the 4th and
        5th elements of ``item_dense`` (dim becomes 5 instead of 3).
      - ``user_dense`` always includes sin/cos DOW computed from
        ``preferred_dow``, making it 8-dim instead of 6.

    Args:
        train_pairs_df:    DataFrame with columns [user_idx, item_idx, confidence_score].
        users_encoded_df:  DataFrame produced by FeatureBuilder.build() for users.
        items_encoded_df:  DataFrame produced by FeatureBuilder.build() for items.
                           Pass ``items_encoded_v2.parquet`` to enable V2 item features.
    """

    # V2 item dense columns (in addition to the original 3)
    _V2_ITEM_DENSE_COLS = ["price_relative_to_cat_avg_scaled", "product_recency_log_scaled"]

    # V3 user dense columns — item centroid (in addition to the 8 V2 dims)
    _CENTROID_DIM: int = 32
    _CENTROID_COLS = [f"item_centroid_{i}" for i in range(_CENTROID_DIM)]

    def __init__(
        self,
        train_pairs_df: pd.DataFrame,
        users_encoded_df: pd.DataFrame,
        items_encoded_df: pd.DataFrame,
    ) -> None:
        self.pairs = train_pairs_df.reset_index(drop=True)

        # ── User feature lookups keyed by user_idx ────────────────────────────
        n_users = int(users_encoded_df["user_idx"].max()) + 1

        # int64: [top_cat_idx, peak_hour_bucket, preferred_dow, has_purchase_history]
        user_cat_arr = np.zeros((n_users, 4), dtype=np.int64)
        user_cat_arr[users_encoded_df["user_idx"].values] = users_encoded_df[
            ["top_cat_idx", "peak_hour_bucket", "preferred_dow", "has_purchase_history"]
        ].values.astype(np.int64)

        # float32: 6 original + sin_dow + cos_dow = 8 dims (V2)
        # If item_centroid_* columns are present (V3 users), append 32 more → 40 dims.
        dow_vals = users_encoded_df["preferred_dow"].values.astype(np.float32)
        sin_dow  = np.sin(2.0 * np.pi * dow_vals / 7.0).reshape(-1, 1)
        cos_dow  = np.cos(2.0 * np.pi * dow_vals / 7.0).reshape(-1, 1)

        base_dense = users_encoded_df[
            ["log_total_events", "months_active", "purchase_rate", "cart_rate",
             "log_n_sessions", "avg_purchase_price_scaled"]
        ].values.astype(np.float32)

        use_centroid = all(c in users_encoded_df.columns for c in self._CENTROID_COLS)
        if use_centroid:
            centroid = users_encoded_df[self._CENTROID_COLS].values.astype(np.float32)
            dense_data = np.hstack([base_dense, sin_dow, cos_dow, centroid])  # (n, 40)
        else:
            dense_data = np.hstack([base_dense, sin_dow, cos_dow])            # (n, 8)

        n_user_dense = dense_data.shape[1]
        user_dense_arr = np.zeros((n_users, n_user_dense), dtype=np.float32)
        user_dense_arr[users_encoded_df["user_idx"].values] = dense_data

        self._user_cat      = user_cat_arr
        self._user_dense    = user_dense_arr
        self._use_centroid  = use_centroid

        # ── Item feature lookups keyed by item_idx ────────────────────────────
        n_items = int(items_encoded_df["item_idx"].max()) + 1

        # int64: [item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket]
        item_cat_arr = np.zeros((n_items, 5), dtype=np.int64)
        item_cat_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
        ].values.astype(np.int64)

        # float32: 3 original + 2 V2 if present
        base_dense_cols = ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
        v2_cols_present = all(c in items_encoded_df.columns for c in self._V2_ITEM_DENSE_COLS)
        if v2_cols_present:
            dense_cols = base_dense_cols + list(self._V2_ITEM_DENSE_COLS)
        else:
            dense_cols = base_dense_cols

        n_item_dense = len(dense_cols)
        item_dense_arr = np.zeros((n_items, n_item_dense), dtype=np.float32)
        item_dense_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            dense_cols
        ].values.astype(np.float32)

        self._item_cat        = item_cat_arr
        self._item_dense      = item_dense_arr
        self._use_v2_items    = v2_cols_present

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row      = self.pairs.iloc[idx]
        user_idx = int(row.user_idx)
        item_idx = int(row.item_idx)

        return {
            "user_idx":    torch.tensor(user_idx,                    dtype=torch.long),
            "item_idx":    torch.tensor(item_idx,                    dtype=torch.long),
            "user_cat":    torch.tensor(self._user_cat[user_idx],    dtype=torch.long),
            "user_dense":  torch.tensor(self._user_dense[user_idx],  dtype=torch.float32),
            "item_cat":    torch.tensor(self._item_cat[item_idx],    dtype=torch.long),
            "item_dense":  torch.tensor(self._item_dense[item_idx],  dtype=torch.float32),
            "confidence":  torch.tensor(float(row.confidence_score), dtype=torch.float32),
        }


class TwoTowerDatasetWithHardNegs(Dataset):
    """Two-Tower dataset with pre-mined hard negatives.

    Hard negative strategy (decided from data investigation), now
    *interaction-aware*: the routing depends on each user's training-pair count.

    For pairs whose positive item has ``cat_l2_idx != 0``:
      * **Cold users** (``user_train_counts[user_idx] < WARM_THRESHOLD``):
        Sample 3 items from the positive's ``cat_l2`` (legacy behaviour).
        At low interaction counts the user's category footprint is small,
        so same-cat negatives are unlikely to be false negatives.
      * **Warm users** (``user_train_counts[user_idx] >= WARM_THRESHOLD``):
        Sample 3 items from the *complement* of the user's interacted
        ``cat_l2`` set — i.e. categories the user has never engaged with.
        This avoids surfacing hard negatives in categories the user has
        already shown preference for, which become likely false negatives
        as interaction history grows.  If the complement pool has fewer
        than ``n_hard_negs`` items (extreme power user who has touched
        almost everything), we fall back to the unfiltered same-cat pool.

    For pairs whose positive item has ``cat_l2_idx == 0`` (unknown):
      Sample 3 items from the positive's ``price_bucket`` (unchanged).

    All mining is done once in ``__init__`` and stored in a ``(N, 3)`` int64
    tensor so that ``__getitem__`` is a simple O(1) lookup.

    Args:
        train_pairs_df:    DataFrame with columns [user_idx, item_idx, confidence_score].
        users_encoded_df:  DataFrame produced by FeatureBuilder.build() for users.
        items_encoded_df:  DataFrame produced by FeatureBuilder.build() for items.
        n_hard_negs:       Hard negatives per positive (default: 3).
        seed:              NumPy random seed for reproducible mining (default: 42).
        hard_neg_cache_path:
            Optional path to a ``.npy`` file of shape ``(N, n_hard_negs)`` int64.
            If the file exists and matches ``len(train_pairs_df)`` and ``n_hard_negs``,
            mining is skipped (fast restart after installing faiss, etc.).
            After a successful mine, the array (and a small ``.meta.json``) is written
            here when this path is provided.

    Class constants:
        WARM_THRESHOLD:    Minimum train-pair count to route a user to the
                           warm (anti-category) strategy.  Below this we
                           keep the legacy same-cat sampling.
    """

    _N_HARD: int = 3   # hard negatives per positive pair
    WARM_THRESHOLD: int = 20

    def __init__(
        self,
        train_pairs_df: pd.DataFrame,
        users_encoded_df: pd.DataFrame,
        items_encoded_df: pd.DataFrame,
        n_hard_negs: int = 3,
        seed: int = 42,
        hard_neg_cache_path: str | pathlib.Path | None = None,
    ) -> None:
        self.pairs = train_pairs_df.reset_index(drop=True)
        self._n_hard = n_hard_negs

        # ── User feature lookups (identical to TwoTowerDataset) ───────────────
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

        self._user_cat   = user_cat_arr
        self._user_dense = user_dense_arr

        # ── Item feature lookups (identical to TwoTowerDataset) ───────────────
        n_items = int(items_encoded_df["item_idx"].max()) + 1

        item_cat_arr = np.zeros((n_items, 5), dtype=np.int64)
        item_cat_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]
        ].values.astype(np.int64)

        item_dense_arr = np.zeros((n_items, 3), dtype=np.float32)
        item_dense_arr[items_encoded_df["item_idx"].values] = items_encoded_df[
            ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
        ].values.astype(np.float32)

        self._item_cat   = item_cat_arr
        self._item_dense = item_dense_arr

        cache_path: pathlib.Path | None = None
        if hard_neg_cache_path is not None:
            cache_path = pathlib.Path(hard_neg_cache_path).expanduser().resolve()

        loaded_from_cache = False
        if cache_path is not None and cache_path.is_file():
            loaded_arr = np.load(cache_path, mmap_mode="r")
            if loaded_arr.shape == (len(self.pairs), self._n_hard) and np.issubdtype(
                loaded_arr.dtype, np.integer
            ):
                self._hard_neg_idxs = torch.from_numpy(
                    np.ascontiguousarray(loaded_arr.astype(np.int64, copy=True))
                )
                meta_path = cache_path.with_suffix(".meta.json")
                if meta_path.is_file():
                    with open(meta_path, encoding="utf-8") as mf:
                        meta = json.load(mf)
                    self._n_cat_l2_pairs        = int(meta.get("n_cat_l2_pairs",        -1))
                    self._n_fallback_pairs      = int(meta.get("n_fallback_pairs",      -1))
                    self._n_warm_users          = int(meta.get("n_warm_users",          -1))
                    self._n_cold_users          = int(meta.get("n_cold_users",          -1))
                    self._n_warm_pairs          = int(meta.get("n_warm_pairs",          -1))
                    self._n_warm_fallback_pairs = int(meta.get("n_warm_fallback_pairs", -1))
                    self._warm_threshold_used   = int(meta.get("warm_threshold",
                                                               self.WARM_THRESHOLD))
                else:
                    self._n_cat_l2_pairs        = -1
                    self._n_fallback_pairs      = -1
                    self._n_warm_users          = -1
                    self._n_cold_users          = -1
                    self._n_warm_pairs          = -1
                    self._n_warm_fallback_pairs = -1
                    self._warm_threshold_used   = self.WARM_THRESHOLD
                loaded_from_cache = True
                print(
                    f"  Loaded hard-negative cache: {cache_path}  "
                    f"shape={tuple(self._hard_neg_idxs.shape)}"
                )
            else:
                print(
                    f"  WARNING: cache {cache_path} has shape {loaded_arr.shape} / dtype "
                    f"{loaded_arr.dtype}; expected ({len(self.pairs)}, {self._n_hard}) int64 "
                    f"— re-mining."
                )

        if not loaded_from_cache:
            # ── Hard-negative candidate pools ─────────────────────────────────
            # col 2 of item_cat_arr = cat_l2_idx; col 4 = price_bucket
            print(
                "  [hard-negs] Stage 1/3: building cat_l2 / price_bucket pools …",
                flush=True,
            )
            t_stage = time.perf_counter()
            items_dedup = items_encoded_df.drop_duplicates("item_idx")

            cat_l2_to_items: dict[int, np.ndarray] = {}
            for cat_l2, grp in items_dedup.groupby("cat_l2_idx"):
                cat_l2_to_items[int(cat_l2)] = grp["item_idx"].values.astype(np.int64)

            price_bucket_to_items: dict[int, np.ndarray] = {}
            for bucket, grp in items_dedup.groupby("price_bucket"):
                price_bucket_to_items[int(bucket)] = grp["item_idx"].values.astype(
                    np.int64
                )
            print(
                f"  [hard-negs] Stage 1 done in {time.perf_counter() - t_stage:.1f}s  "
                f"(cat_l2 bins={len(cat_l2_to_items):,}, "
                f"price_bucket bins={len(price_bucket_to_items):,})",
                flush=True,
            )

            # ── Per-user positive item sets (for exclusion during mining) ─────
            print(
                "  [hard-negs] Stage 2/3: grouping train_pairs by user_idx …",
                flush=True,
            )
            t_stage = time.perf_counter()
            user_positives: dict[int, set[int]] = {}
            for uid, grp in self.pairs.groupby("user_idx"):
                user_positives[int(uid)] = set(grp["item_idx"].values.tolist())
            print(
                f"  [hard-negs] Stage 2 done in {time.perf_counter() - t_stage:.1f}s  "
                f"({len(user_positives):,} users with ≥1 positive)",
                flush=True,
            )

            # ── Stage 2.5: per-user counts + interacted-category footprints ──
            print(
                "  [hard-negs] Stage 2.5/3: per-user counts + interacted cats …",
                flush=True,
            )
            t_stage = time.perf_counter()

            user_train_counts: dict[int, int] = {
                int(uid): int(c)
                for uid, c in self.pairs.groupby("user_idx").size().items()
            }
            user_interacted_cats: dict[int, set[int]] = {
                uid: {int(item_cat_arr[it, 2]) for it in items}
                for uid, items in user_positives.items()
            }

            # All cat_l2 values that have at least one item, excluding 0 (unknown)
            all_cats_arr = np.array(
                [int(c) for c in cat_l2_to_items if int(c) != 0], dtype=np.int64
            )
            cat_sizes: dict[int, int] = {
                int(c): int(len(arr)) for c, arr in cat_l2_to_items.items()
            }

            # Per warm user: array of cat_l2 values they have NOT interacted with
            # (and the total catalog size of that anti-pool, used for the
            # < n_hard fallback check).
            warm_anti_cats: dict[int, np.ndarray] = {}
            warm_anti_size: dict[int, int]        = {}
            n_warm_users = 0
            for uid, count in user_train_counts.items():
                if count >= self.WARM_THRESHOLD:
                    n_warm_users += 1
                    ucats = user_interacted_cats.get(uid, set())
                    anti = all_cats_arr[~np.isin(all_cats_arr, list(ucats))] \
                           if len(ucats) > 0 else all_cats_arr
                    warm_anti_cats[uid] = anti
                    warm_anti_size[uid] = int(
                        sum(cat_sizes[int(c)] for c in anti)
                    )
            n_cold_users = len(user_train_counts) - n_warm_users
            print(
                f"  [hard-negs] Stage 2.5 done in "
                f"{time.perf_counter() - t_stage:.1f}s  "
                f"(warm users ≥{self.WARM_THRESHOLD}: {n_warm_users:,}, "
                f"cold: {n_cold_users:,})",
                flush=True,
            )

            # ── Pre-mine all hard negatives ───────────────────────────────────
            rng = np.random.default_rng(seed)

            n_pairs = len(self.pairs)
            hard_neg_idxs = np.empty((n_pairs, self._n_hard), dtype=np.int64)

            n_cat_l2        = 0   # pairs satisfied by primary (cat_l2) strategy
            n_fallback      = 0   # pairs using price_bucket fallback (cat_l2==0)
            n_warm_pairs    = 0   # cat_l2-path pairs from warm users
            n_warm_fallback = 0   # warm pairs whose anti-cat pool was < n_hard
                                  # and had to fall back to the unfiltered same-cat pool

            pair_user_idxs = self.pairs["user_idx"].values.astype(np.int64)
            pair_item_idxs = self.pairs["item_idx"].values.astype(np.int64)

            # Progress: ~20 lines for large N (Colab-friendly)
            report_every = max(250_000, n_pairs // 20)
            print(
                f"  [hard-negs] Stage 3/3: sampling {self._n_hard} negs per pair "
                f"({n_pairs:,} rows, log every {report_every:,}) …",
                flush=True,
            )
            t_mine0 = time.perf_counter()

            for i in range(n_pairs):
                user_idx = int(pair_user_idxs[i])
                item_idx = int(pair_item_idxs[i])

                cat_l2  = int(item_cat_arr[item_idx, 2])   # col 2 = cat_l2_idx
                bucket  = int(item_cat_arr[item_idx, 4])   # col 4 = price_bucket
                pos_set = user_positives.get(user_idx, set())
                is_warm = user_train_counts.get(user_idx, 0) >= self.WARM_THRESHOLD

                chosen_via_warm = False

                if cat_l2 != 0:
                    n_cat_l2 += 1
                    same_cat_pool = cat_l2_to_items.get(
                        cat_l2, np.empty(0, dtype=np.int64)
                    )

                    if is_warm:
                        n_warm_pairs += 1
                        anti_cats = warm_anti_cats.get(user_idx)
                        anti_size = warm_anti_size.get(user_idx, 0)

                        if (
                            anti_cats is not None
                            and len(anti_cats) > 0
                            and anti_size >= self._n_hard
                        ):
                            # Draw n_hard cats from the anti-cat array, then one
                            # item per cat.  This avoids building a huge
                            # concatenated pool per pair while still sampling
                            # exclusively from cats the user has never touched.
                            cat_choices = rng.choice(
                                anti_cats, size=self._n_hard, replace=True
                            )
                            chosen = np.empty(self._n_hard, dtype=np.int64)
                            for k in range(self._n_hard):
                                chosen[k] = rng.choice(
                                    cat_l2_to_items[int(cat_choices[k])]
                                )
                            hard_neg_idxs[i] = chosen
                            chosen_via_warm = True
                        else:
                            # Anti-pool exhausted → use unfiltered same-cat pool
                            n_warm_fallback += 1
                            pool = same_cat_pool
                    else:
                        # Cold user — legacy same-cat behaviour
                        pool = same_cat_pool
                else:
                    pool = price_bucket_to_items.get(
                        bucket, np.empty(0, dtype=np.int64)
                    )
                    n_fallback += 1

                if not chosen_via_warm:
                    # Exclude the user's own positive items from the candidate pool
                    if len(pos_set) > 0:
                        pool = pool[~np.isin(pool, list(pos_set))]

                    if len(pool) == 0:
                        # Degenerate: pool is empty after exclusion — fall back
                        # to any item.
                        pool = items_dedup["item_idx"].values.astype(np.int64)
                        pool = pool[pool != item_idx]

                    replace = len(pool) < self._n_hard
                    chosen  = rng.choice(pool, size=self._n_hard, replace=replace)
                    hard_neg_idxs[i] = chosen

                done = i + 1
                if done % report_every == 0 or done == n_pairs:
                    elapsed = time.perf_counter() - t_mine0
                    rate    = done / elapsed if elapsed > 0 else 0.0
                    remain  = n_pairs - done
                    eta_s   = remain / rate if rate > 0 else float("nan")
                    pct     = 100.0 * done / n_pairs
                    print(
                        f"  [hard-negs] {done:>10,} / {n_pairs:,}  "
                        f"({pct:5.1f}%)  "
                        f"{rate:,.0f} pairs/s  "
                        f"elapsed {elapsed / 60:.1f} min  "
                        f"ETA ~{eta_s / 60:.1f} min",
                        flush=True,
                    )

            warm_fb_pct = (
                f"{100 * n_warm_fallback / n_warm_pairs:.1f}%"
                if n_warm_pairs > 0
                else "n/a"
            )
            print(
                f"  [hard-negs] Stage 3 done in "
                f"{(time.perf_counter() - t_mine0) / 60:.1f} min  "
                f"(cat_l2 rows={n_cat_l2:,}, "
                f"price_bucket fallback rows={n_fallback:,}, "
                f"warm pairs={n_warm_pairs:,}, "
                f"warm→same-cat fallback={n_warm_fallback:,} ({warm_fb_pct}))",
                flush=True,
            )

            self._hard_neg_idxs = torch.from_numpy(hard_neg_idxs)   # (N, 3) int64

            # Stats for __repr__
            self._n_cat_l2_pairs        = n_cat_l2
            self._n_fallback_pairs      = n_fallback
            self._n_warm_users          = n_warm_users
            self._n_cold_users          = n_cold_users
            self._n_warm_pairs          = n_warm_pairs
            self._n_warm_fallback_pairs = n_warm_fallback
            self._warm_threshold_used   = int(self.WARM_THRESHOLD)

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, hard_neg_idxs)
                meta_path = cache_path.with_suffix(".meta.json")
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(
                        {
                            "n_pairs":              int(n_pairs),
                            "n_hard_negs":          int(self._n_hard),
                            "seed":                 int(seed),
                            "n_cat_l2_pairs":       int(n_cat_l2),
                            "n_fallback_pairs":     int(n_fallback),
                            "warm_threshold":       int(self.WARM_THRESHOLD),
                            "n_warm_users":         int(n_warm_users),
                            "n_cold_users":         int(n_cold_users),
                            "n_warm_pairs":         int(n_warm_pairs),
                            "n_warm_fallback_pairs": int(n_warm_fallback),
                        },
                        mf,
                        indent=2,
                    )
                print(f"  Saved hard-negative cache: {cache_path}")

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row      = self.pairs.iloc[idx]
        user_idx = int(row.user_idx)
        item_idx = int(row.item_idx)

        return {
            "user_idx":      torch.tensor(user_idx,                    dtype=torch.long),
            "item_idx":      torch.tensor(item_idx,                    dtype=torch.long),
            "user_cat":      torch.tensor(self._user_cat[user_idx],    dtype=torch.long),
            "user_dense":    torch.tensor(self._user_dense[user_idx],  dtype=torch.float32),
            "item_cat":      torch.tensor(self._item_cat[item_idx],    dtype=torch.long),
            "item_dense":    torch.tensor(self._item_dense[item_idx],  dtype=torch.float32),
            "confidence":    torch.tensor(float(row.confidence_score), dtype=torch.float32),
            "hard_neg_idxs": self._hard_neg_idxs[idx],   # (3,) int64
        }

    def __repr__(self) -> str:
        n    = len(self.pairs)
        n_l2 = self._n_cat_l2_pairs
        n_fb = self._n_fallback_pairs

        if n_l2 < 0:
            return (
                f"TwoTowerDatasetWithHardNegs("
                f"pairs={n:,}, loaded_from_cache=True, "
                f"hard_negs_per_pair={self._n_hard})"
            )

        ratio = f"{100 * n_l2 / n:.1f}% cat_l2 / {100 * n_fb / n:.1f}% price_bucket"
        parts = [
            f"pairs={n:,}",
            f"cat_l2={n_l2:,}",
            f"price_bucket_fallback={n_fb:,}",
            f"ratio=[{ratio}]",
            f"hard_negs_per_pair={self._n_hard}",
        ]

        n_wu = getattr(self, "_n_warm_users", -1)
        if n_wu >= 0:
            n_cu  = self._n_cold_users
            n_wp  = self._n_warm_pairs
            n_wf  = self._n_warm_fallback_pairs
            thr   = getattr(self, "_warm_threshold_used", self.WARM_THRESHOLD)
            wf_pct = (
                f"{100 * n_wf / n_wp:.1f}%" if n_wp > 0 else "n/a"
            )
            parts.extend([
                f"warm_threshold={thr}",
                f"warm_users={n_wu:,}",
                f"cold_users={n_cu:,}",
                f"warm_pairs={n_wp:,}",
                f"warm_anti_pool_fallback={n_wf:,} ({wf_pct} of warm pairs)",
            ])

        return "TwoTowerDatasetWithHardNegs(" + ", ".join(parts) + ")"


class TwoTowerDatasetWithSeq(TwoTowerDataset):
    """TwoTowerDataset extended with a pre-computed user item-history sequence.

    For each user, the last ``seq_len`` item indices from ``train_pairs_df``
    are stored in ``_user_seq`` (shape ``(n_users, seq_len)``).  Row order in
    ``train_pairs_df`` is assumed to follow interaction time (the FeatureBuilder
    preserves the temporal ordering of the source interaction log).  Items are
    left-padded with 0 for users whose history is shorter than ``seq_len``.

    The extra ``user_seq`` key in each batch feeds ``SequentialUserTower``.

    Args:
        train_pairs_df:   Same as TwoTowerDataset.
        users_encoded_df: Same as TwoTowerDataset.
        items_encoded_df: Same as TwoTowerDataset (V2 parquet recommended).
        seq_len:          History window length.  Default 20 matches the
                          SequentialUserTower default.
    """

    def __init__(
        self,
        train_pairs_df: pd.DataFrame,
        users_encoded_df: pd.DataFrame,
        items_encoded_df: pd.DataFrame,
        seq_len: int = 20,
    ) -> None:
        super().__init__(train_pairs_df, users_encoded_df, items_encoded_df)

        n_users = self._user_dense.shape[0]
        user_seq_arr = np.zeros((n_users, seq_len), dtype=np.int64)

        print(f"  Building user item sequences (seq_len={seq_len})...")
        n_processed = 0
        for uid, grp in train_pairs_df.groupby("user_idx", sort=False)["item_idx"]:
            uid_int = int(uid)
            if uid_int >= n_users:
                continue
            items = grp.values.astype(np.int64)
            if len(items) >= seq_len:
                user_seq_arr[uid_int] = items[-seq_len:]
            else:
                user_seq_arr[uid_int, seq_len - len(items):] = items
            n_processed += 1
            if n_processed % 100_000 == 0:
                print(f"    {n_processed:,} users processed...")

        self._user_seq = user_seq_arr
        print(f"  User sequences ready: {n_users:,} users, {seq_len} items each.")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        base = super().__getitem__(idx)
        user_idx = int(base["user_idx"])
        base["user_seq"] = torch.tensor(self._user_seq[user_idx], dtype=torch.long)
        return base


def build_full_item_tensors(
    items_encoded_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense tensors over ALL items for FAISS index construction at inference time.

    Items are sorted by item_idx ascending so that tensor row i corresponds to item_idx i.
    Auto-detects V2 item features: if ``price_relative_to_cat_avg_scaled`` and
    ``product_recency_log_scaled`` are present in ``items_encoded_df``, they are
    appended to item_dense (making it 5-dim instead of 3).

    Args:
        items_encoded_df: DataFrame produced by FeatureBuilder.build() for items,
                          or the V2 augmented version with extra dense columns.

    Returns:
        item_cat_tensor:   LongTensor of shape (n_items, 5)
                           columns: [item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket]
        item_dense_tensor: FloatTensor of shape (n_items, 3 or 5)
                           columns: [avg_price_scaled, log_confidence_scaled, purchase_rate_scaled,
                                     (price_relative_to_cat_avg_scaled, product_recency_log_scaled)]
    """
    _V2_COLS = ["price_relative_to_cat_avg_scaled", "product_recency_log_scaled"]
    df = items_encoded_df.sort_values("item_idx").reset_index(drop=True)

    item_cat_tensor = torch.tensor(
        df[["item_idx", "cat_l1_idx", "cat_l2_idx", "brand_idx", "price_bucket"]].values,
        dtype=torch.long,
    )

    dense_cols = ["avg_price_scaled", "log_confidence_scaled", "purchase_rate_scaled"]
    if all(c in df.columns for c in _V2_COLS):
        dense_cols = dense_cols + _V2_COLS

    item_dense_tensor = torch.tensor(
        df[dense_cols].values,
        dtype=torch.float32,
    )
    return item_cat_tensor, item_dense_tensor
