"""Build users_encoded_v2.parquet — adds a 32-dim item-centroid feature.

Strategy (YouTube Two-Tower, section 3.1):
  For each user, average the 32-dim item_emb lookup-table vectors of all
  cart+purchase interactions (confidence_score >= 2) from the training set.
  Users with no high-intent history get a zero centroid vector, so they fall
  back gracefully to their other static features.

Why the item_emb table (32-d) and not the item tower output (64-d)?
  - The item tower output is L2-normalised and post-MLP, so it lives in the
    retrieval space but discards low-level embedding geometry.
  - The raw 32-d embedding captures the learned item identity before
    projection, which is closer to how YouTube uses it.
  - Keeps UserTowerV3's MLP input manageable (97-d vs 129-d for 64-d variant).

Usage:
    # Run ONCE before training V6. Requires a trained ItemTower checkpoint.
    python scripts/two_tower/build_users_v2_500k.py \\
        --checkpoint artifacts/500k/checkpoints_v4/epoch_05.pt

Output:
    artifacts/500k/users_encoded_v2.parquet
      — all columns of users_encoded.parquet plus
        item_centroid_0 .. item_centroid_31  (float32)
"""

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

ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "500k"


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
    print("WARNING: No service-account JSON found.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=ARTIFACTS_DIR / "checkpoints_v4" / "epoch_05.pt",
        help="Path to a trained TwoTowerModel checkpoint (.pt). "
             "The item_tower.item_emb weights are extracted from this file. "
             "Default: artifacts/500k/checkpoints_v4/epoch_05.pt",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=ARTIFACTS_DIR / "users_encoded_v2.parquet",
        help="Destination for the augmented users parquet.",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=2.0,
        help="Minimum confidence_score to include a pair in the centroid "
             "(default 2.0 keeps cart and purchase interactions only).",
    )
    return p.parse_args()


def main() -> None:
    _ensure_gcp_credentials()
    args = _parse_args()

    # ── Load artifacts ────────────────────────────────────────────────────────
    print("Loading artifacts...")
    with open(ARTIFACTS_DIR / "vocabs.pkl", "rb") as f:
        vocabs = pickle.load(f)

    users_enc   = pd.read_parquet(ARTIFACTS_DIR / "users_encoded.parquet")
    train_pairs = pd.read_parquet(ARTIFACTS_DIR / "train_pairs.parquet")

    print(f"  users_encoded : {users_enc.shape}")
    print(f"  train_pairs   : {train_pairs.shape}")
    print(f"  n_users (vocab): {len(vocabs['user2idx']):,}")
    print(f"  n_items (vocab): {len(vocabs['item2idx']):,}")

    # ── Load item embedding table from checkpoint ─────────────────────────────
    print(f"\nLoading item_emb weights from {args.checkpoint} ...")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Pass --checkpoint path/to/epoch_05.pt"
        )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_state = ckpt["model_state"]

    # Key: "item_tower.item_emb.weight"  shape (n_items, 32)
    item_emb_key = "item_tower.item_emb.weight"
    if item_emb_key not in model_state:
        available = [k for k in model_state if "item_emb" in k]
        raise KeyError(
            f"'{item_emb_key}' not found in checkpoint state_dict. "
            f"Available keys with 'item_emb': {available}"
        )

    item_emb_weight: np.ndarray = model_state[item_emb_key].numpy()  # (n_items, 32)
    emb_dim = item_emb_weight.shape[1]
    n_items_in_emb = item_emb_weight.shape[0]
    print(f"  item_emb shape : {item_emb_weight.shape}  (n_items={n_items_in_emb:,}, dim={emb_dim})")
    print(f"  checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    # ── Filter to high-intent pairs ───────────────────────────────────────────
    hi_pairs = train_pairs[train_pairs["confidence_score"] >= args.min_confidence].copy()
    print(f"\nHigh-intent pairs (confidence >= {args.min_confidence}): "
          f"{len(hi_pairs):,} of {len(train_pairs):,} "
          f"({100 * len(hi_pairs) / len(train_pairs):.1f}%)")
    print(f"  Unique users in high-intent pairs: {hi_pairs['user_idx'].nunique():,}")

    # ── Build per-user centroid ───────────────────────────────────────────────
    n_users_max = int(users_enc["user_idx"].max()) + 1
    centroid_arr = np.zeros((n_users_max, emb_dim), dtype=np.float32)
    count_arr    = np.zeros(n_users_max, dtype=np.int32)

    print(f"\nComputing item centroids for {n_users_max:,} users ...")
    user_idxs  = hi_pairs["user_idx"].values.astype(np.int64)
    item_idxs  = hi_pairs["item_idx"].values.astype(np.int64)

    # Guard against item indices out of range (shouldn't happen but safe)
    valid_mask = item_idxs < n_items_in_emb
    if not valid_mask.all():
        n_oob = (~valid_mask).sum()
        print(f"  WARNING: {n_oob:,} pairs have item_idx >= {n_items_in_emb} — skipping them.")
        user_idxs = user_idxs[valid_mask]
        item_idxs = item_idxs[valid_mask]

    # Accumulate embeddings (vectorised with np.add.at)
    np.add.at(centroid_arr, user_idxs, item_emb_weight[item_idxs])
    np.add.at(count_arr,    user_idxs, 1)

    # Divide by count where count > 0
    has_history_mask = count_arr > 0
    centroid_arr[has_history_mask] /= count_arr[has_history_mask, None]

    n_with_history = has_history_mask.sum()
    print(f"  Users with ≥1 high-intent interaction : {n_with_history:,}")
    print(f"  Users with zero centroid (all-views)  : {n_users_max - n_with_history:,}")
    print(f"  Centroid L2 norm (non-zero users, mean): "
          f"{np.linalg.norm(centroid_arr[has_history_mask], axis=1).mean():.4f}")

    # ── Build coverage sanity check ───────────────────────────────────────────
    print("\n-- Item coverage sanity check --")
    test_gcs = "gs://recosys-data-bucket/samples/users_sample_500k/test/"
    try:
        test_df = pd.read_parquet(test_gcs)
        item2idx = vocabs["item2idx"]
        feb_purchases = test_df[test_df["event_type"] == "purchase"]["product_id"].value_counts()
        top100 = set(feb_purchases.head(100).index)
        in_train_emb = sum(
            1 for p in top100
            if p in item2idx and item2idx[p] < n_items_in_emb
        )
        print(f"  Top-100 Feb purchase items in training embedding: {in_train_emb}/100")
        if in_train_emb < 90:
            print("  WARNING: <90% of top Feb items are in the trained embedding. "
                  "Coverage gap may limit V6 ceiling.")
    except Exception as exc:
        print(f"  (Skipped coverage check — could not load test set: {exc})")

    # ── Merge into users_encoded ──────────────────────────────────────────────
    print("\nMerging centroid features into users_encoded ...")

    # Build DataFrame of (user_idx, item_centroid_0 .. item_centroid_{dim-1})
    centroid_cols = [f"item_centroid_{i}" for i in range(emb_dim)]
    # Subset to user rows that actually appear in users_enc
    enc_user_idxs = users_enc["user_idx"].values.astype(np.int64)
    centroid_for_enc = centroid_arr[enc_user_idxs]   # (len(users_enc), emb_dim)

    centroid_df = pd.DataFrame(
        centroid_for_enc,
        columns=centroid_cols,
        index=users_enc.index,
    )

    users_enc_v2 = pd.concat([users_enc, centroid_df], axis=1)
    print(f"  users_encoded_v2 shape : {users_enc_v2.shape}")
    print(f"  New columns ({emb_dim}): {centroid_cols[:3]} ... {centroid_cols[-1]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    users_enc_v2.to_parquet(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(f"  Total columns: {len(users_enc_v2.columns)}")
    print("Done.")


if __name__ == "__main__":
    main()
