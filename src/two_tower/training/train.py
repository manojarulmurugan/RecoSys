"""Training loop for the Two-Tower recommendation model.

Uses in-batch negatives: for a batch of B (user, item) positive pairs,
each user is treated as a query against all B items in the batch, so
the model sees B-1 negatives per positive without any extra sampling.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.two_tower.data.dataset import (
    TwoTowerDataset,
    TwoTowerDatasetWithHardNegs,
    build_full_item_tensors,
)
from src.two_tower.models.two_tower import TwoTowerModel


# ── Loss ──────────────────────────────────────────────────────────────────────

def in_batch_loss(
    scores: torch.Tensor,
    confidence_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch negative cross-entropy loss with optional confidence weighting.

    Each row i of `scores` is treated as logits over B classes; the correct
    class is i (the diagonal positive pair).

    When `confidence_weights` is provided, per-sample losses are scaled by
    normalised confidence so higher-confidence interactions (e.g. purchases)
    contribute proportionally more gradient than lower-confidence ones
    (e.g. views). Weights are normalised to mean=1 so overall loss magnitude
    stays comparable to the unweighted case.

    Args:
        scores:             (B, B) scaled dot-product similarity matrix.
        confidence_weights: (B,) float tensor of raw confidence scores, or
                            None for uniform weighting.

    Returns:
        Scalar loss tensor.
    """
    labels = torch.arange(scores.size(0), device=scores.device)

    if confidence_weights is None:
        return F.cross_entropy(scores, labels)

    per_sample_loss = F.cross_entropy(scores, labels, reduction="none")
    norm_weights = confidence_weights / confidence_weights.mean()
    return (per_sample_loss * norm_weights).mean()


def in_batch_loss_with_hard_negs(
    user_embs: torch.Tensor,
    pos_item_embs: torch.Tensor,
    hard_neg_item_embs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """In-batch negative cross-entropy loss augmented with pre-mined hard negatives.

    For each user i the logits contain:
      • B in-batch scores  (diagonal entry i is the true positive)
      • 3 hard-negative scores appended as extra columns

    The label for every row is still i (the diagonal positive), so standard
    cross-entropy over the enlarged (B, B+K) matrix is correct.

    Args:
        user_embs:          (B, 64) L2-normalised user embeddings.
        pos_item_embs:      (B, 64) L2-normalised positive item embeddings.
        hard_neg_item_embs: (B, K, 64) L2-normalised hard negative embeddings.
        temperature:        Softmax temperature scalar.

    Returns:
        Scalar cross-entropy loss.
    """
    B = user_embs.size(0)

    # (B, B) — standard in-batch similarity matrix
    S_inbatch = (user_embs @ pos_item_embs.T) / temperature

    # (B, K) — each user dot-producted against its own K hard negatives
    S_hard = torch.einsum("bd,bkd->bk", user_embs, hard_neg_item_embs) / temperature

    # (B, B+K) — hard negs appended as extra negative columns
    S = torch.cat([S_inbatch, S_hard], dim=1)

    labels = torch.arange(B, device=user_embs.device)
    return F.cross_entropy(S, labels)


# ── Single Epoch ──────────────────────────────────────────────────────────────

def train_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 100,
    use_confidence_weighting: bool = False,
) -> float:
    """Run one full training epoch.

    Args:
        model:                    TwoTowerModel instance.
        dataloader:               DataLoader yielding batches from TwoTowerDataset.
        optimizer:                Optimiser (e.g. AdamW).
        device:                   Target device.
        log_every:                Print a progress line every this many batches.
        use_confidence_weighting: If True, weight each pair's loss by its
                                  normalised confidence score.

    Returns:
        Mean loss across all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        user_idx   = batch["user_idx"].to(device)
        user_cat   = batch["user_cat"].to(device)
        user_dense = batch["user_dense"].to(device)
        item_cat   = batch["item_cat"].to(device)
        item_dense = batch["item_dense"].to(device)
        conf       = batch["confidence"].to(device)

        optimizer.zero_grad()

        _, _, scores = model(user_idx, user_cat, user_dense, item_cat, item_dense)

        # Confidence weighting: disabled by default (neutral on 50k experiments)
        # Set use_confidence_weighting=True to enable — see reports/05_model_experiments_50k.md
        if use_confidence_weighting:
            loss = in_batch_loss(scores, conf)
        else:
            loss = in_batch_loss(scores)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_every == 0:
            print(f"  batch {batch_idx + 1}/{total_batches} — loss: {loss.item():.4f}")

    return total_loss / total_batches


def train_epoch_with_hard_negs(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    device: torch.device,
    log_every: int = 100,
) -> float:
    """Run one full training epoch using pre-mined hard negatives.

    Mirrors train_epoch exactly, with three additions:
      1. Unpacks ``batch['hard_neg_idxs']`` — shape (B, K) of item indices.
      2. Resolves hard-neg feature tensors from the dataset's pre-built numpy
         lookup arrays (``dataset._item_cat`` / ``dataset._item_dense``),
         the same arrays used to serve positive item features in __getitem__.
      3. Calls in_batch_loss_with_hard_negs instead of in_batch_loss.

    The DataLoader must wrap a TwoTowerDatasetWithHardNegs instance.

    Args:
        model:       TwoTowerModel instance.
        dataloader:  DataLoader built from TwoTowerDatasetWithHardNegs.
        optimizer:   Optimiser (e.g. AdamW).
        temperature: Softmax temperature passed to in_batch_loss_with_hard_negs.
        device:      Target device.
        log_every:   Print a progress line every this many batches.

    Returns:
        Mean loss across all batches in the epoch.
    """
    model.train()
    total_loss    = 0.0
    total_batches = len(dataloader)

    # Cache the numpy feature arrays once — same arrays backing __getitem__
    item_cat_arr   = dataloader.dataset._item_cat    # (n_items, 5) int64
    item_dense_arr = dataloader.dataset._item_dense  # (n_items, 3) float32

    for batch_idx, batch in enumerate(dataloader):
        user_idx   = batch["user_idx"].to(device)
        user_cat   = batch["user_cat"].to(device)
        user_dense = batch["user_dense"].to(device)
        item_cat   = batch["item_cat"].to(device)
        item_dense = batch["item_dense"].to(device)

        # hard_neg_idxs: (B, K) int64 — item indices for pre-mined hard negatives
        hard_neg_idxs = batch["hard_neg_idxs"]   # keep on CPU for numpy indexing
        B, K = hard_neg_idxs.shape

        # Resolve hard-neg features using the same lookup arrays as __getitem__
        flat_idxs = hard_neg_idxs.reshape(-1).numpy()          # (B*K,)
        hn_cat_t   = torch.tensor(
            item_cat_arr[flat_idxs], dtype=torch.long
        ).to(device)                                             # (B*K, 5)
        hn_dense_t = torch.tensor(
            item_dense_arr[flat_idxs], dtype=torch.float32
        ).to(device)                                             # (B*K, 3)

        optimizer.zero_grad()

        # Encode users and positive items (mirrors model.forward internals)
        user_embs     = model.user_tower(user_idx, user_cat, user_dense)  # (B, 64)
        pos_item_embs = model.item_tower(item_cat, item_dense)             # (B, 64)

        # Encode all hard negatives in one batched pass, then reshape
        hn_embs_flat  = model.item_tower(hn_cat_t, hn_dense_t)            # (B*K, 64)
        hard_neg_embs = hn_embs_flat.view(B, K, -1)                       # (B, K, 64)

        loss = in_batch_loss_with_hard_negs(
            user_embs, pos_item_embs, hard_neg_embs, temperature
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_every == 0:
            print(f"  batch {batch_idx + 1}/{total_batches} — loss: {loss.item():.4f}")

    return total_loss / total_batches


# ── Item Index Builder ────────────────────────────────────────────────────────

def build_item_index(
    model: TwoTowerModel,
    items_encoded_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all items through the item tower and return L2-normalised embeddings.

    Items are processed in batches to avoid OOM on large catalogues.
    The returned arrays are sorted by item_idx ascending (matching the
    row order of build_full_item_tensors).

    Args:
        model:            TwoTowerModel instance.
        items_encoded_df: DataFrame produced by FeatureBuilder.
        device:           Target device.
        batch_size:       Number of items per forward pass.

    Returns:
        item_embeddings_np: float32 numpy array of shape (n_items, 64),
                            L2-normalised (ready for FAISS IndexFlatIP).
        item_idx_array:     int64 numpy array of shape (n_items,) giving
                            the item_idx for each embedding row.
    """
    import faiss  # optional dependency — imported lazily

    item_cat_tensor, item_dense_tensor = build_full_item_tensors(items_encoded_df)

    # item_idx lives in column 0 of item_cat_tensor (see dataset.py)
    item_idx_array = item_cat_tensor[:, 0].numpy().astype(np.int64)

    n_items = item_cat_tensor.size(0)
    all_embeddings: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            cat_batch   = item_cat_tensor[start : start + batch_size].to(device)
            dense_batch = item_dense_tensor[start : start + batch_size].to(device)
            emb = model.get_item_embeddings(cat_batch, dense_batch)
            all_embeddings.append(emb.cpu().numpy())

    item_embeddings_np = np.vstack(all_embeddings).astype(np.float32)
    faiss.normalize_L2(item_embeddings_np)

    return item_embeddings_np, item_idx_array


# ── Full Training Orchestrator ────────────────────────────────────────────────

def train(
    model: TwoTowerModel,
    train_pairs_df: pd.DataFrame,
    users_encoded_df: pd.DataFrame,
    items_encoded_df: pd.DataFrame,
    n_epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device_str: str = "cpu",
    checkpoint_dir: str = "artifacts/50k/checkpoints/",
    log_every: int = 100,
    eval_every: int = 0,
    eval_fn: Callable[[TwoTowerModel], dict[str, Any]] | None = None,
    lr_eta_min: float = 1e-5,
    use_confidence_weighting: bool = False,
) -> dict[str, list[float]]:
    """Full training loop with per-epoch checkpointing and cosine LR decay.

    A CosineAnnealingLR scheduler decays the learning rate from
    `learning_rate` at epoch 1 down to `lr_eta_min` at epoch `n_epochs`,
    which stabilises late-stage training (prevents the within-epoch loss
    drift observed at a flat high learning rate).

    Args:
        model:                    TwoTowerModel instance (uninitialised or pre-trained).
        train_pairs_df:           DataFrame with [user_idx, item_idx, confidence_score].
        users_encoded_df:         DataFrame produced by FeatureBuilder for users.
        items_encoded_df:         DataFrame produced by FeatureBuilder for items.
        n_epochs:                 Number of training epochs.
        batch_size:               DataLoader batch size.
        learning_rate:            AdamW learning rate (initial / peak value).
        weight_decay:             AdamW weight decay (L2 regularisation).
        device_str:               Device string — 'cpu', 'cuda', or 'mps'.
        checkpoint_dir:           Directory to save per-epoch checkpoints.
        log_every:                Batch logging frequency inside each epoch.
        eval_every:               If > 0, call `eval_fn(model)` every N epochs.
        eval_fn:                  Optional callable taking the model and returning
                                  a metrics dict (expected keys: 'recall_10', 'ndcg_10',
                                  'recall_20', 'ndcg_20').
        lr_eta_min:               Minimum learning rate at the end of cosine decay.
        use_confidence_weighting: If True, scale each pair's loss by its normalised
                                  confidence score (purchases outweigh views).

    Returns:
        history dict: {'train_loss': [float, ...]} — one entry per epoch.
    """
    ckpt_path = pathlib.Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    model.to(device)

    dataset = TwoTowerDataset(train_pairs_df, users_encoded_df, items_encoded_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_eta_min
    )

    history: dict[str, list[float]] = {"train_loss": []}

    print(f"Training on {device} — {n_epochs} epochs, batch {batch_size}, "
          f"lr {learning_rate} → {lr_eta_min:.0e} (cosine), wd {weight_decay}")
    print(f"  Dataset size              : {len(dataset):,} pairs")
    print(f"  Batches/epoch             : {len(dataloader):,}")
    print(f"  Confidence weighting      : {use_confidence_weighting}")
    model.model_summary()

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        epoch_loss = train_epoch(
            model, dataloader, optimizer, device, log_every,
            use_confidence_weighting=use_confidence_weighting,
        )
        history["train_loss"].append(epoch_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}/{n_epochs} — loss: {epoch_loss:.4f}  "
              f"lr: {current_lr:.2e}")

        scheduler.step()

        if eval_every > 0 and eval_fn is not None and epoch % eval_every == 0:
            metrics = eval_fn(model)
            print(f"  → Recall@10: {metrics.get('recall_10', 0):.4f}  "
                  f"NDCG@10: {metrics.get('ndcg_10', 0):.4f}  "
                  f"Recall@20: {metrics.get('recall_20', 0):.4f}  "
                  f"NDCG@20: {metrics.get('ndcg_20', 0):.4f}")

        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss":            epoch_loss,
            },
            ckpt_path / f"epoch_{epoch}.pt",
        )

    return history
