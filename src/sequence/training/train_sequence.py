"""Sampled-softmax training loop for sequence recommenders.

Used by both V7 (GRU4Rec) and V8 (SASRec) — the loop is model-agnostic
because both models expose the same minimal interface
(``forward`` returning per-position embeddings + ``get_item_embeddings``).

Per the eSASRec (RecSys 2025) study, sampled softmax with K~512 uniform
random negatives per position dominates BCE / pairwise BPR / full softmax
on long-tail e-commerce data while staying tractable on a single GPU.
We compute one set of negatives per (batch, position) and concatenate
the positive logit at index 0 for a clean cross-entropy loss.
"""

from __future__ import annotations

import time
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.sequence.data.negative_sampler import UniformNegativeSampler


# ── Light protocol for type hints (avoids circular import) ────────────────────

class SequenceModel(Protocol):
    """Minimal interface a sequence model must implement to plug into the loop."""

    def forward(
        self,
        item_seq:  torch.Tensor,
        event_seq: torch.Tensor,
    ) -> torch.Tensor: ...

    def get_item_embeddings(self) -> torch.Tensor: ...


# ── Optimizer param groups ────────────────────────────────────────────────────

def get_param_groups(
    model: nn.Module,
    lr:           float,
    weight_decay: float,
) -> list[dict]:
    """Two-group AdamW: zero weight-decay on embeddings, ``weight_decay``
    on everything else.

    Same convention as ``scripts/two_tower/train_two_tower_v3.get_param_groups``
    so V7/V8 results are directly comparable to V1-V6.
    """
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


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch_sequence(
    model:        SequenceModel,
    dataloader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    neg_sampler:  UniformNegativeSampler,
    device:       torch.device,
    temperature:  float = 1.0,
    grad_clip:    float | None = 1.0,
    log_every:    int = 100,
) -> float:
    """Run one epoch of sampled-softmax training.

    For every (user, position) pair where the target is a real item:
      1. Encode the user sequence at that position → ``out`` (B, L, D).
      2. Look up the positive item embedding from the model's own
         item-emb table (same lookup used by the FAISS index later).
      3. Sample K random negative item idxs and look up their embeddings.
      4. Cross-entropy over the (1 positive + K negative) logit row,
         scaled by ``1 / temperature``.
      5. Mean over real positions only (PAD positions are masked out).

    Args:
        model:       Object exposing ``forward(item_seq, event_seq)`` →
                     ``(B, L, D)`` and ``get_item_embeddings()`` →
                     ``(n_items, D)``.
        dataloader:  Yields dicts with keys ``input_seq``, ``input_event_seq``,
                     ``target_seq``, ``target_mask`` (see
                     ``SequenceTrainDataset``).
        optimizer:   AdamW (or any ``torch.optim.Optimizer``).
        neg_sampler: Provides ``sample((B, L)) -> LongTensor (B, L, K)`` on
                     the right device.
        device:      Torch device for inputs and model.
        temperature: Logit scaling.  1.0 = unscaled cosine similarity;
                     lower = sharper distribution.  Default 1.0.
        grad_clip:   Max global L2 grad norm (None = disabled).
        log_every:   Print average loss every N batches.  Set 0 to disable.

    Returns:
        Average per-position loss over the epoch.
    """
    model.train()                                       # type: ignore[attr-defined]
    total_loss_weighted = 0.0
    n_pos_total         = 0
    t0                  = time.time()
    last_log_time       = t0

    for step, batch in enumerate(dataloader, start=1):
        input_seq  = batch["input_seq"].to(device,  non_blocking=True)
        event_seq  = batch["input_event_seq"].to(device, non_blocking=True)
        target_seq = batch["target_seq"].to(device, non_blocking=True)
        mask       = batch["target_mask"].to(device, non_blocking=True)

        n_pos = int(mask.sum().item())
        if n_pos == 0:
            # Entire batch was padding (pathological — skip).
            continue

        # Encode user sequence at every position.
        out = model.forward(input_seq, event_seq)           # (B, L, D)

        # Item embedding table — re-fetched each step so the latest
        # weights drive both positive and negative scores.
        item_embs = model.get_item_embeddings()             # (n_items, D)

        # Positive scores at every position.
        pos_emb   = item_embs[target_seq]                   # (B, L, D)
        pos_score = (out * pos_emb).sum(-1) / temperature   # (B, L)

        # Sampled negatives at every position.
        neg_idx   = neg_sampler.sample(out.shape[:2])       # (B, L, K)
        neg_emb   = item_embs[neg_idx]                      # (B, L, K, D)
        neg_score = torch.einsum("bld,blkd->blk", out, neg_emb) / temperature

        # Sampled-softmax CE: positive lives at column 0 in the concat.
        logits = torch.cat([pos_score.unsqueeze(-1), neg_score], dim=-1)
        labels = torch.zeros_like(target_seq)               # (B, L) all-zero

        loss_per_pos = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction = "none",
        ).view_as(target_seq)                                # (B, L)

        loss = (loss_per_pos * mask).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),                          # type: ignore[arg-type]
                max_norm = grad_clip,
            )
        optimizer.step()

        # Track epoch-level mean loss weighted by # of real positions.
        total_loss_weighted += float(loss.item()) * n_pos
        n_pos_total         += n_pos

        if log_every and step % log_every == 0:
            now = time.time()
            running_loss = total_loss_weighted / max(n_pos_total, 1)
            steps_since  = log_every
            steps_per_s  = steps_since / max(now - last_log_time, 1e-6)
            print(
                f"    step {step:>5,}/{len(dataloader):,}  "
                f"loss {running_loss:.4f}  "
                f"({steps_per_s:.1f} steps/s)"
            )
            last_log_time = now

    epoch_loss = total_loss_weighted / max(n_pos_total, 1)
    elapsed    = int(time.time() - t0)
    print(
        f"  epoch loss : {epoch_loss:.4f}  "
        f"({n_pos_total:,} positions, {elapsed // 60}m {elapsed % 60}s)"
    )
    return epoch_loss
