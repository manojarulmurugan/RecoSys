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
    model:          SequenceModel,
    dataloader:     DataLoader,
    optimizer:      torch.optim.Optimizer,
    neg_sampler:    UniformNegativeSampler,
    device:         torch.device,
    temperature:    float = 1.0,
    grad_clip:      float | None = 1.0,
    log_every:      int = 100,
    step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
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
        if step_scheduler is not None:
            step_scheduler.step()

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


# ── Canonical SASRec training loop (V10): sampled softmax CE on raw IP ───────

def train_epoch_sasrec(
    model:          SequenceModel,
    dataloader:     DataLoader,
    optimizer:      torch.optim.Optimizer,
    neg_sampler:    UniformNegativeSampler,
    device:         torch.device,
    temperature:    float = 1.0,
    grad_clip:      float | None = 1.0,
    log_every:      int = 200,
    step_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """One epoch of canonical SASRec training (V10): sampled softmax CE on raw IP.

    Approach follows eSASRec (Petrov & Macdonald 2024):
      - Raw inner-product logits (encoder output is NOT L2-normalised).
      - K uniform random negatives per (batch, position).
      - Cross-entropy over (1 positive + K negative) logits, positive at col 0.
      - Optional fixed temperature scaling.

    Why sampled softmax CE replaced gBCE here:
      gBCE summed K negative-loss terms vs a single β-weighted positive term.
      At our 285K-item catalog with K=256, β ≈ 0.25 (gSASRec formula tuned for
      ≤100K catalogs), producing a K/β ≈ 1024:1 gradient imbalance — the
      model collapsed to neg_logit → -∞ while leaving positives at ≈-2.5
      (true positives scored worse than random items, NDCG below pop).
      Softmax CE pos/neg gradients sum to zero by construction, so the
      imbalance never appears.  Wu et al. (TOIS 2024) prove sampled softmax
      CE is a consistent estimator of full softmax under log-Q correction.

    Why no learnable log_scale (still on model for backward compat with
    earlier checkpoints / sanity checks):
      With L2 norm gone, raw embedding magnitudes are free to grow; the
      emergent natural scale plays the role a learnable scalar would, so
      adding one is redundant and was a degenerate knob in earlier attempts.

    Args:
        model:          SequenceModel exposing ``forward`` + ``get_item_embeddings``.
        dataloader:     Yields input_seq, input_event_seq, target_seq, target_mask.
        optimizer:      AdamW.
        neg_sampler:    Provides ``sample((B, L)) -> LongTensor (B, L, K)``.
        device:         Torch device.
        temperature:    Logit scaling.  Default 1.0 (raw IP).  Drop to 0.5 if
                        logits explode and softmax saturates early in training.
        grad_clip:      Max global L2 grad norm.
        log_every:      Print loss every N steps.
        step_scheduler: Per-step LR scheduler (warmup + cosine).

    Returns:
        Mean per-position cross-entropy loss over the epoch (PAD masked out).
    """
    model.train()                                       # type: ignore[attr-defined]
    total_loss_weighted = 0.0
    n_pos_total         = 0
    t0                  = time.time()
    last_log_time       = t0

    for step, batch in enumerate(dataloader, start=1):
        input_seq  = batch["input_seq"].to(device,        non_blocking=True)
        event_seq  = batch["input_event_seq"].to(device,  non_blocking=True)
        target_seq = batch["target_seq"].to(device,       non_blocking=True)
        mask       = batch["target_mask"].to(device,      non_blocking=True)

        n_pos = int(mask.sum().item())
        if n_pos == 0:
            continue

        out       = model.forward(input_seq, event_seq)         # (B, L, D)
        item_embs = model.get_item_embeddings()                 # (n_items, D)

        pos_emb   = item_embs[target_seq]                       # (B, L, D)
        pos_logit = (out * pos_emb).sum(-1) / temperature       # (B, L)

        neg_idx   = neg_sampler.sample(out.shape[:2])           # (B, L, K)
        neg_emb   = item_embs[neg_idx]                          # (B, L, K, D)
        neg_logit = torch.einsum("bld,blkd->blk", out, neg_emb) / temperature

        # Positive at column 0, K negatives after — standard sampled softmax CE.
        logits = torch.cat([pos_logit.unsqueeze(-1), neg_logit], dim=-1)
        labels = torch.zeros_like(target_seq)

        loss_per_pos = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction = "none",
        ).view_as(target_seq)                                              # (B, L)

        loss = (loss_per_pos * mask).sum() / mask.sum().clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),                             # type: ignore[arg-type]
                max_norm = grad_clip,
            )
        optimizer.step()
        if step_scheduler is not None:
            step_scheduler.step()

        total_loss_weighted += float(loss.item()) * n_pos
        n_pos_total         += n_pos

        if log_every and step % log_every == 0:
            now          = time.time()
            running_loss = total_loss_weighted / max(n_pos_total, 1)
            steps_per_s  = log_every / max(now - last_log_time, 1e-6)
            with torch.no_grad():
                pos_sum  = (pos_logit * mask).sum().item()
                pos_mean = pos_sum / max(float(mask.sum().item()), 1.0)
                neg_mean = float(neg_logit.mean().item())
            print(
                f"    step {step:>5,}/{len(dataloader):,}  "
                f"loss {running_loss:.4f}  "
                f"pos {pos_mean:+.2f}  neg {neg_mean:+.2f}  "
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


# ── Session-based full-softmax training loop (V9) ────────────────────────────

def train_epoch_session(
    model:           SequenceModel,
    dataloader:      DataLoader,
    optimizer:       torch.optim.Optimizer,
    device:          torch.device,
    temperature:     float = 0.07,
    label_smoothing: float = 0.1,
    grad_clip:       float | None = 1.0,
    log_every:       int = 200,
    step_scheduler:  torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """One epoch of full-softmax training for session-based next-item prediction.

    Key differences from ``train_epoch_sequence`` (sampled softmax):
      - Logits computed over ALL catalog items: out @ item_embs.T  (B*L, n_items).
      - Temperature scaling (default 0.07): without it, cosine similarities in
        [-1, 1] give near-zero gradients over 284K items — a well-trained model
        can only reach loss~10.5, making the 11.4 plateau mathematically inevitable.
        At tau=0.07 the gradient is 14x stronger and loss can reach ~1-2.
      - PAD positions excluded via ``ignore_index=0`` in cross_entropy.
      - Label smoothing 0.1 per T4Rec sec. 3.2 convention.
      - No UniformNegativeSampler dependency.

    Memory note: peak activation ~2 x (B * (L-1) * n_items * 4 bytes).
    With B=256, L=20, n_items=284k: ~11 GB — fits on A100.
    Reduce batch_size to 64-128 for V100/T4.

    Args:
        model:           SequenceModel with ``forward`` + ``get_item_embeddings``.
        dataloader:      Yields dicts with keys ``input_seq``, ``input_event_seq``,
                         ``target_seq`` (from ``SessionTrainDataset``).
        optimizer:       AdamW.
        device:          Torch device.
        temperature:     Logit scaling: logits /= temperature before softmax.
                         0.07 is standard for cosine-similarity contrastive losses.
        label_smoothing: CE label-smoothing factor (default 0.1).
        grad_clip:       Max global L2 grad norm (None = disabled).
        log_every:       Print loss every N steps (0 = silent).

    Returns:
        Mean per-position CE loss over the epoch (PAD positions excluded).
    """
    model.train()                                       # type: ignore[attr-defined]
    total_loss_weighted = 0.0
    n_pos_total         = 0
    t0                  = time.time()
    last_log_time       = t0

    for step, batch in enumerate(dataloader, start=1):
        input_seq  = batch["input_seq"].to(device,        non_blocking=True)
        event_seq  = batch["input_event_seq"].to(device,  non_blocking=True)
        target_seq = batch["target_seq"].to(device,       non_blocking=True)

        n_pos = int((target_seq != 0).sum().item())
        if n_pos == 0:
            continue

        # (B, L, D) — per-position L2-normalised user embeddings.
        out = model.forward(input_seq, event_seq)

        # (n_items, D) — tied item embeddings, same as FAISS payload at eval.
        item_embs = model.get_item_embeddings()

        # Full cosine-similarity logits over all catalog items, temperature-scaled.
        # Without temperature, cosine sims in [-1,1] give near-zero gradients at 284K scale.
        logits = (out @ item_embs.T).view(-1, item_embs.size(0)) / temperature  # (B*L, n_items)

        loss = F.cross_entropy(
            logits,
            target_seq.view(-1),
            ignore_index    = 0,
            label_smoothing = label_smoothing,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),                     # type: ignore[arg-type]
                max_norm = grad_clip,
            )
        optimizer.step()
        if step_scheduler is not None:
            step_scheduler.step()

        total_loss_weighted += float(loss.item()) * n_pos
        n_pos_total         += n_pos

        if log_every and step % log_every == 0:
            now          = time.time()
            running_loss = total_loss_weighted / max(n_pos_total, 1)
            steps_per_s  = log_every / max(now - last_log_time, 1e-6)
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
