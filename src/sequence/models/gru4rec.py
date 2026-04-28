"""GRU4Rec — gated-recurrent-unit baseline for sequence recommendation.

Architecture:
    item_emb(item_seq) + event_emb(event_seq)  →  multi-layer GRU
        →  Linear projection back to embed_dim  →  L2-normalised hidden states.

Why this shape:
    * The output projection produces vectors in the same space as the item
      embeddings, so the same matrix is reused as the negative-/positive-
      score lookup table during training and as the FAISS index payload at
      eval time.  Score = dot(user_emb, item_emb).
    * Event type is added (not concatenated) so the embedding dimension is
      preserved end-to-end.  This is the same pattern T4Rec uses for
      side-features (NVIDIA Merlin, 2021).
    * Last-position pooling is used at inference time — for left-padded
      sequences this is always the most recent real interaction.

Interface (shared with SASRec):
    forward(item_seq, event_seq)        → (B, L, D)  — per-position user embs
    encode_sequence(item_seq, event_seq) → (B, D)    — last-position only
    get_item_embeddings()                → (n_items, D) — L2-normalised lookup
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4RecModel(nn.Module):
    """Two-embedding (item + event) GRU sequence encoder.

    Args:
        n_items:      Total catalog size including the PAD token at idx 0.
        n_event_types: Number of event-type ids including PAD (default 4:
                       PAD, view, cart, purchase).
        embed_dim:    Output dim — must match the item-emb dim used in the
                      training loop and FAISS index.  Default 64 (matches V1-V6).
        gru_hidden:   GRU hidden size.  Larger = more capacity, slower.
        n_layers:     Stacked GRU layers.  Dropout is applied between layers
                      when ``n_layers > 1``.
        dropout:      Dropout applied between GRU layers and to the input
                      embedding sum.
    """

    def __init__(
        self,
        n_items: int,
        n_event_types: int = 4,
        embed_dim: int = 64,
        gru_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_items       = int(n_items)
        self.n_event_types = int(n_event_types)
        self.embed_dim     = int(embed_dim)
        self.gru_hidden    = int(gru_hidden)
        self.n_layers      = int(n_layers)

        self.item_emb  = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.event_emb = nn.Embedding(n_event_types, embed_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size  = embed_dim,
            hidden_size = gru_hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(gru_hidden, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight,  mean=0.0, std=0.01)
        nn.init.normal_(self.event_emb.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.item_emb.weight [0].zero_()
            self.event_emb.weight[0].zero_()
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        item_seq: torch.Tensor,
        event_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of sequences position-by-position.

        Args:
            item_seq:  (B, L) long tensor of item idxs (0 = PAD).
            event_seq: (B, L) long tensor of event-type idxs (0 = PAD).

        Returns:
            (B, L, D) float tensor — L2-normalised user embedding at every
            position.  Padded positions still receive a hidden state but
            are excluded from the loss via ``target_mask``.
        """
        x = self.item_emb(item_seq) + self.event_emb(event_seq)   # (B, L, D)
        x = self.input_dropout(x)
        h, _ = self.gru(x)                                         # (B, L, H)
        out = self.proj(h)                                         # (B, L, D)
        return F.normalize(out, dim=-1)

    def encode_sequence(
        self,
        item_seq: torch.Tensor,
        event_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Return one user embedding per row, taken at the last position.

        Left-padding guarantees the last position is the user's most
        recent real interaction.

        Args:
            item_seq:  (B, L) long tensor.
            event_seq: (B, L) long tensor.

        Returns:
            (B, D) L2-normalised float tensor.
        """
        return self.forward(item_seq, event_seq)[:, -1, :]

    def get_item_embeddings(self) -> torch.Tensor:
        """Return ``(n_items, D)`` L2-normalised item embeddings.

        Same lookup is used (a) as the loss-time score table during
        training and (b) as the FAISS index payload at eval time.
        """
        return F.normalize(self.item_emb.weight, dim=-1)

    def model_summary(self) -> None:
        """Print parameter counts and shapes."""
        total  = sum(p.numel() for p in self.parameters())
        emb    = self.item_emb.weight.numel() + self.event_emb.weight.numel()
        rest   = total - emb
        print(f"GRU4RecModel:")
        print(f"  n_items     : {self.n_items:,}")
        print(f"  embed_dim   : {self.embed_dim}")
        print(f"  gru_hidden  : {self.gru_hidden}  (n_layers={self.n_layers})")
        print(f"  params:")
        print(f"    embeddings: {emb:>10,}")
        print(f"    other     : {rest:>10,}")
        print(f"    total     : {total:>10,}")
