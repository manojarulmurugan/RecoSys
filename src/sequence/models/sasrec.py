"""SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Architecture choices in this implementation:
    * Item + event-type + learned positional embeddings, summed (Vaswani 2017
      style; same dim end-to-end so the same item-emb table doubles as the
      negative/positive lookup table at loss time).
    * Pre-LayerNorm + GELU activations — the modern stability-friendly variant
      shown to dominate the post-LN/ReLU original on sparse e-commerce data
      (eSASRec, RecSys 2025).
    * Causal mask + key padding mask so each position attends only to past
      real items.
    * L2-normalised output so dot-product scores in the loss / FAISS index
      are in [-1, 1] and the temperature has the same meaning across runs.

Public interface (shared with GRU4Rec):
    forward(item_seq, event_seq)         → (B, L, D)
    encode_sequence(item_seq, event_seq)  → (B, D)        last position
    get_item_embeddings()                 → (n_items, D)  L2-normalised
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecModel(nn.Module):
    """Self-attentive transformer encoder for next-item prediction.

    Args:
        n_items:       Catalog size including PAD at idx 0.
        n_event_types: Number of event-type ids including PAD (default 4).
        embed_dim:     Token / output dim.  Default 64 to match V1-V6.
        max_seq_len:   Maximum sequence length the model is ever asked to
                       process.  Used to size the positional embedding and
                       precompute the causal mask buffer.  Pass the full
                       eval-time length (e.g. 50), not the training-time
                       L = max_seq_len - 1.
        n_layers:      Number of stacked transformer encoder blocks.
        n_heads:       Multi-head attention heads.  ``embed_dim`` must be
                       divisible by ``n_heads``.
        ffn_dim:       Feed-forward inner dim.
        dropout:       Dropout probability applied to embeddings, attention,
                       and FFN.
    """

    def __init__(
        self,
        n_items: int,
        n_event_types: int = 4,
        embed_dim: int = 64,
        max_seq_len: int = 50,
        n_layers: int = 2,
        n_heads: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.n_items       = int(n_items)
        self.n_event_types = int(n_event_types)
        self.embed_dim     = int(embed_dim)
        self.max_seq_len   = int(max_seq_len)
        self.n_layers      = int(n_layers)
        self.n_heads       = int(n_heads)
        self.ffn_dim       = int(ffn_dim)

        self.item_emb  = nn.Embedding(n_items,       embed_dim, padding_idx=0)
        self.event_emb = nn.Embedding(n_event_types, embed_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_seq_len,   embed_dim)

        self.input_ln      = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = n_heads,
            dim_feedforward = ffn_dim,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,  # pre-LN — more stable than the original
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_ln  = nn.LayerNorm(embed_dim)

        # Precompute causal mask once.  Shape (max_seq_len, max_seq_len),
        # upper-triangle True = "cannot attend to this position" so each
        # position only sees its past + itself.  Bool dtype keeps it
        # consistent with src_key_padding_mask (PyTorch deprecates mixed
        # float/bool mask combinations as of 2.x).
        causal = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight,  mean=0.0, std=0.01)
        nn.init.normal_(self.event_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.01)
        with torch.no_grad():
            self.item_emb.weight [0].zero_()
            self.event_emb.weight[0].zero_()

    def forward(
        self,
        item_seq: torch.Tensor,
        event_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of sequences with causal self-attention.

        Args:
            item_seq:  (B, L) long tensor of item idxs (0 = PAD).
            event_seq: (B, L) long tensor of event-type idxs (0 = PAD).

        Returns:
            (B, L, D) float tensor — L2-normalised user embedding at every
            position.  Padded positions are still computed but excluded
            from the loss via ``target_mask``.
        """
        L = item_seq.size(1)
        if L > self.max_seq_len:
            raise ValueError(
                f"Sequence length {L} exceeds model max_seq_len "
                f"{self.max_seq_len}; rebuild model or truncate input."
            )

        pos = torch.arange(L, device=item_seq.device).unsqueeze(0)  # (1, L)
        x   = (
            self.item_emb(item_seq)
            + self.event_emb(event_seq)
            + self.pos_emb(pos)
        )
        x = self.input_dropout(self.input_ln(x))

        # Padding token = 0; mask True = position to ignore in attention.
        key_padding_mask = item_seq == 0  # (B, L) bool

        # If a row has no real items (entirely PAD), the attention layer
        # would produce NaNs.  Force the rightmost position to be unmasked
        # for any such rows so the encoder still gets at least one query
        # to attend to.  (In practice the dataset filters out users with
        # < 2 events, so all rows have ≥ 1 real item.)
        all_pad_mask = key_padding_mask.all(dim=1)
        if all_pad_mask.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_pad_mask, -1] = False

        h = self.encoder(
            x,
            mask                = self.causal_mask[:L, :L],
            src_key_padding_mask = key_padding_mask,
        )
        return F.normalize(self.out_ln(h), dim=-1)

    def encode_sequence(
        self,
        item_seq: torch.Tensor,
        event_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Return one user embedding per row, taken at the last position.

        Left-padding guarantees the last position holds the user's most
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

        Used by both the training loss (positive/negative score lookup)
        and the FAISS index at eval time.
        """
        return F.normalize(self.item_emb.weight, dim=-1)

    def model_summary(self) -> None:
        """Print parameter counts and config."""
        total  = sum(p.numel() for p in self.parameters())
        emb    = (
            self.item_emb.weight.numel()
            + self.event_emb.weight.numel()
            + self.pos_emb.weight.numel()
        )
        rest   = total - emb
        print(f"SASRecModel:")
        print(f"  n_items     : {self.n_items:,}")
        print(f"  embed_dim   : {self.embed_dim}  (heads={self.n_heads}, "
              f"ffn={self.ffn_dim}, layers={self.n_layers})")
        print(f"  max_seq_len : {self.max_seq_len}")
        print(f"  params:")
        print(f"    embeddings: {emb:>10,}  (item + event + pos)")
        print(f"    other     : {rest:>10,}")
        print(f"    total     : {total:>10,}")
