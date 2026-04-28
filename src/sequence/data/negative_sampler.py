"""Negative samplers for sampled-softmax loss in sequence models.

Per the eSASRec (RecSys 2025) and gSASRec (2023) findings, sampled softmax
with K~512 uniform random negatives per position is the strongest tractable
training objective for transformer/RNN sequence recommenders on long-tail
e-commerce data.

We do **not** exclude items already in the user's history.  In practice
the false-negative rate is very low (a user's history is a tiny subset of
the catalog), and the implementation cost of per-row exclusion outweighs
the metric gain — see eSASRec section 3.2.
"""

from __future__ import annotations

from typing import Sequence

import torch


class UniformNegativeSampler:
    """Uniform random sampling from ``[1, n_items)`` (idx 0 = PAD).

    Negatives are sampled fresh per call; no precomputed pool, no
    exclusion of user history.

    Args:
        n_items: Total catalog size including PAD.  Sampling range is
                 ``[1, n_items)`` so PAD is never returned.
        n_neg:   Number of negatives drawn per call along the last axis.
        device:  Torch device to place samples on.  Pass ``"cuda"`` to
                 keep negatives on GPU and avoid host→device copies in
                 the hot loop.
    """

    def __init__(
        self,
        n_items: int,
        n_neg:   int,
        device:  torch.device | str = "cpu",
    ) -> None:
        if n_items < 2:
            raise ValueError(f"n_items must be > 1; got {n_items}")
        if n_neg < 1:
            raise ValueError(f"n_neg must be > 0; got {n_neg}")
        self.n_items = int(n_items)
        self.n_neg   = int(n_neg)
        self.device  = torch.device(device)

    def sample(self, batch_shape: Sequence[int]) -> torch.LongTensor:
        """Return ``(*batch_shape, n_neg)`` int64 negative item idxs.

        Sampling is uniform over the half-open range ``[1, n_items)``
        so the PAD token (index 0) is never returned as a negative.

        Args:
            batch_shape: Leading shape of the output tensor.  For sequence
                         training this is typically ``(B, L)`` so that each
                         position in each sequence gets its own draw.
        """
        out_shape = (*tuple(batch_shape), self.n_neg)
        return torch.randint(
            low    = 1,
            high   = self.n_items,
            size   = out_shape,
            device = self.device,
            dtype  = torch.long,
        )

    def to(self, device: torch.device | str) -> "UniformNegativeSampler":
        """Move future samples to a different device."""
        self.device = torch.device(device)
        return self

    def __repr__(self) -> str:
        return (
            f"UniformNegativeSampler(n_items={self.n_items:,}, "
            f"n_neg={self.n_neg}, device={self.device})"
        )
