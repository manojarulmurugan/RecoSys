"""Two-Tower neural retrieval model for implicit-feedback recommendation.

Architecture:
  - UserTower: embeds categorical user features + passes dense features
    through an MLP, outputs an L2-normalised 64-d embedding.
  - ItemTower: same pattern for item features.
  - TwoTowerModel: composes both towers and computes an (B, B) in-batch
    similarity matrix scaled by a temperature for contrastive training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def _init_embedding(emb: nn.Embedding) -> None:
    nn.init.normal_(emb.weight, mean=0.0, std=0.01)


def _init_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


# ── User Tower ────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """Encodes per-user features into a unit-norm 64-d embedding.

    Categorical inputs (embedded):
      user_idx, top_cat_idx, peak_hour_bucket, preferred_dow

    Scalar pass-through appended to the dense vector:
      has_purchase_history  (0/1 int treated as a float scalar)

    Dense inputs (pre-scaled, passed through directly):
      log_total_events, months_active, purchase_rate, cart_rate,
      log_n_sessions, avg_purchase_price_scaled

    MLP input dim: embed_dim_user + embed_dim_cat + 2*embed_dim_small
                   + dense_input_dim + 1  (the +1 is has_purchase_history)
    """

    def __init__(
        self,
        n_users: int,
        n_top_cats: int,
        n_hour_buckets: int = 4,
        n_dow: int = 7,
        embed_dim_user: int = 32,
        embed_dim_cat: int = 16,
        embed_dim_small: int = 8,
        dense_input_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.user_emb      = nn.Embedding(n_users,       embed_dim_user)
        self.top_cat_emb   = nn.Embedding(n_top_cats,    embed_dim_cat)
        self.hour_emb      = nn.Embedding(n_hour_buckets, embed_dim_small)
        self.dow_emb       = nn.Embedding(n_dow,          embed_dim_small)

        mlp_input_dim = (
            embed_dim_user + embed_dim_cat + 2 * embed_dim_small
            + dense_input_dim + 1  # +1 for has_purchase_history scalar
        )

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.user_emb, self.top_cat_emb, self.hour_emb, self.dow_emb):
            _init_embedding(emb)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(
        self,
        user_idx: torch.Tensor,   # (B,)
        user_cat: torch.Tensor,   # (B, 4) — [top_cat_idx, peak_hour_bucket, preferred_dow, has_purchase_history]
        user_dense: torch.Tensor, # (B, 6)
    ) -> torch.Tensor:            # (B, output_dim)
        # Embed the three true categoricals; has_purchase_history is a scalar
        e_user     = self.user_emb(user_idx)           # (B, 32)
        e_top_cat  = self.top_cat_emb(user_cat[:, 0])  # (B, 16)
        e_hour     = self.hour_emb(user_cat[:, 1])     # (B, 8)
        e_dow      = self.dow_emb(user_cat[:, 2])      # (B, 8)
        has_purch  = user_cat[:, 3:4].float()          # (B, 1)

        x = torch.cat([e_user, e_top_cat, e_hour, e_dow, user_dense, has_purch], dim=1)
        x = self.mlp(x)
        return F.normalize(x, dim=1)


# ── Item Tower ────────────────────────────────────────────────────────────────

class ItemTower(nn.Module):
    """Encodes per-item features into a unit-norm 64-d embedding.

    Categorical inputs (embedded):
      item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket

    Dense inputs (pre-scaled, passed through directly):
      avg_price_scaled, log_confidence_scaled, purchase_rate_scaled

    MLP input dim: embed_dim_item + 2*embed_dim_cat + embed_dim_cat
                   + embed_dim_small + dense_input_dim
                 = 32 + 16 + 16 + 16 + 8 + 3 = 91
    """

    def __init__(
        self,
        n_items: int,
        n_cat_l1: int,
        n_cat_l2: int,
        n_brands: int,
        n_price_buckets: int = 8,
        embed_dim_item: int = 32,
        embed_dim_cat: int = 16,
        embed_dim_small: int = 8,
        dense_input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.item_emb   = nn.Embedding(n_items,         embed_dim_item)
        self.cat_l1_emb = nn.Embedding(n_cat_l1,        embed_dim_cat)
        self.cat_l2_emb = nn.Embedding(n_cat_l2,        embed_dim_cat)
        self.brand_emb  = nn.Embedding(n_brands,        embed_dim_cat)
        self.price_emb  = nn.Embedding(n_price_buckets, embed_dim_small)

        mlp_input_dim = (
            embed_dim_item + 3 * embed_dim_cat + embed_dim_small + dense_input_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.item_emb, self.cat_l1_emb, self.cat_l2_emb,
                    self.brand_emb, self.price_emb):
            _init_embedding(emb)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(
        self,
        item_cat: torch.Tensor,   # (B, 5) — [item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket]
        item_dense: torch.Tensor, # (B, 3)
    ) -> torch.Tensor:            # (B, output_dim)
        e_item   = self.item_emb(item_cat[:, 0])    # (B, 32)
        e_cat_l1 = self.cat_l1_emb(item_cat[:, 1]) # (B, 16)
        e_cat_l2 = self.cat_l2_emb(item_cat[:, 2]) # (B, 16)
        e_brand  = self.brand_emb(item_cat[:, 3])   # (B, 16)
        e_price  = self.price_emb(item_cat[:, 4])   # (B, 8)

        x = torch.cat([e_item, e_cat_l1, e_cat_l2, e_brand, e_price, item_dense], dim=1)
        x = self.mlp(x)
        return F.normalize(x, dim=1)


# ── Two-Tower Model ───────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """Composes UserTower and ItemTower into a retrieval model trained with
    in-batch negatives via scaled dot-product similarity.

    The temperature is a fixed float (not a learned parameter) and controls
    the sharpness of the softmax distribution during contrastive training.

    Args:
        user_tower:  Instantiated UserTower.
        item_tower:  Instantiated ItemTower.
        temperature: Softmax temperature for scaling similarity scores.
                     Typical value: 0.07 (following SimCLR / CLIP convention).
    """

    def __init__(
        self,
        user_tower: UserTower,
        item_tower: ItemTower,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.user_tower  = user_tower
        self.item_tower  = item_tower
        self.temperature = temperature

    def forward(
        self,
        user_idx: torch.Tensor,   # (B,)
        user_cat: torch.Tensor,   # (B, 4)
        user_dense: torch.Tensor, # (B, 6)
        item_cat: torch.Tensor,   # (B, 5)
        item_dense: torch.Tensor, # (B, 3)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both towers and return embeddings plus the in-batch score matrix.

        Returns:
            user_embeddings: (B, 64)
            item_embeddings: (B, 64)
            scores:          (B, B) scaled dot-product matrix
        """
        user_embeddings = self.user_tower(user_idx, user_cat, user_dense)  # (B, 64)
        item_embeddings = self.item_tower(item_cat, item_dense)             # (B, 64)
        scores = (user_embeddings @ item_embeddings.T) / self.temperature   # (B, B)
        return user_embeddings, item_embeddings, scores

    def get_user_embedding(
        self,
        user_idx: torch.Tensor,   # (B,)
        user_cat: torch.Tensor,   # (B, 4)
        user_dense: torch.Tensor, # (B, 6)
    ) -> torch.Tensor:            # (B, 64)
        """Encode users only. Used at inference time."""
        return self.user_tower(user_idx, user_cat, user_dense)

    def get_item_embeddings(
        self,
        item_cat: torch.Tensor,   # (N, 5)
        item_dense: torch.Tensor, # (N, 3)
    ) -> torch.Tensor:            # (N, 64)
        """Encode items only. Used to build the FAISS index."""
        return self.item_tower(item_cat, item_dense)

    def model_summary(self) -> None:
        """Print total and per-tower trainable parameter counts."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total      = _count(self)
        user_count = _count(self.user_tower)
        item_count = _count(self.item_tower)

        print("=" * 40)
        print("TwoTowerModel — parameter summary")
        print("=" * 40)
        print(f"  User tower : {user_count:>12,}")
        print(f"  Item tower : {item_count:>12,}")
        print(f"  Total      : {total:>12,}")
        print("=" * 40)
