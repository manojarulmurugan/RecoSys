"""Two-Tower neural retrieval model for implicit-feedback recommendation.

Architecture:
  - UserTower: embeds categorical user features + passes dense features
    through an MLP, outputs an L2-normalised 64-d embedding.
  - ItemTower: same pattern for item features.
  - TwoTowerModel: composes both towers and computes an (B, B) in-batch
    similarity matrix scaled by a temperature for contrastive training.

V2 / V4 additions:
  - UserTowerV2: replaces integer dow_emb with sin/cos DOW in the dense
    vector (dense_input_dim 6 → 8).  MLP input 71 → 65.
  - ItemTowerV2: adds two extra dense features (price_relative_to_cat_avg,
    product_recency_log; dense_input_dim 3 → 5) and uses a hierarchical
    cat residual: cat_l2_final = cat_l2_emb + cat_l1_emb.  MLP input
    91 → 93.

V3 / V5 addition:
  - SequentialUserTower: encodes the user's last-N interacted items with a
    GRU and concatenates the hidden state with static dense features.
    item_seq_emb weights are tied to ItemTower.item_emb after construction.

V6 addition:
  - UserTowerV3: extends UserTowerV2 by appending a pre-computed 32-dim
    item-centroid vector (mean of item_emb over the user's high-intent
    history) to the dense vector (dense_input_dim 8 → 40).
    MLP input: 32 + 16 + 8 + 40 + 1 = 97.
    Users with no high-intent history receive a zero centroid and fall
    back gracefully to their static features.
    Run scripts/two_tower/build_users_v2_500k.py to build the centroid
    columns before training.
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


# ── User Tower V2 (sin/cos DOW, no dow_emb) ───────────────────────────────────

class UserTowerV2(nn.Module):
    """V2 user tower: replaces the integer day-of-week embedding with cyclic
    sin/cos encoding baked into the dense vector.

    Categorical inputs (embedded):
      user_idx, top_cat_idx, peak_hour_bucket

    Scalar appended to the dense vector:
      has_purchase_history  (0/1 int treated as float)

    Dense inputs (pre-scaled, 8-dim):
      log_total_events, months_active, purchase_rate, cart_rate,
      log_n_sessions, avg_purchase_price_scaled,
      sin(2π·preferred_dow/7),  cos(2π·preferred_dow/7)

    MLP input dim: embed_dim_user + embed_dim_cat + embed_dim_small
                   + dense_input_dim + 1  (the +1 is has_purchase_history)
                 = 32 + 16 + 8 + 8 + 1 = 65  (with defaults)
    """

    def __init__(
        self,
        n_users: int,
        n_top_cats: int,
        n_hour_buckets: int = 4,
        embed_dim_user: int = 32,
        embed_dim_cat: int = 16,
        embed_dim_small: int = 8,
        dense_input_dim: int = 8,   # 6 original + 2 sin/cos DOW
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.user_emb    = nn.Embedding(n_users,       embed_dim_user)
        self.top_cat_emb = nn.Embedding(n_top_cats,    embed_dim_cat)
        self.hour_emb    = nn.Embedding(n_hour_buckets, embed_dim_small)

        mlp_input_dim = (
            embed_dim_user + embed_dim_cat + embed_dim_small
            + dense_input_dim + 1   # +1 for has_purchase_history scalar
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
        for emb in (self.user_emb, self.top_cat_emb, self.hour_emb):
            _init_embedding(emb)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(
        self,
        user_idx: torch.Tensor,   # (B,)
        user_cat: torch.Tensor,   # (B, 4) — [top_cat_idx, hour_bucket, preferred_dow, has_purchase]
        user_dense: torch.Tensor, # (B, 8) — 6 original + sin_dow + cos_dow
        user_seq: torch.Tensor | None = None,  # ignored; accepted for API compat
    ) -> torch.Tensor:            # (B, output_dim)
        e_user    = self.user_emb(user_idx)           # (B, 32)
        e_top_cat = self.top_cat_emb(user_cat[:, 0])  # (B, 16)
        e_hour    = self.hour_emb(user_cat[:, 1])     # (B, 8)
        has_purch = user_cat[:, 3:4].float()          # (B, 1)

        x = torch.cat([e_user, e_top_cat, e_hour, user_dense, has_purch], dim=1)
        x = self.mlp(x)
        return F.normalize(x, dim=1)


# ── Item Tower V2 (hierarchical cat residual + 2 extra dense features) ─────────

class ItemTowerV2(nn.Module):
    """V2 item tower: hierarchical category residual + two new dense features.

    Categorical inputs (embedded):
      item_idx, cat_l1_idx, cat_l2_idx, brand_idx, price_bucket

    Hierarchical cat residual:
      cat_l2_final = cat_l2_emb(cat_l2_idx) + cat_l1_emb(cat_l1_idx)
      This initialises sub-category embeddings biased toward their parent,
      reducing the learning burden for rare sub-categories.

    Dense inputs (pre-scaled, 5-dim):
      avg_price_scaled, log_confidence_scaled, purchase_rate_scaled,
      price_relative_to_cat_avg_scaled, product_recency_log_scaled

    MLP input dim: embed_dim_item + 3*embed_dim_cat + embed_dim_small
                   + dense_input_dim
                 = 32 + 48 + 8 + 5 = 93  (with defaults)
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
        dense_input_dim: int = 5,   # 3 original + price_rel + recency
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
        item_dense: torch.Tensor, # (B, 5)
    ) -> torch.Tensor:            # (B, output_dim)
        e_item   = self.item_emb(item_cat[:, 0])    # (B, 32)
        e_cat_l1 = self.cat_l1_emb(item_cat[:, 1]) # (B, 16)
        # Hierarchical residual: cat_l2 embedding gets its parent's signal added
        e_cat_l2 = self.cat_l2_emb(item_cat[:, 2]) + e_cat_l1  # (B, 16)
        e_brand  = self.brand_emb(item_cat[:, 3])   # (B, 16)
        e_price  = self.price_emb(item_cat[:, 4])   # (B, 8)

        x = torch.cat([e_item, e_cat_l1, e_cat_l2, e_brand, e_price, item_dense], dim=1)
        x = self.mlp(x)
        return F.normalize(x, dim=1)


# ── Sequential User Tower (GRU over item history) ─────────────────────────────

class SequentialUserTower(nn.Module):
    """User tower that encodes the user's last-N interacted items with a GRU.

    Architecture:
      1. Item sequence (last seq_len items, left-padded with 0):
           item_seq_emb → GRU → last hidden state  (gru_hidden,)
      2. Static features:
           user_emb(user_idx)  (embed_dim_user,)
           dense vector        (dense_input_dim,)  — same 8-dim as UserTowerV2
           has_purchase_history scalar  (1,)
      3. Concat → MLP → output_dim, L2-normalised.

    Weight tying: after model construction in the training script, set
        model.user_tower.item_seq_emb.weight = model.item_tower.item_emb.weight
    so that history items and candidate items share the same embedding space.

    Args:
        n_users:        Vocabulary size for user IDs.
        n_items:        Vocabulary size for item IDs (must match ItemTower).
        seq_len:        Length of the item history sequence (default 20).
        gru_hidden:     GRU hidden state dimension (default 128).
        item_embed_dim: Embedding dimension for sequence items; should equal
                        ItemTower.embed_dim_item so weight tying works (default 32).
        embed_dim_user: User ID embedding dimension (default 32).
        dense_input_dim: Static dense feature dimension (default 8, same as V2).
        hidden_dim:     MLP hidden layer width (default 256).
        output_dim:     Final embedding dimension (default 64).
        dropout:        Dropout rate in the MLP (default 0.1).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        seq_len: int = 20,
        gru_hidden: int = 128,
        item_embed_dim: int = 32,
        embed_dim_user: int = 32,
        dense_input_dim: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq_len    = seq_len
        self.gru_hidden = gru_hidden

        # Item history embedding — weight will be tied to item_tower.item_emb
        self.item_seq_emb = nn.Embedding(n_items, item_embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size  = item_embed_dim,
            hidden_size = gru_hidden,
            batch_first = True,
        )

        self.user_emb = nn.Embedding(n_users, embed_dim_user)

        # MLP: user_emb + gru_hidden + dense + has_purchase
        mlp_input_dim = embed_dim_user + gru_hidden + dense_input_dim + 1

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        _init_embedding(self.user_emb)
        _init_embedding(self.item_seq_emb)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(
        self,
        user_idx: torch.Tensor,   # (B,)
        user_cat: torch.Tensor,   # (B, 4) — has_purchase_history at index 3
        user_dense: torch.Tensor, # (B, 8) — same 8-dim dense as UserTowerV2
        user_seq: torch.Tensor,   # (B, seq_len) — item_idx history, 0 = padding
    ) -> torch.Tensor:            # (B, output_dim)
        # ── Sequence branch ──────────────────────────────────────────────────
        seq_emb = self.item_seq_emb(user_seq)       # (B, seq_len, item_embed_dim)
        _, hidden = self.gru(seq_emb)               # hidden: (1, B, gru_hidden)
        gru_out = hidden.squeeze(0)                 # (B, gru_hidden)

        # ── Static branch ────────────────────────────────────────────────────
        e_user    = self.user_emb(user_idx)         # (B, 32)
        has_purch = user_cat[:, 3:4].float()        # (B, 1)

        x = torch.cat([e_user, gru_out, user_dense, has_purch], dim=1)
        x = self.mlp(x)
        return F.normalize(x, dim=1)


# ── User Tower V3 (sin/cos DOW + 32-dim item centroid) ────────────────────────

class UserTowerV3(nn.Module):
    """V3 user tower: UserTowerV2 + a pre-computed 32-dim item-centroid feature.

    The centroid is the mean of ``item_tower.item_emb`` vectors over the user's
    cart+purchase history, computed offline by
    ``scripts/two_tower/build_users_v2_500k.py`` and stored as
    ``item_centroid_0 .. item_centroid_31`` in ``users_encoded_v2.parquet``.

    Categorical inputs (embedded):
      user_idx, top_cat_idx, peak_hour_bucket

    Dense inputs (40-dim):
      [0:6]   log_total_events, months_active, purchase_rate, cart_rate,
              log_n_sessions, avg_purchase_price_scaled
      [6:8]   sin(2π·preferred_dow/7),  cos(2π·preferred_dow/7)
      [8:40]  item_centroid_0 .. item_centroid_31  (zero if no high-intent history)

    Scalar appended separately:
      has_purchase_history  (0/1 → float)

    MLP input dim: embed_dim_user + embed_dim_cat + embed_dim_small
                   + dense_input_dim + 1
                 = 32 + 16 + 8 + 40 + 1 = 97  (with defaults)
    """

    CENTROID_DIM: int = 32   # must match item_tower.item_emb embed_dim (32)

    def __init__(
        self,
        n_users: int,
        n_top_cats: int,
        n_hour_buckets: int = 4,
        embed_dim_user: int = 32,
        embed_dim_cat: int = 16,
        embed_dim_small: int = 8,
        dense_input_dim: int = 40,   # 8 (V2) + 32 (centroid)
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.user_emb    = nn.Embedding(n_users,        embed_dim_user)
        self.top_cat_emb = nn.Embedding(n_top_cats,     embed_dim_cat)
        self.hour_emb    = nn.Embedding(n_hour_buckets, embed_dim_small)

        mlp_input_dim = (
            embed_dim_user + embed_dim_cat + embed_dim_small
            + dense_input_dim + 1   # +1 for has_purchase_history scalar
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
        for emb in (self.user_emb, self.top_cat_emb, self.hour_emb):
            _init_embedding(emb)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                _init_linear(module)

    def forward(
        self,
        user_idx: torch.Tensor,   # (B,)
        user_cat: torch.Tensor,   # (B, 4) — [top_cat_idx, hour_bucket, preferred_dow, has_purchase]
        user_dense: torch.Tensor, # (B, 40) — 8 V2 features + 32-dim centroid
        user_seq: torch.Tensor | None = None,  # ignored; accepted for API compat
    ) -> torch.Tensor:            # (B, output_dim)
        e_user    = self.user_emb(user_idx)           # (B, 32)
        e_top_cat = self.top_cat_emb(user_cat[:, 0])  # (B, 16)
        e_hour    = self.hour_emb(user_cat[:, 1])     # (B, 8)
        has_purch = user_cat[:, 3:4].float()          # (B, 1)

        x = torch.cat([e_user, e_top_cat, e_hour, user_dense, has_purch], dim=1)
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
        user_tower: "UserTower | UserTowerV2 | UserTowerV3 | SequentialUserTower",
        item_tower: "ItemTower | ItemTowerV2",
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.user_tower  = user_tower
        self.item_tower  = item_tower
        self.temperature = temperature

    def forward(
        self,
        user_idx: torch.Tensor,            # (B,)
        user_cat: torch.Tensor,            # (B, 4)
        user_dense: torch.Tensor,          # (B, 6 or 8)
        item_cat: torch.Tensor,            # (B, 5)
        item_dense: torch.Tensor,          # (B, 3 or 5)
        user_seq: torch.Tensor | None = None,  # (B, seq_len) — sequential tower only
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run both towers and return embeddings plus the in-batch score matrix.

        Returns:
            user_embeddings: (B, 64)
            item_embeddings: (B, 64)
            scores:          (B, B) scaled dot-product matrix
        """
        if user_seq is not None:
            user_embeddings = self.user_tower(user_idx, user_cat, user_dense, user_seq)
        else:
            user_embeddings = self.user_tower(user_idx, user_cat, user_dense)
        item_embeddings = self.item_tower(item_cat, item_dense)
        scores = (user_embeddings @ item_embeddings.T) / self.temperature
        return user_embeddings, item_embeddings, scores

    def get_user_embedding(
        self,
        user_idx: torch.Tensor,            # (B,)
        user_cat: torch.Tensor,            # (B, 4)
        user_dense: torch.Tensor,          # (B, 6 or 8)
        user_seq: torch.Tensor | None = None,  # (B, seq_len) — sequential tower only
    ) -> torch.Tensor:                     # (B, 64)
        """Encode users only. Used at inference time."""
        if user_seq is not None:
            return self.user_tower(user_idx, user_cat, user_dense, user_seq)
        return self.user_tower(user_idx, user_cat, user_dense)

    def get_item_embeddings(
        self,
        item_cat: torch.Tensor,   # (N, 5)
        item_dense: torch.Tensor, # (N, 3 or 5)
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

        tower_name = type(self.user_tower).__name__

        print("=" * 40)
        print("TwoTowerModel — parameter summary")
        print("=" * 40)
        print(f"  User tower ({tower_name})")
        print(f"           : {user_count:>12,}")
        print(f"  Item tower : {item_count:>12,}")
        print(f"  Total      : {total:>12,}")
        print("=" * 40)
