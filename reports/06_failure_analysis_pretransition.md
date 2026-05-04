# RecoSys Failure Analysis — Pre-Session-Based Transition

**Scope:** This report consolidates and replaces `reports/06_prior_work_comparative_analysis.md` and `reports/07_sequence_model_results.md`. It documents all modeling approaches attempted before the session-based reframe, their results, root causes of failure, and the reasoning that led to the pivot.

---

## 1. Dataset & Problem Context

**Dataset:** REES46 eCommerce (Kaggle: ecommerce-behavior-data-from-multi-category-store)
- 5 months of data (Oct 2019 – Mar 2020), 279.9M raw rows cleaned
- **500k user sample:** 445,150 users, 284,523 items (after iterative k-core-3 filtering), 18.5M events
- Events: `view` (confidence 1), `cart` (confidence 2), `purchase` (confidence 4)
- Temporal split: Train Oct 2019 – Jan 2020, Test Feb 2020
- Eval users: 39,516 (users with ≥1 cart/purchase event in Feb 2020)
- Ground truth: cart + purchase events in Feb, deduplicated per (user, item)

**Preprocessing advantages over all published sources:**
- Only project with explicit bot removal (>300 events/day threshold, 285 users removed)
- Only project with iterative k=3 user+item core filtering
- 1-second near-dedup window; $1.00 price floor; "unknown" encoding for null categories
- Sampling by user (full user histories preserved), not by time — correct for user-based retrieval

---

## 2. Prior Work Baseline (T4Rec Paper — Same Dataset)

The Transformers4Rec paper (Moreira et al., RecSys 2021) is the primary published benchmark on REES46:

| Model | NDCG@20 | HR@20 |
|---|---|---|
| V-SkNN | 0.2187 | 0.4662 |
| GRU4Rec (Full Training) | 0.2231 | 0.4414 |
| XLNet (PLM) | 0.2422 | 0.4760 |
| **XLNet (RTD) — their best** | **0.2546** | **0.4886** |

**Important caveats for comparison:**
- T4Rec uses Oct 2019 only (1 month, 156K items, 3.2M sessions)
- T4Rec uses incremental day-by-day evaluation (Average over Time protocol)
- Our project uses 5 months, 284K items, single temporal holdout
- T4Rec GRU4Rec is item-id only; ours includes event-type side features
- These are **session-based** metrics (next-item in session), not user-based retrieval

---

## 3. Phase 7: Two-Tower User-Based Retrieval (V1–V6) — Failed to Beat Popularity

**Objective:** Given a user's aggregate interaction history (Oct–Jan), retrieve the top-K items they will interact with in Feb 2020.

**Task framing:** User embedding (tower 1) × Item embedding (tower 2) → dot product → FAISS retrieval. In-batch negatives with L2 normalization.

**Evaluation:** Recall@10, NDCG@10 against Feb 2020 cart+purchase ground truth, filter_seen=True (items seen in training removed from candidates), 39,516 eval users.

**Baselines established:**

| Baseline | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---|---|---|---|
| Global Popularity | **0.0646** | 0.0411 | 0.0890 | 0.0480 |
| CoOcc kNN | 0.0598 | 0.0375 | 0.0864 | 0.0452 |
| Per-Category Popularity | 0.0341 | 0.0231 | 0.0467 | 0.0267 |

**Two-Tower results (best runs):**

| Version | R@10 | NDCG@10 | R@20 | Notes |
|---|---|---|---|---|
| V5 | 0.0515 | 0.0331 | 0.0757 | Best early config |
| V6 | 0.0517 | 0.0329 | 0.0748 | Final two-tower attempt |

**All Two-Tower versions failed to beat Global Popularity (R@10=0.0646).**

**Root cause analysis:** The user-based retrieval task conditions on aggregate Oct–Jan history to predict Feb behavior. The signal is sparse — 62.8% of users appear in only one month, sessions are short (T4Rec paper reports avg=5.49 for REES46). User aggregate features do not capture the temporal session context that drives next-item decisions in e-commerce. The popularity baseline is very strong because the catalog is long-tail (Gini≈0.86) and recent purchase intent is better captured by session context than user history aggregates.

---

## 4. Phase 8 (First Iteration): Sequential Models on User-History Objective — Failed

After the Two-Tower failure, sequential models (GRU4Rec V7, SASRec V8) were trained on the same user-history objective with sampled softmax loss.

**Config:** embed_dim=64, max_seq_len=50, K_neg=512, temperature=0.07, AdamW, CosineAnnealingLR 30 epochs, eval_every=5 epochs on Feb test set, filter_seen=True.

### GRU4Rec V7 — User History Objective

| Epoch | Train Loss | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---|---|---|---|---|
| 5 | 0.7548 | **0.0535** | 0.0335 | 0.0762 | 0.0401 |
| 10 | 0.6822 | 0.0529 | 0.0327 | 0.0758 | 0.0394 |
| 15 | 0.6446 | 0.0524 | 0.0327 | 0.0764 | 0.0397 |
| 20 | 0.6174 | 0.0518 | 0.0321 | 0.0761 | 0.0392 |
| 25 | 0.5994 | 0.0520 | 0.0321 | 0.0762 | 0.0391 |
| 30 | 0.5909 | 0.0515 | 0.0319 | 0.0764 | 0.0391 |

Best: R@10=0.0535 at epoch 5. **Below Global Popularity (0.0646).** Performance degrades after epoch 5 — classic overfitting pattern. Warm users (51+ training interactions) perform worst (R@10=0.0208), indicating the model fits to interaction frequency rather than purchase intent.

### SASRec V8 — User History Objective

| Epoch | Train Loss | R@10 | NDCG@10 |
|---|---|---|---|
| 5 | 0.7586 | 0.0113 | 0.0074 |

Killed at epoch 6. R@10=0.0113 — 5× worse than GRU4Rec at same epoch. **Dramatically below popularity.**

**Root cause:** Attention rank collapse. With L2 normalization and sampled softmax, SASRec's transformer encoder produces near-constant output vectors across users (uniform attention weights on long user histories with sparse interaction signal). Established in literature: SASRec is designed for longer user-history tasks (ML-1M, Steam avg 73 interactions); on REES46's avg 5.49 interactions per session it has insufficient sequential context to leverage attention effectively.

---

## 5. Why the Session-Based Reframe Was Necessary

**Key insight:** The evaluation failure was not a model hyperparameter problem — it was a task formulation problem.

The user-history objective asks: "Given a user's full Oct–Jan history, what will they buy in Feb?" This is a poor task for REES46 because:
1. Sessions are the unit of purchase intent, not user histories. A user who bought electronics in October and furniture in January has fundamentally different purchase intent in each session.
2. 62.8% of users appear in only one calendar month — their "history" is a single short session.
3. The global popularity baseline (R@10=0.0646) is hard to beat because item frequency is heavily skewed (Gini=0.86) and without session context, predicting popular items is the best strategy.

**The session-based framing** directly addresses all of these: given a user's within-session history (up to 20 interactions), predict the next item. This is what the T4Rec paper optimizes. It captures real-time purchase intent and leverages the most informative signal available.

---

## 6. Summary of All Pre-Transition Results

| Model | Task | Best R@10 | Best NDCG@20 | vs. Pop Baseline | Status |
|---|---|---|---|---|---|
| Two-Tower V1–V4 | User-based retrieval | <0.05 | <0.035 | Below | Killed |
| Two-Tower V5 | User-based retrieval | 0.0515 | — | Below (0.0646) | Killed |
| Two-Tower V6 | User-based retrieval | 0.0517 | 0.0394 | Below | Final attempt |
| GRU4Rec V7 | User-history sequential | 0.0535 | 0.0401 | Below | Best pre-pivot |
| SASRec V8 | User-history sequential | 0.0113 | 0.0090 | Dramatically below | Killed ep.6 |

**Decision:** Reframe to session-based next-item prediction, matching the task for which published baselines and the dataset's natural structure are optimized.
