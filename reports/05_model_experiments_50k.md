# Two-Tower Model Experiments — 50k Sample

## Setup
- Dataset: events_sample_50k (44,559 users, 105,266 items in training)
- Evaluation: Recall@10, NDCG@10, Precision@10
- Ground truth: cart + purchase events in Feb 2020 test split
- Eval users: 4,037 (users with training history who bought/carted in Feb)
- FAISS index: scoped to trained items only (105,266 items)

---

## Experiment 1 — Baseline (checkpoints_v2)
**Config:**
- Epochs: 30
- Batch size: 1024
- Learning rate: 1e-3 → 1e-5 (cosine)
- Temperature: 0.05
- Confidence weighting: False

**Results:**
| Epoch | Train Loss | Recall@10 | NDCG@10 | Prec@10 |
|-------|-----------|-----------|---------|---------|
| 5     | 4.2615    | 0.0061    | 0.0035  | 0.0011  |
| 10    | 3.6441    | 0.0068    | 0.0037  | 0.0013  |
| 15    | 3.3491    | 0.0077    | 0.0043  | 0.0014  |
| 20    | 3.1625    | 0.0077    | 0.0041  | 0.0014  |
| 25    | 3.0483    | 0.0097    | 0.0048  | 0.0017  |
| 30    | 3.0020    | 0.0093    | 0.0047  | 0.0016  |

**Best:** Epoch 25 — Recall@10: 0.0097

---

## Experiment 2 — Confidence-Weighted Loss (checkpoints_v3)
**Config:** Same as Experiment 1 except:
- Confidence weighting: True
  (per-sample loss weighted by normalised confidence score)

**Results:**
| Epoch | Train Loss | Recall@10 | NDCG@10 | Prec@10 |
|-------|-----------|-----------|---------|---------|
| 5     | 4.1388    | 0.0098    | 0.0059  | 0.0017  |
| 10    | 3.4833    | 0.0089    | 0.0050  | 0.0015  |
| 15    | 3.1770    | 0.0097    | 0.0053  | 0.0016  |
| 20    | 2.9879    | 0.0088    | 0.0053  | 0.0015  |
| 25    | 2.8732    | 0.0084    | 0.0048  | 0.0015  |
| 30    | 2.8268    | 0.0089    | 0.0047  | 0.0015  |

**Best:** Epoch 5 — Recall@10: 0.0098

---

## Key Findings

**Confidence weighting: neutral on 50k**
Peak Recall@10 of 0.0098 (weighted) vs 0.0097 (unweighted) — difference
of 0.0001 (1 user out of 4,037). Not meaningful.

Weighting produced stronger early learning (epoch 5: 0.0098 vs 0.0061)
but degraded and plateaued lower in later epochs. Hypothesis: weighting
causes overfit to purchase-heavy pairs at the expense of view coverage,
which hurts late-stage generalisation.

**50k ceiling confirmed**
Both experiments plateau in the 0.009–0.010 range regardless of config.
Data scale is the binding constraint — 750,814 training pairs with only
44,559 users is insufficient for this task.

**Decision for 500k**
- Use confidence weighting: False (neutral evidence, simpler is better)
- FAISS scoped to trained items: True (carry this fix forward)
- Ground truth: cart + purchase (carry this forward)

---

## What Changes for 500k
- Training pairs: ~7.5M (10x more)
- Trained items in FAISS: ~220K (2x more coverage)
- Eval users: ~40K (10x more, more stable metrics)
- Batch size: 2048 (larger GPU batches = more in-batch negatives)
- Device: GPU (Colab L4)
