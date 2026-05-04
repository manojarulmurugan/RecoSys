# Sequence Model Results — GRU4Rec (V7) & SASRec (V8)

---

## Dataset

- **Source:** REES46 e-commerce dataset, 500k user sample
- **Catalog:** 284,523 items (after k-core-3 filtering), 445,150 users
- **Events:** `view` (confidence 1), `cart` (confidence 2), `purchase` (confidence 4)
- **Total training interactions:** 7,473,130 rows in `train_pairs.parquet`
- **Raw test rows:** 3,451,452 → 2,722,179 after id-mapping (152,037 users, 133,098 items)

---

## Temporal Splits

| Split | Date range | Purpose |
|---|---|---|
| **Training** | Oct 2019 – Jan 31 2020 | Model training (all sequence events) |
| **Test** | Feb 2020 | Evaluation only — never seen by model during training |

The training sequences are stored in `full_train_seqs.parquet` (435,953 users with ≥1 event in training window). The Feb test raw events are loaded directly from GCS at evaluation time.

---

## Training Protocol (both models)

**Sequence construction:**
- Per-user events are sorted chronologically and truncated to the last 50 interactions (`max_seq_len=50`)
- Sequences are **left-padded** with zeros to length 50 (PAD token = 0)
- Event type is encoded as a side feature: `view=1, cart=2, purchase=3, PAD=0`

**Training objective — Sampled Softmax:**
- At each position `t` in the sequence, the model sees items `0..t-1` (input) and must predict item `t` (target)
- This is implemented as a **shifted-target** approach: `input_seq = seq[:-1]`, `target_seq = seq[1:]`
- Loss = cross-entropy over 1 positive + K=512 uniformly sampled random negatives
- Padded positions are masked from the loss
- Optimizer: AdamW, embedding weight decay = 0, all other params weight decay = 1e-5
- LR schedule: CosineAnnealingLR over 30 epochs, eta_min=1e-5

**FAISS index at eval:**
- Item embeddings are L2-normalised before indexing
- Only the 201,648 items present in `train_pairs.parquet` are indexed (items with ≥1 interaction in training)
- User embeddings are computed from the **full training sequence** (`full_train_seqs`) via the model's `encode_sequence()` method (last-position output)

---

## Evaluation Protocol (both models)

**Ground truth:** cart and purchase events from Feb 2020, deduplicated per (user, item). Users with no cart/purchase events in Feb are excluded. **Eval users: 39,516.**

**Filter:** `filter_seen=True` — items the user interacted with in `train_pairs` (any event type, Oct–Jan) are removed from the FAISS retrieval results before computing metrics. This matches the V1–V6 Two-Tower protocol. 34.1% of ground-truth items are in the seen-item set and are therefore invisible to retrieval.

**Retrieval:** FAISS IndexFlatIP (inner product on L2-normalised vectors). Top-N candidates are retrieved per user (`n_faiss_candidates`), seen items are filtered, and the top-20 remaining items are used for metric computation.

**Metrics:** Recall@K and NDCG@K at K=10 and K=20, averaged over eval users. Stratified by training interaction count: Cold (3–10), Medium (11–50), Warm (51+).

**Periodic evaluation:** Every 5 epochs, the Feb test set is evaluated using the current model weights. This is the same test set used for the final evaluation — it is not a held-out validation set.

---

## Baselines to Beat

| Model | R@10 | NDCG@10 | R@20 | NDCG@20 | Hit%@10 |
|---|---|---|---|---|---|
| Baseline 1: Global Popularity | **0.0646** | 0.0411 | 0.0890 | 0.0480 | 10.35% |
| Baseline 3: CoOcc kNN | 0.0598 | 0.0375 | 0.0864 | 0.0452 | 9.64% |
| Two-Tower V6 (best) | 0.0517 | 0.0329 | 0.0748 | 0.0394 | — |
| Two-Tower V5 | 0.0515 | 0.0331 | 0.0757 | — | — |
| Baseline 2: Per-Category Popularity | 0.0341 | 0.0231 | 0.0467 | 0.0267 | 5.63% |

---

## GRU4Rec (V7) — Run 1

**Config:** embed_dim=64, gru_hidden=128, layers=2, max_seq_len=50,
batch_size=256, lr=1e-3, wd=1e-5 (zero on embeddings), K_neg=512,
**T=0.07**, n_faiss_candidates=200, epochs=30, eval_every=5  
Training data: `full_train_seqs.parquet` (Oct–Jan)

### Periodic test evals

| Epoch | Train Loss | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---|---|---|---|---|
| 5 | 0.7548 | **0.0535** | 0.0335 | 0.0762 | 0.0401 |
| 10 | 0.6822 | 0.0529 | 0.0327 | 0.0758 | 0.0394 |
| 15 | 0.6446 | 0.0524 | 0.0327 | 0.0764 | 0.0397 |
| 20 | 0.6174 | 0.0518 | 0.0321 | 0.0761 | 0.0392 |
| 25 | 0.5994 | 0.0520 | 0.0321 | 0.0762 | 0.0391 |
| 30 | 0.5909 | 0.0515 | 0.0319 | 0.0764 | 0.0391 |

**Best epoch: 5** — R@10 = 0.0535 (`epoch_05.pt`)

### Final test — epoch 30 checkpoint (stratified)

| Cohort | N users | R@10 | NDCG@10 | R@20 | NDCG@20 | Hit%@10 |
|---|---|---|---|---|---|---|
| Overall | 39,516 | 0.0515 | 0.0319 | 0.0764 | 0.0391 | 8.3% |
| Cold (3–10) | 9,378 | 0.0728 | 0.0449 | 0.1084 | 0.0551 | 11.0% |
| Medium (11–50) | 16,634 | 0.0393 | 0.0249 | 0.0601 | 0.0311 | 7.1% |
| Warm (51+) | 9,030 | 0.0208 | 0.0137 | 0.0305 | 0.0167 | 4.5% |

---

## SASRec (V8) — Run 1

**Config:** embed_dim=64, heads=2, ffn=256, layers=2, max_seq_len=50,
batch_size=256, lr=1e-3, wd=1e-5 (zero on embeddings), K_neg=512,
**T=0.07**, n_faiss_candidates=200, epochs=30 (killed at epoch 6)  
Training data: `full_train_seqs.parquet` (Oct–Jan)

| Epoch | Train Loss | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---|---|---|---|---|
| 5 | 0.7586 | 0.0113 | 0.0074 | 0.0166 | 0.0090 |

Run killed at epoch 6 step 200.

---

## GRU4Rec (V7) — Run 2

**Config:** same as Run 1 except **T=0.20**, n_faiss_candidates=400  
Training data: `full_train_seqs.parquet` (Oct–Jan)

| Epoch | Train Loss | R@10 | NDCG@10 | R@20 | NDCG@20 |
|---|---|---|---|---|---|
| 5 | 1.7020 | 0.0365 | 0.0222 | 0.0566 | 0.0280 |

Run killed at epoch 6 step 200.

---

## Summary Table

| Run | Model | T | n_faiss_cand | Best R@10 | Best epoch |
|---|---|---|---|---|---|
| V7 Run 1 | GRU4Rec | 0.07 | 200 | 0.0535 | 5 |
| V8 Run 1 | SASRec | 0.07 | 200 | 0.0113 | 5 |
| V7 Run 2 | GRU4Rec | 0.20 | 400 | 0.0365 | 5 |

All three runs are **below Global Popularity (0.0646)**.  
GRU4Rec Run 1 (T=0.07) is the best sequential result so far: **R@10 = 0.0535** at epoch 5.
