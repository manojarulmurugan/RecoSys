# GRU4Rec V9 — 1M Sample Training Results
## Vertex AI Job: `3348631089810767872`
**Date:** 2026-05-03 | **Duration:** 10h 46m | **Hardware:** A100 40GB (a2-highgpu-1g)

---

## 1. Objective

Scale GRU4Rec V9 (previously validated on 500k users, NDCG@20=0.2606) to 1M users to confirm performance gains from larger training data and meet T4Rec paper benchmarks.

---

## 2. Configuration

| Parameter | Value |
|---|---|
| Model | GRU4Rec V9 |
| embed_dim | 128 |
| gru_hidden | 256 |
| n_layers | 1 |
| dropout | 0.3 |
| batch_size | 256 |
| learning_rate | 3e-4 |
| weight_decay | 1e-5 |
| temperature | 0.07 |
| label_smoothing | 0.1 |
| grad_clip | 1.0 |
| scheduler | cosine (lr_min=1e-5) |
| max_epochs | 30 |
| patience | 5 |
| seed | 42 |
| loss | Full softmax |

---

## 3. Dataset Statistics

| Metric | Value |
|---|---|
| Users | 890,736 |
| Items (incl PAD) | 222,864 |
| Train sessions | 2,884,945 |
| Val sessions | 151,177 |
| max_seq_len | 20 |
| Train batches/epoch | 11,270 |
| FAISS index items | 209,092 |
| Training speed | ~9.4–9.5 steps/s |
| Time per epoch | ~19m 52s |

---

## 4. Training Progression (Val NDCG@20 per epoch)

| Epoch | Train Loss | NDCG@20 | Best |
|---|---|---|---|
| 1 | 7.6711 | 0.2381 | ← |
| 2 | 7.2016 | 0.2492 | ← |
| 3 | 7.1148 | 0.2537 | ← |
| 4 | 7.0667 | 0.2588 | ← |
| 5 | 7.0366 | 0.2602 | ← |
| 6 | 7.0149 | 0.2617 | ← |
| 7 | 6.9982 | 0.2614 | |
| 8 | 6.9847 | 0.2628 | ← |
| 9 | 6.9729 | 0.2638 | ← |
| 10 | 6.9631 | 0.2646 | ← |
| 11 | 6.9543 | 0.2652 | ← |
| 12 | 6.9467 | 0.2650 | |
| 13 | 6.9394 | 0.2652 | ← |
| 14 | 6.9327 | 0.2659 | ← |
| 15 | 6.9267 | 0.2666 | ← |
| 16 | 6.9210 | 0.2663 | |
| 17 | 6.9160 | 0.2667 | ← |
| 18 | 6.9111 | 0.2661 | |
| 19 | 6.9062 | 0.2665 | |
| 20 | 6.9018 | 0.2674 | ← |
| 21 | 6.8979 | 0.2670 | |
| 22 | 6.8942 | 0.2673 | |
| 23 | 6.8907 | 0.2672 | |
| **24** | **6.8873** | **0.2676** | **← BEST** |
| 25 | 6.8844 | 0.2671 | |
| 26 | 6.8820 | 0.2672 | |
| 27 | 6.8796 | 0.2673 | |
| 28 | 6.8779 | 0.2675 | |
| 29 | 6.8768 | 0.2674 | patience 5/5 |

Early stopping triggered after epoch 29. Best checkpoint: epoch 24.

---

## 5. Final Metrics (Best Checkpoint — Epoch 24)

| Metric | Model | Pop Baseline | T4Rec Target | Status |
|---|---|---|---|---|
| HR@10 | 0.3803 | 0.0579 | — | — |
| NDCG@10 | 0.2420 | 0.0296 | — | — |
| HR@20 | 0.4815 | 0.0806 | ≥ 0.44 | ✅ |
| NDCG@20 | **0.2676** | 0.0353 | ≥ 0.22 | ✅ |

Model outperforms population baseline by **6.0× on HR@20** and **7.6× on NDCG@20**.

---

## 6. Comparison: 500k vs 1M Sample

| Metric | GRU4Rec V9 (500k) | GRU4Rec V9 (1M) | Δ |
|---|---|---|---|
| NDCG@20 | 0.2606 | 0.2676 | +0.0070 (+2.7%) |
| HR@20 | ~0.449 (est.) | 0.4815 | +~0.033 |
| Best Epoch | ~15 (est.) | 24 | |
| Train Sessions | 1,452,472 | 2,884,945 | +99% |

Scaling from 500k to 1M users provides a consistent +2.7% NDCG@20 improvement, confirming that larger data benefits GRU4Rec.

---

## 7. Artifacts

| Artifact | Location |
|---|---|
| Best checkpoint | `gs://recosys-data-bucket/models/gru4rec_session_v9_1M/best_checkpoint.pt` |
| Latest checkpoint | `gs://recosys-data-bucket/models/gru4rec_session_v9_1M/latest_checkpoint.pt` |
| Training log (JSON) | `gs://recosys-data-bucket/models/gru4rec_session_v9_1M/training_log.json` |
| Hyperparams | `gs://recosys-data-bucket/models/gru4rec_session_v9_1M/hparams.json` |
| Training data | `gs://recosys-data-bucket/data/1M/` |
| Docker image | `us-central1-docker.pkg.dev/recosys-489001/recosys-images/gru4rec:v9` |

---

## 8. Infrastructure Notes

- **GPU**: NVIDIA A100 40GB (`a2-highgpu-1g`, us-central1)
- **Vertex AI Job ID**: `3348631089810767872`
- **Container**: `gru4rec:v9` (PyTorch 2.1.0-cuda11.8, faiss-cpu==1.7.4)
- **Key fix from prior attempts**: batch_size reduced 512→256 to prevent CUDA OOM (full softmax over 222,864 items requires ~4GB VRAM at batch=256 vs ~9GB at batch=512)
- **GCS IAM**: `roles/storage.objectAdmin` granted to Vertex AI control-plane SA (`service-921967012784@gcp-sa-aiplatform-cc.iam.gserviceaccount.com`) for checkpoint uploads
- Artifacts downloaded from GCS to `/tmp/gru4rec_artifacts/` at job start; checkpoints synced to GCS after each epoch

---

## 9. Conclusion

GRU4Rec V9 trained on 1M users achieves **NDCG@20 = 0.2676** and **HR@20 = 0.4815**, both exceeding T4Rec paper targets (≥0.22 and ≥0.44 respectively). The 2.7% NDCG improvement over the 500k baseline confirms data scaling benefits. Training converged stably over 29 epochs with cosine LR annealing, early stopping triggered at epoch 29. The model is ready for serving (Days 8–14: Cloud Run deployment, MLflow experiment tracking, drift monitoring).
