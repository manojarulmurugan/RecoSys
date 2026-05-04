# RecoSys Day 5–14 Continuation Brief

**Date:** 2026-05-02  
**Current status:** Day 4 complete. GRU4Rec V9 session-based model trained and evaluated on 500k sample. SASRec confirmed negative result (5 attempts). Ready to scale and deploy.

---

## What Has Been Done (Days 1–4)

### Data Pipeline
- REES46 eCommerce dataset, 5 months (Oct 2019 – Mar 2020), 279.9M rows
- Cleaning: null handling, 1s near-dedup, $1 price floor, bot removal (>300 events/day), iterative k=3 core filtering
- 500k user sample: 445,150 users, 284,523 items, 18.5M events
- Session construction: group events into sessions (30-min gap), min_len=2, max_len=20, consecutive repeats removed
- Session parquets: 1,450,480 train sessions, 76,304 val sessions, 266,913 test sessions
- Artifacts at: `artifacts/500k/sequences_v2/` (train/val/test parquets + metadata.json + vocabs.pkl)

### Models Built

**Two-Tower V1–V6 (user-history retrieval):** All failed to beat Global Popularity (R@10=0.0646). Best was V6 at R@10=0.0517. Task mismatch: aggregate user history is a poor signal for session-driven purchase intent on REES46.

**GRU4Rec V7 + SASRec V8 (user-history sequential):** GRU4Rec best R@10=0.0535 (below pop), SASRec R@10=0.0113. Both below popularity on Feb 2020 holdout with filter_seen=True.

**GRU4Rec V9 (session-based) — WINNER:**
- Architecture: embed_dim=128, gru_hidden=256, 2 layers, event-type features, full softmax, L2 norm, temp=0.07
- Training: 1,450,480 sessions, warmup+cosine LR, AdamW
- **Result: NDCG@20 = 0.2606** — exceeds T4Rec paper's best model (XLNet+RTD 0.2546) on the same REES46 dataset
- GCS checkpoint: `gs://recosys-data-bucket/models/gru4rec_session_v9/`

**SASRec V10 (session-based) — Confirmed Negative Result:**
- 5 attempts, all failed (best NDCG@20=0.0044 vs pop 0.034)
- Root causes: attention rank collapse, geometric loss floor, gradient imbalance, embedding collapse
- Consistent with Ludewig & Jannach (2019), Hidasi & Czapp (2023)
- Documented in `reports/09_session_model_results.md`

### Key Files
- `src/sequence/models/gru4rec.py` — GRU4Rec model
- `src/sequence/models/sasrec.py` — SASRec model
- `src/sequence/training/train_sequence.py` — training loops
- `src/sequence/data/session_dataset.py` — SessionTrainDataset, SessionEvalDataset
- `src/sequence/evaluation/evaluate_sequence.py` — evaluate_sessions (normalize=True for GRU4Rec)
- `scripts/sequence/train_gru4rec_session.py` — GRU4Rec session training script
- `scripts/sequence/train_sasrec_session.py` — SASRec session training script

---

## What Needs to Be Done (Days 5–14)

### Revised Scope

The original plan included SASRec as a parallel track. That track is closed (negative result documented). Remaining work:

1. **Day 5: 1M Sample Creation** — Scale up from 500k to 1M users
2. **Day 6–7: Vertex AI Training** — Submit GRU4Rec V9 session training to Google Cloud Vertex AI
3. **Day 8–9: Model Serving** — Cloud Run serving endpoint with FAISS index
4. **Day 10–11: MLflow Experiment Tracking** — Log all metrics, hyperparams, artifacts
5. **Day 12–13: Drift Monitoring** — Basic feature drift detection pipeline
6. **Day 14: End-to-End Demo** — Full pipeline demo, documentation, portfolio writeup

---

## Day 5: 1M Sample Creation

**Goal:** Create `artifacts/1M/` with same structure as `artifacts/500k/`, scaled to 1M users.

**Steps:**
1. Pull full cleaned REES46 data from GCS (`gs://recosys-data-bucket/data/cleaned/`)
2. Sample 1M users (random seed=42 for reproducibility)
3. Run session construction pipeline (`scripts/data/build_session_sequences.py`)
4. Produce `artifacts/1M/sequences_v2/` with train/val/test parquets + metadata.json + vocabs.pkl
5. Upload to GCS: `gs://recosys-data-bucket/data/1M/`

**Expected scale:**
- ~1M users × ~40 interactions/user = ~40M events
- Training sessions: ~3–4M
- Val sessions: ~150–200K
- Items: ~284K (same catalog; k-core at 500k covers most items)

**Script to create:** `scripts/data/build_1m_sample.py`

---

## Day 6–7: Vertex AI Training

**Goal:** Submit GRU4Rec V9 session training job to Vertex AI with 1M dataset.

**Architecture decision:** GRU4Rec V9 is the model to scale. Same hyperparameters that worked at 500k; possibly increase batch_size for 1M scale.

**Steps:**
1. Create `Dockerfile.gru4rec` — PyTorch GPU container with src/ and scripts/ packaged
2. Build and push to Google Container Registry: `gcr.io/recosys-project/gru4rec:v9`
3. Create Vertex AI training job spec (`configs/vertex_gru4rec_1m.yaml`)
4. Submit via `gcloud ai custom-jobs create` or Python SDK
5. Monitor training, pull best checkpoint to `gs://recosys-data-bucket/models/gru4rec_session_v9_1M/`

**Key Dockerfile requirements:**
- Base: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- Copy: `src/`, `scripts/`, `requirements.txt`
- Install: `faiss-gpu`, `pandas`, `pyarrow`, `google-cloud-storage`
- Entrypoint: `python scripts/sequence/train_gru4rec_session.py --artifacts-dir gs://... --checkpoint-dir gs://...`

**Expected result:** NDCG@20 ≥ 0.27 (more training data should improve over 500k's 0.2606)

---

## Day 8–9: Cloud Run Serving

**Goal:** Serve GRU4Rec recommendations via REST API on Cloud Run.

**Architecture:**
- Load best checkpoint from GCS at startup
- Build FAISS IndexFlatIP over train-seen item embeddings (L2 normalized)
- Endpoint: `POST /recommend` — accepts session item sequence, returns top-K item IDs
- Endpoint: `GET /health` — returns model version, index size

**Steps:**
1. Create `src/serving/recommend.py` — FastAPI app
2. Create `Dockerfile.serving`
3. Build, push to GCR, deploy: `gcloud run deploy`
4. Test with sample session queries

---

## Day 10–11: MLflow Experiment Tracking

**Goal:** Log all experiments (GRU4Rec V9 500k, GRU4Rec V9 1M) with full metrics and artifact pointers.

**Setup:**
- MLflow tracking server on Cloud Run (or managed MLflow on Vertex AI)
- Log: hyperparams, per-epoch metrics, best NDCG@20, model artifact GCS path
- Compare 500k vs 1M run side-by-side

The `train_gru4rec_session.py` already has MLflow hooks (`_mlflow_start`, `_mlflow_log_params`, `_mlflow_log_metrics`) — just pass `--mlflow-tracking-uri`.

---

## Day 12–13: Drift Monitoring

**Goal:** Demonstrate concept drift detection.

**Plan:**
- Use Mar–Apr 2020 data (COVID-19 period, withheld during training) as the distribution-shifted test set
- Compare item popularity distribution (Jan vs. Mar 2020) using Jensen-Shannon divergence
- Compare model NDCG@20 on Mar vs. Feb holdout — expect significant degradation
- Implement `scripts/monitoring/check_drift.py`
- Portfolio narrative: "We set up an MLOps pipeline that detects when model performance degrades due to real-world distribution shift (COVID-19 in this case)"

---

## Day 14: End-to-End Demo & Documentation

**Deliverables:**
- Running Cloud Run endpoint serving GRU4Rec recommendations
- MLflow dashboard with all experiment runs
- Drift monitoring report (Feb vs. Mar 2020 performance degradation)
- Portfolio write-up including:
  - GRU4Rec V9 NDCG@20=0.2606 (vs T4Rec paper's XLNet+RTD 0.2546 on same dataset)
  - SASRec negative result (5 attempts, documented root causes)
  - 1M-scale training on Vertex AI
  - Full MLOps pipeline (Cloud Run + MLflow + drift detection)

---

## Key GCS Paths

| Resource | GCS Path |
|---|---|
| Raw cleaned data | `gs://recosys-data-bucket/data/cleaned/` |
| 500k artifacts | `gs://recosys-data-bucket/data/500k/` |
| 1M artifacts (to create) | `gs://recosys-data-bucket/data/1M/` |
| GRU4Rec V9 500k checkpoint | `gs://recosys-data-bucket/models/gru4rec_session_v9/` |
| GRU4Rec V9 1M checkpoint (to create) | `gs://recosys-data-bucket/models/gru4rec_session_v9_1M/` |

---

## Key Technical Decisions (Do Not Revisit)

1. **GRU4Rec V9 is the production model.** Architecture frozen. Same hyperparameters.
2. **SASRec is closed.** Do not attempt further. See `reports/09_session_model_results.md` for documentation.
3. **Evaluation protocol:** T4Rec protocol (no filter_seen), FAISS over train-seen items, HR@20 and NDCG@20.
4. **T4Rec comparison is directionally valid** (same REES46 dataset) but note different eval protocols in all public claims.
5. **1M scale target:** NDCG@20 ≥ 0.27.
